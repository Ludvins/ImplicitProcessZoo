from __future__ import annotations

import torch
from torch import nn

from ..priors.function_bank import PriorFunctionBank
from ..utils.empirical_covariance import empirical_cross_cov, empirical_mean, stabilize_covariance
from ..utils.linalg import right_cholesky_solve, safe_cholesky
from .kernels import RBFKernel, initialize_rbf_lengthscale


def _freeze_prior(prior: nn.Module) -> None:
    if hasattr(prior, "freeze_parameters"):
        prior.freeze_parameters()
    for param in prior.parameters():
        param.requires_grad_(False)


class BaseMatheronOperator(nn.Module):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        learn_Z: bool = False,
        enforce_exact_Z_identity: bool = True,
    ):
        super().__init__()
        if inducing_points.ndim != 2:
            raise ValueError("inducing_points must have shape [M, D].")
        if inducing_points.shape[0] < 1:
            raise ValueError("At least one inducing point is required.")
        self.learn_Z = bool(learn_Z)
        self.enforce_exact_Z_identity = bool(enforce_exact_Z_identity)
        if self.learn_Z:
            self.Z_param = nn.Parameter(inducing_points.detach().clone())
        else:
            self.register_buffer("Z_buffer", inducing_points.detach().clone())

    @property
    def Z(self) -> torch.Tensor:
        return self.Z_param if self.learn_Z else self.Z_buffer

    @property
    def num_inducing(self) -> int:
        return int(self.Z.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.Z.shape[1])

    def is_exact_inducing_input(self, X: torch.Tensor) -> bool:
        return (
            self.enforce_exact_Z_identity
            and X.shape == self.Z.shape
            and X.device == self.Z.device
            and X.dtype == self.Z.dtype
            and torch.equal(X, self.Z)
        )

    def psi(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inducing_mean(self) -> torch.Tensor:
        raise NotImplementedError

    def inducing_scale_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def mean_at(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def whitened_to_inducing(self, a: torch.Tensor) -> torch.Tensor:
        mu_Z = self.inducing_mean()
        D_Z = self.inducing_scale_matrix()
        return mu_Z + a.matmul(D_Z.T)

    def apply(
        self,
        X: torch.Tensor,
        g_X: torch.Tensor,
        g_Z: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        a = a.to(dtype=self.Z.dtype, device=self.Z.device)
        u = self.whitened_to_inducing(a)
        if self.is_exact_inducing_input(X):
            return u
        psi = self.psi(X)
        return g_X + (u - g_Z).matmul(psi.T)

    def apply_components(
        self,
        X: torch.Tensor,
        b_X: torch.Tensor,
        b_Z: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        if a.ndim != 3:
            raise ValueError("a must have shape [S, R, M].")
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        S, R, M = a.shape
        if M != self.num_inducing:
            raise ValueError("a last dimension must equal num_inducing.")
        u = self.whitened_to_inducing(a.reshape(S * R, M)).reshape(S, R, M)
        if self.is_exact_inducing_input(X):
            return u
        psi = self.psi(X)
        return b_X[:, None, :] + (u - b_Z[:, None, :]).matmul(psi.T)


class EmpiricalCovarianceMatheronOperator(BaseMatheronOperator):
    def __init__(
        self,
        base_prior: nn.Module,
        inducing_points: torch.Tensor,
        num_bank_samples: int = 512,
        jitter: float = 1e-5,
        shrinkage: float = 1e-4,
        learn_Z: bool = False,
        detach_bank_values: bool = True,
        detach_prior_grad: bool = False,
        freeze_base_prior: bool = True,
        seed: int | None = None,
        enforce_exact_Z_identity: bool = True,
    ):
        if num_bank_samples < 2:
            raise ValueError("num_bank_samples must be at least 2.")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1].")
        if freeze_base_prior:
            _freeze_prior(base_prior)
        super().__init__(
            inducing_points,
            learn_Z=learn_Z,
            enforce_exact_Z_identity=enforce_exact_Z_identity,
        )
        self.base_prior = base_prior
        self.num_bank_samples = int(num_bank_samples)
        self.jitter = float(jitter)
        self.shrinkage = float(shrinkage)
        self.detach_bank_values = bool(detach_bank_values and not learn_Z)
        self.detach_prior_grad = bool(detach_prior_grad)
        self.bank = PriorFunctionBank(
            prior=base_prior,
            num_bank_samples=self.num_bank_samples,
            seed=seed,
            detach=self.detach_bank_values,
            detach_prior_grad=self.detach_prior_grad,
        )
        bank_Z, mu_Z, K_ZZ_raw, K_ZZ, L_ZZ = self._compute_Z_moments()
        self.register_buffer("bank_Z", bank_Z.detach().clone())
        self.register_buffer("mu_Z", mu_Z.detach().clone())
        self.register_buffer("K_ZZ_raw", K_ZZ_raw.detach().clone())
        self.register_buffer("K_ZZ", K_ZZ.detach().clone())
        self.register_buffer("L_ZZ", L_ZZ.detach().clone())

    def _compute_Z_moments(self):
        bank_Z = self.bank.evaluate(self.Z)
        mu_Z = empirical_mean(bank_Z)
        K_ZZ_raw = empirical_cross_cov(bank_Z, bank_Z, mu_Z, mu_Z)
        K_ZZ = stabilize_covariance(K_ZZ_raw, jitter=self.jitter, shrinkage=self.shrinkage)
        L_ZZ = safe_cholesky(K_ZZ, initial_jitter=self.jitter)
        return bank_Z, mu_Z, K_ZZ_raw, K_ZZ, L_ZZ

    def _current_Z_moments(self):
        if self.learn_Z or not self.detach_bank_values:
            return self._compute_Z_moments()
        return self.bank_Z, self.mu_Z, self.K_ZZ_raw, self.K_ZZ, self.L_ZZ

    def _evaluate_bank_at_XZ(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        XZ = torch.cat([X, self.Z], dim=0)
        bank_XZ = self.bank.evaluate(XZ)
        num_X = int(X.shape[0])
        return bank_XZ[:, :num_X], bank_XZ[:, num_X:]

    def compute_cross_covariance(self, X: torch.Tensor, bank_Z: torch.Tensor | None = None, mu_Z: torch.Tensor | None = None) -> torch.Tensor:
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        if bank_Z is None or mu_Z is None:
            bank_X, bank_Z = self._evaluate_bank_at_XZ(X)
            mu_Z = empirical_mean(bank_Z)
        else:
            bank_X = self.bank.evaluate(X)
        mu_X = empirical_mean(bank_X)
        return empirical_cross_cov(bank_X, bank_Z, mu_X, mu_Z)

    def psi(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        if self.is_exact_inducing_input(X):
            return torch.eye(self.num_inducing, dtype=self.Z.dtype, device=self.Z.device)
        _, _, _, _, L_ZZ = self._current_Z_moments()
        K_XZ = self.compute_cross_covariance(X)
        return right_cholesky_solve(K_XZ, L_ZZ)

    def inducing_mean(self) -> torch.Tensor:
        if self.learn_Z or not self.detach_bank_values:
            _, mu_Z, _, _, _ = self._compute_Z_moments()
            return mu_Z
        return self.mu_Z

    def inducing_scale_matrix(self) -> torch.Tensor:
        if self.learn_Z or not self.detach_bank_values:
            _, _, _, _, L_ZZ = self._compute_Z_moments()
            return L_ZZ
        return self.L_ZZ

    def mean_at(self, X: torch.Tensor) -> torch.Tensor:
        return empirical_mean(self.bank.evaluate(X.to(dtype=self.Z.dtype, device=self.Z.device)))


class RBFCardinalMatheronOperator(BaseMatheronOperator):
    def __init__(
        self,
        base_prior: nn.Module,
        inducing_points: torch.Tensor,
        input_dim: int | None = None,
        num_moment_samples: int = 512,
        jitter: float = 1e-5,
        shrinkage: float = 1e-4,
        learn_Z: bool = False,
        learn_kernel: bool = True,
        ard: bool = True,
        init_lengthscale: float | torch.Tensor | str = "median",
        init_outputscale: float | str = "prior_marginal",
        inducing_scale: str = "prior_cholesky",
        mean_mode: str = "prior_sample",
        freeze_base_prior: bool = True,
        detach_moment_values: bool = True,
        detach_prior_grad: bool = False,
        seed: int | None = None,
        enforce_exact_Z_identity: bool = True,
    ):
        if freeze_base_prior:
            _freeze_prior(base_prior)
        super().__init__(
            inducing_points,
            learn_Z=learn_Z,
            enforce_exact_Z_identity=enforce_exact_Z_identity,
        )
        self.base_prior = base_prior
        self.jitter = float(jitter)
        self.shrinkage = float(shrinkage)
        self.mean_mode = str(mean_mode)
        self.requested_inducing_scale = str(inducing_scale)
        self.inducing_scale = self.requested_inducing_scale
        self.num_moment_samples = int(num_moment_samples)
        self.detach_moment_values = bool(detach_moment_values and not learn_Z)
        self.detach_prior_grad = bool(detach_prior_grad)
        self._input_dim = int(input_dim or inducing_points.shape[1])
        if self._input_dim != inducing_points.shape[1]:
            raise ValueError("input_dim must match inducing_points.shape[1].")
        if self.mean_mode not in {"zero", "prior_sample", "prior_api"}:
            raise ValueError("mean_mode must be 'zero', 'prior_sample', or 'prior_api'.")
        if self.requested_inducing_scale not in {"prior_cholesky", "rbf_cholesky", "prior_diag", "identity"}:
            raise ValueError("inducing_scale must be 'prior_cholesky', 'rbf_cholesky', 'prior_diag', or 'identity'.")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1].")
        if self.num_moment_samples < 2:
            raise ValueError("num_moment_samples must be at least 2.")

        self.moment_bank = PriorFunctionBank(
            prior=base_prior,
            num_bank_samples=self.num_moment_samples,
            seed=seed,
            detach=self.detach_moment_values,
            detach_prior_grad=self.detach_prior_grad,
        )
        moment_Z, moment_mean, prior_K_ZZ_raw, prior_K_ZZ, prior_L_ZZ = self._compute_prior_Z_moments()
        moment_var = torch.diagonal(prior_K_ZZ_raw).clamp_min(1e-8)

        if self.mean_mode == "prior_sample":
            mu_Z = moment_mean
        elif self.mean_mode == "prior_api":
            if not hasattr(base_prior, "mean"):
                raise ValueError("mean_mode='prior_api' requires base_prior.mean.")
            mu = base_prior.mean(self.Z)
            if mu.ndim == 2 and mu.shape[-1] == 1:
                mu = mu[..., 0]
            mu_Z = mu.to(dtype=self.Z.dtype, device=self.Z.device)
        else:
            mu_Z = torch.zeros(self.num_inducing, dtype=self.Z.dtype, device=self.Z.device)
        self.register_buffer("mu_Z", mu_Z.detach().clone())
        self.register_buffer("moment_Z", moment_Z.detach().clone())
        self.register_buffer("moment_var_Z", moment_var.detach().clone())
        self.register_buffer("prior_K_ZZ_raw", prior_K_ZZ_raw.detach().clone())
        self.register_buffer("prior_K_ZZ", prior_K_ZZ.detach().clone())
        self.register_buffer("prior_L_ZZ", prior_L_ZZ.detach().clone())

        lengthscale = initialize_rbf_lengthscale(
            self.Z,
            ard=ard,
            value=init_lengthscale,
        )
        if init_outputscale == "prior_marginal":
            outputscale = moment_var.mean().clamp_min(1e-8)
        else:
            outputscale = torch.as_tensor(
                init_outputscale,
                dtype=self.Z.dtype,
                device=self.Z.device,
            ).reshape(()).clamp_min(1e-8)
        self.kernel = RBFKernel(
            input_dim=self._input_dim,
            lengthscale=lengthscale,
            outputscale=outputscale,
            ard=ard,
            learn_kernel=learn_kernel,
            device=self.Z.device,
            dtype=self.Z.dtype,
        )

    def _K_ZZ(self) -> torch.Tensor:
        K_ZZ = self.kernel(self.Z, self.Z)
        return 0.5 * (K_ZZ + K_ZZ.T)

    def _L_ZZ(self) -> torch.Tensor:
        return safe_cholesky(self._K_ZZ(), initial_jitter=self.jitter)

    def _compute_prior_Z_moments(self):
        moment_Z = self.moment_bank.evaluate(self.Z)
        moment_mean = empirical_mean(moment_Z)
        prior_K_ZZ_raw = empirical_cross_cov(moment_Z, moment_Z, moment_mean, moment_mean)
        prior_K_ZZ = stabilize_covariance(
            prior_K_ZZ_raw,
            jitter=self.jitter,
            shrinkage=self.shrinkage,
        )
        prior_L_ZZ = safe_cholesky(prior_K_ZZ, initial_jitter=self.jitter)
        return moment_Z, moment_mean, prior_K_ZZ_raw, prior_K_ZZ, prior_L_ZZ

    def _current_prior_Z_moments(self):
        if self.learn_Z or not self.detach_moment_values:
            return self._compute_prior_Z_moments()
        return (
            self.moment_Z,
            self.mu_Z if self.mean_mode == "prior_sample" else empirical_mean(self.moment_Z),
            self.prior_K_ZZ_raw,
            self.prior_K_ZZ,
            self.prior_L_ZZ,
        )

    def _psi_from_cholesky(self, X: torch.Tensor, L_ZZ: torch.Tensor) -> torch.Tensor:
        K_XZ = self.kernel(X, self.Z)
        return right_cholesky_solve(K_XZ, L_ZZ)

    def _inducing_scale_matrix(self) -> torch.Tensor:
        if self.inducing_scale == "identity":
            return torch.eye(self.num_inducing, dtype=self.Z.dtype, device=self.Z.device)
        if self.inducing_scale == "rbf_cholesky":
            return self._L_ZZ()
        _, _, _, prior_K_ZZ, prior_L_ZZ = self._current_prior_Z_moments()
        if self.inducing_scale == "prior_cholesky":
            return prior_L_ZZ
        if self.inducing_scale == "prior_diag":
            diag = torch.diagonal(prior_K_ZZ).clamp_min(torch.finfo(self.Z.dtype).eps).sqrt()
            return torch.diag(diag)
        raise ValueError("inducing_scale must be 'prior_cholesky', 'rbf_cholesky', 'prior_diag', or 'identity'.")

    def _needs_cholesky_for_apply(self, X: torch.Tensor) -> bool:
        return not self.is_exact_inducing_input(X)

    def psi(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        if self.is_exact_inducing_input(X):
            return torch.eye(self.num_inducing, dtype=self.Z.dtype, device=self.Z.device)
        return self._psi_from_cholesky(X, self._L_ZZ())

    def inducing_mean(self) -> torch.Tensor:
        if (
            self.mean_mode == "prior_sample"
            and self.moment_bank is not None
            and (self.learn_Z or not self.detach_moment_values)
        ):
            _, moment_mean, _, _, _ = self._compute_prior_Z_moments()
            return moment_mean
        if self.mean_mode == "prior_api":
            mu = self.base_prior.mean(self.Z)
            if mu.ndim == 2 and mu.shape[-1] == 1:
                mu = mu[..., 0]
            return mu.to(dtype=self.Z.dtype, device=self.Z.device)
        return self.mu_Z

    def inducing_scale_matrix(self) -> torch.Tensor:
        return self._inducing_scale_matrix()

    def apply(
        self,
        X: torch.Tensor,
        g_X: torch.Tensor,
        g_Z: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        a = a.to(dtype=self.Z.dtype, device=self.Z.device)
        L_ZZ = self._L_ZZ() if self._needs_cholesky_for_apply(X) else None
        D_Z = self._inducing_scale_matrix()
        u = self.inducing_mean() + a.matmul(D_Z.T)
        if self.is_exact_inducing_input(X):
            return u
        psi = self._psi_from_cholesky(X, L_ZZ)
        return g_X + (u - g_Z).matmul(psi.T)

    def apply_components(
        self,
        X: torch.Tensor,
        b_X: torch.Tensor,
        b_Z: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        if a.ndim != 3:
            raise ValueError("a must have shape [S, R, M].")
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        S, R, M = a.shape
        if M != self.num_inducing:
            raise ValueError("a last dimension must equal num_inducing.")
        L_ZZ = self._L_ZZ() if self._needs_cholesky_for_apply(X) else None
        D_Z = self._inducing_scale_matrix()
        u = self.inducing_mean() + a.reshape(S * R, M).matmul(D_Z.T)
        u = u.reshape(S, R, M)
        if self.is_exact_inducing_input(X):
            return u
        psi = self._psi_from_cholesky(X, L_ZZ)
        return b_X[:, None, :] + (u - b_Z[:, None, :]).matmul(psi.T)

    def mean_at(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        if self.mean_mode == "zero":
            return torch.zeros(X.shape[0], dtype=self.Z.dtype, device=self.Z.device)
        if self.mean_mode == "prior_api":
            mu = self.base_prior.mean(X)
            if mu.ndim == 2 and mu.shape[-1] == 1:
                mu = mu[..., 0]
            return mu.to(dtype=self.Z.dtype, device=self.Z.device)
        if self.moment_bank is None:
            raise RuntimeError("moment_bank is required for mean_mode='prior_sample'.")
        return empirical_mean(self.moment_bank.evaluate(X))
