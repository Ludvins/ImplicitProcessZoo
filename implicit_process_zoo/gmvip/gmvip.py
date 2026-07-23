from __future__ import annotations

import torch
from torch import nn

from ..priors.function_bank import CoherentPriorFunctionSampler
from ..utils.likelihood import multiclass_logp
from ..utils.random import preserve_constructor_rng
from .likelihoods import GaussianRegressionLikelihood
from .operators import EmpiricalCovarianceMatheronOperator, RBFCardinalMatheronOperator
from .posteriors import CholeskyGaussianCoefficientPosterior, RealNVPCoefficientPosterior


@preserve_constructor_rng
class GeneralizedMatheronVIP(nn.Module):
    """
    Generalized Matheron Variational Implicit Process.

    This model constructs posterior function samples by replacing the values
    of a coherent prior function sample at inducing locations Z. Given a prior
    sample g ~ p0(f), whitened coefficients a, inducing mean mu_Z, inducing
    scale D_Z, and a Matheron/cardinal operator Psi_Z, the sampled function is

        f(X) = g(X) + Psi_Z(X) [mu_Z + D_Z a - g(Z)].

    With ``path_mode="inducing_only"``, the coherent sampled prior path is
    replaced by its mean, giving the inducing-only ablation

        f_ind(X) = mu(X) + Psi_Z(X) D_Z a.

    The corresponding prior uses a ~ N(0, I). The Gaussian posterior variant
    uses q(a) = N(m, L_q L_q^T), giving a closed-form coefficient KL. The
    RealNVP variant uses affine coupling layers over a and estimates the same
    latent-variable KL with reparameterized Monte Carlo samples.

    """

    def __init__(
        self,
        base_prior: nn.Module,
        inducing_points: torch.Tensor,
        operator_type: str = "rbf",
        posterior_type: str = "gaussian",
        likelihood: str = "regression",
        num_operator_bank_samples: int = 512,
        learn_noise: bool = True,
        init_log_noise: float = -2.0,
        min_log_noise: float | None = None,
        max_log_noise: float | None = None,
        freeze_base_prior: bool = True,
        detach_prior_samples: bool = True,
        detach_operator_prior_grad: bool = False,
        jitter: float = 1e-5,
        shrinkage: float = 1e-4,
        learn_Z: bool = False,
        learn_kernel: bool = True,
        ard: bool = True,
        init_lengthscale: float | torch.Tensor | str = "median",
        init_outputscale: float | str = "prior_marginal",
        inducing_scale: str = "prior_cholesky",
        mean_mode: str = "prior_sample",
        enforce_exact_Z_identity: bool = True,
        posterior_init_mean: float = 0.0,
        posterior_init_log_std: float = 0.0,
        posterior_min_log_std: float | None = -8.0,
        posterior_max_log_std: float | None = 4.0,
        flow_depth: int = 4,
        flow_hidden_dim: int = 128,
        flow_num_layers: int = 2,
        flow_dropout: float = 0.0,
        flow_scale_bound: float = 2.0,
        antithetic_samples: bool = True,
        num_data: int | None = None,
        num_train_samples: int = 8,
        beta: float = 1.0,
        beta_warmup_steps: int = 0,
        data_alpha: float = 0.0,
        max_grad_norm: float | None = 10.0,
        operator_bank_seed: int | None = None,
        bank_seed: int | None = None,
        output_dim: int = 1,
        joint_output_covariance: bool = False,
        num_classes: int | None = None,
        likelihood_type: str | None = None,
        path_mode: str = "full",
    ):
        super().__init__()
        if inducing_points.ndim != 2:
            raise ValueError("inducing_points must have shape [M, D].")
        likelihood = self._normalize_likelihood(likelihood, likelihood_type)
        if likelihood not in ("regression", "multiclass"):
            raise ValueError(f"likelihood must be 'regression' or 'multiclass', got '{likelihood}'")
        if likelihood == "multiclass":
            if num_classes is None:
                raise ValueError("num_classes is required for multiclass likelihood")
            if int(output_dim) == 1:
                output_dim = int(num_classes)
            if int(output_dim) != int(num_classes):
                raise ValueError("output_dim must equal num_classes for multiclass likelihood")
            if posterior_type == "realnvp":
                raise NotImplementedError("RealNVP q(a) is not supported for multiclass GMVIP.")
        elif int(output_dim) > 1 and posterior_type == "realnvp":
            raise NotImplementedError(
                "RealNVP q(a) is not supported for vector-output regression GMVIP."
            )
        self.base_prior = base_prior
        self.operator_type = str(operator_type)
        self.posterior_type = str(posterior_type)
        self.path_mode = str(path_mode)
        if self.path_mode not in {"full", "inducing_only"}:
            raise ValueError("path_mode must be 'full' or 'inducing_only'.")
        self.likelihood_type = str(likelihood)
        self.output_dim = int(output_dim)
        self.joint_output_covariance = bool(joint_output_covariance and self.output_dim > 1)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.epsilon = 1e-3
        self.detach_prior_samples = bool(detach_prior_samples)
        self.detach_operator_prior_grad = bool(detach_operator_prior_grad)
        self.antithetic_samples = bool(antithetic_samples)
        if bank_seed is not None:
            if operator_bank_seed is None:
                operator_bank_seed = bank_seed

        if self.operator_type == "empirical":
            if mean_mode != "prior_sample":
                raise ValueError("empirical operator only supports mean_mode='prior_sample'.")
            if inducing_scale != "prior_cholesky":
                raise ValueError(
                    "empirical operator only supports inducing_scale='prior_cholesky'."
                )
            self.operator = EmpiricalCovarianceMatheronOperator(
                base_prior=base_prior,
                inducing_points=inducing_points,
                num_bank_samples=num_operator_bank_samples,
                jitter=jitter,
                shrinkage=shrinkage,
                learn_Z=learn_Z,
                detach_bank_values=detach_prior_samples,
                detach_prior_grad=self.detach_operator_prior_grad,
                freeze_base_prior=freeze_base_prior,
                seed=operator_bank_seed,
                enforce_exact_Z_identity=enforce_exact_Z_identity,
                joint_outputs=self.joint_output_covariance,
            )
        elif self.operator_type == "rbf":
            self.operator = RBFCardinalMatheronOperator(
                base_prior=base_prior,
                inducing_points=inducing_points,
                input_dim=inducing_points.shape[1],
                num_moment_samples=num_operator_bank_samples,
                jitter=jitter,
                shrinkage=shrinkage,
                learn_Z=learn_Z,
                learn_kernel=learn_kernel,
                ard=ard,
                init_lengthscale=init_lengthscale,
                init_outputscale=init_outputscale,
                inducing_scale=inducing_scale,
                mean_mode=mean_mode,
                freeze_base_prior=freeze_base_prior,
                detach_moment_values=detach_prior_samples,
                detach_prior_grad=self.detach_operator_prior_grad,
                seed=operator_bank_seed,
                enforce_exact_Z_identity=enforce_exact_Z_identity,
            )
        else:
            raise ValueError("operator_type must be 'empirical' or 'rbf'.")

        if self.posterior_type == "gaussian":
            self.posterior = CholeskyGaussianCoefficientPosterior(
                num_inducing=self.operator.num_inducing,
                output_dim=self.output_dim,
                init_mean=posterior_init_mean,
                init_log_std=posterior_init_log_std,
                min_log_std=posterior_min_log_std,
                max_log_std=posterior_max_log_std,
                joint_output_covariance=self.joint_output_covariance,
                device=self.Z.device,
                dtype=self.Z.dtype,
            )
        elif self.posterior_type == "realnvp":
            self.posterior = RealNVPCoefficientPosterior(
                num_inducing=self.operator.num_inducing,
                num_flows=flow_depth,
                hidden_dim=flow_hidden_dim,
                num_layers=flow_num_layers,
                dropout=flow_dropout,
                scale_bound=flow_scale_bound,
                device=self.Z.device,
                dtype=self.Z.dtype,
            )
        else:
            raise ValueError("posterior_type must be 'gaussian' or 'realnvp'.")

        self.likelihood = None
        if self.likelihood_type == "regression":
            self.likelihood = GaussianRegressionLikelihood(
                init_log_noise=init_log_noise,
                learn_noise=learn_noise,
                min_log_noise=min_log_noise,
                max_log_noise=max_log_noise,
                device=self.Z.device,
                dtype=self.Z.dtype,
            )
        self.residual_sampler = CoherentPriorFunctionSampler(base_prior)
        self.num_data = None if num_data is None else int(num_data)
        self.num_train_samples = int(num_train_samples)
        self.beta = float(beta)
        self.beta_warmup_steps = int(beta_warmup_steps)
        self.data_alpha = float(data_alpha)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self._step = 0
        self._elbo_buffer: list[torch.Tensor] = []
        self._expected_log_lik_buffer: list[torch.Tensor] = []
        self._kl_buffer: list[torch.Tensor] = []
        self.data_terms: list[torch.Tensor] = []
        self.function_terms: list[torch.Tensor] = []
        self.betas: list[float] = []
        self._last_train_metrics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _normalize_likelihood(likelihood: str, likelihood_type: str | None) -> str:
        if likelihood_type is not None:
            if str(likelihood) != "regression" and str(likelihood_type) != str(likelihood):
                raise ValueError("Specify only one of likelihood or likelihood_type.")
            likelihood = likelihood_type
        if str(likelihood) == "gaussian":
            return "regression"
        return str(likelihood)

    @property
    def Z(self) -> torch.Tensor:
        return self.operator.Z

    @property
    def num_inducing(self) -> int:
        return self.operator.num_inducing

    @property
    def coefficients(self):
        if self.posterior_type != "gaussian":
            raise AttributeError("coefficients is only available for posterior_type='gaussian'.")
        return self.posterior

    @property
    def log_noise(self) -> torch.Tensor:
        if self.likelihood is None:
            raise AttributeError("log_noise is only available for regression likelihood.")
        return self.likelihood.log_noise

    @property
    def clamped_log_noise(self) -> torch.Tensor:
        if self.likelihood is None:
            raise AttributeError("clamped_log_noise is only available for regression likelihood.")
        return self.likelihood.clamped_log_noise

    @property
    def noise_std(self) -> torch.Tensor:
        if self.likelihood is None:
            raise AttributeError("noise_std is only available for regression likelihood.")
        return self.likelihood.noise_std

    def _make_generator(self, seed: int | None):
        if seed is None:
            return None
        generator = torch.Generator(device=self.Z.device)
        generator.manual_seed(int(seed))
        return generator

    def _as_model_input(self, X: torch.Tensor) -> torch.Tensor:
        return X.to(dtype=self.Z.dtype, device=self.Z.device)

    def _as_target(self, y: torch.Tensor) -> torch.Tensor:
        if self.likelihood_type == "multiclass":
            y = y.to(device=self.Z.device)
            if y.ndim >= 2 and y.shape[-1] == self.num_classes:
                return y.to(dtype=self.Z.dtype)
            if y.ndim == 2 and y.shape[-1] == 1:
                y = y[..., 0]
            if y.ndim != 1:
                raise ValueError("Multiclass targets must have shape [N], [N, 1], or [N, K].")
            return y.long()
        y = y.to(dtype=self.Z.dtype, device=self.Z.device)
        if self.output_dim == 1 and y.ndim == 2 and y.shape[-1] == 1:
            y = y[..., 0]
        if self.output_dim == 1:
            if y.ndim != 1:
                raise ValueError("Scalar-output regression targets must have shape [N] or [N, 1].")
            return y
        if y.ndim != 2 or y.shape[-1] != self.output_dim:
            raise ValueError(
                "Vector-output regression targets must have shape "
                f"[N, {self.output_dim}], got {tuple(y.shape)}."
            )
        return y

    def vi_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def prepare_for_training(self, train_loader) -> None:
        if self.num_data is None and hasattr(train_loader, "dataset"):
            self.num_data = len(train_loader.dataset)

    def _scheduled_beta(self) -> float:
        if self.beta_warmup_steps <= 0:
            return self.beta
        return self.beta * min(1.0, self._step / float(self.beta_warmup_steps))

    def sample_fresh_prior_values(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        num_samples: int,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X = self._as_model_input(X)
        XZ = torch.cat([X, Z.to(dtype=X.dtype, device=X.device)], dim=0)
        values = self.residual_sampler.sample_values(XZ, int(num_samples), seed=seed)
        n_eval = X.shape[0]
        return values[:, :n_eval], values[:, n_eval:]

    def sample_residual_prior_values(
        self,
        X: torch.Tensor,
        num_samples: int,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample_fresh_prior_values(X, self.Z, num_samples, seed=seed)

    def inducing_values_from_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.operator.whitened_to_inducing(
            coefficients.to(dtype=self.Z.dtype, device=self.Z.device)
        )

    def posterior_values_from_coefficients(
        self,
        X: torch.Tensor,
        coefficients: torch.Tensor,
        g_X: torch.Tensor | None = None,
        g_Z: torch.Tensor | None = None,
        residual_seed: int | None = None,
    ) -> torch.Tensor:
        X = self._as_model_input(X)
        coefficients = coefficients.to(dtype=self.Z.dtype, device=self.Z.device)
        if self.path_mode == "inducing_only":
            g_X = self.operator.mean_at(X)
            g_Z = self.operator.inducing_mean()
        elif g_X is None or g_Z is None:
            g_X, g_Z = self.sample_residual_prior_values(
                X,
                coefficients.shape[0],
                seed=residual_seed,
            )
        return self.operator.apply(X, g_X, g_Z, coefficients)

    def compute_interpolation_matrix(self, X: torch.Tensor) -> torch.Tensor:
        return self.operator.psi(self._as_model_input(X))

    def compute_cross_covariance(self, X: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.operator, "compute_cross_covariance"):
            raise AttributeError(
                "compute_cross_covariance is only available for empirical operator."
            )
        return self.operator.compute_cross_covariance(self._as_model_input(X))

    def sample_posterior_values(
        self,
        X: torch.Tensor,
        num_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        return self._sample_posterior_values_gaussian(X, num_samples, seed=seed)

    def _sample_posterior_values_gaussian(
        self,
        X: torch.Tensor,
        num_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        generator = self._make_generator(seed)
        coefficients = self.posterior.rsample(
            int(num_samples),
            generator=generator,
            antithetic=self.antithetic_samples,
        )
        residual_seed = None if seed is None else int(seed) + 104729
        return self.posterior_values_from_coefficients(
            X,
            coefficients,
            residual_seed=residual_seed,
        )

    def sample_posterior_values_with_kl(
        self,
        X: torch.Tensor,
        num_samples: int,
        seed: int | None = None,
    ):
        if self.posterior_type not in {"gaussian", "realnvp"}:
            raise NotImplementedError(
                "sample_posterior_values_with_kl requires gaussian or realnvp q(a)."
            )
        generator = self._make_generator(seed)
        coefficients, kl_terms, diagnostics = self.posterior.rsample_with_kl(
            int(num_samples),
            generator=generator,
            antithetic=self.antithetic_samples,
        )
        residual_seed = None if seed is None else int(seed) + 104729
        values = self.posterior_values_from_coefficients(
            X,
            coefficients,
            residual_seed=residual_seed,
        )
        return values, kl_terms, diagnostics

    def _alpha_sample_log_likelihood(
        self,
        log_prob: torch.Tensor,
        data_alpha: float,
    ) -> torch.Tensor:
        if abs(float(data_alpha)) < 1e-12:
            return log_prob.reshape(log_prob.shape[0], -1).sum(dim=-1).mean()
        sample_count = int(log_prob.shape[0])
        log_mean_exp = torch.logsumexp(float(data_alpha) * log_prob, dim=0) - torch.log(
            torch.tensor(sample_count, dtype=log_prob.dtype, device=log_prob.device)
        )
        return log_mean_exp.sum() / float(data_alpha)

    def sample_prior_values(
        self,
        X: torch.Tensor,
        num_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        return self._sample_prior_values_gaussian(X, num_samples, seed=seed)

    def _sample_prior_values_gaussian(
        self,
        X: torch.Tensor,
        num_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        generator = self._make_generator(seed)
        coefficients = self.posterior.sample_prior(
            int(num_samples),
            generator=generator,
            antithetic=self.antithetic_samples,
        )
        residual_seed = None if seed is None else int(seed) + 104729
        return self.posterior_values_from_coefficients(
            X,
            coefficients,
            residual_seed=residual_seed,
        )

    def kl_divergence(self) -> torch.Tensor:
        if self.posterior_type == "gaussian":
            return self.posterior.kl_to_standard_normal()
        if self.posterior_type == "realnvp":
            return self.posterior.kl_to_standard_normal(
                antithetic=self.antithetic_samples,
            )
        raise ValueError("posterior_type must be 'gaussian' or 'realnvp'.")

    def elbo(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        num_samples: int,
        num_data: int | None = None,
        beta: float = 1.0,
        seed: int | None = None,
        data_alpha: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.likelihood_type == "regression":
            return self._elbo_gaussian(
                X_batch, y_batch, num_samples, num_data, beta, seed=seed, data_alpha=data_alpha
            )
        return self._elbo_multiclass(
            X_batch, y_batch, num_samples, num_data, beta, seed=seed, data_alpha=data_alpha
        )

    def _elbo_gaussian(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        num_samples: int,
        num_data: int | None,
        beta: float,
        seed: int | None = None,
        data_alpha: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        X_batch = self._as_model_input(X_batch)
        y_batch = self._as_target(y_batch)
        f_samples, kl_terms, diagnostics = self.sample_posterior_values_with_kl(
            X_batch,
            int(num_samples),
            seed=seed,
        )
        log_prob = self.likelihood.log_prob(y_batch.unsqueeze(0), f_samples)
        expected_log_lik = self._alpha_sample_log_likelihood(log_prob, data_alpha=data_alpha)
        if num_data is not None:
            expected_log_lik = expected_log_lik * (float(num_data) / float(X_batch.shape[0]))
        kl = kl_terms.mean()
        elbo = expected_log_lik - float(beta) * kl
        q_std_mean = diagnostics.get("q_std_mean")
        if q_std_mean is None:
            q_std_mean = self.posterior.std.mean()
        coefficient_displacement = diagnostics.get("coefficient_displacement")
        if coefficient_displacement is None:
            coefficient_displacement = self.posterior.loc.square().mean()
        metrics = self._base_metrics(elbo, expected_log_lik, kl, beta, data_alpha=data_alpha)
        metrics.update(
            {
                "q_std_mean": q_std_mean.detach(),
                "coefficient_displacement": coefficient_displacement.detach(),
            }
        )
        for key in ("flow_logdet_mean", "flow_kl_std"):
            if key in diagnostics:
                metrics[key] = diagnostics[key].detach()
        return elbo, metrics

    def _elbo_multiclass(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        num_samples: int,
        num_data: int | None,
        beta: float,
        seed: int | None = None,
        data_alpha: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        X_batch = self._as_model_input(X_batch)
        y_batch = self._as_target(y_batch)
        f_samples, kl_terms, diagnostics = self.sample_posterior_values_with_kl(
            X_batch,
            int(num_samples),
            seed=seed,
        )
        if f_samples.ndim != 3 or f_samples.shape[-1] != self.num_classes:
            raise RuntimeError("Multiclass GMVIP posterior samples must have shape [S, N, K].")
        log_prob = multiclass_logp(f_samples, y_batch, self.num_classes, self.epsilon).sum(dim=-1)
        expected_log_lik = self._alpha_sample_log_likelihood(log_prob, data_alpha=data_alpha)
        if num_data is not None:
            expected_log_lik = expected_log_lik * (float(num_data) / float(X_batch.shape[0]))
        kl = kl_terms.mean()
        elbo = expected_log_lik - float(beta) * kl
        q_std_mean = diagnostics.get("q_std_mean")
        if q_std_mean is None:
            q_std_mean = self.posterior.std.mean()
        coefficient_displacement = diagnostics.get("coefficient_displacement")
        if coefficient_displacement is None:
            coefficient_displacement = self.posterior.loc.square().mean()
        metrics = self._base_metrics(elbo, expected_log_lik, kl, beta, data_alpha=data_alpha)
        metrics.update(
            {
                "q_std_mean": q_std_mean.detach(),
                "coefficient_displacement": coefficient_displacement.detach(),
            }
        )
        return elbo, metrics

    def _base_metrics(
        self,
        elbo: torch.Tensor,
        expected_log_lik: torch.Tensor,
        kl: torch.Tensor,
        beta: float,
        data_alpha: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        beta_tensor = torch.tensor(float(beta), dtype=self.Z.dtype, device=self.Z.device)
        data_alpha_tensor = torch.tensor(
            float(data_alpha), dtype=self.Z.dtype, device=self.Z.device
        )
        metrics = {
            "elbo": elbo.detach(),
            "expected_log_lik": expected_log_lik.detach(),
            "expected_loglik": expected_log_lik.detach(),
            "data_nll": (-expected_log_lik).detach(),
            "kl": kl.detach(),
            "beta": beta_tensor,
            "data_alpha": data_alpha_tensor,
        }
        if self.likelihood_type == "regression":
            metrics.update(
                {
                    "noise": self.noise_std.detach(),
                    "noise_std": self.noise_std.detach(),
                }
            )
        return metrics

    def nelbo(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_samples: int = 8,
        num_data: int | None = None,
        beta: float = 1.0,
        seed: int | None = None,
        data_alpha: float = 0.0,
    ) -> torch.Tensor:
        return -self.elbo(
            X,
            y,
            num_samples=num_samples,
            num_data=num_data,
            beta=beta,
            seed=seed,
            data_alpha=data_alpha,
        )[0]

    def elbo_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_samples: int = 8,
        num_data: int | None = None,
        beta: float = 1.0,
        seed: int | None = None,
        data_alpha: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value, metrics = self.elbo(
            X,
            y,
            num_samples=num_samples,
            num_data=num_data,
            beta=beta,
            seed=seed,
            data_alpha=data_alpha,
        )
        loss = -value
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        return loss, metrics

    @torch.no_grad()
    def predict_summary(
        self,
        X: torch.Tensor,
        num_samples: int = 128,
        include_noise: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Summarize posterior function and observation moments.

        Parameters
        ----------
        X : torch.Tensor
            Inputs with shape ``[N, input_dim]``.
        num_samples : int, default=128
            Number of Monte Carlo function samples.
        include_noise : bool, default=True
            Whether regression observation variance includes likelihood noise.

        Returns
        -------
        dict of str to torch.Tensor
            Samples and predictive moments. Multiclass results additionally
            contain logits and class probabilities.
        """
        was_training = self.training
        self.eval()
        try:
            X = self._as_model_input(X)
            f_samples = self.sample_posterior_values(X, int(num_samples))
            if self.likelihood_type == "multiclass":
                f_mean = f_samples.mean(dim=0)
                f_var = f_samples.var(dim=0, unbiased=False)
                probs = torch.softmax(f_samples, dim=-1).mean(dim=0)
                return {
                    "f_samples": f_samples,
                    "logits_samples": f_samples,
                    "f_mean": f_mean,
                    "f_var": f_var,
                    "logits": f_mean,
                    "probs": probs,
                    "y_mean": probs,
                    "y_var": probs * (1.0 - probs),
                }
            f_mean = f_samples.mean(dim=0)
            f_var = f_samples.var(dim=0, unbiased=False)
            y_var = f_var + self.noise_std.square() if include_noise else f_var
            return {
                "f_samples": f_samples,
                "f_mean": f_mean,
                "f_var": f_var,
                "y_mean": f_mean,
                "y_var": y_var,
            }
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict_y_samples(
        self,
        X: torch.Tensor,
        num_samples: int,
        *,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Draw predictive observation samples.

        Parameters
        ----------
        X : torch.Tensor
            Inputs with shape ``[N, input_dim]``.
        num_samples : int
            Number of predictive samples.
        seed : int, optional
            Local random seed.

        Returns
        -------
        torch.Tensor
            Observation samples or multiclass logits with shape
            ``[S, N, D]``.
        """
        samples = self.sample_posterior_values(X, int(num_samples), seed=seed)
        if self.likelihood_type == "multiclass":
            return samples
        if seed is None:
            noise = torch.randn_like(samples)
        else:
            generator = torch.Generator(device=samples.device).manual_seed(int(seed) + 1)
            noise = torch.randn(
                samples.shape,
                generator=generator,
                dtype=samples.dtype,
                device=samples.device,
            )
        samples = samples + self.noise_std * noise
        if self.output_dim > 1:
            return samples
        return samples.unsqueeze(-1)

    def predict_samples(
        self,
        X: torch.Tensor,
        num_samples: int = 128,
        noisy: bool = False,
    ) -> torch.Tensor:
        """Backward-compatible sampling helper; prefer the common methods."""
        if noisy:
            return self.predict_y_samples(X, num_samples)
        return self.predict_f_samples(X, num_samples)

    @torch.no_grad()
    def predict(self, X: torch.Tensor, num_samples: int, *, seed=None) -> torch.Tensor:
        return self.predict_y_samples(X, num_samples, seed=seed)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.likelihood_type == "multiclass":
            return self.predict_f_samples(X, 128), None
        pred = self.predict_summary(X, num_samples=128, include_noise=True)
        if self.output_dim > 1:
            return pred["y_mean"], pred["y_var"].sqrt()
        return pred["y_mean"].unsqueeze(-1), pred["y_var"].sqrt().unsqueeze(-1)

    @torch.no_grad()
    def predict_f_samples(
        self, X: torch.Tensor, num_samples: int, *, seed: int | None = None
    ) -> torch.Tensor:
        """Draw latent posterior function samples.

        Parameters
        ----------
        X : torch.Tensor
            Inputs with shape ``[N, input_dim]``.
        num_samples : int
            Number of posterior samples.
        seed : int, optional
            Local random seed.

        Returns
        -------
        torch.Tensor
            Function samples with shape ``[S, N, D]``.
        """
        samples = self.sample_posterior_values(X, int(num_samples), seed=seed)
        if self.output_dim == 1 and samples.ndim == 2:
            samples = samples.unsqueeze(-1)
        return samples

    def forward_prior(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
        samples = self.sample_prior_values(X, int(num_samples))
        if self.likelihood_type == "multiclass":
            return samples
        if self.output_dim == 1:
            samples = samples.unsqueeze(-1)
        y_mean = getattr(self, "y_mean", None)
        y_std = getattr(self, "y_std", None)
        if y_mean is not None and y_std is not None:
            samples = samples * y_std + y_mean
        return samples

    def _train_step(
        self, optimizer: torch.optim.Optimizer, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        X = self._as_model_input(X)
        y = self._as_target(y)
        self._step += 1
        beta = self._scheduled_beta()

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.elbo_loss(
            X,
            y,
            num_samples=self.num_train_samples,
            num_data=self.num_data,
            beta=beta,
            data_alpha=self.data_alpha,
        )
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        optimizer.step()

        self._elbo_buffer.append(metrics["elbo"])
        self._expected_log_lik_buffer.append(metrics["expected_log_lik"])
        self._kl_buffer.append(metrics["kl"])
        self.data_terms.append(metrics["data_nll"])
        self.function_terms.append(metrics["kl"])
        self.betas.append(beta)
        self._last_train_metrics = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in metrics.items()
        }
        self._last_train_metrics["loss"] = loss.detach()
        return loss

    @property
    def last_train_metrics(self) -> dict[str, torch.Tensor]:
        return self._last_train_metrics

    @property
    def KLs(self):
        return self._kl_buffer

    @property
    def expected_log_liks(self):
        return self._expected_log_lik_buffer
