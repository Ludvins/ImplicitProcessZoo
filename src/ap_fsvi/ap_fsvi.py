import math
import warnings

import torch
from tqdm import tqdm

from ..priors.generative_functions import ExactGP
from ..utils.likelihood import bernoulli_logp, gaussian_logp, inv_probit, multiclass_logp
from ..utils.utils import infinite_loader


class FunctionDiscrepancy:
    """Function-space discrepancy over finite measurement projections.

    All sample-based discrepancies operate on flattened values
    ``[S, M, D] -> [S, M * D]``. The ``stein`` option is a GP-prior KSD and
    uses posterior samples plus the prior score, so it does not require prior
    samples.
    """

    VALID_KINDS = (
        "mmd",
        "energy",
        "sliced_wasserstein",
        "stein",
        "sinkhorn",
        "prior_whitened_gaussian_kl",
        "prior_whitened_sliced_kl",
        "spectral_sliced_kl",
        "spectral_projected_kl",
        "sample_sliced_kl",
        "sample_sliced_knn_kl",
        "sample_sliced_gaussian_kl",
        "sample_sliced_quantile_transport_kl",
    )

    def __init__(
        self,
        kind="mmd",
        bandwidth="median",
        estimator="biased",
        min_bandwidth=1e-6,
        num_projections=64,
        sinkhorn_epsilon=1.0,
        sinkhorn_iterations=50,
        sinkhorn_debiased=False,
        distance_eps=1e-12,
        spectral_num_modes=None,
        spectral_estimator="gaussian",
        spectral_cumulant_weights=(0.05, 0.01),
        spectral_detach_prior_eig=True,
        spectral_mode_weighting="uniform",
        spectral_eigenvalue_power=1.0,
        spectral_cov_shrinkage=0.05,
        spectral_knn_k=3,
        sample_knn_k=3,
        sample_gaussian_shrinkage=0.05,
        sample_projection_mode="random",
        quantile_transport_k=3,
    ):
        self.kind = _normalize_discrepancy_kind(kind)
        self.bandwidth = bandwidth
        self.estimator = estimator
        self.min_bandwidth = min_bandwidth
        self.num_projections = num_projections
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_debiased = sinkhorn_debiased
        self.distance_eps = distance_eps
        self.spectral_num_modes = spectral_num_modes or num_projections
        self.spectral_estimator = spectral_estimator
        self.spectral_cumulant_weights = spectral_cumulant_weights
        self.spectral_detach_prior_eig = spectral_detach_prior_eig
        self.spectral_mode_weighting = spectral_mode_weighting
        self.spectral_eigenvalue_power = spectral_eigenvalue_power
        self.spectral_cov_shrinkage = spectral_cov_shrinkage
        self.spectral_knn_k = spectral_knn_k
        if sample_knn_k <= 0:
            raise ValueError("sample_knn_k must be positive.")
        self.sample_knn_k = int(sample_knn_k)
        self.sample_gaussian_shrinkage = sample_gaussian_shrinkage
        if quantile_transport_k <= 0:
            raise ValueError("quantile_transport_k must be positive.")
        self.quantile_transport_k = int(quantile_transport_k)
        self.sample_projection_mode = _normalize_sample_projection_mode(
            sample_projection_mode
        )
        self._fixed_projection_cache = {}

    @classmethod
    def build(
        cls,
        kind="mmd",
        bandwidth="median",
        estimator="biased",
        min_bandwidth=1e-6,
        num_projections=64,
        sinkhorn_epsilon=1.0,
        sinkhorn_iterations=50,
        sinkhorn_debiased=False,
        distance_eps=1e-12,
        spectral_num_modes=None,
        spectral_estimator="gaussian",
        spectral_cumulant_weights=(0.05, 0.01),
        spectral_detach_prior_eig=True,
        spectral_mode_weighting="uniform",
        spectral_eigenvalue_power=1.0,
        spectral_cov_shrinkage=0.05,
        spectral_knn_k=3,
        sample_knn_k=3,
        sample_gaussian_shrinkage=0.05,
        sample_projection_mode="random",
        quantile_transport_k=3,
    ):
        return cls(
            kind=kind,
            bandwidth=bandwidth,
            estimator=estimator,
            min_bandwidth=min_bandwidth,
            num_projections=num_projections,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_debiased=sinkhorn_debiased,
            distance_eps=distance_eps,
            spectral_num_modes=spectral_num_modes,
            spectral_estimator=spectral_estimator,
            spectral_cumulant_weights=spectral_cumulant_weights,
            spectral_detach_prior_eig=spectral_detach_prior_eig,
            spectral_mode_weighting=spectral_mode_weighting,
            spectral_eigenvalue_power=spectral_eigenvalue_power,
            spectral_cov_shrinkage=spectral_cov_shrinkage,
            spectral_knn_k=spectral_knn_k,
            sample_knn_k=sample_knn_k,
            sample_gaussian_shrinkage=sample_gaussian_shrinkage,
            sample_projection_mode=sample_projection_mode,
            quantile_transport_k=quantile_transport_k,
        )

    @property
    def requires_prior_samples(self):
        return self.kind in (
            "mmd",
            "energy",
            "sliced_wasserstein",
            "sinkhorn",
            "sample_sliced_kl",
            "sample_sliced_knn_kl",
            "sample_sliced_gaussian_kl",
            "sample_sliced_quantile_transport_kl",
        )

    def __call__(
        self,
        posterior_values,
        prior_values=None,
        measurement_inputs=None,
        prior_function=None,
    ):
        if self.kind == "stein":
            return self._stein_ksd(posterior_values, measurement_inputs, prior_function)
        if self.kind == "prior_whitened_gaussian_kl":
            return self._prior_whitened_gaussian_kl(
                posterior_values, measurement_inputs, prior_function
            )
        if self.kind == "prior_whitened_sliced_kl":
            return self._prior_whitened_sliced_kl(
                posterior_values, measurement_inputs, prior_function
            )
        if self.kind == "spectral_sliced_kl":
            return self._spectral_sliced_kl(
                posterior_values,
                prior_values,
                measurement_inputs,
                prior_function,
            )
        if self.kind == "spectral_projected_kl":
            return self._spectral_projected_kl(
                posterior_values,
                prior_values,
                measurement_inputs,
                prior_function,
            )
        if prior_values is None:
            raise ValueError(f"{self.kind} discrepancy requires prior_values.")

        z = _flatten_function_values(posterior_values)
        w = _flatten_function_values(prior_values).detach()
        if z.shape[-1] != w.shape[-1]:
            raise ValueError("posterior and prior projections must have the same dimension.")

        if self.kind == "mmd":
            return self._mmd(z, w)
        if self.kind == "energy":
            return self._energy(z, w)
        if self.kind == "sliced_wasserstein":
            return self._sliced_wasserstein(z, w)
        if self.kind == "sinkhorn":
            return self._sinkhorn_divergence(z, w)
        if self.kind == "sample_sliced_kl":
            return self._sample_sliced_kl(z, w)
        if self.kind == "sample_sliced_knn_kl":
            return self._sample_sliced_knn_kl(z, w)
        if self.kind == "sample_sliced_gaussian_kl":
            return self._sample_sliced_gaussian_kl(z, w)
        if self.kind == "sample_sliced_quantile_transport_kl":
            return self._sample_sliced_quantile_transport_kl(z, w)
        raise ValueError(f"Unknown function discrepancy: {self.kind!r}")

    def _mmd(self, z, w):
        sigma2 = self._bandwidth_squared(z, w)
        k_zz = _rbf_gram(z, z, sigma2)
        k_ww = _rbf_gram(w, w, sigma2)
        k_zw = _rbf_gram(z, w, sigma2)

        if self.estimator == "biased":
            return k_zz.mean() + k_ww.mean() - 2.0 * k_zw.mean()
        if self.estimator == "unbiased":
            if z.shape[0] < 2 or w.shape[0] < 2:
                return k_zz.mean() + k_ww.mean() - 2.0 * k_zw.mean()
            zz = _off_diagonal_mean(k_zz)
            ww = _off_diagonal_mean(k_ww)
            return zz + ww - 2.0 * k_zw.mean()
        raise ValueError(f"Unknown MMD estimator: {self.estimator!r}")

    def _energy(self, z, w):
        d_zz = _pairwise_distances(z, z, self.distance_eps)
        d_ww = _pairwise_distances(w, w, self.distance_eps)
        d_zw = _pairwise_distances(z, w, self.distance_eps)

        if self.estimator == "biased":
            value = 2.0 * d_zw.mean() - d_zz.mean() - d_ww.mean()
        elif self.estimator == "unbiased":
            if z.shape[0] < 2 or w.shape[0] < 2:
                value = 2.0 * d_zw.mean() - d_zz.mean() - d_ww.mean()
            else:
                value = 2.0 * d_zw.mean() - _off_diagonal_mean(d_zz) - _off_diagonal_mean(d_ww)
        else:
            raise ValueError(f"Unknown energy estimator: {self.estimator!r}")
        return value.clamp_min(0.0)

    def _sliced_wasserstein(self, z, w):
        if self.num_projections <= 0:
            raise ValueError("num_projections must be positive.")
        directions = torch.randn(
            z.shape[-1],
            self.num_projections,
            dtype=z.dtype,
            device=z.device,
        )
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)

        z_proj = z @ directions
        w_proj = w @ directions
        z_sorted = torch.sort(z_proj, dim=0).values
        w_sorted = torch.sort(w_proj, dim=0).values
        z_quant, w_quant = _match_sorted_quantiles(z_sorted, w_sorted)
        return (z_quant - w_quant).square().mean()

    def _sinkhorn_divergence(self, z, w):
        if self.sinkhorn_epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be positive.")
        if self.sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive.")
        cost_zw = _sinkhorn_cost(
            z,
            w,
            epsilon=self.sinkhorn_epsilon,
            iterations=self.sinkhorn_iterations,
        )
        if not self.sinkhorn_debiased:
            return cost_zw
        cost_zz = _sinkhorn_cost(
            z,
            z,
            epsilon=self.sinkhorn_epsilon,
            iterations=self.sinkhorn_iterations,
        )
        cost_ww = _sinkhorn_cost(
            w,
            w,
            epsilon=self.sinkhorn_epsilon,
            iterations=self.sinkhorn_iterations,
        )
        return (cost_zw - 0.5 * cost_zz - 0.5 * cost_ww).clamp_min(0.0)

    def _stein_ksd(self, posterior_values, measurement_inputs, prior_function):
        if measurement_inputs is None:
            raise ValueError("Stein discrepancy requires measurement_inputs.")
        if prior_function is None:
            raise ValueError("Stein discrepancy requires prior_function.")
        z = _flatten_function_values(posterior_values)
        score = _gp_prior_score(posterior_values, measurement_inputs, prior_function)

        sigma2 = self._bandwidth_squared(z, z)
        k = _rbf_gram(z, z, sigma2)
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        diff2 = diff.square().sum(dim=-1)
        score_dot = score @ score.T
        score_diff = ((score.unsqueeze(1) - score.unsqueeze(0)) * diff).sum(dim=-1)
        dim = torch.as_tensor(z.shape[-1], dtype=z.dtype, device=z.device)
        stein_kernel = k * (
            score_dot
            + score_diff / sigma2
            + dim / sigma2
            - diff2 / sigma2.square()
        )
        if self.estimator == "biased":
            return stein_kernel.mean().clamp_min(0.0)
        if self.estimator == "unbiased":
            if z.shape[0] < 2:
                return stein_kernel.mean().clamp_min(0.0)
            return _off_diagonal_mean(stein_kernel).clamp_min(0.0)
        raise ValueError(f"Unknown Stein estimator: {self.estimator!r}")

    def _prior_whitened_gaussian_kl(
        self, posterior_values, measurement_inputs, prior_function
    ):
        u = self._prior_whitened_flattened(
            posterior_values, measurement_inputs, prior_function
        )
        n, dim = u.shape
        mean = u.mean(dim=0)
        centered = u - mean
        if n < 2:
            variance = centered.square().mean(dim=0).clamp_min(self.min_bandwidth**2)
            kl = 0.5 * (variance.sum() + mean.square().sum() - dim - variance.log().sum())
            return kl.clamp_min(0.0)

        cov = centered.T @ centered / (n - 1)
        eye = torch.eye(dim, dtype=u.dtype, device=u.device)
        cov = cov + self.min_bandwidth**2 * eye
        sign, logdet = torch.linalg.slogdet(cov)
        if torch.any(sign <= 0):
            cov = cov + 10.0 * self.min_bandwidth**2 * eye
            sign, logdet = torch.linalg.slogdet(cov)
        kl = 0.5 * (torch.trace(cov) + mean.square().sum() - dim - logdet)
        return kl.clamp_min(0.0)

    def _prior_whitened_sliced_kl(
        self, posterior_values, measurement_inputs, prior_function
    ):
        u = self._prior_whitened_flattened(
            posterior_values, measurement_inputs, prior_function
        )
        if self.num_projections <= 0:
            raise ValueError("num_projections must be positive.")
        directions = _random_unit_directions(
            u.shape[-1], self.num_projections, dtype=u.dtype, device=u.device
        )
        projected = u @ directions
        cross_entropy = 0.5 * (
            projected.square() + math.log(2.0 * math.pi)
        ).mean(dim=0)
        entropy = self._kde_entropy_1d(projected)
        # A raw sliced KL is an average of one-dimensional projected KLs.
        # Multiplying by the function-vector dimension puts the estimator on
        # the scale of a full finite projected KL for isotropic Gaussian cases,
        # so beta=1 has the intended VI interpretation.
        return (u.shape[-1] * (cross_entropy - entropy).mean()).clamp_min(0.0)

    def _sample_sliced_kl(self, z, w):
        if self.num_projections <= 0:
            raise ValueError("num_projections must be positive.")
        z, w = _diagonal_standardize_by_reference(z, w, self.min_bandwidth)
        directions = _sample_sliced_projection_directions(
            z,
            w,
            self.num_projections,
            mode=self.sample_projection_mode,
            min_bandwidth=self.min_bandwidth,
            cache=self._fixed_projection_cache,
        )
        z_proj = z @ directions
        w_proj = w @ directions
        log_q = self._kde_log_density_1d(
            z_proj, z_proj, leave_one_out=True
        )
        log_p = self._kde_log_density_1d(
            z_proj, w_proj, leave_one_out=False
        )
        # Same dimensional scaling as prior_whitened_sliced_kl. This remains a
        # projected/sample-only KL surrogate, but avoids requiring beta to
        # compensate for averaging over one-dimensional projections.
        return (z.shape[-1] * (log_q - log_p).mean()).clamp_min(0.0)

    def _sample_sliced_knn_kl(self, z, w):
        """Sliced KL with one-dimensional kNN spacing density estimates."""
        if self.num_projections <= 0:
            raise ValueError("num_projections must be positive.")
        z, w = _diagonal_standardize_by_reference(z, w, self.min_bandwidth)
        directions = _sample_sliced_projection_directions(
            z,
            w,
            self.num_projections,
            mode=self.sample_projection_mode,
            min_bandwidth=self.min_bandwidth,
            cache=self._fixed_projection_cache,
        )
        z_proj = z @ directions
        w_proj = w @ directions
        if z_proj.shape[0] < 2 or w_proj.shape[0] < 1:
            return self._sample_sliced_gaussian_kl(z, w)

        log_q = _knn_log_density_1d(
            z_proj,
            z_proj,
            k=self.sample_knn_k,
            min_width=self.min_bandwidth,
            leave_one_out=True,
        )
        log_p = _knn_log_density_1d(
            z_proj,
            w_proj.detach(),
            k=self.sample_knn_k,
            min_width=self.min_bandwidth,
            leave_one_out=False,
        )
        return (z.shape[-1] * (log_q - log_p).mean()).clamp_min(0.0)

    def _sample_sliced_gaussian_kl(self, z, w):
        if self.num_projections <= 0:
            raise ValueError("num_projections must be positive.")
        z, w = _diagonal_standardize_by_reference(z, w, self.min_bandwidth)
        directions = _sample_sliced_projection_directions(
            z,
            w,
            self.num_projections,
            mode=self.sample_projection_mode,
            min_bandwidth=self.min_bandwidth,
            cache=self._fixed_projection_cache,
        )
        z_proj = z @ directions
        w_proj = w @ directions

        q_mean = z_proj.mean(dim=0)
        p_mean = w_proj.detach().mean(dim=0)
        q_var = z_proj.var(dim=0, unbiased=False)
        p_var = w_proj.detach().var(dim=0, unbiased=False)

        shrinkage = float(self.sample_gaussian_shrinkage)
        if shrinkage < 0.0 or shrinkage >= 1.0:
            raise ValueError("sample_gaussian_shrinkage must be in [0, 1).")
        if shrinkage > 0.0:
            target = torch.ones_like(p_var)
            q_var = (1.0 - shrinkage) * q_var + shrinkage * target
            p_var = (1.0 - shrinkage) * p_var + shrinkage * target

        q_var = q_var.clamp_min(self.min_bandwidth**2)
        p_var = p_var.clamp_min(self.min_bandwidth**2)
        kl = 0.5 * (
            q_var / p_var
            + (q_mean - p_mean).square() / p_var
            - 1.0
            + p_var.log()
            - q_var.log()
        )
        return (z.shape[-1] * kl.mean()).clamp_min(0.0)

    def _sample_sliced_quantile_transport_kl(self, z, w):
        """Sliced KL using 1-D quantile transport and spacing densities.

        For each projection, build the monotone transport map from posterior
        projected samples to prior projected samples by matching sorted
        quantiles. The density ratio is then estimated from the transport
        slope plus a spacing-density correction under the prior samples.
        """
        if self.num_projections <= 0:
            raise ValueError("num_projections must be positive.")
        z, w = _diagonal_standardize_by_reference(z, w, self.min_bandwidth)
        directions = _sample_sliced_projection_directions(
            z,
            w,
            self.num_projections,
            mode=self.sample_projection_mode,
            min_bandwidth=self.min_bandwidth,
            cache=self._fixed_projection_cache,
        )
        z_proj = z @ directions
        w_proj = w @ directions
        if z_proj.shape[0] < 2 or w_proj.shape[0] < 2:
            return self._sample_sliced_gaussian_kl(z, w)

        z_sorted = torch.sort(z_proj, dim=0).values
        w_sorted = torch.sort(w_proj.detach(), dim=0).values
        transported = _interpolate_sorted(w_sorted, z_sorted.shape[0])
        slope = _local_quantile_slopes(
            z_sorted,
            transported,
            k=self.quantile_transport_k,
            min_width=self.min_bandwidth,
        )
        log_p_transport = _spacing_log_density_1d(
            transported,
            w_sorted,
            k=self.quantile_transport_k,
            min_width=self.min_bandwidth,
        )
        log_p_z = _spacing_log_density_1d(
            z_sorted,
            w_sorted,
            k=self.quantile_transport_k,
            min_width=self.min_bandwidth,
        )
        kl = (log_p_transport - log_p_z + slope.log()).mean(dim=0)
        return (z.shape[-1] * kl.mean()).clamp_min(0.0)

    def _spectral_sliced_kl(
        self, posterior_values, prior_values, measurement_inputs, prior_function
    ):
        coeffs, eigvals = _prior_spectral_coefficients(
            posterior_values,
            measurement_inputs=measurement_inputs,
            prior_function=prior_function,
            prior_values=prior_values,
            num_modes=self.spectral_num_modes,
            jitter=self.min_bandwidth,
            detach_prior_eig=self.spectral_detach_prior_eig,
        )
        kl_per_mode = self._spectral_kl_per_mode(coeffs)
        weights = _spectral_mode_weights(
            eigvals,
            kl_per_mode.shape[0],
            mode=self.spectral_mode_weighting,
            power=self.spectral_eigenvalue_power,
        )
        return (weights * kl_per_mode).sum().clamp_min(0.0)

    def _spectral_kl_per_mode(self, coeffs):
        if coeffs.ndim != 2:
            raise ValueError("spectral coefficients must have shape [S, K].")
        if self.spectral_estimator not in ("gaussian", "cumulant"):
            raise ValueError(
                "spectral_estimator must be 'gaussian' or 'cumulant', "
                f"got {self.spectral_estimator!r}."
            )
        mean = coeffs.mean(dim=0)
        centered = coeffs - mean.view(1, -1)
        variance = coeffs.var(dim=0, unbiased=False).clamp_min(1e-6)
        kl = 0.5 * (mean.square() + variance - variance.log() - 1.0)
        if self.spectral_estimator == "gaussian":
            return kl

        alpha_3, alpha_4 = self.spectral_cumulant_weights
        standardized = centered / variance.sqrt().view(1, -1).clamp_min(1e-6)
        skew = standardized.pow(3).mean(dim=0)
        kurt = standardized.pow(4).mean(dim=0)
        return kl + alpha_3 * skew.square() + alpha_4 * (kurt - 3.0).square()

    def _spectral_projected_kl(
        self, posterior_values, prior_values, measurement_inputs, prior_function
    ):
        coeffs, _ = _prior_spectral_coefficients(
            posterior_values,
            measurement_inputs=measurement_inputs,
            prior_function=prior_function,
            prior_values=prior_values,
            num_modes=self.spectral_num_modes,
            jitter=self.min_bandwidth,
            detach_prior_eig=self.spectral_detach_prior_eig,
        )
        if self.spectral_estimator in ("gaussian", "full_gaussian"):
            return _full_gaussian_kl_to_standard(
                coeffs,
                jitter=self.min_bandwidth,
                shrinkage=self.spectral_cov_shrinkage,
            )
        if self.spectral_estimator == "knn_entropy":
            return _knn_kl_to_standard_normal(
                coeffs,
                k=self.spectral_knn_k,
                distance_eps=self.distance_eps,
            )
        raise ValueError(
            "spectral_projected_kl estimator must be 'full_gaussian' "
            f"or 'knn_entropy', got {self.spectral_estimator!r}."
        )

    def _prior_whitened_flattened(
        self, posterior_values, measurement_inputs, prior_function
    ):
        if measurement_inputs is None:
            raise ValueError(f"{self.kind} requires measurement_inputs.")
        if prior_function is None:
            raise ValueError(f"{self.kind} requires prior_function.")
        return _flatten_function_values(
            _prior_whiten_values(posterior_values, measurement_inputs, prior_function)
        )

    def _kde_entropy_1d(self, values):
        log_density = self._kde_log_density_1d(
            values, values, leave_one_out=values.shape[0] > 1
        )
        return -log_density.mean(dim=0)

    def _kde_log_density_1d(self, query, reference, leave_one_out=False):
        if query.ndim != 2 or reference.ndim != 2:
            raise ValueError("KDE inputs must have shape [num_samples, num_projections].")
        if query.shape[1] != reference.shape[1]:
            raise ValueError("KDE query/reference projection counts must match.")
        ref_count = reference.shape[0]
        if ref_count == 0:
            raise ValueError("KDE reference set must be non-empty.")
        if leave_one_out and ref_count < 2:
            return self._gaussian_log_density_1d(query)

        bandwidth = self._projection_bandwidth(query, reference)
        diff = query.unsqueeze(1) - reference.unsqueeze(0)
        log_kernel = (
            -0.5 * (diff / bandwidth.view(1, 1, -1)).square()
            - bandwidth.log().view(1, 1, -1)
            - 0.5 * math.log(2.0 * math.pi)
        )
        normalizer = ref_count
        if leave_one_out:
            if query.shape != reference.shape:
                raise ValueError("leave_one_out=True requires matching query/reference shapes.")
            mask = torch.eye(ref_count, dtype=torch.bool, device=query.device).unsqueeze(-1)
            log_kernel = log_kernel.masked_fill(mask, -torch.inf)
            normalizer = ref_count - 1
        return torch.logsumexp(log_kernel, dim=1) - math.log(normalizer)

    def _projection_bandwidth(self, query, reference):
        if isinstance(self.bandwidth, str):
            rule = self.bandwidth.lower()
            values = torch.cat([query.detach(), reference.detach()], dim=0)
            if rule == "silverman":
                std = values.std(dim=0, unbiased=False)
                n = max(values.shape[0], 1)
                scale = (4.0 / (3.0 * n)) ** 0.2
                return (scale * std).clamp_min(self.min_bandwidth)
            if rule == "median":
                diffs = (values.unsqueeze(1) - values.unsqueeze(0)).abs()
                positive = diffs.masked_fill(diffs <= 0, torch.nan)
                median = torch.nanmedian(positive, dim=0).values
                median = torch.nanmedian(median, dim=0).values
                std_fallback = values.std(dim=0, unbiased=False)
                fallback = std_fallback.clamp_min(self.min_bandwidth)
                median = torch.where(torch.isfinite(median), median, fallback)
                return median.clamp_min(self.min_bandwidth)
            raise ValueError(f"Unknown bandwidth rule: {self.bandwidth!r}")
        bandwidth = torch.as_tensor(self.bandwidth, dtype=query.dtype, device=query.device)
        if torch.any(bandwidth <= 0):
            raise ValueError("bandwidth must be positive.")
        return bandwidth.clamp_min(self.min_bandwidth).expand(query.shape[1])

    def _gaussian_log_density_1d(self, values):
        mean = values.detach().mean(dim=0)
        std = values.detach().std(dim=0, unbiased=False).clamp_min(self.min_bandwidth)
        return (
            -0.5 * ((values - mean) / std).square()
            - std.log()
            - 0.5 * math.log(2.0 * math.pi)
        )

    def _bandwidth_squared(self, z, w):
        if isinstance(self.bandwidth, str):
            if self.bandwidth != "median":
                raise ValueError(f"Unknown bandwidth rule: {self.bandwidth!r}")
            combined = torch.cat([z.detach(), w.detach()], dim=0)
            sqdist = _pairwise_squared_distances(combined, combined)
            positive = sqdist[sqdist > 0]
            if positive.numel() == 0:
                return torch.as_tensor(self.min_bandwidth**2, dtype=z.dtype, device=z.device)
            return positive.median().clamp_min(self.min_bandwidth**2)
        bandwidth = torch.as_tensor(self.bandwidth, dtype=z.dtype, device=z.device)
        if torch.any(bandwidth <= 0):
            raise ValueError("bandwidth must be positive.")
        return bandwidth.pow(2).clamp_min(self.min_bandwidth**2)


class MMDivergence(FunctionDiscrepancy):
    """Backward-compatible RBF-kernel MMD wrapper."""

    def __init__(self, bandwidth="median", estimator="biased", min_bandwidth=1e-6):
        super().__init__(
            kind="mmd",
            bandwidth=bandwidth,
            estimator=estimator,
            min_bandwidth=min_bandwidth,
        )


class APFSVI(torch.nn.Module):
    """Adaptive Projective Function-Space Variational Inference.

    This implementation is the proposed AP-FSVI-GP model for regression and
    classification:

    * supplied stochastic generative-function posterior,
    * finite measurement projections sampled from data/near-data/domain,
    * RBF GP function prior,
    * configurable regularizer over projected function values,
    * Gaussian, Bernoulli-probit, or multiclass softmax likelihood.
    """

    def __init__(
        self,
        generative_function,
        prior_function=None,
        input_dim=None,
        output_dim=1,
        likelihood="regression",
        num_classes=None,
        num_data=1,
        num_samples=16,
        num_prior_samples=None,
        num_measurement=64,
        beta=0.1,
        beta_start=None,
        beta_warmup_steps=0,
        data_pretrain_steps=0,
        data_loss="expected_nll",
        measurement_weights=(0.2, 0.2, 0.6),
        near_data_noise=0.1,
        domain_bounds=None,
        domain_std=2.0,
        domain_gap_sampling=False,
        domain_gap_candidate_multiplier=8,
        adaptive_measure_points=False,
        adaptive_measure_steps=3,
        adaptive_measure_lr=0.05,
        adaptive_measure_normalize_grad=True,
        adaptive_measure_domain_limit=None,
        adaptive_measure_mode="gradient",
        adaptive_measure_every=1,
        reuse_adaptive_measure_points=False,
        adaptive_measure_reuse_fraction=1.0,
        adaptive_candidate_pool_multiplier=4,
        adaptive_candidate_pool_size=None,
        adaptive_num_samples=None,
        adaptive_num_prior_samples=None,
        adaptive_num_projections=None,
        function_discrepancy="mmd",
        mmd_bandwidth="median",
        mmd_estimator="biased",
        discrepancy_num_projections=64,
        sinkhorn_epsilon=1.0,
        sinkhorn_iterations=50,
        sinkhorn_debiased=False,
        spectral_num_modes=None,
        spectral_estimator="gaussian",
        spectral_cumulant_weights=(0.05, 0.01),
        spectral_detach_prior_eig=True,
        spectral_mode_weighting="uniform",
        spectral_eigenvalue_power=1.0,
        spectral_cov_shrinkage=0.05,
        spectral_knn_k=3,
        sample_knn_k=3,
        sample_gaussian_shrinkage=0.05,
        sample_projection_mode="random",
        quantile_transport_k=3,
        fixed_measure_points=False,
        prior_kernel_amp=1.0,
        prior_kernel_length=1.0,
        prior_jitter=1e-6,
        reservoir_size=1000,
        y_mean=0.0,
        y_std=1.0,
        log_variance_init=-2.0,
        classification_epsilon=1e-3,
        center_multiclass_logits=True,
        max_grad_norm=None,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__()
        if likelihood not in ("regression", "binary", "multiclass"):
            raise ValueError(
                "likelihood must be 'regression', 'binary', or 'multiclass', "
                f"got {likelihood!r}."
            )
        if generative_function is None:
            raise ValueError("APFSVI requires a generative_function posterior.")
        if input_dim is None:
            input_dim = getattr(generative_function, "input_dim", None)
        if input_dim is None and prior_function is None:
            raise ValueError("input_dim is required when prior_function is not provided.")
        output_dim = getattr(generative_function, "output_dim", output_dim)
        if likelihood == "binary" and output_dim != 1:
            raise ValueError(
                "binary APFSVI expects a scalar logit posterior with output_dim=1, "
                f"got output_dim={output_dim}."
            )
        if likelihood == "multiclass":
            if num_classes is None:
                num_classes = output_dim
            if num_classes != output_dim:
                raise ValueError(
                    "multiclass APFSVI expects output_dim to equal num_classes, "
                    f"got output_dim={output_dim}, num_classes={num_classes}."
                )
            if classification_epsilon <= 0 or classification_epsilon >= 1:
                raise ValueError("classification_epsilon must be in (0, 1).")

        self.likelihood_type = likelihood
        self.num_classes = num_classes
        self.epsilon = classification_epsilon
        self.num_data = num_data
        self.num_samples = num_samples
        self.num_prior_samples = num_prior_samples or num_samples
        self.num_measurement = num_measurement
        self.beta = beta
        self.beta_start = beta if beta_start is None else beta_start
        self.beta_warmup_steps = beta_warmup_steps
        self.data_pretrain_steps = data_pretrain_steps
        self.data_loss = data_loss
        self.near_data_noise = near_data_noise
        self.domain_std = domain_std
        self.domain_gap_sampling = domain_gap_sampling
        self.domain_gap_candidate_multiplier = domain_gap_candidate_multiplier
        self.adaptive_measure_points = adaptive_measure_points
        self.adaptive_measure_steps = adaptive_measure_steps
        self.adaptive_measure_lr = adaptive_measure_lr
        self.adaptive_measure_normalize_grad = adaptive_measure_normalize_grad
        self.adaptive_measure_domain_limit = adaptive_measure_domain_limit
        self.adaptive_measure_mode = adaptive_measure_mode
        self.adaptive_measure_every = adaptive_measure_every
        self.reuse_adaptive_measure_points = reuse_adaptive_measure_points
        self.adaptive_measure_reuse_fraction = adaptive_measure_reuse_fraction
        self.adaptive_candidate_pool_multiplier = adaptive_candidate_pool_multiplier
        self.adaptive_candidate_pool_size = adaptive_candidate_pool_size
        self.adaptive_num_samples = adaptive_num_samples or min(num_samples, 8)
        self.adaptive_num_prior_samples = adaptive_num_prior_samples or min(
            self.num_prior_samples, 8
        )
        self.adaptive_num_projections = adaptive_num_projections or min(
            discrepancy_num_projections, 16
        )
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.dtype = dtype
        self.output_dim = output_dim
        self.center_multiclass_logits = center_multiclass_logits
        self.fixed_measure_points = fixed_measure_points
        self._step = 0
        self._reservoir_size = reservoir_size
        self._reservoir = None
        self._last_adaptive_measure = None
        self._fixed_measurement_set = None

        if self.adaptive_measure_steps < 0:
            raise ValueError("adaptive_measure_steps must be non-negative.")
        if self.adaptive_measure_lr < 0:
            raise ValueError("adaptive_measure_lr must be non-negative.")
        if self.adaptive_measure_mode not in (
            "gradient",
            "candidate",
            "candidate_then_one_step",
        ):
            raise ValueError(
                "adaptive_measure_mode must be 'gradient', 'candidate', or "
                "'candidate_then_one_step'."
            )
        if self.adaptive_measure_every <= 0:
            raise ValueError("adaptive_measure_every must be positive.")
        if (
            self.adaptive_measure_reuse_fraction < 0.0
            or self.adaptive_measure_reuse_fraction > 1.0
        ):
            raise ValueError("adaptive_measure_reuse_fraction must be in [0, 1].")
        if self.adaptive_candidate_pool_multiplier < 1:
            raise ValueError("adaptive_candidate_pool_multiplier must be >= 1.")
        if (
            self.adaptive_candidate_pool_size is not None
            and self.adaptive_candidate_pool_size <= 0
        ):
            raise ValueError("adaptive_candidate_pool_size must be positive.")
        if self.adaptive_num_samples <= 0:
            raise ValueError("adaptive_num_samples must be positive.")
        if self.adaptive_num_prior_samples <= 0:
            raise ValueError("adaptive_num_prior_samples must be positive.")
        if self.adaptive_num_projections <= 0:
            raise ValueError("adaptive_num_projections must be positive.")
        if (
            self.adaptive_measure_domain_limit is not None
            and self.adaptive_measure_domain_limit <= 0
        ):
            raise ValueError("adaptive_measure_domain_limit must be positive.")
        if self.domain_gap_candidate_multiplier < 1:
            raise ValueError("domain_gap_candidate_multiplier must be >= 1.")

        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=dtype, device=device))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=dtype, device=device))
        self.register_buffer(
            "measurement_weights",
            _normalize_weights(measurement_weights, dtype=dtype, device=device),
        )
        if domain_bounds is None:
            self.domain_bounds = None
        else:
            bounds = torch.as_tensor(domain_bounds, dtype=dtype, device=device)
            if bounds.ndim == 1 and bounds.numel() == 2:
                bounds = bounds.view(1, 2)
            self.register_buffer("domain_bounds", bounds)

        self._generator_device = _generator_device(device)
        self.generator = torch.Generator(self._generator_device)
        self.generator.manual_seed(seed)
        self._cpu_generator = torch.Generator()
        self._cpu_generator.manual_seed(seed)

        self.generative_function = generative_function

        if prior_function is None:
            prior_function = ExactGP(
                num_samples=self.num_prior_samples,
                input_dim=input_dim,
                output_dim=output_dim,
                kernel_amp=prior_kernel_amp,
                kernel_length=prior_kernel_length,
                jitter=prior_jitter,
                fix_random_noise=False,
                device=device,
                dtype=dtype,
                seed=seed + 1,
            )
            prior_function.freeze_parameters()
        self.prior_function = prior_function
        for param in self.prior_function.parameters():
            param.requires_grad = False

        self.function_discrepancy = _normalize_discrepancy_kind(function_discrepancy)
        if (
            self.function_discrepancy == "stein"
            and not _prior_supports_stein_score(self.prior_function)
        ):
            warnings.warn(
                "AP-FSVI Stein discrepancy needs a prior score. The current "
                "built-in Stein score is GP-specific, and the supplied "
                "prior_function does not expose score(X, values) or an "
                "ExactGP-like _rbf method. Training will fail when the Stein "
                "regularizer is evaluated unless the prior supplies a score.",
                UserWarning,
                stacklevel=2,
            )
        self.divergence = FunctionDiscrepancy.build(
            kind=self.function_discrepancy,
            bandwidth=mmd_bandwidth,
            estimator=mmd_estimator,
            num_projections=discrepancy_num_projections,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_debiased=sinkhorn_debiased,
            spectral_num_modes=spectral_num_modes,
            spectral_estimator=spectral_estimator,
            spectral_cumulant_weights=spectral_cumulant_weights,
            spectral_detach_prior_eig=spectral_detach_prior_eig,
            spectral_mode_weighting=spectral_mode_weighting,
            spectral_eigenvalue_power=spectral_eigenvalue_power,
            spectral_cov_shrinkage=spectral_cov_shrinkage,
            spectral_knn_k=spectral_knn_k,
            sample_knn_k=sample_knn_k,
            sample_gaussian_shrinkage=sample_gaussian_shrinkage,
            sample_projection_mode=sample_projection_mode,
            quantile_transport_k=quantile_transport_k,
        )
        if self.likelihood_type == "regression":
            log_variance = torch.as_tensor(log_variance_init, dtype=dtype, device=device)
            if log_variance.ndim == 0:
                log_variance = log_variance.expand(output_dim).clone()
            elif log_variance.shape != (output_dim,):
                raise ValueError(
                    "log_variance_init must be a scalar or have shape "
                    f"({output_dim},), got {tuple(log_variance.shape)}"
                )
            self.log_variance = torch.nn.Parameter(log_variance.clone())

        self.data_terms = []
        self.KLs = []
        self.function_terms = []
        self.betas = []
        self.adaptive_measure_displacement_means = []
        self.adaptive_measure_displacement_maxes = []
        self.adaptive_measure_relative_displacement_means = []

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def predict_f_samples(self, X, S):
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        sample = getattr(self.generative_function, "sample", None)
        if callable(sample):
            return sample(X, S)
        return self._forward_generator(X, S)

    def _forward_generator(self, X, S):
        try:
            return self.generative_function(X, num_samples=S)
        except TypeError:
            pass

        old_states = _set_num_samples_recursive(self.generative_function, S)
        try:
            try:
                return self.generative_function(X)
            except TypeError:
                return self.generative_function(X, S)
        finally:
            _restore_num_samples_recursive(old_states)

    def predict_y_samples(self, X, S):
        F = self.predict_f_samples(X, S)
        if self.likelihood_type == "binary":
            return inv_probit(F)
        if self.likelihood_type == "multiclass":
            return F
        std = torch.sqrt(torch.exp(self.log_variance)).view(1, 1, -1)
        return F + std * torch.randn(
            F.shape, generator=self.generator, dtype=F.dtype, device=F.device
        )

    def forward(self, X):
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.likelihood_type != "regression":
            return self.predict_y_samples(X, self.num_samples)
        F = self.predict_f_samples(X, self.num_samples)
        samples = F * self.y_std + self.y_mean
        std = torch.sqrt(torch.exp(self.log_variance)).view(1, 1, -1)
        std = std.expand_as(F) * self.y_std
        return samples, std

    def predict(self, X, S):
        self.eval()
        with torch.no_grad():
            if self.dtype != X.dtype:
                X = X.to(self.dtype)
            Y = self.predict_y_samples(X, S)
            if self.likelihood_type != "regression":
                return Y
            return Y * self.y_std + self.y_mean

    def forward_prior(self, X, num_samples):
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        prior = self._sample_prior(X, num_samples)
        if self.likelihood_type != "regression":
            return prior
        return prior * self.y_std + self.y_mean

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nelbo(self, X, y):
        X = X.to(dtype=self.dtype, device=self.device)
        y = self._prepare_targets(y)
        N_batch = X.shape[0]

        beta = self._scheduled_beta()

        X_measure = self._sample_measurement_set(X)
        X_measure = self._reuse_or_fresh_measurement_set(X_measure)
        M = X_measure.shape[0]
        if M > 0 and beta > 0 and self._should_adapt_measurement_points():
            X_measure = self._adapt_measurement_points(X_measure)
            if self.reuse_adaptive_measure_points:
                self._last_adaptive_measure = X_measure.detach()
        if M > 0:
            X_joint = torch.cat([X_measure, X], dim=0)
        else:
            X_joint = X
        F_joint = self.predict_f_samples(X_joint, self.num_samples)
        F_measure = F_joint[:, :M, :] if M > 0 else None
        F_batch = F_joint[:, M:, :]

        logpdf = self._logp(F_batch, y)
        if self.data_loss in ("expected_nll", "expected", "elbo"):
            ve = torch.mean(logpdf, dim=0)
        elif self.data_loss in ("predictive_nll", "mixture_nll", "log_mean_exp"):
            ve = torch.logsumexp(logpdf, dim=0) - math.log(F_batch.shape[0])
        else:
            raise ValueError(f"Unknown data_loss mode: {self.data_loss!r}")
        ve = torch.sum(ve)
        scale = self.num_data / N_batch
        data_term = -scale * ve

        if M > 0:
            prior_values = (
                self._sample_prior(X_measure, self.num_prior_samples)
                if self._regularizer_requires_prior_samples()
                else None
            )
            prior_loss = self.divergence(
                self._regularizer_values(F_measure),
                self._regularizer_values(prior_values),
                measurement_inputs=X_measure,
                prior_function=self.prior_function,
            )
        else:
            prior_loss = torch.zeros((), dtype=self.dtype, device=self.device)

        loss = data_term + beta * prior_loss
        self.data_terms.append(data_term.detach())
        self.KLs.append(prior_loss.detach())
        self.function_terms.append(prior_loss.detach())
        self.betas.append(beta)
        return loss

    def _prepare_targets(self, y):
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        target_device = self.device if self.device is not None else y.device
        if self.likelihood_type == "multiclass":
            return y.to(device=target_device)
        return y.to(dtype=self.dtype, device=target_device)

    def _logp(self, F, y):
        if self.likelihood_type == "regression":
            return gaussian_logp(F, y, self.log_variance)
        if self.likelihood_type == "binary":
            return bernoulli_logp(F, y)
        return multiclass_logp(F, y, self.num_classes, self.epsilon)

    def _regularizer_values(self, values):
        if values is None:
            return None
        if (
            self.likelihood_type == "multiclass"
            and self.center_multiclass_logits
            and self.function_discrepancy != "stein"
        ):
            return values - values.mean(dim=-1, keepdim=True)
        return values

    def _sample_prior(self, X, S):
        try:
            return self.prior_function(X, num_samples=S)
        except TypeError:
            states = _set_num_samples_recursive(self.prior_function, S)
            try:
                return self.prior_function(X)
            finally:
                _restore_num_samples_recursive(states)

    def _regularizer_requires_prior_samples(self):
        if self.divergence.requires_prior_samples:
            return True
        return (
            self.function_discrepancy
            in ("spectral_sliced_kl", "spectral_projected_kl")
            and not _prior_has_covariance(self.prior_function)
        )

    def _scheduled_beta(self):
        step = self._step
        if step <= self.data_pretrain_steps:
            return 0.0
        if self.beta_warmup_steps <= 0:
            return self.beta
        warmup_step = min(max(step - self.data_pretrain_steps, 0), self.beta_warmup_steps)
        progress = warmup_step / self.beta_warmup_steps
        return self.beta_start + progress * (self.beta - self.beta_start)

    # ------------------------------------------------------------------
    # Measurement sets
    # ------------------------------------------------------------------

    def _sample_measurement_set(self, X_batch):
        if self.num_measurement <= 0:
            return X_batch[:0]
        if self.fixed_measure_points and self._fixed_measurement_set is not None:
            return self._fixed_measurement_set.to(
                dtype=X_batch.dtype,
                device=X_batch.device,
            )
        return self._sample_measurement_set_for_count(X_batch, self.num_measurement)

    def _sample_measurement_set_for_count(self, X_batch, count):
        if count <= 0:
            return X_batch[:0]
        counts = _allocate_counts(self.measurement_weights, count)
        parts = []

        if counts[0] > 0:
            parts.append(self._sample_data_points(X_batch, counts[0]))
        if counts[1] > 0:
            base = self._sample_data_points(X_batch, counts[1])
            near = base + self.near_data_noise * torch.randn(
                base.shape, generator=self.generator, dtype=base.dtype, device=base.device
            )
            parts.append(self._clip_to_domain(near))
        if counts[2] > 0:
            parts.append(self._sample_domain_points(X_batch, counts[2]))

        X_measure = torch.cat(parts, dim=0)
        perm = torch.randperm(X_measure.shape[0], generator=self.generator, device=X_measure.device)
        return X_measure[perm]

    def _initialize_fixed_measurement_set(self, train_loader):
        if not self.fixed_measure_points or self.num_measurement <= 0:
            self._fixed_measurement_set = None
            return
        try:
            X_batch, _ = next(iter(train_loader))
        except StopIteration:
            self._fixed_measurement_set = None
            return
        target_device = self.device if self.device is not None else X_batch.device
        X_batch = X_batch.to(dtype=self.dtype, device=target_device)
        self._fixed_measurement_set = self._sample_measurement_set_for_count(
            X_batch,
            self.num_measurement,
        ).detach()

    def _reuse_or_fresh_measurement_set(self, X_measure):
        if not self.reuse_adaptive_measure_points:
            return X_measure
        cached = self._last_adaptive_measure
        if cached is None:
            return X_measure
        if cached.shape != X_measure.shape:
            return X_measure
        fraction = self.adaptive_measure_reuse_fraction
        if fraction <= 0.0 or X_measure.shape[0] == 0:
            return X_measure
        cached = cached.to(dtype=X_measure.dtype, device=X_measure.device).detach()
        if fraction >= 1.0:
            return cached

        num_cached = int(round(fraction * X_measure.shape[0]))
        num_cached = min(max(num_cached, 0), X_measure.shape[0])
        if num_cached == 0:
            return X_measure
        if num_cached == X_measure.shape[0]:
            return cached

        perm = torch.randperm(
            X_measure.shape[0],
            generator=self.generator,
            device=X_measure.device,
        )
        mixed = X_measure.clone()
        mixed[perm[:num_cached]] = cached[perm[:num_cached]]
        return mixed

    def _adapt_measurement_points(self, X_measure):
        if (
            not self.adaptive_measure_points
            or (
                self.adaptive_measure_mode != "candidate"
                and (self.adaptive_measure_steps == 0 or self.adaptive_measure_lr == 0)
            )
            or X_measure.numel() == 0
        ):
            return X_measure.detach()

        if self.adaptive_measure_mode == "candidate":
            return self._select_adaptive_candidates(X_measure).detach()
        if self.adaptive_measure_mode == "candidate_then_one_step":
            X_selected = self._select_adaptive_candidates(X_measure)
            return self._gradient_adapt_measurement_points(X_selected, steps=1)
        return self._gradient_adapt_measurement_points(
            X_measure, steps=self.adaptive_measure_steps
        )

    def _should_adapt_measurement_points(self):
        if not self.adaptive_measure_points:
            return False
        return self._step % self.adaptive_measure_every == 0

    def _select_adaptive_candidates(self, X_measure):
        M = X_measure.shape[0]
        pool_size = self.adaptive_candidate_pool_size
        if pool_size is None:
            pool_size = int(M * self.adaptive_candidate_pool_multiplier)
        pool_size = max(M, pool_size)

        X_pool = self._sample_measurement_set_for_count(X_measure, pool_size)
        with torch.enable_grad():
            scores = self._pointwise_adaptation_scores(X_pool)
        topk = torch.topk(scores.detach(), k=M, largest=True).indices
        return X_pool[topk].detach()

    def _pointwise_adaptation_scores(self, X_pool):
        S = min(self.adaptive_num_samples, self.num_samples)
        S_prior = min(self.adaptive_num_prior_samples, self.num_prior_samples)
        F_pool = self.predict_f_samples(X_pool, S)
        values = self._regularizer_values(F_pool)

        if self.function_discrepancy == "prior_whitened_sliced_kl":
            return _prior_whitened_point_scores(values, X_pool, self.prior_function)
        if (
            self.function_discrepancy
            in ("spectral_sliced_kl", "spectral_projected_kl")
            and _prior_has_covariance(self.prior_function)
        ):
            return _prior_whitened_point_scores(values, X_pool, self.prior_function)

        prior_values = (
            self._sample_prior(X_pool, S_prior)
            if self._regularizer_requires_prior_samples()
            else None
        )
        prior_values = self._regularizer_values(prior_values)
        if self.function_discrepancy in (
            "sample_sliced_kl",
            "sample_sliced_knn_kl",
            "sample_sliced_gaussian_kl",
            "sample_sliced_quantile_transport_kl",
        ):
            return _sample_sliced_point_scores(
                values,
                prior_values,
                min_bandwidth=self.divergence.min_bandwidth,
            )

        return self._cheap_adaptation_divergence(
            values,
            prior_values,
            X_pool,
        )

    def _cheap_adaptation_divergence(self, posterior_values, prior_values, X_pool):
        cheap = FunctionDiscrepancy.build(
            kind=self.function_discrepancy,
            bandwidth=self.divergence.bandwidth,
            estimator=self.divergence.estimator,
            min_bandwidth=self.divergence.min_bandwidth,
            num_projections=self.adaptive_num_projections,
            sinkhorn_epsilon=self.divergence.sinkhorn_epsilon,
            sinkhorn_iterations=max(1, min(self.divergence.sinkhorn_iterations, 10)),
            sinkhorn_debiased=False,
            distance_eps=self.divergence.distance_eps,
            spectral_num_modes=min(
                self.divergence.spectral_num_modes,
                self.adaptive_num_projections,
            ),
            spectral_estimator=self.divergence.spectral_estimator,
            spectral_cumulant_weights=self.divergence.spectral_cumulant_weights,
            spectral_detach_prior_eig=self.divergence.spectral_detach_prior_eig,
            spectral_mode_weighting=self.divergence.spectral_mode_weighting,
            spectral_eigenvalue_power=self.divergence.spectral_eigenvalue_power,
            spectral_cov_shrinkage=self.divergence.spectral_cov_shrinkage,
            spectral_knn_k=self.divergence.spectral_knn_k,
            sample_knn_k=self.divergence.sample_knn_k,
            sample_projection_mode=self.divergence.sample_projection_mode,
            quantile_transport_k=self.divergence.quantile_transport_k,
        )
        values = []
        for i in range(X_pool.shape[0]):
            post_i = posterior_values[:, i : i + 1, :]
            prior_i = prior_values[:, i : i + 1, :] if prior_values is not None else None
            x_i = X_pool[i : i + 1]
            values.append(
                cheap(
                    post_i,
                    prior_i,
                    measurement_inputs=x_i,
                    prior_function=self.prior_function,
                )
            )
        return torch.stack(values)

    def _gradient_adapt_measurement_points(self, X_measure, steps):
        X_adv = X_measure.detach()
        X_start = X_adv
        for _ in range(steps):
            X_adv = X_adv.detach().requires_grad_(True)
            with torch.enable_grad():
                F_adv = self.predict_f_samples(
                    X_adv, min(self.adaptive_num_samples, self.num_samples)
                )
                prior_values = (
                    self._sample_prior(
                        X_adv,
                        min(self.adaptive_num_prior_samples, self.num_prior_samples),
                    )
                    if self._regularizer_requires_prior_samples()
                    else None
                )
                score = self._adaptation_divergence()(
                    self._regularizer_values(F_adv),
                    self._regularizer_values(prior_values),
                    measurement_inputs=X_adv,
                    prior_function=self.prior_function,
                )
                if not score.requires_grad:
                    break
                grad = torch.autograd.grad(
                    score,
                    X_adv,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
            if grad is None:
                break
            if self.adaptive_measure_normalize_grad:
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                grad = grad / grad_norm
            X_adv = X_adv + self.adaptive_measure_lr * grad
            X_adv = self._project_adaptive_measurement_points(X_adv)
        X_adv = X_adv.detach()
        self._record_adaptive_measurement_displacement(X_start, X_adv)
        return X_adv

    def _record_adaptive_measurement_displacement(self, X_before, X_after):
        if X_before.numel() == 0 or X_before.shape != X_after.shape:
            return
        delta = (X_after - X_before).detach()
        distances = delta.reshape(delta.shape[0], -1).norm(dim=-1)
        base_scale = X_before.detach().reshape(X_before.shape[0], -1).std(
            dim=0,
            unbiased=False,
        )
        relative_scale = base_scale.square().sum().sqrt().clamp_min(1e-12)
        self.adaptive_measure_displacement_means.append(distances.mean().detach())
        self.adaptive_measure_displacement_maxes.append(distances.max().detach())
        self.adaptive_measure_relative_displacement_means.append(
            (distances.mean() / relative_scale).detach()
        )

    def _adaptation_divergence(self):
        if not hasattr(self.divergence, "num_projections"):
            return self.divergence
        if (
            self.adaptive_num_projections == self.divergence.num_projections
            and self.function_discrepancy not in ("sinkhorn",)
        ):
            return self.divergence
        return FunctionDiscrepancy.build(
            kind=self.function_discrepancy,
            bandwidth=self.divergence.bandwidth,
            estimator=self.divergence.estimator,
            min_bandwidth=self.divergence.min_bandwidth,
            num_projections=self.adaptive_num_projections,
            sinkhorn_epsilon=self.divergence.sinkhorn_epsilon,
            sinkhorn_iterations=max(1, min(self.divergence.sinkhorn_iterations, 10)),
            sinkhorn_debiased=False,
            distance_eps=self.divergence.distance_eps,
            spectral_num_modes=min(
                self.divergence.spectral_num_modes,
                self.adaptive_num_projections,
            ),
            spectral_estimator=self.divergence.spectral_estimator,
            spectral_cumulant_weights=self.divergence.spectral_cumulant_weights,
            spectral_detach_prior_eig=self.divergence.spectral_detach_prior_eig,
            spectral_mode_weighting=self.divergence.spectral_mode_weighting,
            spectral_eigenvalue_power=self.divergence.spectral_eigenvalue_power,
            spectral_cov_shrinkage=self.divergence.spectral_cov_shrinkage,
            spectral_knn_k=self.divergence.spectral_knn_k,
            sample_knn_k=self.divergence.sample_knn_k,
            sample_projection_mode=self.divergence.sample_projection_mode,
            quantile_transport_k=self.divergence.quantile_transport_k,
        )

    def _project_adaptive_measurement_points(self, X):
        if self.domain_bounds is not None:
            return self._clip_to_domain(X)
        limit = self.adaptive_measure_domain_limit
        if limit is None and self.domain_std is not None:
            limit = 3.0 * self.domain_std
        if limit is None:
            return X
        return X.clamp(min=-limit, max=limit)

    def _sample_data_points(self, X_batch, count):
        source = self._reservoir
        if source is None:
            source = X_batch.detach()
        source = source.to(dtype=X_batch.dtype, device=X_batch.device)
        idx = torch.randint(
            source.shape[0],
            (count,),
            generator=self.generator,
            device=X_batch.device,
        )
        return source[idx]

    def _sample_domain_points(self, X_batch, count):
        if self.domain_gap_sampling and self._reservoir is not None and count > 0:
            return self._sample_low_density_domain_points(X_batch, count)
        return self._sample_uniform_domain_points(X_batch, count)

    def _sample_uniform_domain_points(self, X_batch, count):
        D = X_batch.shape[-1]
        if self.domain_bounds is None:
            return torch.randn(
                count, D, generator=self.generator,
                dtype=X_batch.dtype, device=X_batch.device,
            ) * self.domain_std
        bounds = self.domain_bounds.to(dtype=X_batch.dtype, device=X_batch.device)
        if bounds.shape[0] == 1:
            low = bounds[:, 0].expand(1, D)
            high = bounds[:, 1].expand(1, D)
        else:
            low = bounds[:, 0].view(1, D)
            high = bounds[:, 1].view(1, D)
        unit = torch.rand(count, D, generator=self.generator, dtype=X_batch.dtype, device=X_batch.device)
        return low + unit * (high - low)

    def _sample_low_density_domain_points(self, X_batch, count):
        pool_size = max(count, int(count * self.domain_gap_candidate_multiplier))
        X_pool = self._sample_uniform_domain_points(X_batch, pool_size)
        reservoir = self._reservoir.to(dtype=X_batch.dtype, device=X_batch.device)
        if reservoir.numel() == 0:
            return X_pool[:count]

        # Select domain candidates farthest from the nearest observed input.
        # Chunking avoids a large candidate-by-reservoir distance matrix on
        # higher-dimensional problems.
        scores = []
        chunk_size = 512
        for start in range(0, X_pool.shape[0], chunk_size):
            chunk = X_pool[start:start + chunk_size]
            dist2 = torch.cdist(chunk, reservoir).pow(2)
            scores.append(dist2.min(dim=1).values)
        scores = torch.cat(scores, dim=0)
        topk = torch.topk(scores, k=count, largest=True).indices
        return X_pool[topk]

    def _clip_to_domain(self, X):
        if self.domain_bounds is None:
            return X
        bounds = self.domain_bounds.to(dtype=X.dtype, device=X.device)
        D = X.shape[-1]
        if bounds.shape[0] == 1:
            low = bounds[:, 0].expand(1, D)
            high = bounds[:, 1].expand(1, D)
        else:
            low = bounds[:, 0].view(1, D)
            high = bounds[:, 1].view(1, D)
        return torch.maximum(torch.minimum(X, high), low)

    def _fill_reservoir(self, train_loader):
        R = self._reservoir_size
        if R == 0:
            return
        inputs = []
        seen = 0
        reservoir = None
        for batch_X, _ in train_loader:
            batch_X = batch_X.detach()
            if reservoir is None:
                reservoir = torch.empty(R, batch_X.shape[-1], dtype=batch_X.dtype)
            for row in batch_X:
                if seen < R:
                    reservoir[seen] = row
                else:
                    j = torch.randint(0, seen + 1, (1,), generator=self._cpu_generator).item()
                    if j < R:
                        reservoir[j] = row
                seen += 1
        if reservoir is None:
            self._reservoir = None
        else:
            self._reservoir = reservoir[: min(seen, R)]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_loader, optimizer=None, lr=0.001, epochs=None,
            iterations=None, use_tqdm=False, return_loss=False,
            cosine_annealing=False):
        self._fill_reservoir(train_loader)
        self._initialize_fixed_measurement_set(train_loader)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if epochs is None and iterations is None:
            raise ValueError("Either epochs or iterations must be set.")

        scheduler = None
        if cosine_annealing:
            T_max = epochs if epochs is not None else max(1, iterations // len(train_loader))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=lr / 100
            )

        self.train()
        losses = []
        if epochs is not None:
            loop = tqdm(range(epochs), unit=" epoch", desc="Training") if use_tqdm else range(epochs)
            for _ in loop:
                for inputs, target in train_loader:
                    loss = self._train_step(optimizer, inputs, target)
                    if return_loss:
                        losses.append(loss.detach().cpu().numpy())
                if scheduler is not None:
                    scheduler.step()

        if iterations is not None:
            data_stream = infinite_loader(train_loader)
            iters_per_epoch = len(train_loader)
            loop = tqdm(range(iterations), unit=" iter", desc="Training") if use_tqdm else range(iterations)
            for i in loop:
                inputs, target = next(data_stream)
                loss = self._train_step(optimizer, inputs, target)
                if return_loss:
                    losses.append(loss.detach().cpu().numpy())
                if scheduler is not None and (i + 1) % iters_per_epoch == 0:
                    scheduler.step()
        return losses

    def _train_step(self, optimizer, X, y):
        X = X.to(dtype=self.dtype, device=self.device)
        y = self._prepare_targets(y)
        self._step += 1
        optimizer.zero_grad(set_to_none=True)
        loss = self.nelbo(X, y)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()
        return loss


def _generator_device(device):
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _normalize_discrepancy_kind(kind):
    aliases = {
        "mmd": "mmd",
        "energy": "energy",
        "energy_distance": "energy",
        "sliced_wasserstein": "sliced_wasserstein",
        "sliced-wasserstein": "sliced_wasserstein",
        "sw": "sliced_wasserstein",
        "stein": "stein",
        "ksd": "stein",
        "stein_ksd": "stein",
        "sinkhorn": "sinkhorn",
        "sinkhorn_ot": "sinkhorn",
        "sinkhorn-ot": "sinkhorn",
        "prior_whitened_gaussian_kl": "prior_whitened_gaussian_kl",
        "prior-whitened-gaussian-kl": "prior_whitened_gaussian_kl",
        "pwgkl": "prior_whitened_gaussian_kl",
        "prior_whitened_sliced_kl": "prior_whitened_sliced_kl",
        "prior-whitened-sliced-kl": "prior_whitened_sliced_kl",
        "pwskl": "prior_whitened_sliced_kl",
        "spectral_sliced_kl": "spectral_sliced_kl",
        "spectral-sliced-kl": "spectral_sliced_kl",
        "prior_whitened_spectral_sliced_kl": "spectral_sliced_kl",
        "prior-whitened-spectral-sliced-kl": "spectral_sliced_kl",
        "spectral": "spectral_sliced_kl",
        "spkl": "spectral_sliced_kl",
        "spectral_projected_kl": "spectral_projected_kl",
        "spectral-projected-kl": "spectral_projected_kl",
        "prior_whitened_spectral_projected_kl": "spectral_projected_kl",
        "prior-whitened-spectral-projected-kl": "spectral_projected_kl",
        "projected_spectral_kl": "spectral_projected_kl",
        "projected-spectral-kl": "spectral_projected_kl",
        "spjkl": "spectral_projected_kl",
        "sample_sliced_kl": "sample_sliced_kl",
        "sample-sliced-kl": "sample_sliced_kl",
        "sskl": "sample_sliced_kl",
        "sample_sliced_gaussian_kl": "sample_sliced_gaussian_kl",
        "sample-sliced-gaussian-kl": "sample_sliced_gaussian_kl",
        "sample_gaussian_sliced_kl": "sample_sliced_gaussian_kl",
        "sample-gaussian-sliced-kl": "sample_sliced_gaussian_kl",
        "ssgkl": "sample_sliced_gaussian_kl",
        "sample_sliced_knn_kl": "sample_sliced_knn_kl",
        "sample-sliced-knn-kl": "sample_sliced_knn_kl",
        "sliced_knn_kl": "sample_sliced_knn_kl",
        "sliced-knn-kl": "sample_sliced_knn_kl",
        "sample_knn_kl": "sample_sliced_knn_kl",
        "sample-knn-kl": "sample_sliced_knn_kl",
        "sample_sliced_spacing_kl": "sample_sliced_knn_kl",
        "sample-sliced-spacing-kl": "sample_sliced_knn_kl",
        "sliced_spacing_kl": "sample_sliced_knn_kl",
        "sliced-spacing-kl": "sample_sliced_knn_kl",
        "ssknnkl": "sample_sliced_knn_kl",
        "sample_sliced_quantile_transport_kl": "sample_sliced_quantile_transport_kl",
        "sample-sliced-quantile-transport-kl": "sample_sliced_quantile_transport_kl",
        "sliced_quantile_transport_kl": "sample_sliced_quantile_transport_kl",
        "sliced-quantile-transport-kl": "sample_sliced_quantile_transport_kl",
        "sample_quantile_transport_kl": "sample_sliced_quantile_transport_kl",
        "sample-quantile-transport-kl": "sample_sliced_quantile_transport_kl",
        "sqtkl": "sample_sliced_quantile_transport_kl",
    }
    normalized = aliases.get(str(kind).lower())
    if normalized is None:
        valid = ", ".join(FunctionDiscrepancy.VALID_KINDS)
        raise ValueError(f"Unknown function_discrepancy {kind!r}. Valid options: {valid}.")
    return normalized


def _normalize_sample_projection_mode(mode):
    aliases = {
        "random": "random",
        "rand": "random",
        "fixed_random": "fixed_random",
        "fixed-random": "fixed_random",
        "fixed_rand": "fixed_random",
        "fixed-rand": "fixed_random",
        "prior_pca": "prior_pca",
        "prior-pca": "prior_pca",
        "prior": "prior_pca",
        "pca": "prior_pca",
        "discrepancy_pca": "discrepancy_pca",
        "discrepancy-pca": "discrepancy_pca",
        "diff_pca": "discrepancy_pca",
        "diff-pca": "discrepancy_pca",
        "delta_pca": "discrepancy_pca",
        "delta-pca": "discrepancy_pca",
        "fixed_orthogonal": "fixed_orthogonal",
        "fixed-orthogonal": "fixed_orthogonal",
        "orthogonal": "fixed_orthogonal",
        "ortho": "fixed_orthogonal",
    }
    normalized = aliases.get(str(mode).lower().strip(), str(mode).lower().strip())
    if normalized not in (
        "random",
        "fixed_random",
        "prior_pca",
        "discrepancy_pca",
        "fixed_orthogonal",
    ):
        raise ValueError(
            "sample_projection_mode must be 'random', 'fixed_random', "
            "'prior_pca', 'discrepancy_pca', or 'fixed_orthogonal', "
            f"got {mode!r}."
        )
    return normalized


def _prior_supports_stein_score(prior_function):
    return hasattr(prior_function, "score") or hasattr(prior_function, "_rbf")


def _prior_has_covariance(prior_function):
    marginal = getattr(prior_function, "marginal", None)
    return callable(marginal) or hasattr(prior_function, "_rbf")


def _set_num_samples_recursive(module, S):
    states = []
    for child in module.modules():
        if hasattr(child, "num_samples"):
            old = child.num_samples
            if old != S:
                child.num_samples = S
                states.append((child, old))
                if getattr(child, "fix_random_noise", False) and hasattr(child, "get_noise"):
                    child.noise = child.get_noise(first_call=True)
    return states


def _restore_num_samples_recursive(states):
    for module, old in reversed(states):
        module.num_samples = old
        if getattr(module, "fix_random_noise", False) and hasattr(module, "get_noise"):
            module.noise = module.get_noise(first_call=True)


def _normalize_weights(weights, dtype, device):
    weights = torch.as_tensor(weights, dtype=dtype, device=device)
    if weights.numel() != 3:
        raise ValueError("measurement_weights must contain data/near/domain weights.")
    if torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("measurement_weights must be non-negative with positive sum.")
    return weights / weights.sum()


def _allocate_counts(weights, total):
    raw = weights * total
    counts = torch.floor(raw).to(torch.long)
    remainder = total - int(counts.sum().item())
    if remainder > 0:
        fractions = raw - counts
        order = torch.argsort(fractions, descending=True)
        counts[order[:remainder]] += 1
    return [int(v.item()) for v in counts]


def _flatten_function_values(values):
    if values.ndim != 3:
        raise ValueError("function values must have shape [S, N, D].")
    return values.reshape(values.shape[0], -1)


def _random_unit_directions(dim, num_projections, dtype, device):
    directions = torch.randn(dim, num_projections, dtype=dtype, device=device)
    return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


def _sample_sliced_projection_directions(
    z,
    w,
    num_projections,
    mode,
    min_bandwidth,
    cache=None,
):
    dim = z.shape[-1]
    if mode == "random":
        return _random_unit_directions(
            dim, num_projections, dtype=z.dtype, device=z.device
        )
    if mode in ("fixed_random", "fixed_orthogonal"):
        return _fixed_unit_directions(
            dim,
            num_projections,
            dtype=z.dtype,
            device=z.device,
            mode=mode,
            cache=cache,
        )
    if mode == "prior_pca":
        covariance = _empirical_covariance(w.detach(), min_bandwidth)
        return _top_eigen_directions(covariance, num_projections, min_bandwidth)
    if mode == "discrepancy_pca":
        z_det = z.detach()
        w_det = w.detach()
        mean_delta = z_det.mean(dim=0) - w_det.mean(dim=0)
        cov_delta = (
            _empirical_covariance(z_det, min_bandwidth)
            - _empirical_covariance(w_det, min_bandwidth)
        )
        discrepancy_matrix = (
            torch.outer(mean_delta, mean_delta) + cov_delta @ cov_delta.T
        )
        return _top_eigen_directions(
            discrepancy_matrix,
            num_projections,
            min_bandwidth,
        )
    raise ValueError(f"Unknown sample_projection_mode: {mode!r}.")


def _fixed_unit_directions(dim, num_projections, dtype, device, mode, cache=None):
    key = (mode, dim, num_projections, str(dtype), str(device))
    if cache is not None and key in cache:
        return cache[key]

    cpu_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    generator = torch.Generator(device="cpu")
    seed = 1729 + 37 * dim + 1009 * num_projections
    if mode == "fixed_orthogonal":
        seed += 7919
    generator.manual_seed(seed)

    if mode == "fixed_random":
        directions = torch.randn(
            dim,
            num_projections,
            dtype=cpu_dtype,
            generator=generator,
        )
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    elif mode == "fixed_orthogonal":
        repeats = math.ceil(num_projections / dim)
        blocks = []
        for _ in range(repeats):
            matrix = torch.randn(dim, dim, dtype=cpu_dtype, generator=generator)
            q, r = torch.linalg.qr(matrix, mode="reduced")
            signs = torch.sign(torch.diagonal(r)).clamp(min=0.0) * 2.0 - 1.0
            blocks.append(q * signs.view(1, -1))
        directions = torch.cat(blocks, dim=1)[:, :num_projections]
    else:
        raise ValueError(f"Unknown fixed projection mode: {mode!r}.")

    directions = directions.to(dtype=dtype, device=device)
    if cache is not None:
        cache[key] = directions
    return directions


def _empirical_covariance(values, jitter):
    if values.ndim != 2:
        raise ValueError("empirical covariance input must have shape [S, D].")
    _, dim = values.shape
    centered = values - values.mean(dim=0, keepdim=True)
    denom = max(values.shape[0] - 1, 1)
    covariance = centered.T @ centered / denom
    eye = torch.eye(dim, dtype=values.dtype, device=values.device)
    return 0.5 * (covariance + covariance.T) + float(jitter) * eye


def _top_eigen_directions(matrix, num_projections, jitter):
    dim = matrix.shape[0]
    matrix = 0.5 * (matrix + matrix.T)
    eye = torch.eye(dim, dtype=matrix.dtype, device=matrix.device)
    eigvals, eigvecs = torch.linalg.eigh(matrix + float(jitter) * eye)
    order = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, order].detach()
    if num_projections <= dim:
        return eigvecs[:, :num_projections]
    repeats = math.ceil(num_projections / dim)
    return eigvecs.repeat(1, repeats)[:, :num_projections]


def _diagonal_standardize_by_reference(z, w, min_scale):
    mean = w.detach().mean(dim=0, keepdim=True)
    std = w.detach().std(dim=0, unbiased=False, keepdim=True).clamp_min(min_scale)
    return (z - mean) / std, (w - mean) / std


def _prior_whitened_point_scores(values, measurement_inputs, prior_function):
    """Cheap pointwise GP-prior score for selecting adaptive measurements."""
    if values.ndim != 3:
        raise ValueError("function values must have shape [S, N, D].")
    X = measurement_inputs[0] if measurement_inputs.ndim == 3 else measurement_inputs
    S, M, D = values.shape
    mean = torch.zeros(M, D, dtype=values.dtype, device=values.device)

    marginal = getattr(prior_function, "marginal", None)
    if callable(marginal):
        prior_mean, covariance = marginal(X)
        prior_mean = torch.as_tensor(
            prior_mean, dtype=values.dtype, device=values.device
        )
        covariance = torch.as_tensor(
            covariance, dtype=values.dtype, device=values.device
        )
        if prior_mean.numel() == M * D:
            mean = prior_mean.reshape(M, D)
        if covariance.shape == (M * D, M * D):
            variance = covariance.diagonal().reshape(M, D)
        elif covariance.shape == (M, M):
            variance = covariance.diagonal().view(M, 1).expand(M, D)
        else:
            raise ValueError(
                "prior_function.marginal returned an unsupported covariance shape "
                f"{tuple(covariance.shape)}."
            )
    elif hasattr(prior_function, "_rbf"):
        covariance = prior_function._rbf(X)
        variance = covariance.diagonal().view(M, 1).expand(M, D)
    else:
        raise ValueError(
            "prior_whitened point selection requires an ExactGP-like prior "
            "or prior_function.marginal(X)."
        )

    jitter = getattr(prior_function, "jitter", 1e-6)
    std = (variance + jitter).clamp_min(1e-12).sqrt()
    u = (values - mean.view(1, M, D)) / std.view(1, M, D)
    return _pointwise_gaussian_kl_to_standard(u)


def _pointwise_gaussian_kl_to_standard(values):
    mean = values.mean(dim=0)
    variance = values.var(dim=0, unbiased=False).clamp_min(1e-6)
    kl = 0.5 * (variance + mean.square() - 1.0 - variance.log())
    return kl.sum(dim=-1)


def _sample_sliced_point_scores(posterior_values, prior_values, min_bandwidth):
    if prior_values is None:
        raise ValueError("sample_sliced_kl point scoring requires prior_values.")
    if posterior_values.ndim != 3 or prior_values.ndim != 3:
        raise ValueError("function values must have shape [S, N, D].")
    if posterior_values.shape[1:] != prior_values.shape[1:]:
        raise ValueError("posterior and prior point samples must share [N, D].")

    z, w = _pointwise_standardize_by_reference(
        posterior_values,
        prior_values.detach(),
        min_bandwidth,
    )
    if z.shape[-1] == 1:
        z = z[..., 0]
        w = w[..., 0]
        log_q = _kde_log_density_pointwise_scalar(
            z,
            z,
            min_bandwidth,
            leave_one_out=True,
        )
        log_p = _kde_log_density_pointwise_scalar(
            z,
            w,
            min_bandwidth,
            leave_one_out=False,
        )
        return (log_q - log_p).mean(dim=0).clamp_min(0.0)
    return _pointwise_sample_gaussian_kl_vectorized(z, w)


def _pointwise_standardize_by_reference(z, w, min_scale):
    mean = w.detach().mean(dim=0, keepdim=True)
    std = w.detach().std(dim=0, unbiased=False, keepdim=True).clamp_min(min_scale)
    return (z - mean) / std, (w - mean) / std


def _kde_log_density_pointwise_scalar(
    query, reference, min_bandwidth, leave_one_out=False
):
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("pointwise scalar KDE inputs must have shape [S, N].")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("pointwise scalar KDE inputs must share N.")
    ref_count = reference.shape[0]
    if ref_count == 0:
        raise ValueError("KDE reference set must be non-empty.")
    if leave_one_out and ref_count < 2:
        std = query.detach().std(dim=0, unbiased=False, keepdim=True)
        std = std.clamp_min(min_bandwidth)
        mean = query.detach().mean(dim=0, keepdim=True)
        return (
            -0.5 * ((query - mean) / std).square()
            - std.log()
            - 0.5 * math.log(2.0 * math.pi)
        )

    values = torch.cat([query.detach(), reference.detach()], dim=0)
    bandwidth = values.std(dim=0, unbiased=False).clamp_min(min_bandwidth)
    diff = query.unsqueeze(1) - reference.unsqueeze(0)
    log_kernel = (
        -0.5 * (diff / bandwidth.view(1, 1, -1)).square()
        - bandwidth.log().view(1, 1, -1)
        - 0.5 * math.log(2.0 * math.pi)
    )
    normalizer = ref_count
    if leave_one_out:
        if query.shape != reference.shape:
            raise ValueError("leave_one_out=True requires matching query/reference shapes.")
        mask = torch.eye(ref_count, dtype=torch.bool, device=query.device).unsqueeze(-1)
        log_kernel = log_kernel.masked_fill(mask, -torch.inf)
        normalizer = ref_count - 1
    return torch.logsumexp(log_kernel, dim=1) - math.log(normalizer)


def _pointwise_sample_gaussian_kl_vectorized(z, w):
    q_mean = z.mean(dim=0)
    q_var = z.var(dim=0, unbiased=False).clamp_min(1e-6)
    p_mean = w.detach().mean(dim=0)
    p_var = w.detach().var(dim=0, unbiased=False).clamp_min(1e-6)
    kl = 0.5 * (
        q_var / p_var
        + (q_mean - p_mean).square() / p_var
        - 1.0
        + p_var.log()
        - q_var.log()
    )
    return kl.sum(dim=-1).clamp_min(0.0)


def _pointwise_sample_gaussian_kl(z, w):
    q_mean = z.mean(dim=0)
    q_var = z.var(dim=0, unbiased=False).clamp_min(1e-6)
    p_mean = w.detach().mean(dim=0)
    p_var = w.detach().var(dim=0, unbiased=False).clamp_min(1e-6)
    kl = 0.5 * (
        q_var / p_var
        + (q_mean - p_mean).square() / p_var
        - 1.0
        + p_var.log()
        - q_var.log()
    )
    return kl.sum().clamp_min(0.0)


def _kde_log_density_scalar(query, reference, min_bandwidth, leave_one_out=False):
    ref_count = reference.shape[0]
    if leave_one_out and ref_count < 2:
        std = query.detach().std(unbiased=False).clamp_min(min_bandwidth)
        mean = query.detach().mean()
        return (
            -0.5 * ((query - mean) / std).square()
            - std.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
    values = torch.cat([query.detach(), reference.detach()], dim=0)
    bandwidth = values.std(unbiased=False).clamp_min(min_bandwidth)
    diff = query.view(-1, 1) - reference.view(1, -1)
    log_kernel = (
        -0.5 * (diff / bandwidth).square()
        - bandwidth.log()
        - 0.5 * math.log(2.0 * math.pi)
    )
    normalizer = ref_count
    if leave_one_out:
        mask = torch.eye(ref_count, dtype=torch.bool, device=query.device)
        log_kernel = log_kernel.masked_fill(mask, -torch.inf)
        normalizer = ref_count - 1
    return torch.logsumexp(log_kernel, dim=1) - math.log(normalizer)


def _prior_spectral_coefficients(
    values,
    measurement_inputs,
    prior_function,
    prior_values,
    num_modes,
    jitter,
    detach_prior_eig,
):
    if values.ndim != 3:
        raise ValueError("function values must have shape [S, N, D].")
    S, M, D = values.shape
    if num_modes <= 0:
        raise ValueError("spectral_num_modes must be positive.")

    if measurement_inputs is not None and _prior_has_covariance(prior_function):
        mean, covariance = _prior_mean_and_covariance(
            values,
            measurement_inputs,
            prior_function,
        )
        if covariance.shape == (M, M):
            coeffs, eigvals = _spectral_coefficients_independent_outputs(
                values,
                mean,
                covariance,
                num_modes,
                jitter,
                detach_prior_eig,
            )
            return coeffs, eigvals
        if covariance.shape == (M * D, M * D):
            mean = mean.reshape(M * D)
            flat = values.reshape(S, M * D)
            coeffs, eigvals = _spectral_coefficients_flat(
                flat,
                mean,
                covariance,
                num_modes,
                jitter,
                detach_prior_eig,
            )
            return coeffs, eigvals
        raise ValueError(
            "prior covariance has unsupported shape "
            f"{tuple(covariance.shape)} for values shape {(S, M, D)}."
        )

    if prior_values is None:
        raise ValueError(
            "spectral_sliced_kl requires measurement_inputs plus an analytic "
            "prior covariance, or prior_values for empirical covariance."
        )
    if prior_values.ndim != 3 or prior_values.shape[1:] != values.shape[1:]:
        raise ValueError("prior_values must have shape [T, M, D].")
    prior_flat = prior_values.detach().reshape(prior_values.shape[0], M * D)
    mean = prior_flat.mean(dim=0)
    centered = prior_flat - mean.view(1, -1)
    denom = max(prior_flat.shape[0] - 1, 1)
    covariance = centered.T @ centered / denom
    return _spectral_coefficients_flat(
        values.reshape(S, M * D),
        mean,
        covariance,
        num_modes,
        jitter,
        detach_prior_eig,
    )


def _prior_mean_and_covariance(values, measurement_inputs, prior_function):
    X = measurement_inputs[0] if measurement_inputs.ndim == 3 else measurement_inputs
    _, M, D = values.shape
    marginal = getattr(prior_function, "marginal", None)
    if callable(marginal):
        mean, covariance = marginal(X)
        mean = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
        covariance = torch.as_tensor(covariance, dtype=values.dtype, device=values.device)
        if mean.ndim == 1 and mean.numel() == M:
            mean = mean.view(M, 1).expand(M, D)
        elif mean.numel() == M * D:
            mean = mean.reshape(M, D)
        return mean, covariance

    if not hasattr(prior_function, "_rbf"):
        raise ValueError(
            "spectral_sliced_kl analytic mode requires prior_function.marginal(X) "
            "or an ExactGP-like prior exposing _rbf(X)."
        )
    mean = torch.zeros(M, D, dtype=values.dtype, device=values.device)
    covariance = prior_function._rbf(X).to(dtype=values.dtype, device=values.device)
    return mean, covariance


def _spectral_coefficients_independent_outputs(
    values,
    mean,
    covariance,
    num_modes,
    jitter,
    detach_prior_eig,
):
    S, M, D = values.shape
    if mean.ndim == 1:
        mean = mean.view(M, 1).expand(M, D)
    if mean.shape != (M, D):
        mean = mean.reshape(M, D)
    eigvals, eigvecs = _leading_eigh(
        covariance,
        num_modes,
        jitter,
        detach_prior_eig,
    )
    centered = values - mean.to(dtype=values.dtype, device=values.device).view(1, M, D)
    coeffs = torch.einsum("smd,mr->srd", centered, eigvecs)
    coeffs = coeffs / eigvals.sqrt().view(1, -1, 1).clamp_min(1e-6)
    eigvals = eigvals.repeat_interleave(D)
    return coeffs.reshape(S, -1), eigvals


def _spectral_coefficients_flat(
    values,
    mean,
    covariance,
    num_modes,
    jitter,
    detach_prior_eig,
):
    eigvals, eigvecs = _leading_eigh(
        covariance,
        num_modes,
        jitter,
        detach_prior_eig,
    )
    centered = values - mean.to(dtype=values.dtype, device=values.device).view(1, -1)
    coeffs = centered @ eigvecs
    coeffs = coeffs / eigvals.sqrt().view(1, -1).clamp_min(1e-6)
    return coeffs, eigvals


def _leading_eigh(covariance, num_modes, jitter, detach):
    covariance = 0.5 * (covariance + covariance.T)
    dim = covariance.shape[0]
    eye = torch.eye(dim, dtype=covariance.dtype, device=covariance.device)
    covariance = covariance + float(jitter) * eye
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    idx = torch.argsort(eigvals, descending=True)
    r = min(int(num_modes), dim)
    eigvals = eigvals[idx][:r].clamp_min(float(jitter))
    eigvecs = eigvecs[:, idx][:, :r]
    if detach:
        eigvals = eigvals.detach()
        eigvecs = eigvecs.detach()
    return eigvals, eigvecs


def _spectral_mode_weights(eigvals, num_coefficients, mode, power):
    if mode not in ("uniform", "eigenvalue"):
        raise ValueError(
            "spectral_mode_weighting must be 'uniform' or 'eigenvalue', "
            f"got {mode!r}."
        )
    if mode == "uniform":
        return torch.full(
            (num_coefficients,),
            1.0 / num_coefficients,
            dtype=eigvals.dtype,
            device=eigvals.device,
        )
    repeats = max(num_coefficients // eigvals.numel(), 1)
    weights = eigvals.pow(power).repeat_interleave(repeats)[:num_coefficients]
    return weights / weights.sum().clamp_min(1e-12)


def _full_gaussian_kl_to_standard(values, jitter, shrinkage):
    if values.ndim != 2:
        raise ValueError("projected coefficients must have shape [S, R].")
    if shrinkage < 0 or shrinkage > 1:
        raise ValueError("spectral_cov_shrinkage must be in [0, 1].")
    n, dim = values.shape
    mean = values.mean(dim=0)
    centered = values - mean.view(1, -1)
    eye = torch.eye(dim, dtype=values.dtype, device=values.device)
    if n < 2:
        variance = centered.square().mean(dim=0).clamp_min(float(jitter))
        cov = torch.diag(variance)
    else:
        cov = centered.T @ centered / (n - 1)
        diag = torch.diag(torch.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * diag
    cov = 0.5 * (cov + cov.T) + float(jitter) * eye
    sign, logdet = torch.linalg.slogdet(cov)
    if torch.any(sign <= 0):
        cov = cov + 10.0 * float(jitter) * eye
        sign, logdet = torch.linalg.slogdet(cov)
    kl = 0.5 * (torch.trace(cov) + mean.square().sum() - dim - logdet)
    return kl.clamp_min(0.0)


def _knn_kl_to_standard_normal(values, k, distance_eps):
    if values.ndim != 2:
        raise ValueError("projected coefficients must have shape [S, R].")
    n, dim = values.shape
    if n < 2:
        return _full_gaussian_kl_to_standard(values, jitter=1e-6, shrinkage=1.0)
    k = min(max(int(k), 1), n - 1)
    distances = _pairwise_distances(values, values, distance_eps)
    inf = torch.as_tensor(torch.inf, dtype=values.dtype, device=values.device)
    distances = distances.masked_fill(
        torch.eye(n, dtype=torch.bool, device=values.device),
        inf,
    )
    kth = torch.kthvalue(distances, k, dim=1).values.clamp_min(distance_eps)
    dim_tensor = torch.as_tensor(float(dim), dtype=values.dtype, device=values.device)
    n_tensor = torch.as_tensor(float(n), dtype=values.dtype, device=values.device)
    k_tensor = torch.as_tensor(float(k), dtype=values.dtype, device=values.device)
    log_unit_ball = (
        0.5 * dim_tensor * math.log(math.pi)
        - torch.lgamma(0.5 * dim_tensor + 1.0)
    )
    entropy = (
        torch.special.digamma(n_tensor)
        - torch.special.digamma(k_tensor)
        + log_unit_ball
        + dim_tensor * kth.log().mean()
    )
    cross_entropy = 0.5 * (
        values.square().sum(dim=1).mean()
        + dim * math.log(2.0 * math.pi)
    )
    return (cross_entropy - entropy).clamp_min(0.0)


def _prior_whiten_values(values, measurement_inputs, prior_function):
    if values.ndim != 3:
        raise ValueError("function values must have shape [S, N, D].")
    X = measurement_inputs[0] if measurement_inputs.ndim == 3 else measurement_inputs
    S, M, D = values.shape

    marginal = getattr(prior_function, "marginal", None)
    if callable(marginal):
        mean, covariance = marginal(X)
        mean = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
        covariance = torch.as_tensor(covariance, dtype=values.dtype, device=values.device)
        if covariance.shape == (M, M):
            return _whiten_independent_outputs(values, mean, covariance, prior_function)
        if covariance.shape == (M * D, M * D):
            flat_mean = mean.reshape(-1)
            flat = values.reshape(S, M * D) - flat_mean.view(1, -1)
            jitter = getattr(prior_function, "jitter", 1e-6)
            eye = torch.eye(M * D, dtype=values.dtype, device=values.device)
            L = _cholesky_with_jitter(covariance, eye, jitter)
            solved = torch.linalg.solve_triangular(L, flat.T, upper=False).T
            return solved.reshape(S, M, D)
        raise ValueError(
            "prior_function.marginal returned an unsupported covariance shape "
            f"{tuple(covariance.shape)}."
        )

    if not hasattr(prior_function, "_rbf"):
        raise ValueError(
            "Prior-whitened KL discrepancies require prior_function.marginal(X) "
            "or an ExactGP-like prior exposing _rbf(X). Use sample_sliced_kl "
            "for implicit sample-only priors such as BNN priors."
        )
    mean = torch.zeros(M, D, dtype=values.dtype, device=values.device)
    covariance = prior_function._rbf(X)
    return _whiten_independent_outputs(values, mean, covariance, prior_function)


def _whiten_independent_outputs(values, mean, covariance, prior_function):
    S, M, D = values.shape
    if mean.ndim == 1:
        mean = mean.view(M, 1).expand(M, D)
    if mean.shape != (M, D):
        mean = mean.reshape(M, D)
    jitter = getattr(prior_function, "jitter", 1e-6)
    covariance = covariance.to(dtype=values.dtype, device=values.device)
    eye = torch.eye(M, dtype=values.dtype, device=values.device)
    L = _cholesky_with_jitter(covariance, eye, jitter)
    centered = values - mean.to(dtype=values.dtype, device=values.device).view(1, M, D)
    rhs = centered.permute(1, 0, 2).reshape(M, S * D)
    solved = torch.linalg.solve_triangular(L, rhs, upper=False)
    return solved.reshape(M, S, D).permute(1, 0, 2)


def _cholesky_with_jitter(matrix, eye, jitter, max_tries=6):
    jitter_value = float(jitter)
    last_error = None
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(matrix + jitter_value * eye)
        except torch._C._LinAlgError as exc:
            last_error = exc
            jitter_value *= 10.0
    raise last_error


def _pairwise_squared_distances(x, y):
    x_norm = x.square().sum(dim=-1, keepdim=True)
    y_norm = y.square().sum(dim=-1, keepdim=True).T
    return (x_norm + y_norm - 2.0 * x @ y.T).clamp_min(0.0)


def _pairwise_distances(x, y, eps):
    return _pairwise_squared_distances(x, y).clamp_min(eps).sqrt()


def _rbf_gram(x, y, bandwidth_squared):
    return torch.exp(-0.5 * _pairwise_squared_distances(x, y) / bandwidth_squared)


def _off_diagonal_mean(matrix):
    n = matrix.shape[0]
    return (matrix.sum() - matrix.diagonal().sum()) / (n * (n - 1))


def _match_sorted_quantiles(x_sorted, y_sorted):
    n = max(x_sorted.shape[0], y_sorted.shape[0])
    return _interpolate_sorted(x_sorted, n), _interpolate_sorted(y_sorted, n)


def _interpolate_sorted(values, n):
    if values.shape[0] == n:
        return values
    if values.shape[0] == 1:
        return values.expand(n, -1)
    pos = torch.linspace(
        0,
        values.shape[0] - 1,
        n,
        dtype=values.dtype,
        device=values.device,
    )
    lo = torch.floor(pos).to(torch.long)
    hi = torch.ceil(pos).to(torch.long)
    weight = (pos - lo.to(values.dtype)).unsqueeze(-1)
    return values[lo] * (1.0 - weight) + values[hi] * weight


def _local_quantile_slopes(source_sorted, target_sorted, k, min_width):
    if source_sorted.ndim != 2 or target_sorted.ndim != 2:
        raise ValueError("quantile slope inputs must have shape [N, P].")
    if source_sorted.shape != target_sorted.shape:
        raise ValueError("source and target quantiles must have matching shapes.")
    n = source_sorted.shape[0]
    if n < 2:
        return torch.ones_like(source_sorted)
    k = min(max(int(k), 1), max(1, (n - 1) // 2))
    idx = torch.arange(n, device=source_sorted.device)
    left = (idx - k).clamp_min(0)
    right = (idx + k).clamp_max(n - 1)
    same = right == left
    right = torch.where(same & (right < n - 1), right + 1, right)
    left = torch.where(same & (left > 0), left - 1, left)
    dx = (source_sorted[right] - source_sorted[left]).abs().clamp_min(min_width)
    dy = (target_sorted[right] - target_sorted[left]).abs().clamp_min(min_width)
    return (dy / dx).clamp_min(min_width)


def _knn_log_density_1d(query, reference, k, min_width, leave_one_out=False):
    """One-dimensional kNN ball-density estimate for each projection column."""
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("kNN density inputs must have shape [N, P].")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference projection counts must match.")
    ref_count = reference.shape[0]
    if ref_count == 0:
        raise ValueError("kNN density reference set must be non-empty.")
    if leave_one_out:
        if query.shape != reference.shape:
            raise ValueError("leave_one_out=True requires matching query/reference shapes.")
        if ref_count < 2:
            return torch.zeros_like(query)
        normalizer = ref_count - 1
    else:
        normalizer = ref_count

    k = min(max(int(k), 1), normalizer)
    distances = (query.unsqueeze(1) - reference.unsqueeze(0)).abs()
    if leave_one_out:
        mask = torch.eye(ref_count, dtype=torch.bool, device=query.device).unsqueeze(-1)
        distances = distances.masked_fill(mask, torch.inf)
    radius = torch.topk(distances, k, dim=1, largest=False).values[:, -1, :]
    radius = radius.clamp_min(min_width)
    return math.log(k) - math.log(normalizer) - math.log(2.0) - radius.log()


def _spacing_log_density_1d(query, reference_sorted, k, min_width):
    """Piecewise-linear 1-D spacing log density for each projection column."""
    if query.ndim != 2 or reference_sorted.ndim != 2:
        raise ValueError("spacing density inputs must have shape [N, P].")
    if query.shape[1] != reference_sorted.shape[1]:
        raise ValueError("query/reference projection counts must match.")
    n_ref, num_proj = reference_sorted.shape
    if n_ref < 2:
        return torch.zeros_like(query)

    k = min(max(int(k), 1), max(1, (n_ref - 1) // 2))
    idx = torch.arange(n_ref, device=reference_sorted.device)
    left = (idx - k).clamp_min(0)
    right = (idx + k).clamp_max(n_ref - 1)
    same = right == left
    right = torch.where(same & (right < n_ref - 1), right + 1, right)
    left = torch.where(same & (left > 0), left - 1, left)
    width = (reference_sorted[right] - reference_sorted[left]).abs().clamp_min(
        min_width
    )
    count = (right - left).to(dtype=reference_sorted.dtype).view(n_ref, 1)
    log_density_grid = count.log() - math.log(n_ref) - width.log()

    columns = []
    for j in range(num_proj):
        ref = reference_sorted[:, j].detach().contiguous()
        grid = log_density_grid[:, j].detach().contiguous()
        q = query[:, j].contiguous()
        insert = torch.searchsorted(ref, q).clamp(1, n_ref - 1)
        lo = insert - 1
        hi = insert
        ref_lo = ref[lo]
        ref_hi = ref[hi]
        denom = (ref_hi - ref_lo).abs().clamp_min(min_width)
        weight = ((q - ref_lo) / denom).clamp(0.0, 1.0)
        log_density = grid[lo] * (1.0 - weight) + grid[hi] * weight

        scale = ref.std(unbiased=False).clamp_min(min_width)
        lower_excess = (ref[0] - q).clamp_min(0.0)
        upper_excess = (q - ref[-1]).clamp_min(0.0)
        tail_excess = torch.maximum(lower_excess, upper_excess)
        log_density = log_density - 0.5 * (tail_excess / scale).square()
        columns.append(log_density)
    return torch.stack(columns, dim=1)


def _sinkhorn_cost(x, y, epsilon, iterations):
    cost = _pairwise_squared_distances(x, y)
    n, m = cost.shape
    log_a = -math.log(n)
    log_b = -math.log(m)
    f = torch.zeros(n, dtype=x.dtype, device=x.device)
    g = torch.zeros(m, dtype=x.dtype, device=x.device)
    for _ in range(iterations):
        f = epsilon * (
            log_a - torch.logsumexp((g.unsqueeze(0) - cost) / epsilon, dim=1)
        )
        g = epsilon * (
            log_b - torch.logsumexp((f.unsqueeze(1) - cost) / epsilon, dim=0)
        )
    log_transport = (f.unsqueeze(1) + g.unsqueeze(0) - cost) / epsilon
    return (torch.exp(log_transport) * cost).sum()


def _gp_prior_score(values, measurement_inputs, prior_function):
    score_fn = getattr(prior_function, "score", None)
    if callable(score_fn):
        return _flatten_function_values(score_fn(measurement_inputs, values))
    if not hasattr(prior_function, "_rbf"):
        raise ValueError(
            "Stein discrepancy currently requires an ExactGP-like prior with "
            "an _rbf method, or a prior_function.score(X, values) method."
        )
    X = measurement_inputs
    if X.ndim == 3:
        X = X[0]
    K = prior_function._rbf(X)
    jitter = getattr(prior_function, "jitter", 1e-6)
    K = K + jitter * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
    S, M, D = values.shape
    rhs = values.permute(1, 0, 2).reshape(M, S * D)
    solved = torch.linalg.solve(K, rhs)
    score = -solved.reshape(M, S, D).permute(1, 0, 2)
    return _flatten_function_values(score)
