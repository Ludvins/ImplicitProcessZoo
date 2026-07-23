from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.common import (
    build_flow as build_common_flow,
)
from experiments.common import (
    fix_gaussian_noise,
    write_csv_rows,
    write_json,
)
from experiments.common.electricity_data import (
    load_electricity_tasks,
    load_synthetic_tasks,
    processed_exists,
)
from experiments.common.electricity_metrics import (
    coerce_regions,
    forecast_regions,
    metrics_by_region,
)
from implicit_process_zoo.ftip import FTIP
from implicit_process_zoo.gmvip import GeneralizedMatheronVIP
from implicit_process_zoo.utils.random import fork_torch_rng
from implicit_process_zoo.vip import VIP

METHODS = (
    "analog",
    "seasonal_naive",
    "empirical_gaussian",
    "gmvip_empirical_exact",
    "vip",
    "vip_512",
    "ftip",
    "gmvip_empirical",
)
PAPER_METHODS = ("analog", "vip", "ftip", "empirical_gaussian", "gmvip_empirical")
LEARNABLE_NOISE_METHODS = {"vip", "vip_512", "ftip", "gmvip_empirical"}
EVALUATION_SAMPLES = 1024
FIXED_NOISE_NLL = "equal_weight_gaussian_mixture_with_fixed_observation_variance"
LEARNED_NOISE_NLL = "equal_weight_gaussian_mixture_with_learned_observation_variance"


DEFAULT_CONFIG: dict = {
    "experiment": "eld_forecasting",
    "method": ",".join(PAPER_METHODS),
    "data": {
        "root": "data/electricity_load_diagrams",
        "split": "test",
        "n_targets": 25,
        "window_length": 192,
        "prefix_length": 96,
        "noise_std_norm": 0.05,
        "min_nonzero_fraction": 0.9,
        "min_prefix_std": 1.0e-3,
        "prior_years": [2011, 2012, 2013],
        "target_years": [2014],
        "target_manifest": "bundled:paper_targets.csv",
    },
    "prior": {
        "bank_size": 2048,
        "selection": "calendar_prefix_nn",
    },
    "gmvip": {
        "num_inducing": 96,
        "jitter": 1.0e-5,
        "shrinkage": 0.02,
        "beta": 1.0,
        "posterior_init_mean": 0.0,
        "posterior_init_log_std": 0.0,
    },
    "ftip": {
        "flow_type": "affine",
        "flow_depth": 1,
        "flow_num_bins": 8,
        "flow_domain": 5.0,
    },
    "training": {
        "learning_rate": 5.0e-3,
        "max_steps": 500,
        "n_mc_train": 8,
        "n_mc_eval": EVALUATION_SAMPLES,
        "batch_size": "full",
        "regression_coeffs": 20,
        "max_grad_norm": 10.0,
        "disable_tqdm": True,
    },
    "likelihood": {
        "learn_observation_noise": True,
    },
    "metrics": {
        "levels": [0.8, 0.9, 0.95],
        "regions": {"test_forecast": {"start": 96, "stop": 192}},
    },
    "plots": {
        "skip": True,
    },
}


SMOKE_CONFIG: dict = {
    **copy.deepcopy(DEFAULT_CONFIG),
    "method": "analog,vip,gmvip_empirical",
    "data": {
        **copy.deepcopy(DEFAULT_CONFIG["data"]),
        "n_targets": 1,
        "window_length": 48,
        "prefix_length": 8,
        "target_manifest": None,
    },
    "prior": {"bank_size": 16, "selection": "calendar"},
    "gmvip": {
        "num_inducing": 8,
        "jitter": 1.0e-5,
        "shrinkage": 0.02,
        "beta": 1.0,
    },
    "training": {
        **copy.deepcopy(DEFAULT_CONFIG["training"]),
        "max_steps": 2,
        "n_mc_train": 2,
        "n_mc_eval": 4,
        "regression_coeffs": 8,
        "disable_tqdm": True,
    },
    "metrics": {"levels": [0.8, 0.9, 0.95], "regions": None},
}


def _training_config(config: dict) -> SimpleNamespace:
    return SimpleNamespace(**dict(config.get("training", {})))


def _fixed_log_variance(noise_std: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.log(noise_std.clamp_min(1e-8))


def _learn_observation_noise(config: dict) -> bool:
    return bool(config.get("likelihood", {}).get("learn_observation_noise", True))


def _configure_model_noise(model, noise_std: torch.Tensor, *, learn: bool) -> None:
    if hasattr(model, "log_variance"):
        value = _fixed_log_variance(noise_std).to(
            dtype=model.log_variance.dtype,
            device=model.log_variance.device,
        )
        model.log_variance = torch.nn.Parameter(value.detach().clone(), requires_grad=learn)
        return
    fix_gaussian_noise(model, noise_std)


def _model_noise_std_norm(model, task, config: dict) -> torch.Tensor:
    if not _learn_observation_noise(config) or getattr(model, "is_fixed_predictive", False):
        return task.noise_std
    if getattr(model, "likelihood", None) is not None and hasattr(model.likelihood, "noise_std"):
        value = model.likelihood.noise_std
    elif hasattr(model, "effective_log_variance"):
        value = torch.exp(0.5 * model.effective_log_variance())
    elif hasattr(model, "log_variance"):
        value = torch.exp(0.5 * model.log_variance)
    else:
        raise RuntimeError(f"{type(model).__name__} has no Gaussian noise parameter.")
    value = value.reshape(-1)
    if value.numel() != 1 or not torch.isfinite(value).all() or torch.any(value <= 0):
        raise RuntimeError(f"Invalid learned electricity observation noise: {value}.")
    return value


def _make_flow(config: dict, input_dim: int, *, seed: int, device, dtype) -> torch.nn.Module:
    ftip_cfg = dict(config.get("ftip", {}))
    flow_type = str(ftip_cfg.get("flow_type", "affine")).lower()
    if flow_type in {"spline_1x1", "spline-1x1", "glow"}:
        flow_type = "spline_1x1"
    return build_common_flow(
        flow_type,
        depth=int(ftip_cfg.get("flow_depth", 1)),
        input_dim=input_dim,
        device=device,
        dtype=dtype,
        seed=seed,
        num_bins=int(ftip_cfg.get("flow_num_bins", 8)),
        domain=float(ftip_cfg.get("flow_domain", 5.0)),
    )


def _inducing_points(task, num_inducing: int, *, device, dtype) -> torch.Tensor:
    observed = task.X_train[:, 0].detach().to(dtype=dtype, device=device)
    if int(num_inducing) <= observed.numel():
        idx = (
            torch.linspace(
                0, observed.numel() - 1, int(num_inducing), dtype=torch.float64, device=device
            )
            .round()
            .long()
        )
        return observed[idx].unique(sorted=True).unsqueeze(-1)
    future_count = int(num_inducing) - int(observed.numel())
    future = torch.linspace(
        float(observed.max()), 1.0, future_count + 1, dtype=dtype, device=device
    )[1:]
    return torch.cat([observed, future]).unique(sorted=True).unsqueeze(-1)


class AnalogPriorPredictive(torch.nn.Module):
    is_fixed_predictive = True

    def __init__(self, prior, *, seed: int):
        super().__init__()
        self.prior = prior
        self.seed = int(seed)

    def predict_f_samples(
        self, X: torch.Tensor, num_samples: int, *, seed: int | None = None
    ) -> torch.Tensor:
        return self.prior.sample(X, int(num_samples), seed=self.seed if seed is None else int(seed))


class EmpiricalGaussianPredictive(torch.nn.Module):
    """Original analytic Gaussian approximation used in the paper table."""

    is_fixed_predictive = True

    def __init__(self, task, *, jitter: float = 1.0e-10):
        super().__init__()
        windows = task.prior.windows[..., 0]
        if windows.ndim != 2:
            raise ValueError("Historical prior windows must have shape [P, T, 1].")
        if int(windows.shape[0]) < 2:
            raise ValueError("The empirical Gaussian baseline needs at least two prior paths.")

        dtype = windows.dtype
        device = windows.device
        prefix_length = int(task.X_train.shape[0])
        if prefix_length <= 0 or prefix_length >= int(windows.shape[1]):
            raise ValueError("The observed prefix must be non-empty and shorter than the window.")

        mean = windows.mean(dim=0)
        centered = windows - mean
        covariance = centered.mT @ centered / float(windows.shape[0] - 1)
        covariance = 0.5 * (covariance + covariance.mT)

        observed_covariance = covariance[:prefix_length, :prefix_length]
        noise_variance = task.noise_std.to(dtype=dtype, device=device).reshape(-1)[0].square()
        scale = covariance.diagonal().mean().clamp_min(torch.finfo(dtype).eps)
        numerical_jitter = max(float(jitter), float(torch.finfo(dtype).eps)) * scale
        observed_system = observed_covariance + (noise_variance + numerical_jitter) * torch.eye(
            prefix_length, dtype=dtype, device=device
        )
        observed_cholesky = torch.linalg.cholesky(observed_system)

        cross_covariance = covariance[:, :prefix_length]
        residual = task.y_train[:, 0].to(dtype=dtype, device=device) - mean[:prefix_length]
        conditional_mean = mean + cross_covariance @ torch.cholesky_solve(
            residual.unsqueeze(-1), observed_cholesky
        ).squeeze(-1)
        conditional_covariance = covariance - cross_covariance @ torch.cholesky_solve(
            cross_covariance.mT, observed_cholesky
        )
        conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.mT)

        eigenvalues, eigenvectors = torch.linalg.eigh(conditional_covariance)
        factor = eigenvectors * eigenvalues.clamp_min(0.0).sqrt().unsqueeze(0)

        self.window_length = int(windows.shape[1])
        self.device = device
        self.dtype = dtype
        self.register_buffer("prior_mean", mean)
        self.register_buffer("prior_covariance", covariance)
        self.register_buffer("posterior_mean", conditional_mean)
        self.register_buffer("posterior_covariance", conditional_covariance)
        self.register_buffer("posterior_factor", factor)

    def predict_f_samples(
        self, X: torch.Tensor, num_samples: int, *, seed: int | None = None
    ) -> torch.Tensor:
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(0 if seed is None else int(seed))
        standard_normal = torch.randn(
            num_samples,
            self.window_length,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        full_samples = self.posterior_mean.unsqueeze(0) + standard_normal @ self.posterior_factor.mT
        positions = (
            (X[:, 0].to(dtype=self.dtype, device=self.device).clamp(-1.0, 1.0) + 1.0)
            * 0.5
            * float(self.window_length - 1)
        )
        indices = positions.round().long().clamp(0, self.window_length - 1)
        return full_samples[:, indices].unsqueeze(-1)


class ExactEmpiricalMatheronPredictive(torch.nn.Module):
    """Exact-q(a) GMVIP predictor retaining empirical historical residuals."""

    is_fixed_predictive = True

    def __init__(self, task, *, jitter: float = 1.0e-10):
        super().__init__()
        windows = task.prior.windows[..., 0]
        if windows.ndim != 2:
            raise ValueError("Historical prior windows must have shape [P, T, 1].")
        if int(windows.shape[0]) < 2:
            raise ValueError("The empirical Gaussian baseline needs at least two prior paths.")

        dtype = windows.dtype
        device = windows.device
        prefix_length = int(task.X_train.shape[0])
        if prefix_length <= 0 or prefix_length >= int(windows.shape[1]):
            raise ValueError("The observed prefix must be non-empty and shorter than the window.")

        mean = windows.mean(dim=0)
        centered = windows - mean
        raw_covariance = centered.mT @ centered / float(windows.shape[0] - 1)
        raw_covariance = 0.5 * (raw_covariance + raw_covariance.mT)

        covariance = raw_covariance
        observed_covariance = covariance[:prefix_length, :prefix_length]
        noise_variance = task.noise_std.to(dtype=dtype, device=device).reshape(-1)[0].square()
        inducing_scale, cholesky_info = torch.linalg.cholesky_ex(observed_covariance)
        if int(cholesky_info.item()) != 0:
            scale = raw_covariance.diagonal().mean().clamp_min(torch.finfo(dtype).eps)
            numerical_jitter = max(float(jitter), float(torch.finfo(dtype).eps)) * scale
            covariance = raw_covariance + numerical_jitter * torch.eye(
                int(windows.shape[1]), dtype=dtype, device=device
            )
            observed_covariance = covariance[:prefix_length, :prefix_length]
            inducing_scale = torch.linalg.cholesky(observed_covariance)

        cross_covariance = covariance[:, :prefix_length]
        residual = task.y_train[:, 0].to(dtype=dtype, device=device) - mean[:prefix_length]

        # Exact full-covariance Gaussian posterior in whitened coordinates:
        #   S_a = (I + sigma^-2 L^T L)^-1,
        #   m_a = sigma^-2 S_a L^T (y - mu).
        identity = torch.eye(prefix_length, dtype=dtype, device=device)
        coefficient_precision = identity + inducing_scale.mT @ inducing_scale / noise_variance
        coefficient_precision_factor = torch.linalg.cholesky(coefficient_precision)
        coefficient_covariance = torch.cholesky_inverse(coefficient_precision_factor)
        coefficient_mean = torch.cholesky_solve(
            (inducing_scale.mT @ residual / noise_variance).unsqueeze(-1),
            coefficient_precision_factor,
        ).squeeze(-1)

        inducing_posterior_mean = mean[:prefix_length] + inducing_scale @ coefficient_mean
        inducing_posterior_covariance = inducing_scale @ coefficient_covariance @ inducing_scale.mT
        inducing_posterior_covariance = 0.5 * (
            inducing_posterior_covariance + inducing_posterior_covariance.mT
        )
        inducing_eigenvalues, inducing_eigenvectors = torch.linalg.eigh(
            inducing_posterior_covariance
        )
        inducing_factor = inducing_eigenvectors * inducing_eigenvalues.clamp_min(
            0.0
        ).sqrt().unsqueeze(0)

        projection = torch.cholesky_solve(cross_covariance.mT, inducing_scale).mT
        projection[:prefix_length] = identity

        # The empirical Matheron residual vanishes at the inducing locations,
        # so adding it preserves the exact posterior draw over the prefix.
        path_residuals = centered - centered[:, :prefix_length] @ projection.mT
        path_residuals[:, :prefix_length] = 0.0
        path_residuals = path_residuals - path_residuals.mean(dim=0, keepdim=True)

        self.window_length = int(windows.shape[1])
        self.prefix_length = prefix_length
        self.device = device
        self.dtype = dtype
        self.register_buffer("prior_mean", mean)
        self.register_buffer("prior_covariance", covariance)
        self.register_buffer("inducing_scale", inducing_scale)
        self.register_buffer("coefficient_posterior_mean", coefficient_mean)
        self.register_buffer("coefficient_posterior_covariance", coefficient_covariance)
        self.register_buffer("inducing_posterior_mean", inducing_posterior_mean)
        self.register_buffer("inducing_posterior_covariance", inducing_posterior_covariance)
        self.register_buffer("inducing_posterior_factor", inducing_factor)
        self.register_buffer("conditional_projection", projection)
        self.register_buffer("empirical_residuals", path_residuals)

    def predict_f_samples(
        self, X: torch.Tensor, num_samples: int, *, seed: int | None = None
    ) -> torch.Tensor:
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(0 if seed is None else int(seed))
        inducing_noise = torch.randn(
            num_samples,
            self.prefix_length,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        inducing_values = (
            self.inducing_posterior_mean.unsqueeze(0)
            + inducing_noise @ self.inducing_posterior_factor.mT
        )
        conditional_mean = (
            self.prior_mean.unsqueeze(0)
            + (inducing_values - self.prior_mean[: self.prefix_length].unsqueeze(0))
            @ self.conditional_projection.mT
        )
        residual_indices = torch.randint(
            0,
            int(self.empirical_residuals.shape[0]),
            (num_samples,),
            dtype=torch.long,
            device=self.device,
            generator=generator,
        )
        residual_samples = self.empirical_residuals[residual_indices]
        full_samples = conditional_mean + residual_samples
        # Preserve exact cardinality despite eigensolver roundoff.
        full_samples[:, : self.prefix_length] = inducing_values
        positions = (
            (X[:, 0].to(dtype=self.dtype, device=self.device).clamp(-1.0, 1.0) + 1.0)
            * 0.5
            * float(self.window_length - 1)
        )
        indices = positions.round().long().clamp(0, self.window_length - 1)
        return full_samples[:, indices].unsqueeze(-1)


class SeasonalNaivePredictive(torch.nn.Module):
    is_fixed_predictive = True

    def __init__(self, task):
        super().__init__()
        values = task.metadata.get("seasonal_window_norm")
        if values is None:
            last = task.y_train[-1:].detach().repeat(task.X_plot.shape[0], 1)
            values_tensor = last
        else:
            values_tensor = torch.as_tensor(
                values, dtype=task.X_plot.dtype, device=task.X_plot.device
            ).reshape(-1, 1)
        self.register_buffer("values", values_tensor)

    def predict_f_samples(
        self, X: torch.Tensor, num_samples: int, *, seed: int | None = None
    ) -> torch.Tensor:
        n = X.shape[0]
        if n == self.values.shape[0]:
            values = self.values
        else:
            pos = (X[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5 * float(self.values.shape[0] - 1)
            idx = pos.round().long().clamp(0, self.values.shape[0] - 1)
            values = self.values[idx]
        return values.unsqueeze(0).repeat(int(num_samples), 1, 1)


def build_model(method: str, task, config: dict, *, seed: int, device, dtype):
    train_cfg = _training_config(config)
    output_dim = int(task.y_train.shape[-1])
    noise_std = task.noise_std.to(dtype=dtype, device=device)
    learn_noise = _learn_observation_noise(config) and method in LEARNABLE_NOISE_METHODS
    if method == "analog":
        return AnalogPriorPredictive(task.prior, seed=seed + 11)
    if method == "empirical_gaussian":
        return EmpiricalGaussianPredictive(task)
    if method == "gmvip_empirical_exact":
        return ExactEmpiricalMatheronPredictive(task)
    if method == "seasonal_naive":
        return SeasonalNaivePredictive(task)
    regression_coeffs = int(train_cfg.regression_coeffs)
    if method == "vip_512":
        regression_coeffs = 512
    if method in {"vip", "vip_512"}:
        prior = task.prior.clone_with_normalization(num_samples=regression_coeffs, seed=seed + 21)
        model = VIP(
            generative_function=prior,
            num_regression_coeffs=regression_coeffs,
            output_dim=output_dim,
            likelihood="regression",
            num_data=int(task.X_train.shape[0]),
            bb_alpha=0.0,
            y_mean=np.zeros((1, output_dim), dtype=np.float64),
            y_std=np.ones((1, output_dim), dtype=np.float64),
            device=device,
            dtype=dtype,
            seed=seed + 22,
        )
        _configure_model_noise(model, noise_std, learn=learn_noise)
        return model
    if method == "ftip":
        prior = task.prior.clone_with_normalization(num_samples=regression_coeffs, seed=seed + 21)
        flow = _make_flow(
            config,
            input_dim=regression_coeffs * output_dim,
            seed=seed + 26,
            device=device,
            dtype=dtype,
        )
        model = FTIP(
            generative_function=prior,
            num_regression_coeffs=regression_coeffs,
            output_dim=output_dim,
            flow=flow,
            likelihood="regression",
            num_data=int(task.X_train.shape[0]),
            num_samples=int(train_cfg.n_mc_train),
            bb_alpha=0.0,
            y_mean=np.zeros((1, output_dim), dtype=np.float64),
            y_std=np.ones((1, output_dim), dtype=np.float64),
            max_grad_norm=float(train_cfg.max_grad_norm),
            device=device,
            dtype=dtype,
            seed=seed + 27,
        )
        _configure_model_noise(model, noise_std, learn=learn_noise)
        return model
    if method == "gmvip_empirical":
        gmvip_cfg = dict(config.get("gmvip", {}))
        bank_size = int(config.get("prior", {}).get("bank_size", 512))
        prior = task.prior.clone_with_normalization(
            num_samples=max(bank_size, int(train_cfg.n_mc_train), 2),
            seed=seed + 31,
        )
        return GeneralizedMatheronVIP(
            base_prior=prior,
            inducing_points=_inducing_points(
                task, int(gmvip_cfg.get("num_inducing", 96)), device=device, dtype=dtype
            ),
            operator_type="empirical",
            posterior_type="gaussian",
            likelihood="regression",
            num_operator_bank_samples=bank_size,
            learn_noise=learn_noise,
            init_log_noise=torch.log(noise_std.clamp_min(1e-8)),
            min_log_noise=math.log(1e-8),
            freeze_base_prior=True,
            detach_prior_samples=True,
            jitter=float(gmvip_cfg.get("jitter", 1e-5)),
            shrinkage=float(gmvip_cfg.get("shrinkage", 0.02)),
            learn_Z=False,
            learn_kernel=False,
            ard=True,
            inducing_scale="prior_cholesky",
            mean_mode="prior_sample",
            posterior_init_mean=float(gmvip_cfg.get("posterior_init_mean", 0.0)),
            posterior_init_log_std=float(gmvip_cfg.get("posterior_init_log_std", 0.0)),
            antithetic_samples=True,
            num_data=int(task.X_train.shape[0]),
            num_train_samples=int(train_cfg.n_mc_train),
            beta=float(gmvip_cfg.get("beta", 1.0)),
            beta_warmup_steps=0,
            data_alpha=0.0,
            max_grad_norm=float(train_cfg.max_grad_norm),
            output_dim=output_dim,
            operator_bank_seed=seed + 101,
        )
    raise ValueError(f"Unknown method {method!r}.")


def predictive_function_samples(
    model, method: str, X: torch.Tensor, n_samples: int, seed: int
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if method in {
            "analog",
            "seasonal_naive",
            "empirical_gaussian",
            "gmvip_empirical_exact",
        }:
            values = model.predict_f_samples(X, int(n_samples), seed=seed)
        elif method in {"vip", "vip_512"}:
            values = model.predict_f_samples(X, int(n_samples), seed=seed)
        elif method == "ftip":
            values = model.predict_f_samples(X, int(n_samples), seed=seed)
        elif method == "gmvip_empirical":
            values = model.sample_posterior_values(X, int(n_samples), seed=seed)
        else:
            raise ValueError(f"Unknown method {method!r}.")
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    return values


def fit_model(model, task, config: dict, *, device) -> dict:
    if getattr(model, "is_fixed_predictive", False):
        return {
            "train_time_sec": 0.0,
            "steps": 0,
            "loss_start": None,
            "loss_end": None,
            "checkpoint": "fixed_predictive",
        }
    train_cfg = _training_config(config)
    dataset = TensorDataset(task.X_train, task.y_train)
    full_batch = train_cfg.batch_size == "full"
    batch_size = len(dataset) if full_batch else int(train_cfg.batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not full_batch, num_workers=0)
    if hasattr(model, "prepare_for_training"):
        model.prepare_for_training(loader)
    params = model.vi_parameters() if hasattr(model, "vi_parameters") else model.parameters()
    params = [param for param in params if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(train_cfg.learning_rate))
    losses = []
    start = time.time()
    stream = iter(loader)
    disable = bool(config.get("training", {}).get("disable_tqdm", False))
    loop = tqdm(range(int(train_cfg.max_steps)), desc="eld train", unit=" step", disable=disable)
    for _step in loop:
        if full_batch:
            xb, yb = task.X_train, task.y_train
        else:
            try:
                xb, yb = next(stream)
            except StopIteration:
                stream = iter(loader)
                xb, yb = next(stream)
        loss = model._train_step(optimizer, xb.to(device), yb.to(device))
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        loop.set_postfix(loss=f"{loss_value:.3f}")
    return {
        "train_time_sec": time.time() - start,
        "steps": len(losses),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "checkpoint": "final_step",
    }


def _unnormalize(task, values: torch.Tensor) -> torch.Tensor:
    y_mean = torch.as_tensor(
        task.metadata["y_mean"], dtype=values.dtype, device=values.device
    ).reshape(1, 1)
    y_std = torch.as_tensor(
        task.metadata["y_std"], dtype=values.dtype, device=values.device
    ).reshape(1, 1)
    return values * y_std + y_mean


def evaluate_target(
    model,
    method: str,
    task,
    config: dict,
    *,
    run_seed: int,
    target_seed: int,
    out_dir: Path,
) -> list[dict]:
    eval_samples = int(_training_config(config).n_mc_eval)
    if eval_samples != EVALUATION_SAMPLES and not bool(config.get("smoke", False)):
        raise RuntimeError(
            f"Standard electricity evaluation requires exactly {EVALUATION_SAMPLES} samples."
        )
    evaluation_seed = int(target_seed) + 501
    start = time.time()
    samples_norm = predictive_function_samples(
        model, method, task.X_plot, eval_samples, evaluation_seed
    )
    samples = _unnormalize(task, samples_norm)
    y_true = _unnormalize(task, task.y_plot_true)
    t_grid = torch.as_tensor(task.metadata["t_grid"], dtype=samples.dtype, device=samples.device)
    noise_std_norm = _model_noise_std_norm(model, task, config).to(
        dtype=samples.dtype, device=samples.device
    )
    physical_scale = torch.as_tensor(
        [float(task.metadata["y_std"])], dtype=samples.dtype, device=samples.device
    )
    noise_std = noise_std_norm * physical_scale
    region_config = config.get("metrics", {}).get("regions")
    if region_config is None:
        region_config = forecast_regions(
            int(task.metadata["window_length"]), int(task.metadata["prefix_length"])
        )
    regions = coerce_regions(region_config)
    metric_rows = metrics_by_region(
        samples,
        y_true,
        t_grid,
        noise_std,
        levels=tuple(config.get("metrics", {}).get("levels", [0.8, 0.9, 0.95])),
        regions=regions,
    )

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    samples_np = samples.detach().cpu().numpy()
    np.savez_compressed(
        pred_dir / f"target_{task.metadata['target_id']}.npz",
        target_id=np.asarray(int(task.metadata["target_id"]), dtype=np.int64),
        client_id=np.asarray(str(task.metadata["client_id"])),
        start_time=np.asarray(str(task.metadata["start_time"])),
        run_seed=np.asarray(int(run_seed), dtype=np.int64),
        target_seed=np.asarray(int(target_seed), dtype=np.int64),
        forecast_start_hour=np.asarray(
            float(task.metadata["forecast_start_hour"]), dtype=np.float64
        ),
        t=np.asarray(task.metadata["t_grid"], dtype=np.float64),
        truth=y_true.detach().cpu().numpy(),
        context_t=np.asarray(task.metadata["t_grid"], dtype=np.float64)[
            : int(task.metadata["prefix_length"])
        ],
        context_y=np.asarray(task.metadata["y_context_physical"], dtype=np.float64),
        train_t=np.asarray(task.metadata["t_grid"], dtype=np.float64)[
            : int(task.metadata["prefix_length"])
        ],
        train_y=np.asarray(task.metadata["y_train_physical"], dtype=np.float64),
        samples=samples_np,
        mean=samples_np.mean(axis=0),
        std=samples_np.std(axis=0),
        observation_noise_std=noise_std.detach().cpu().numpy(),
        observation_noise_std_norm=noise_std_norm.detach().cpu().numpy(),
        evaluation_seed=np.asarray(evaluation_seed, dtype=np.int64),
        evaluation_samples=np.asarray(eval_samples, dtype=np.int64),
    )

    rows = []
    stress = bool(task.metadata.get("stress", False))
    for region, values in metric_rows.items():
        rows.append(
            {
                "experiment": "eld_forecasting",
                "method": method,
                "target_id": int(task.metadata["target_id"]),
                "client_id": task.metadata["client_id"],
                "start_time": task.metadata["start_time"],
                "split": task.metadata["split"],
                "protocol": task.metadata.get("protocol"),
                "prior_years": ",".join(str(year) for year in task.metadata.get("prior_years", [])),
                "target_years": ",".join(
                    str(year) for year in task.metadata.get("target_years", [])
                ),
                "prior_selection": task.metadata.get("prior_selection"),
                "prior_selected_rule": task.metadata.get("prior_selected_rule"),
                "prior_candidate_count": task.metadata.get("prior_candidate_count"),
                "prior_requested_candidate_count": task.metadata.get(
                    "prior_requested_candidate_count"
                ),
                "prior_fallback_tier": task.metadata.get("prior_fallback_tier"),
                "prior_actual_calendar_constraint": task.metadata.get(
                    "prior_actual_calendar_constraint"
                ),
                "prior_actual_client_constraint": task.metadata.get(
                    "prior_actual_client_constraint"
                ),
                "prior_neighbor_distance_mean": task.metadata.get("prior_neighbor_distance_mean"),
                "context_points": int(
                    task.metadata.get("context_points", task.metadata["prefix_length"])
                ),
                "forecast_points": int(task.metadata.get("forecast_points", task.X_test.shape[0])),
                "last_observed_hour": float(task.metadata["last_observed_hour"]),
                "forecast_start_hour": float(task.metadata["forecast_start_hour"]),
                "run_seed": int(run_seed),
                "target_seed": int(target_seed),
                "evaluation_seed": evaluation_seed,
                "evaluation_samples": eval_samples,
                "observation_noise_std_norm": float(noise_std_norm[0].detach().cpu()),
                "observation_noise_std": float(noise_std[0].detach().cpu()),
                "region": region,
                "stress": stress,
                "stress_prefix_cv": task.metadata.get("stress_prefix_cv"),
                "stress_prefix_ramp_standardized": task.metadata.get(
                    "stress_prefix_ramp_standardized"
                ),
                "stress_prefix_cv_rank": task.metadata.get("stress_prefix_cv_rank"),
                "stress_prefix_ramp_rank": task.metadata.get("stress_prefix_ramp_rank"),
                "stress_score": task.metadata.get("stress_score"),
                "stress_threshold": task.metadata.get("stress_threshold"),
                "eval_time_sec": time.time() - start,
                **values,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    write_csv_rows(path, rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_manifest(config: dict) -> dict:
    if bool(config.get("smoke", False)):
        payload = {
            "kind": "deterministic_synthetic_smoke",
            "data": config.get("data", {}),
            "prior": config.get("prior", {}),
        }
        return {**payload, "sha256": _stable_hash(payload)}
    root = Path(config["data"]["root"]).resolve()
    processed = root / "processed"
    files = [
        processed / "values_float32.npy",
        processed / "timestamps_ns.npy",
        processed / "clients.json",
        processed / "metadata.json",
        Path("experiments/common/paper_targets.csv"),
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Electricity dataset manifest is incomplete: {missing}")
    hashes = {path.name: _sha256_file(path) for path in files}
    return {
        "root": str(root),
        "files": hashes,
        "sha256": _stable_hash(hashes),
    }


def _method_manifest(method: str, config: dict, *, seed: int, basis_size: int) -> dict:
    learn_noise = _learn_observation_noise(config) and method in LEARNABLE_NOISE_METHODS
    protocol = {
        "schema_version": 2,
        "experiment": "electricity_forecasting",
        "method": method,
        "seed": int(seed),
        "vip_basis_size": int(basis_size),
        "nll": LEARNED_NOISE_NLL if learn_noise else FIXED_NOISE_NLL,
        "observation_noise": {
            "mode": "learned_scalar" if learn_noise else "fixed_scalar",
            "initialization": "0.05_normalized_units",
            "reported_units": "physical_load_units",
        },
        "checkpoint_selection": "none_final_step_only",
        "data_usage": {
            "training": "indices_[0,96)",
            "validation": "none",
            "test": "indices_[96,192)",
        },
        "evaluation_samples": int(config["training"]["n_mc_eval"]),
        "config": config,
        "dataset": _dataset_manifest(config),
    }
    return {
        **protocol,
        "protocol_hash": _stable_hash(protocol),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "status": "running",
        "completed_targets": [],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _summarize(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["region"])), []).append(row)
    metrics = (
        "rmse",
        "nll",
        "crps",
        "cqm",
        "cov80",
        "cov90",
        "cov95",
        "width80",
        "width90",
        "width95",
        "peak_magnitude_error",
        "peak_timing_error_hours",
    )
    summary = {}
    for (method, region), group in groups.items():
        key = f"{method}|{region}"
        summary[key] = {}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group if metric in row], dtype=np.float64)
            if values.size:
                summary[key][metric] = {
                    "median": float(np.nanmedian(values)),
                    "q25": float(np.nanquantile(values, 0.25)),
                    "q75": float(np.nanquantile(values, 0.75)),
                    "mean": float(np.nanmean(values)),
                    "stderr": (
                        float(np.nanstd(values, ddof=1) / math.sqrt(values.size))
                        if values.size > 1
                        else 0.0
                    ),
                }
    return summary


def _requested_methods(value: str) -> list[str]:
    if str(value) == "all":
        return list(PAPER_METHODS)
    methods = [item.strip() for item in str(value).split(",") if item.strip()]
    return list(dict.fromkeys(methods))


def _parse_years(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _load_expected_targets(path_value, *, run_seed: int) -> dict[int, tuple[str, str]] | None:
    if path_value is None:
        return None
    path = (
        Path("experiments/common/paper_targets.csv")
        if str(path_value) == "bundled:paper_targets.csv"
        else Path(path_value)
    )
    if not path.is_file():
        raise FileNotFoundError(f"Frozen ELD target manifest not found: {path}")
    expected: dict[int, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["run_seed"]) != int(run_seed):
                continue
            target_id = int(row["target_id"])
            if target_id in expected:
                raise ValueError(
                    f"Frozen ELD target manifest repeats seed {run_seed}, target {target_id}."
                )
            expected[target_id] = (str(row["client_id"]), str(row["start_time"]))
    if not expected:
        raise ValueError(f"Frozen ELD target manifest has no entries for run seed {run_seed}.")
    return expected


def _requested_target_ids(args: argparse.Namespace, n_targets: int) -> list[int]:
    ids: list[int] = []
    for token in str(args.target_ids).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start, stop = token.split(":", 1)
            ids.extend(range(int(start or 0), int(stop or n_targets)))
        elif "-" in token:
            start, stop = token.split("-", 1)
            ids.extend(range(int(start), int(stop) + 1))
        else:
            ids.append(int(token))
    resolved = list(dict.fromkeys(idx for idx in ids if 0 <= idx < n_targets))
    if not resolved:
        raise ValueError("Target selection did not resolve to any valid target ids.")
    return resolved


def _load_tasks(config: dict, args: argparse.Namespace, *, seed: int, device, dtype):
    data_cfg = dict(config.get("data", {}))
    prior_cfg = dict(config.get("prior", {}))
    bank_size = int(prior_cfg.get("bank_size", 512))
    n_targets = int(data_cfg.get("n_targets", 1 if args.smoke else 25))
    requested_ids = _requested_target_ids(args, n_targets)
    if args.smoke:
        tasks = load_synthetic_tasks(
            seed=seed,
            n_targets=n_targets,
            bank_size=bank_size,
            window_length=int(data_cfg.get("window_length", 48)),
            prefix_length=int(data_cfg.get("prefix_length", 8)),
            noise_std_norm=float(data_cfg.get("noise_std_norm", 0.05)),
            device=device,
            dtype=dtype,
        )
        tasks = [tasks[index] for index in requested_ids]
        _mark_stress_tasks(tasks)
        return tasks
    root = data_cfg.get("root", "data/electricity_load_diagrams")
    if not processed_exists(root):
        raise FileNotFoundError(
            f"Processed ELD data not found under {root!r}. Prepare it with "
            "the UCI ElectricityLoadDiagrams20112014 archive before running the benchmark."
        )
    tasks = load_electricity_tasks(
        root,
        seed=seed,
        n_targets=n_targets,
        split=str(data_cfg.get("split", "test")),
        bank_size=bank_size,
        prior_years=_parse_years(data_cfg.get("prior_years")),
        target_years=_parse_years(data_cfg.get("target_years")),
        window_length=int(data_cfg.get("window_length", 192)),
        prefix_length=int(data_cfg.get("prefix_length", 32)),
        noise_std_norm=float(data_cfg.get("noise_std_norm", 0.05)),
        min_nonzero_fraction=float(data_cfg.get("min_nonzero_fraction", 0.9)),
        min_prefix_std=float(data_cfg.get("min_prefix_std", 1e-3)),
        prior_selection=str(prior_cfg.get("selection", "calendar")),
        target_ids=requested_ids,
        expected_targets=_load_expected_targets(data_cfg.get("target_manifest"), run_seed=seed),
        device=device,
        dtype=dtype,
    )
    return tasks


def _mark_stress_tasks(tasks) -> None:
    if not tasks:
        return
    prefix_cv = np.asarray(
        [float(task.metadata["stress_prefix_cv"]) for task in tasks], dtype=np.float64
    )
    prefix_ramp = np.asarray(
        [float(task.metadata["stress_prefix_ramp_standardized"]) for task in tasks],
        dtype=np.float64,
    )

    def percentile_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = (np.arange(values.size, dtype=np.float64) + 1.0) / values.size
        return ranks

    cv_ranks = percentile_ranks(prefix_cv)
    ramp_ranks = percentile_ranks(prefix_ramp)
    scores = 0.5 * (cv_ranks + ramp_ranks)
    threshold = float(np.quantile(scores, 0.8))
    for task, cv_rank, ramp_rank, score in zip(tasks, cv_ranks, ramp_ranks, scores):
        task.metadata["stress_prefix_cv_rank"] = float(cv_rank)
        task.metadata["stress_prefix_ramp_rank"] = float(ramp_rank)
        task.metadata["stress_score"] = float(score)
        task.metadata["stress_threshold"] = threshold
        task.metadata["stress"] = bool(score >= threshold)


def run_method(method: str, config: dict, args: argparse.Namespace, *, tasks=None) -> dict:
    seed = int(args.seed)
    dtype = torch.float64
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = tasks or _load_tasks(config, args, seed=seed, device=device, dtype=dtype)
    ids = [int(task.metadata["target_id"]) for task in tasks]
    basis_size = int(args.vip_basis_size)
    out_dir = Path(args.output_root) / f"seed_{seed}" / f"S_{basis_size}" / method
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.yaml"
    config_text = yaml.safe_dump(config, sort_keys=False)
    resume_artifacts = bool(args.resume)
    if resume_artifacts and config_path.exists():
        if config_path.read_text(encoding="utf-8") != config_text:
            raise ValueError("Existing electricity config differs; resume in a new output root.")
    else:
        config_path.write_text(config_text, encoding="utf-8")

    csv_path = out_dir / "metrics_per_target_region.csv"
    runtime_path = out_dir / "runtime.json"
    manifest_path = out_dir / "manifest.json"
    expected_manifest = _method_manifest(method, config, seed=seed, basis_size=basis_size)
    if resume_artifacts and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("protocol_hash") != expected_manifest["protocol_hash"]:
            raise ValueError("Existing electricity manifest differs from the requested protocol.")
        manifest = existing_manifest
    else:
        manifest = expected_manifest
        write_json(manifest_path, manifest)
    if resume_artifacts and csv_path.exists() and runtime_path.exists():
        existing_rows = _read_csv(csv_path)
        existing_runtimes = json.loads(runtime_path.read_text(encoding="utf-8"))
        row_targets = {int(row["target_id"]) for row in existing_rows}
        runtime_targets = {int(row["target_id"]) for row in existing_runtimes}
        completed = row_targets & runtime_targets
        rows = [row for row in existing_rows if int(row["target_id"]) in completed]
        runtimes = [row for row in existing_runtimes if int(row["target_id"]) in completed]
    else:
        completed = set()
        rows = []
        runtimes = []

    def flush() -> dict:
        completed_ids = sorted({int(row["target_id"]) for row in runtimes})
        metrics = {
            "method": method,
            "run_seed": seed,
            "targets": completed_ids,
            "summary": _summarize(rows),
        }
        _write_csv(csv_path, rows)
        write_json(runtime_path, runtimes)
        write_json(out_dir / "metrics.json", metrics)
        manifest["completed_targets"] = completed_ids
        manifest["status"] = "complete" if set(ids).issubset(completed_ids) else "partial"
        write_json(manifest_path, manifest)
        return metrics

    for task in tasks:
        target_idx = int(task.metadata["target_id"])
        if target_idx in completed:
            continue
        target_seed = seed + 1000 * target_idx
        with fork_torch_rng(target_seed):
            model = build_model(method, task, config, seed=target_seed, device=device, dtype=dtype)
            train_info = fit_model(model, task, config, device=device)
            target_rows = evaluate_target(
                model,
                method,
                task,
                config,
                run_seed=seed,
                target_seed=target_seed,
                out_dir=out_dir,
            )
            checkpoint_dir = out_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "method": method,
                    "run_seed": seed,
                    "target_seed": target_seed,
                    "target_id": target_idx,
                    "step": int(train_info["steps"]),
                    "checkpoint": train_info["checkpoint"],
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                checkpoint_dir / f"target_{target_idx}.pt",
            )
        for row in target_rows:
            row["train_time_sec"] = float(train_info["train_time_sec"])
            row["train_steps"] = int(train_info["steps"])
            row["loss_start"] = train_info["loss_start"]
            row["loss_end"] = train_info["loss_end"]
        rows.extend(target_rows)
        runtimes.append({"target_id": target_idx, **train_info})
        flush()

    metrics = flush()
    requested = set(ids)
    if not requested.issubset(set(metrics["targets"])):
        raise RuntimeError("ELD artifact flush did not include every requested target.")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical electricity benchmark.",
        epilog=(
            "Canonical protocol: B=2048, GMVIP M=96, VIP/FTIP S from "
            "--vip-basis-size (20 reported), 500 steps, 8 training samples, "
            "1,024 evaluation samples, learned noise for VIP/FTIP/GMVIP, "
            "and no validation or checkpoint selection. Run seeds 0, 1, and 2."
        ),
    )
    parser.add_argument(
        "--methods",
        required=True,
        help=f"Comma-separated method names. Choices: {','.join(METHODS)}",
    )
    parser.add_argument("--vip-basis-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-ids", default="0:25")
    parser.add_argument(
        "--learn-observation-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn scalar observation noise for trainable methods (default).",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--output-root", default="results/electricity")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if int(args.vip_basis_size) <= 1:
        raise ValueError("--vip-basis-size must be greater than one.")
    config = copy.deepcopy(SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG)
    config["method"] = args.methods
    config["smoke"] = bool(args.smoke)
    config["training"]["regression_coeffs"] = int(args.vip_basis_size)
    config["training"]["n_mc_eval"] = (
        int(config["training"]["n_mc_eval"]) if args.smoke else EVALUATION_SAMPLES
    )
    config["likelihood"]["learn_observation_noise"] = bool(args.learn_observation_noise)
    if args.disable_tqdm:
        config.setdefault("training", {})["disable_tqdm"] = True
    methods = _requested_methods(args.methods)
    dtype = torch.float64
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = _load_tasks(config, args, seed=int(args.seed), device=device, dtype=dtype)
    results = {}
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")
        results[method] = run_method(method, config, args, tasks=tasks)
    return results


if __name__ == "__main__":
    main()
