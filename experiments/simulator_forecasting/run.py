from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.simulator_forecasting.datasets import load_damped_oscillator_tasks
from experiments.simulator_forecasting.generate import generate_dataset
from experiments.simulator_forecasting.metrics import coerce_regions, metrics_by_region
from experiments.simulator_forecasting.plots import (
    plot_metric_by_region,
    plot_posterior_forecast,
    plot_prior_samples,
)
from experiments.simulator_forecasting.priors import DampedOscillatorPrior
from implicit_process_zoo.fbnn import FBNN
from implicit_process_zoo.flows import CouplingFlow, SplineCoupling1x1Flow, SplineCouplingFlow
from implicit_process_zoo.ftip import FTIP
from implicit_process_zoo.gmvip import GeneralizedMatheronVIP
from implicit_process_zoo.map_baseline import DeterministicMAP
from implicit_process_zoo.mfvi import MFVI
from implicit_process_zoo.priors.generative_functions import GP, BayesianNN, BayesLinear
from implicit_process_zoo.sip import SIP
from implicit_process_zoo.tfsvi import TFSVI
from implicit_process_zoo.vip import VIP

METHODS = (
    "gmvip",
    "gmvip_rbf",
    "vip",
    "ftip",
    "sip",
    "map",
    "deep_ensemble",
    "mfvi",
    "fbnn_observed",
    "fbnn_full",
    "tfsvi_observed",
    "tfsvi_full",
)

METHOD_ALIASES = {
    "gmvip_cov": "gmvip",
}


DEFAULT_SIMULATOR_FORECASTING_CONFIG: dict = {
    "experiment": "simulator_forecasting",
    "method": "gmvip",
    "data": {
        "root": "data/simprior/simulator_forecasting",
        "n_eval_targets": 100,
        "n_train": [8, 16, 32],
        "n_test": 500,
        "t_obs": 8.0,
        "t_max": 30.0,
        "sigma_y": 0.05,
        "sigma_u": 0.05,
        "rho": 0.98,
        "forcing_delta": 0.1,
        "misspecified": False,
        "context_points": 128,
    },
    "prior": {
        "bank_size": 1024,
    },
    "gmvip": {
        "num_inducing": 32,
        "jitter": 1.0e-4,
        "shrinkage": 0.02,
        "rbf_lengthscale": "median",
        "learn_kernel": False,
        "beta": 1.0,
    },
    "ftip": {
        "flow_type": "affine",
        "flow_depth": 2,
        "flow_num_bins": 8,
        "flow_domain": 5.0,
        "warm_start_from_vip": False,
        "learnable_affine": False,
        "warm_start_steps": None,
        "fine_tune_steps": None,
        "fine_tune_lr": None,
    },
    "sip": {
        "num_inducing": 32,
        "num_prior_samples": 128,
        "num_train_samples": 16,
        "fresh_prior_samples": True,
        "learn_inducing": False,
        "detach_covariances": True,
        "jitter": 1.0e-4,
        "beta": 1.0,
        "beta_warmup_steps": 0,
        "critic_hidden_dim": 64,
        "critic_lr": 1.0e-3,
        "critic_steps": 1,
        "posterior_noise_dim": 64,
        "posterior_hidden_dim": 64,
        "posterior_depth": 2,
    },
    "neural": {
        "ensemble_size": 5,
        "fbnn_num_measurement": 16,
        "fbnn_num_context": 32,
        "fbnn_context_std": 1.0,
        "fbnn_lambda_kl": 1.0,
        "tfsvi_S_ctx": 2,
        "tfsvi_K_ctx": 32,
        "tfsvi_sigma_prior": 1.0,
    },
    "training": {
        "optimizer": "adam",
        "learning_rate": 5.0e-3,
        "max_steps": 10000,
        "n_mc_train": 16,
        "n_mc_eval": 1000,
        "batch_size": "full",
        "hidden_dims": [128, 128, 128],
        "activation": "tanh",
        "regression_coeffs": 256,
        "weight_log_sigma_init": -5.0,
        "weight_decay": 1.0e-4,
        "max_grad_norm": 10.0,
        "disable_tqdm": False,
    },
    "plots": {
        "skip": False,
        "n_posterior_samples": 20,
        "n_prior_samples": 20,
    },
}


DEV_SIMULATOR_FORECASTING_CONFIG: dict = {
    **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG),
    "data": {
        **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG["data"]),
        "root": "data/simprior/simulator_forecasting_dev",
        "n_eval_targets": 50,
        "n_train": [8, 16, 32],
    },
    "prior": {"bank_size": 256},
    "training": {
        **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG["training"]),
        "max_steps": 1000,
        "n_mc_eval": 256,
        "regression_coeffs": 128,
    },
}


SMOKE_SIMULATOR_FORECASTING_CONFIG: dict = {
    **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG),
    "data": {
        **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG["data"]),
        "root": "data/simprior/simulator_forecasting_smoke",
        "n_eval_targets": 1,
        "n_train": [4],
        "n_test": 61,
        "context_points": 12,
    },
    "prior": {"bank_size": 8},
    "gmvip": {
        "num_inducing": 4,
        "jitter": 1.0e-4,
        "shrinkage": 0.02,
        "rbf_lengthscale": "median",
        "learn_kernel": False,
        "beta": 1.0,
    },
    "ftip": {
        "flow_type": "affine",
        "flow_depth": 1,
        "flow_num_bins": 8,
        "flow_domain": 5.0,
        "warm_start_from_vip": False,
        "learnable_affine": False,
        "warm_start_steps": None,
        "fine_tune_steps": None,
        "fine_tune_lr": None,
    },
    "sip": {
        "num_inducing": 4,
        "num_prior_samples": 8,
        "num_train_samples": 4,
        "fresh_prior_samples": True,
        "learn_inducing": False,
        "detach_covariances": True,
        "jitter": 1.0e-4,
        "beta": 1.0,
        "beta_warmup_steps": 0,
        "critic_hidden_dim": 8,
        "critic_lr": 1.0e-3,
        "critic_steps": 1,
        "posterior_noise_dim": 8,
        "posterior_hidden_dim": 8,
        "posterior_depth": 2,
    },
    "neural": {
        "ensemble_size": 2,
        "fbnn_num_measurement": 2,
        "fbnn_num_context": 2,
        "fbnn_context_std": 1.0,
        "fbnn_lambda_kl": 0.1,
        "tfsvi_S_ctx": 1,
        "tfsvi_K_ctx": 3,
        "tfsvi_sigma_prior": 1.0,
    },
    "training": {
        "optimizer": "adam",
        "learning_rate": 1.0e-3,
        "max_steps": 2,
        "n_mc_train": 2,
        "n_mc_eval": 4,
        "batch_size": "full",
        "hidden_dims": [8],
        "activation": "tanh",
        "regression_coeffs": 8,
        "weight_log_sigma_init": -5.0,
        "weight_decay": 0.0,
        "max_grad_norm": 5.0,
        "disable_tqdm": True,
    },
    "plots": {
        "skip": True,
        "n_posterior_samples": 2,
        "n_prior_samples": 2,
    },
}


MISSPECIFIED_SIMULATOR_FORECASTING_CONFIG: dict = {
    **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG),
    "data": {
        **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG["data"]),
        "root": "data/simprior/simulator_forecasting_misspecified",
        "misspecified": True,
    },
}


TOBS15_VIP_FTIP_GMVIP_20TARGET_CONFIG: dict = {
    **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG),
    "method": "gmvip",
    "data": {
        **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG["data"]),
        "root": "data/simprior/simulator_forecasting_tobs15_20targets",
        "n_eval_targets": 20,
        "n_train": [64],
        "n_test": 500,
        "t_obs": 15.0,
        "t_max": 30.0,
        "context_points": 128,
    },
    "prior": {"bank_size": 1024},
    "gmvip": {
        "num_inducing": 32,
        "jitter": 1.0e-4,
        "shrinkage": 0.02,
        "rbf_lengthscale": "median",
        "learn_kernel": False,
        "beta": 1.0,
    },
    "ftip": {
        "flow_type": "affine",
        "flow_depth": 2,
        "flow_num_bins": 8,
        "flow_domain": 5.0,
        "warm_start_from_vip": True,
        "learnable_affine": False,
        "warm_start_steps": 3000,
        "warm_start_lr": 5.0e-3,
        "fine_tune_steps": 3000,
        "fine_tune_lr": 1.0e-4,
    },
    "training": {
        **copy.deepcopy(DEFAULT_SIMULATOR_FORECASTING_CONFIG["training"]),
        "learning_rate": 5.0e-3,
        "max_steps": 3000,
        "n_mc_train": 16,
        "n_mc_eval": 1000,
        "regression_coeffs": 256,
        "disable_tqdm": True,
    },
    "plots": {
        "skip": True,
        "n_posterior_samples": 30,
        "n_prior_samples": 30,
    },
    "metrics": {
        "regions": {
            "interpolation": {"lo": 0.0, "hi": 15.0, "include_left": True},
            "near_extrapolation": {"lo": 15.0, "hi": 20.0, "include_left": False},
            "far_extrapolation": {"lo": 20.0, "hi": 30.0, "include_left": False},
        },
    },
}


TOBS15_VIP_FTIP_GMVIP_FIGURE_CONFIG: dict = {
    **copy.deepcopy(TOBS15_VIP_FTIP_GMVIP_20TARGET_CONFIG),
    "data": {
        **copy.deepcopy(TOBS15_VIP_FTIP_GMVIP_20TARGET_CONFIG["data"]),
        "root": "data/simprior/simulator_forecasting_tobs15",
        "n_eval_targets": 1,
        "n_train": [64],
    },
    "plots": {
        "skip": True,
        "n_posterior_samples": 30,
        "n_prior_samples": 30,
    },
}


CONFIG_PRESETS = {
    "simulator_forecasting_smoke": SMOKE_SIMULATOR_FORECASTING_CONFIG,
    "simulator_forecasting_dev": DEV_SIMULATOR_FORECASTING_CONFIG,
    "simulator_forecasting_paper": DEFAULT_SIMULATOR_FORECASTING_CONFIG,
    "simulator_forecasting_misspecified": MISSPECIFIED_SIMULATOR_FORECASTING_CONFIG,
    "simulator_forecasting_tobs15_vip_ftip_gmvip_figure": TOBS15_VIP_FTIP_GMVIP_FIGURE_CONFIG,
    "simulator_forecasting_tobs15_vip_ftip_gmvip_20targets": TOBS15_VIP_FTIP_GMVIP_20TARGET_CONFIG,
}


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_runner_config(args: argparse.Namespace) -> dict:
    try:
        config = copy.deepcopy(CONFIG_PRESETS[str(args.preset)])
    except KeyError as exc:
        raise ValueError(
            f"Unknown preset {args.preset!r}; expected one of {tuple(CONFIG_PRESETS)}."
        ) from exc
    if args.config is not None:
        override = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        if not isinstance(override, dict):
            raise ValueError(f"Config override {args.config!r} must contain a YAML mapping.")
        _deep_update(config, override)
    config["preset"] = str(args.preset)
    return config


def _as_namespace(mapping: dict) -> SimpleNamespace:
    return SimpleNamespace(**mapping)


def _training_config(config: dict) -> SimpleNamespace:
    training = dict(config.get("training", {}))
    return _as_namespace(
        {
            "learning_rate": training.get("learning_rate", 1e-3),
            "max_steps": training.get("max_steps", 10000),
            "n_mc_train": training.get("n_mc_train", 8),
            "n_mc_eval": training.get("n_mc_eval", 256),
            "batch_size": training.get("batch_size", "full"),
            "hidden_dims": training.get("hidden_dims", [128, 128, 128]),
            "activation": training.get("activation", "tanh"),
            "regression_coeffs": training.get("regression_coeffs", 256),
            "weight_log_sigma_init": training.get("weight_log_sigma_init", -5.0),
            "weight_decay": training.get("weight_decay", 0.0),
            "max_grad_norm": training.get("max_grad_norm", 10.0),
        }
    )


def _metric_regions(config: dict) -> dict[str, tuple[float, float, bool]]:
    return coerce_regions(config.get("metrics", {}).get("regions"))


def _tensor_to_json(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return float(value)
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _tensor_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_to_json(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _activation(name: str):
    if str(name).lower() == "relu":
        return torch.relu
    return torch.tanh


def _set_bnn_fix_random_noise(model: torch.nn.Module, value: bool) -> None:
    for module in model.modules():
        if hasattr(module, "fix_random_noise"):
            module.fix_random_noise = bool(value)


def _fixed_log_variance(noise_std_norm: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.log(noise_std_norm.clamp_min(1e-8))


def _fix_model_noise(model: torch.nn.Module, noise_std_norm: torch.Tensor) -> None:
    log_var = _fixed_log_variance(noise_std_norm).detach().clone()
    if hasattr(model, "log_variance"):
        param = torch.nn.Parameter(
            log_var.to(dtype=model.log_variance.dtype, device=model.log_variance.device)
        )
        model.log_variance = param
        model.log_variance.requires_grad_(False)


def _make_bnn(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: str,
    num_samples: int,
    seed: int,
    device,
    dtype,
    fix_random_noise: bool,
    zero_mean_prior: bool,
    weight_log_sigma_init: float,
) -> BayesianNN:
    return BayesianNN(
        input_dim=input_dim,
        output_dim=output_dim,
        structure=list(hidden_dims),
        activation=_activation(activation),
        num_samples=int(num_samples),
        layer_model=BayesLinear,
        dropout=0.0,
        fix_random_noise=fix_random_noise,
        zero_mean_prior=zero_mean_prior,
        weight_log_sigma_init=weight_log_sigma_init,
        device=device,
        seed=seed,
        dtype=dtype,
    )


def _inducing_grid(num_inducing: int, *, device, dtype) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, int(num_inducing), dtype=dtype, device=device).unsqueeze(-1)


def _make_flow(config: dict, input_dim: int, *, seed: int, device, dtype) -> torch.nn.Module:
    ftip_cfg = dict(config.get("ftip", {}))
    common = {
        "depth": int(ftip_cfg.get("flow_depth", 2)),
        "input_dim": int(input_dim),
        "device": device,
        "dtype": dtype,
        "seed": int(seed),
    }
    flow_type = str(ftip_cfg.get("flow_type", "affine")).lower()
    if flow_type == "spline":
        return SplineCouplingFlow(
            **common,
            num_bins=int(ftip_cfg.get("flow_num_bins", 8)),
            B=float(ftip_cfg.get("flow_domain", 5.0)),
        )
    if flow_type in {"spline_1x1", "spline-1x1", "glow"}:
        return SplineCoupling1x1Flow(
            **common,
            num_bins=int(ftip_cfg.get("flow_num_bins", 8)),
            B=float(ftip_cfg.get("flow_domain", 5.0)),
        )
    return CouplingFlow(**common)


class DeepEnsemble(torch.nn.Module):
    def __init__(self, members: list[DeterministicMAP]):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.is_deep_ensemble = True

    def predict_f_samples(self, X: torch.Tensor, S: int) -> torch.Tensor:
        values = [member.predict_f(X) for member in self.members]
        stacked = torch.stack(values, dim=0)
        if int(S) <= stacked.shape[0]:
            return stacked[: int(S)]
        repeats = math.ceil(int(S) / stacked.shape[0])
        return stacked.repeat((repeats, 1, 1))[: int(S)]


class ContextFBNN(FBNN):
    def __init__(self, *args, context_pool: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("context_pool", context_pool.detach().clone())

    def _sample_measurement_set(self, X_batch):
        parts = []
        if self.num_context > 0 and self.context_pool.numel() > 0:
            count = min(int(self.num_context), int(self.context_pool.shape[0]))
            idx = torch.randperm(self.context_pool.shape[0], device=self.context_pool.device)[
                :count
            ]
            parts.append(self.context_pool[idx].to(dtype=X_batch.dtype, device=X_batch.device))
        if self._reservoir is not None and self.num_measurement > 0:
            count = min(int(self.num_measurement), int(self._reservoir.shape[0]))
            idx = torch.randperm(self._reservoir.shape[0])[:count]
            parts.append(self._reservoir[idx].to(dtype=X_batch.dtype, device=X_batch.device))
        parts.append(X_batch)
        return torch.cat(parts, dim=0)


class FreshDampedOscillatorSIPPrior(torch.nn.Module):
    def __init__(
        self,
        base_prior: DampedOscillatorPrior,
        *,
        num_samples: int,
        seed: int,
        fresh_prior_samples: bool = True,
    ):
        super().__init__()
        self.base_prior = base_prior
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.fresh_prior_samples = bool(fresh_prior_samples)
        self.register_buffer(
            "_sample_counter", torch.zeros((), dtype=torch.long, device=base_prior.device)
        )

    @property
    def input_dim(self) -> int:
        return int(self.base_prior.input_dim)

    @property
    def output_dim(self) -> int:
        return int(self.base_prior.output_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.base_prior.dtype

    @property
    def device(self) -> torch.device:
        return self.base_prior.device

    def forward(self, X: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        sample_count = self.num_samples if num_samples is None else int(num_samples)
        seed = self.seed
        if self.fresh_prior_samples:
            seed = self.seed + int(self._sample_counter.item())
            self._sample_counter.add_(1)
        latents = self.base_prior.sample_latents(
            sample_count, seed=seed, cache=not self.fresh_prior_samples
        )
        return self.base_prior.evaluate(X, latents)

    def sample(self, X: torch.Tensor, n: int, seed: int | None = None) -> torch.Tensor:
        if seed is None:
            return self.forward(X, int(n))
        return self.base_prior.sample(X, int(n), seed=int(seed))

    def freeze_parameters(self) -> None:
        self.base_prior.freeze_parameters()


def _context_pool(method: str, task):
    if method.endswith("_observed"):
        return task.X_context_observed
    return task.X_context_full


def build_model(method: str, task, config: dict, *, seed: int, device, dtype):
    train_cfg = _training_config(config)
    neural_cfg = dict(config.get("neural", {}))
    output_dim = int(task.y_train.shape[-1])
    noise_std_norm = task.noise_std.to(dtype=dtype, device=device)

    def build_map_member(member_seed: int) -> DeterministicMAP:
        model = DeterministicMAP(
            input_dim=1,
            output_dim=output_dim,
            structure=list(train_cfg.hidden_dims),
            activation=_activation(train_cfg.activation),
            num_data=int(task.X_train.shape[0]),
            l2=float(train_cfg.weight_decay),
            y_mean=np.zeros((1, output_dim), dtype=np.float64),
            y_std=np.ones((1, output_dim), dtype=np.float64),
            log_variance_init=_fixed_log_variance(noise_std_norm),
            device=device,
            dtype=dtype,
            seed=member_seed,
        )
        model.log_variance.requires_grad_(False)
        return model

    if method == "map":
        return build_map_member(seed)

    if method == "deep_ensemble":
        members = [
            build_map_member(seed + 1009 * idx)
            for idx in range(int(neural_cfg.get("ensemble_size", 5)))
        ]
        return DeepEnsemble(members)

    if method == "mfvi":
        bnn = _make_bnn(
            input_dim=1,
            output_dim=output_dim,
            hidden_dims=train_cfg.hidden_dims,
            activation=train_cfg.activation,
            num_samples=train_cfg.regression_coeffs,
            seed=seed + 11,
            device=device,
            dtype=dtype,
            fix_random_noise=False,
            zero_mean_prior=False,
            weight_log_sigma_init=train_cfg.weight_log_sigma_init,
        )
        _set_bnn_fix_random_noise(bnn, False)
        model = MFVI(
            generative_function=bnn,
            output_dim=output_dim,
            likelihood="regression",
            num_data=int(task.X_train.shape[0]),
            num_samples=int(train_cfg.n_mc_train),
            bb_alpha=0.0,
            y_mean=np.zeros((1, output_dim), dtype=np.float64),
            y_std=np.ones((1, output_dim), dtype=np.float64),
            device=device,
            dtype=dtype,
        )
        _fix_model_noise(model, noise_std_norm)
        return model

    if method == "vip":
        prior = task.prior.clone_with_normalization(
            y_mean=np.asarray(task.metadata["y_mean"], dtype=np.float64).reshape(1, output_dim),
            y_std=np.asarray(task.metadata["y_std"], dtype=np.float64).reshape(1, output_dim),
            num_samples=int(train_cfg.regression_coeffs),
            seed=seed + 21,
            sample_drag=False,
        )
        model = VIP(
            generative_function=prior,
            num_regression_coeffs=int(train_cfg.regression_coeffs),
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
        _fix_model_noise(model, noise_std_norm)
        return model

    if method == "ftip":
        prior = task.prior.clone_with_normalization(
            y_mean=np.asarray(task.metadata["y_mean"], dtype=np.float64).reshape(1, output_dim),
            y_std=np.asarray(task.metadata["y_std"], dtype=np.float64).reshape(1, output_dim),
            num_samples=int(train_cfg.regression_coeffs),
            seed=seed + 25,
            sample_drag=False,
        )
        flow = _make_flow(
            config,
            input_dim=int(train_cfg.regression_coeffs) * output_dim,
            seed=seed + 26,
            device=device,
            dtype=dtype,
        )
        model = FTIP(
            generative_function=prior,
            num_regression_coeffs=int(train_cfg.regression_coeffs),
            output_dim=output_dim,
            flow=flow,
            likelihood="regression",
            num_data=int(task.X_train.shape[0]),
            num_samples=int(train_cfg.n_mc_train),
            bb_alpha=0.0,
            y_mean=np.zeros((1, output_dim), dtype=np.float64),
            y_std=np.ones((1, output_dim), dtype=np.float64),
            max_grad_norm=train_cfg.max_grad_norm,
            device=device,
            dtype=dtype,
            seed=seed + 27,
        )
        _fix_model_noise(model, noise_std_norm)
        return model

    if method == "sip":
        sip_cfg = dict(config.get("sip", {}))
        num_prior_samples = int(sip_cfg.get("num_prior_samples", 128))
        prior = task.prior.clone_with_normalization(
            y_mean=np.asarray(task.metadata["y_mean"], dtype=np.float64).reshape(1, output_dim),
            y_std=np.asarray(task.metadata["y_std"], dtype=np.float64).reshape(1, output_dim),
            num_samples=num_prior_samples,
            seed=seed + 29,
            sample_drag=False,
        )
        prior_adapter = FreshDampedOscillatorSIPPrior(
            prior,
            num_samples=num_prior_samples,
            seed=seed + 30,
            fresh_prior_samples=bool(sip_cfg.get("fresh_prior_samples", True)),
        )
        model = SIP(
            generative_function=prior_adapter,
            inducing_inputs=_inducing_grid(
                int(sip_cfg.get("num_inducing", 32)), device=device, dtype=dtype
            ),
            output_dim=output_dim,
            likelihood="regression",
            num_data=int(task.X_train.shape[0]),
            num_prior_samples=num_prior_samples,
            num_train_samples=sip_cfg.get("num_train_samples"),
            num_eval_samples=int(train_cfg.n_mc_eval),
            bb_alpha=0.0,
            beta=float(sip_cfg.get("beta", 1.0)),
            beta_warmup_steps=int(sip_cfg.get("beta_warmup_steps", 0)),
            learn_inducing=bool(sip_cfg.get("learn_inducing", False)),
            detach_covariances=bool(sip_cfg.get("detach_covariances", True)),
            critic_hidden_dim=int(sip_cfg.get("critic_hidden_dim", 64)),
            critic_lr=float(sip_cfg.get("critic_lr", 1e-3)),
            critic_steps=int(sip_cfg.get("critic_steps", 1)),
            posterior_noise_dim=int(sip_cfg.get("posterior_noise_dim", 64)),
            posterior_hidden_dim=int(sip_cfg.get("posterior_hidden_dim", 64)),
            posterior_depth=int(sip_cfg.get("posterior_depth", 2)),
            fresh_prior_samples=bool(sip_cfg.get("fresh_prior_samples", True)),
            y_mean=np.zeros((1, output_dim), dtype=np.float64),
            y_std=np.ones((1, output_dim), dtype=np.float64),
            jitter=float(sip_cfg.get("jitter", 1e-4)),
            log_variance_init=float(sip_cfg.get("log_variance_init", -5.0)),
            min_log_variance=sip_cfg.get("min_log_variance"),
            device=device,
            dtype=dtype,
            seed=seed + 31,
        )
        _fix_model_noise(model, noise_std_norm)
        return model

    if method in {"gmvip", "gmvip_cov", "gmvip_rbf"}:
        gmvip_cfg = dict(config.get("gmvip", {}))
        operator = "rbf" if method == "gmvip_rbf" else "empirical"
        bank_size = int(config.get("prior", {}).get("bank_size", 1024))
        prior = task.prior.clone_with_normalization(
            y_mean=np.asarray(task.metadata["y_mean"], dtype=np.float64).reshape(1, output_dim),
            y_std=np.asarray(task.metadata["y_std"], dtype=np.float64).reshape(1, output_dim),
            num_samples=max(bank_size, int(train_cfg.n_mc_train), 2),
            seed=seed + 31,
            sample_drag=False,
        )
        return GeneralizedMatheronVIP(
            base_prior=prior,
            inducing_points=_inducing_grid(
                int(gmvip_cfg.get("num_inducing", 32)), device=device, dtype=dtype
            ),
            operator_type=operator,
            posterior_type="gaussian",
            likelihood="regression",
            num_operator_bank_samples=bank_size,
            learn_noise=False,
            init_log_noise=torch.log(noise_std_norm.clamp_min(1e-8)),
            min_log_noise=math.log(1e-8),
            freeze_base_prior=True,
            detach_prior_samples=True,
            jitter=float(gmvip_cfg.get("jitter", 1e-4)),
            shrinkage=float(gmvip_cfg.get("shrinkage", 0.02)),
            learn_Z=False,
            learn_kernel=bool(gmvip_cfg.get("learn_kernel", operator == "rbf")),
            ard=True,
            init_lengthscale=gmvip_cfg.get("rbf_lengthscale", "median"),
            init_outputscale="prior_marginal",
            inducing_scale="prior_cholesky" if operator == "empirical" else "prior_cholesky",
            mean_mode="prior_sample",
            posterior_init_mean=0.0,
            posterior_init_log_std=0.0,
            antithetic_samples=True,
            num_data=int(task.X_train.shape[0]),
            num_train_samples=int(train_cfg.n_mc_train),
            beta=float(gmvip_cfg.get("beta", 1.0)),
            beta_warmup_steps=0,
            data_alpha=0.0,
            max_grad_norm=train_cfg.max_grad_norm,
            output_dim=output_dim,
            operator_bank_seed=seed + 101,
        )

    if method in {"fbnn_observed", "fbnn_full"}:
        bnn = _make_bnn(
            input_dim=1,
            output_dim=output_dim,
            hidden_dims=train_cfg.hidden_dims,
            activation=train_cfg.activation,
            num_samples=max(int(train_cfg.n_mc_train), 2),
            seed=seed + 41,
            device=device,
            dtype=dtype,
            fix_random_noise=True,
            zero_mean_prior=False,
            weight_log_sigma_init=train_cfg.weight_log_sigma_init,
        )
        prior = GP(
            input_dim=1,
            output_dim=output_dim,
            inner_layer_dim=64,
            kernel_amp=1.0,
            kernel_length=0.5,
            seed=seed + 42,
            device=device,
            dtype=dtype,
        )
        model = ContextFBNN(
            generative_function=bnn,
            prior_function=prior,
            output_dim=output_dim,
            likelihood="regression",
            num_data=int(task.X_train.shape[0]),
            num_samples=max(int(train_cfg.n_mc_train), 2),
            num_measurement=int(neural_cfg.get("fbnn_num_measurement", 16)),
            num_context=int(neural_cfg.get("fbnn_num_context", 32)),
            context_std=float(neural_cfg.get("fbnn_context_std", 1.0)),
            bb_alpha=0.0,
            lambda_kl=float(neural_cfg.get("fbnn_lambda_kl", 1.0)),
            y_mean=np.zeros((1, output_dim), dtype=np.float64),
            y_std=np.ones((1, output_dim), dtype=np.float64),
            freeze_prior=True,
            context_pool=_context_pool(method, task),
            device=device,
            dtype=dtype,
        )
        _fix_model_noise(model, noise_std_norm)
        return model

    if method in {"tfsvi_observed", "tfsvi_full"}:
        model = TFSVI(
            input_dim=1,
            output_dim=output_dim,
            structure=list(train_cfg.hidden_dims),
            activation=_activation(train_cfg.activation),
            likelihood="regression",
            num_data=int(task.X_train.shape[0]),
            sigma_prior=float(neural_cfg.get("tfsvi_sigma_prior", 1.0)),
            num_samples=max(int(train_cfg.n_mc_train), 2),
            bb_alpha=0.0,
            S_ctx=int(neural_cfg.get("tfsvi_S_ctx", 2)),
            K_ctx=int(neural_cfg.get("tfsvi_K_ctx", 32)),
            y_mean=np.zeros((1, output_dim), dtype=np.float64),
            y_std=np.ones((1, output_dim), dtype=np.float64),
            device=device,
            dtype=dtype,
        )
        model._train_inputs = _context_pool(method, task).detach().clone()
        _fix_model_noise(model, noise_std_norm)
        return model

    raise ValueError(f"Unknown method {method!r}.")


def vip_pathwise_samples(model: VIP, X: torch.Tensor, samples: int) -> torch.Tensor:
    if model.dtype != X.dtype:
        X = X.to(model.dtype)
    f = model.generative_function(X)
    m = f.mean(dim=0, keepdim=True)
    phi = (f - m) / model._sqrt_coeffs_m1
    q_sqrt = torch.zeros_like(model._q_sqrt_buf)
    q_sqrt[model._tril_row, model._tril_col] = model.q_sqrt_tri
    eps = torch.randn(
        int(samples),
        model.num_coeffs,
        model.output_dim,
        generator=model.generator,
        dtype=model.dtype,
        device=model.device,
    )
    coeffs = model.q_mu.unsqueeze(0) + torch.einsum("sid,asd->aid", q_sqrt, eps)
    return torch.einsum("ind,aid->and", phi, coeffs) + m.squeeze(0)


def predictive_function_samples(
    model, method: str, X: torch.Tensor, n_samples: int, seed: int
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if method in {"map", "deep_ensemble"}:
            values = model.predict_f_samples(X, S=int(n_samples))
        elif method == "mfvi":
            values = model.predict_f_samples(X, int(n_samples))
        elif method == "vip":
            values = vip_pathwise_samples(model, X, int(n_samples))
        elif method == "ftip":
            requested = int(n_samples)
            call_samples = requested if requested % 2 == 0 else requested + 1
            values = model.predict_y(X, call_samples)[:requested]
        elif method == "sip":
            values = model.predict_f_samples(X, int(n_samples))
        elif method in {"gmvip", "gmvip_cov", "gmvip_rbf"}:
            values = model.sample_posterior_values(X, int(n_samples), seed=seed)
        elif method in {"fbnn_observed", "fbnn_full", "tfsvi_observed", "tfsvi_full"}:
            values = model.predict_f_samples(X, int(n_samples))
        else:
            raise ValueError(f"Unknown method {method!r}.")
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    return values


def _train_one_model(model, task, config: dict, *, device) -> dict:
    train_cfg = _training_config(config)
    dataset = TensorDataset(task.X_train, task.y_train)
    full_batch = train_cfg.batch_size == "full"
    batch_size = len(dataset) if full_batch else int(train_cfg.batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not full_batch, num_workers=0)
    if hasattr(model, "_fill_reservoir"):
        model._fill_reservoir(loader)
    if hasattr(model, "_train_inputs") and model._train_inputs is None:
        model._train_inputs = task.X_train.detach().clone()
    params = model.vi_parameters() if hasattr(model, "vi_parameters") else model.parameters()
    params = [param for param in params if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(train_cfg.learning_rate))
    losses = []
    start = time.time()
    stream = iter(loader)
    disable = bool(config.get("training", {}).get("disable_tqdm", False))
    loop = tqdm(range(int(train_cfg.max_steps)), desc="train", unit=" step", disable=disable)
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
        losses.append(float(loss.detach().cpu()))
        loop.set_postfix(loss=f"{losses[-1]:.3f}")
    return {
        "train_time_sec": time.time() - start,
        "steps": len(losses),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
    }


def fit_model(model, method: str, task, config: dict, *, device) -> dict:
    if getattr(model, "is_deep_ensemble", False):
        infos = []
        start = time.time()
        for member in model.members:
            infos.append(_train_one_model(member, task, config, device=device))
        return {
            "train_time_sec": time.time() - start,
            "steps": sum(int(info["steps"]) for info in infos),
            "loss_start": infos[0]["loss_start"] if infos else None,
            "loss_end": infos[-1]["loss_end"] if infos else None,
        }
    return _train_one_model(model, task, config, device=device)


def _unnormalize(task, values: torch.Tensor) -> torch.Tensor:
    y_mean = torch.as_tensor(
        task.metadata["y_mean"], dtype=values.dtype, device=values.device
    ).reshape(1, 1)
    y_std = torch.as_tensor(
        task.metadata["y_std"], dtype=values.dtype, device=values.device
    ).reshape(1, 1)
    return values * y_std + y_mean


def evaluate_target(
    model, method: str, task, config: dict, *, seed: int, out_dir: Path
) -> list[dict]:
    eval_samples = int(_training_config(config).n_mc_eval)
    start = time.time()
    samples_norm = predictive_function_samples(model, method, task.X_plot, eval_samples, seed + 501)
    samples = _unnormalize(task, samples_norm)
    y_true = _unnormalize(task, task.y_plot_true)
    t_grid = torch.as_tensor(task.metadata["t_grid"], dtype=samples.dtype, device=samples.device)
    noise_std = torch.as_tensor(
        [float(task.metadata["sigma_y"])], dtype=samples.dtype, device=samples.device
    )
    regions = _metric_regions(config)
    metric_rows = metrics_by_region(
        samples, y_true, t_grid, noise_std, levels=(0.9, 0.95), regions=regions
    )

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    train_t = np.asarray(task.metadata["train_t"], dtype=np.float64)
    train_y = np.asarray(task.metadata["y_train_physical"], dtype=np.float64)
    samples_np = samples.detach().cpu().numpy()
    np.savez_compressed(
        pred_dir / f"target_{task.metadata['target_id']}_ntrain_{task.metadata['n_train']}.npz",
        t=np.asarray(task.metadata["t_grid"], dtype=np.float64),
        truth=y_true.detach().cpu().numpy(),
        train_t=train_t,
        train_y=train_y,
        samples=samples_np,
        mean=samples_np.mean(axis=0),
        std=samples_np.std(axis=0),
        q05=np.quantile(samples_np, 0.05, axis=0),
        q95=np.quantile(samples_np, 0.95, axis=0),
    )

    if not bool(config.get("plots", {}).get("skip", False)):
        fig_dir = out_dir / "figures"
        try:
            plot_posterior_forecast(
                fig_dir
                / f"posterior_target_{task.metadata['target_id']}_ntrain_{task.metadata['n_train']}",
                t=np.asarray(task.metadata["t_grid"], dtype=np.float64),
                y_true=y_true.detach().cpu().numpy(),
                train_t=train_t,
                train_y=train_y,
                samples=samples_np,
                method=method,
                n_samples=int(config.get("plots", {}).get("n_posterior_samples", 20)),
                t_obs=float(task.metadata["t_obs"]),
            )
        except ImportError as exc:
            print(f"Skipping plot generation: {exc}")

    rows = []
    for region, values in metric_rows.items():
        lo, hi, include_left = regions[region]
        rows.append(
            {
                "experiment": "simulator_forecasting",
                "method": method,
                "seed": int(seed),
                "target_id": int(task.metadata["target_id"]),
                "n_train": int(task.metadata["n_train"]),
                "region": region,
                "region_start": lo,
                "region_end": hi,
                "region_include_left": include_left,
                "eval_time_sec": time.time() - start,
                **values,
            }
        )
    return rows


def _target_ids(cli_args, n_tasks: int) -> list[int]:
    if cli_args.target_ids:
        ids = [int(value) for value in cli_args.target_ids.split(",") if value.strip()]
    else:
        start = 0 if cli_args.target_start is None else int(cli_args.target_start)
        stop = n_tasks if cli_args.target_stop is None else min(int(cli_args.target_stop), n_tasks)
        ids = list(range(start, stop))
    return [idx for idx in ids if 0 <= idx < n_tasks]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("method"), row.get("n_train"), row.get("region"))
        groups.setdefault(key, []).append(row)
    summary = {}
    metrics = ("rmse", "nlpd", "crps", "cov90", "cov95", "width90", "width95")
    for key, group in groups.items():
        method, n_train, region = key
        name = f"{method}|n_train={n_train}|{region}"
        summary[name] = {}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group if metric in row], dtype=np.float64)
            if values.size:
                summary[name][metric] = {
                    "mean": float(np.nanmean(values)),
                    "stderr": float(np.nanstd(values) / max(1.0, math.sqrt(values.size))),
                }
    return summary


def _ensure_dataset(config: dict, seed: int) -> None:
    data_cfg = dict(config.get("data", {}))
    root = Path(data_cfg.get("root", "data/simprior/simulator_forecasting"))
    if (root / "target_paths.npz").exists() and (root / "metadata.json").exists():
        return
    generate_dataset(
        root,
        n_targets=int(data_cfg.get("n_eval_targets", 20)),
        n_prior=int(config.get("prior", {}).get("bank_size", 1024)),
        n_test=int(data_cfg.get("n_test", 500)),
        t_max=float(data_cfg.get("t_max", 30.0)),
        forcing_delta=float(data_cfg.get("forcing_delta", 0.1)),
        rho=float(data_cfg.get("rho", 0.98)),
        sigma_u=float(data_cfg.get("sigma_u", 0.05)),
        sigma_y=float(data_cfg.get("sigma_y", 0.05)),
        misspecified=bool(data_cfg.get("misspecified", False)),
        seed=seed,
    )


def _n_train_values(config: dict) -> list[int]:
    values = config.get("data", {}).get("n_train", [16])
    if isinstance(values, int):
        return [int(values)]
    return [int(value) for value in values]


def _requested_methods(value: str) -> list[str]:
    value = str(value)
    if value == "all":
        return list(METHODS)
    methods = [
        METHOD_ALIASES.get(item.strip(), item.strip()) for item in value.split(",") if item.strip()
    ]
    if not methods:
        raise ValueError("--method must name at least one method.")
    return list(dict.fromkeys(methods))


def _config_with_training(config: dict, **updates) -> dict:
    result = copy.deepcopy(config)
    result.setdefault("training", {})
    for key, value in updates.items():
        if value is not None:
            result["training"][key] = value
    return result


def _fit_ftip_with_optional_warm_start(
    model, task, config: dict, *, seed: int, device, dtype
) -> dict:
    ftip_cfg = dict(config.get("ftip", {}))
    if not bool(ftip_cfg.get("warm_start_from_vip", False)):
        return fit_model(model, "ftip", task, config, device=device)

    train_cfg = _training_config(config)
    warm_steps = int(ftip_cfg.get("warm_start_steps") or train_cfg.max_steps)
    fine_steps = int(ftip_cfg.get("fine_tune_steps") or train_cfg.max_steps)
    warm_lr = float(ftip_cfg.get("warm_start_lr") or train_cfg.learning_rate)
    fine_lr = float(ftip_cfg.get("fine_tune_lr") or train_cfg.learning_rate)

    vip_config = _config_with_training(config, max_steps=warm_steps, learning_rate=warm_lr)
    vip_model = build_model("vip", task, config, seed=seed + 7919, device=device, dtype=dtype)
    vip_info = fit_model(vip_model, "vip", task, vip_config, device=device)

    model.warm_start_from_vip(
        vip_model, learnable_affine=bool(ftip_cfg.get("learnable_affine", False))
    )
    ftip_config = _config_with_training(config, max_steps=fine_steps, learning_rate=fine_lr)
    ftip_info = fit_model(model, "ftip", task, ftip_config, device=device)
    return {
        "train_time_sec": float(vip_info["train_time_sec"]) + float(ftip_info["train_time_sec"]),
        "steps": int(vip_info["steps"]) + int(ftip_info["steps"]),
        "loss_start": ftip_info["loss_start"],
        "loss_end": ftip_info["loss_end"],
        "warm_start_steps": int(vip_info["steps"]),
        "fine_tune_steps": int(ftip_info["steps"]),
        "warm_start_loss_start": vip_info["loss_start"],
        "warm_start_loss_end": vip_info["loss_end"],
    }


def run_method(method: str, config: dict, cli_args) -> dict:
    seed = int(cli_args.seed)
    dtype = torch.float64 if str(cli_args.dtype) == "float64" else torch.float32
    device = torch.device(cli_args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _ensure_dataset(config, seed)
    data_cfg = dict(config.get("data", {}))
    prior_cfg = dict(config.get("prior", {}))
    out_dir = (
        Path(cli_args.output_dir or "results/simprior")
        / "simulator_forecasting"
        / method
        / f"seed_{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    rows = []
    runtimes = []
    for n_train in _n_train_values(config):
        tasks = load_damped_oscillator_tasks(
            data_cfg.get("root", "data/simprior/simulator_forecasting"),
            seed=seed,
            n_eval_targets=int(data_cfg.get("n_eval_targets", 20)),
            n_train=int(n_train),
            t_obs=float(data_cfg.get("t_obs", 8.0)),
            sigma_y=float(data_cfg.get("sigma_y", 0.05)),
            prior_bank_size=prior_cfg.get("bank_size"),
            context_points=int(data_cfg.get("context_points", 128)),
            device=device,
            dtype=dtype,
        )
        ids = _target_ids(cli_args, len(tasks))
        for target_idx in ids:
            task = tasks[target_idx]
            model_seed = seed + 1000 * target_idx + 100_000 * int(n_train)
            model = build_model(method, task, config, seed=model_seed, device=device, dtype=dtype)
            if method == "ftip":
                train_info = _fit_ftip_with_optional_warm_start(
                    model, task, config, seed=model_seed, device=device, dtype=dtype
                )
            else:
                train_info = fit_model(model, method, task, config, device=device)
            target_rows = evaluate_target(
                model, method, task, config, seed=model_seed, out_dir=out_dir
            )
            for row in target_rows:
                row["train_time_sec"] = float(train_info["train_time_sec"])
                row["train_steps"] = int(train_info["steps"])
                row["loss_start"] = train_info["loss_start"]
                row["loss_end"] = train_info["loss_end"]
            rows.extend(target_rows)
            runtimes.append({"target_id": target_idx, "n_train": int(n_train), **train_info})

    metrics = {"method": method, "seed": seed, "summary": _summarize(rows)}
    (out_dir / "metrics.json").write_text(
        json.dumps(_tensor_to_json(metrics), indent=2), encoding="utf-8"
    )
    (out_dir / "runtime.json").write_text(
        json.dumps(_tensor_to_json(runtimes), indent=2), encoding="utf-8"
    )
    _write_csv(out_dir / "metrics_per_target_region.csv", rows)

    if rows and not bool(config.get("plots", {}).get("skip", False)):
        try:
            plot_metric_by_region(out_dir / "figures" / "nlpd_by_region", rows=rows, metric="nlpd")
            plot_metric_by_region(
                out_dir / "figures" / "coverage90_by_region", rows=rows, metric="cov90"
            )
            first_task = load_damped_oscillator_tasks(
                data_cfg.get("root", "data/simprior/simulator_forecasting"),
                seed=seed,
                n_eval_targets=1,
                n_train=_n_train_values(config)[0],
                t_obs=float(data_cfg.get("t_obs", 8.0)),
                prior_bank_size=prior_cfg.get("bank_size"),
                device=device,
                dtype=dtype,
            )[0]
            prior_ids = first_task.prior.sample_indices(
                int(config.get("plots", {}).get("n_prior_samples", 20)), seed=seed + 77
            )
            prior_samples = (
                first_task.prior.evaluate_raw(first_task.X_plot, prior_ids).detach().cpu().numpy()
            )
            plot_prior_samples(
                out_dir / "figures" / "prior_samples",
                t=np.asarray(first_task.metadata["t_grid"], dtype=np.float64),
                samples=prior_samples,
                n_samples=int(config.get("plots", {}).get("n_prior_samples", 20)),
                t_obs=float(first_task.metadata["t_obs"]),
            )
        except ImportError as exc:
            print(f"Skipping plot generation: {exc}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulator-prior forecasting experiments.")
    parser.add_argument(
        "--preset", choices=tuple(CONFIG_PRESETS), default="simulator_forecasting_dev"
    )
    parser.add_argument("--config", default=None, help="Optional YAML override.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", "--methods", dest="method", default=None)
    parser.add_argument("--num-inducing", type=int, default=None)
    parser.add_argument("--prior-bank-size", type=int, default=None)
    parser.add_argument("--target-start", type=int, default=None)
    parser.add_argument("--target-stop", type=int, default=None)
    parser.add_argument("--target-ids", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    config = _load_runner_config(args)
    if args.method is not None:
        config["method"] = args.method
    if args.num_inducing is not None:
        config.setdefault("gmvip", {})["num_inducing"] = int(args.num_inducing)
    if args.prior_bank_size is not None:
        config.setdefault("prior", {})["bank_size"] = int(args.prior_bank_size)
    if args.skip_plots:
        config.setdefault("plots", {})["skip"] = True
    if args.disable_tqdm:
        config.setdefault("training", {})["disable_tqdm"] = True
    if config.get("experiment") != "simulator_forecasting":
        raise ValueError("This runner only supports experiment: simulator_forecasting.")
    requested = config.get("method", "gmvip")
    methods = _requested_methods(requested)
    results = {}
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")
        results[method] = run_method(method, config, args)
    return results


if __name__ == "__main__":
    main()
