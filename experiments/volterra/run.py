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

from experiments.volterra.datasets import load_lotka_volterra_tasks
from experiments.volterra.metrics import (
    crps_from_samples,
    gaussian_nll_from_samples,
    interval_coverage,
    interval_width,
    lotka_volterra_residual_score,
    nearest_prior_mse,
    rmse,
)
from experiments.volterra.plots import (
    plot_lv_phase_portrait,
    plot_lv_posterior_trajectory,
    plot_lv_prior_vs_posterior,
)
from experiments.volterra.priors import LotkaVolterraPrior
from src.flows import CouplingFlow, SplineCoupling1x1Flow, SplineCouplingFlow
from src.ftip import FTIP
from src.gmvip import GeneralizedMatheronVIP
from src.map_baseline import DeterministicMAP
from src.mfvi import MFVI
from src.priors.generative_functions import BayesianNN, BayesLinear
from src.sip import SIP
from src.vip import VIP


METHODS = ("map", "mfvi", "vip", "ftip", "sip", "gmvip_empirical", "gmvip_rbf", "oracle_prior_bank")
METHOD_ALIASES = {"oracle": "oracle_prior_bank"}


DEFAULT_LOTKA_VOLTERRA_CONFIG: dict = {
    "experiment": "lotka_volterra",
    "method": "gmvip_empirical",
    "data": {
        "root": "data/simprior/lotka_volterra",
        "n_eval_targets": 20,
        "n_train_times": 80,
        "noise_scale": 0.03,
    },
    "prior": {
        "bank_size": 256,
        "normalize_outputs": True,
    },
    "oracle_prior_bank": {
        # None means sample and weight reference_bank_size ODE prior functions.
        "bank_size": None,
    },
    "gmvip": {
        "operator": "empirical",
        "num_inducing": 96,
        "prior_bank_size": 512,
        "rbf_lengthscale": 0.25,
        "jitter": 1.0e-5,
        "shrinkage": 0.02,
        "learn_kernel": False,
        "beta": 1.0,
        "training_overrides": {
            "max_steps": 800,
            "early_stopping_patience": 801,
            "eval_interval": 100,
        },
    },
    "ftip": {
        "flow_type": "affine",
        "flow_depth": 1,
        "flow_num_bins": 8,
        "flow_domain": 5.0,
        "warm_start_from_vip": True,
        "warm_start_learnable_affine": False,
        "training_overrides": {
            "regression_coeffs": 128,
            "n_mc_train": 8,
        },
        "fine_tune_training": {
            "learning_rate": 2.0e-4,
            "max_steps": 625,
            "early_stopping_patience": 626,
            "eval_interval": 100,
        },
    },
    "sip": {
        "num_inducing": 32,
        "num_prior_samples": 128,
        "num_train_samples": 16,
        "num_eval_samples": 256,
        "fresh_prior_samples": True,
        "learn_inducing": False,
        "detach_covariances": True,
        "jitter": 1.0e-5,
        "beta": 1.0,
        "critic_hidden_dim": 64,
        "critic_lr": 1.0e-3,
        "critic_steps": 1,
        "posterior_noise_dim": 64,
        "posterior_hidden_dim": 64,
        "posterior_depth": 2,
    },
    "training": {
        "optimizer": "adam",
        "learning_rate": 2.0e-3,
        "max_steps": 400,
        "early_stopping_patience": 401,
        "eval_interval": 100,
        "kl_warmup_steps": 200,
        "n_mc_train": 4,
        "n_mc_eval": 256,
        "batch_size": "full",
        "hidden_dims": [32, 32],
        "activation": "tanh",
        "regression_coeffs": 512,
        "disable_tqdm": False,
    },
    "metrics": {
        "levels": [0.5, 0.8, 0.9, 0.95],
    },
    "plots": {
        "n_posterior_samples": 10,
        "n_prior_samples": 20,
    },
}


SMOKE_LOTKA_VOLTERRA_CONFIG: dict = {
    **copy.deepcopy(DEFAULT_LOTKA_VOLTERRA_CONFIG),
    "data": {
        "root": "data/simprior/lotka_volterra_smoke",
        "n_eval_targets": 1,
        "n_train_times": 8,
        "noise_scale": 0.03,
    },
    "prior": {
        "bank_size": 16,
        "normalize_outputs": True,
    },
    "gmvip": {
        "operator": "rbf",
        "num_inducing": 6,
        "rbf_lengthscale": 0.25,
        "jitter": 1.0e-5,
        "shrinkage": 0.02,
        "learn_kernel": False,
    },
    "sip": {
        "num_inducing": 4,
        "num_prior_samples": 8,
        "num_train_samples": 4,
        "num_eval_samples": 4,
        "fresh_prior_samples": True,
        "learn_inducing": False,
        "detach_covariances": True,
        "jitter": 1.0e-5,
        "beta": 1.0,
        "critic_hidden_dim": 8,
        "critic_lr": 1.0e-3,
        "critic_steps": 1,
        "posterior_noise_dim": 8,
        "posterior_hidden_dim": 8,
        "posterior_depth": 2,
    },
    "training": {
        "optimizer": "adam",
        "learning_rate": 1.0e-3,
        "max_steps": 3,
        "early_stopping_patience": 3,
        "eval_interval": 1,
        "kl_warmup_steps": 0,
        "n_mc_train": 2,
        "n_mc_eval": 4,
        "batch_size": "full",
        "hidden_dims": [8],
        "activation": "tanh",
        "regression_coeffs": 8,
        "disable_tqdm": True,
    },
    "plots": {
        "skip": True,
        "n_posterior_samples": 2,
        "n_prior_samples": 2,
    },
}


CONFIG_PRESETS = {
    "lotka_volterra": DEFAULT_LOTKA_VOLTERRA_CONFIG,
    "lotka_volterra_smoke": SMOKE_LOTKA_VOLTERRA_CONFIG,
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
        raise ValueError(f"Unknown preset {args.preset!r}; expected one of {tuple(CONFIG_PRESETS)}.") from exc
    if args.config is not None:
        override = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        if not isinstance(override, dict):
            raise ValueError(f"Config override {args.config!r} must contain a YAML mapping.")
        _deep_update(config, override)
    config["preset"] = str(args.preset)
    return config


def _activation(name: str):
    if str(name).lower() == "relu":
        return torch.relu
    return torch.tanh


def _set_bnn_fix_random_noise(model: torch.nn.Module, value: bool) -> None:
    for module in model.modules():
        if hasattr(module, "fix_random_noise"):
            module.fix_random_noise = bool(value)


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


def _as_namespace(mapping: dict) -> SimpleNamespace:
    return SimpleNamespace(**mapping)


def _training_config(config: dict) -> SimpleNamespace:
    training = dict(config.get("training", {}))
    return _as_namespace(
        {
            "learning_rate": training.get("learning_rate", 1e-3),
            "max_steps": training.get("max_steps", training.get("iterations", 10_000)),
            "early_stopping_patience": training.get("early_stopping_patience", 1000),
            "eval_interval": training.get("eval_interval", 50),
            "n_mc_train": training.get("n_mc_train", 8),
            "n_mc_eval": training.get("n_mc_eval", 256),
            "kl_warmup_steps": training.get("kl_warmup_steps", 2000),
            "batch_size": training.get("batch_size", "full"),
            "hidden_dims": training.get("hidden_dims", [32, 32]),
            "activation": training.get("activation", "tanh"),
            "regression_coeffs": training.get("regression_coeffs", 64),
            "weight_log_sigma_init": training.get("weight_log_sigma_init", -1.0),
            "max_grad_norm": training.get("max_grad_norm", 10.0),
        }
    )


def _with_training_overrides(config: dict, overrides: dict | None) -> dict:
    updated = copy.deepcopy(config)
    updated.setdefault("training", {}).update(dict(overrides or {}))
    return updated


def _ftip_base_config(config: dict) -> dict:
    ftip_cfg = dict(config.get("ftip", {}))
    return _with_training_overrides(config, ftip_cfg.get("training_overrides", {}))


def _gmvip_base_config(config: dict) -> dict:
    gmvip_cfg = dict(config.get("gmvip", {}))
    updated = _with_training_overrides(config, gmvip_cfg.get("training_overrides", {}))
    if "prior_bank_size" in gmvip_cfg:
        current_bank = updated.setdefault("prior", {}).get("bank_size")
        default_bank = DEFAULT_LOTKA_VOLTERRA_CONFIG["prior"]["bank_size"]
        if current_bank in {None, default_bank}:
            updated["prior"]["bank_size"] = int(gmvip_cfg["prior_bank_size"])
    return updated


def _fixed_log_variance(noise_std_norm: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.log(noise_std_norm.clamp_min(1e-8))


def _fix_model_noise(model: torch.nn.Module, noise_std_norm: torch.Tensor) -> None:
    log_var = _fixed_log_variance(noise_std_norm).detach().clone()
    if hasattr(model, "log_variance"):
        param = torch.nn.Parameter(log_var.to(dtype=model.log_variance.dtype, device=model.log_variance.device))
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


class OraclePriorBankPosterior(torch.nn.Module):
    """Discrete posterior over a Monte Carlo bank sampled from the ODE prior."""

    def __init__(self, task, *, bank_size: int | None, seed: int, device, dtype):
        super().__init__()
        self.is_oracle_prior_bank = True
        self.prior = task.prior
        self.seed = int(seed)
        bank_size = task.prior.num_paths if bank_size is None else int(bank_size)
        bank_latents = task.prior.sample_latents(bank_size, seed=seed).to(device=device, dtype=dtype)
        self.register_buffer("bank_latents", bank_latents)

        with torch.no_grad():
            X_train = task.X_train.to(device=device, dtype=dtype)
            y_train = task.y_train.to(device=device, dtype=dtype)
            noise_var = task.noise_std.to(device=device, dtype=dtype).square().clamp_min(1e-12).reshape(1, 1, -1)
            bank_train = self.prior.evaluate(X_train, self.bank_latents)
            log_lik = -0.5 * ((bank_train - y_train.unsqueeze(0)).square() / noise_var + torch.log(2.0 * math.pi * noise_var)).sum(
                dim=(1, 2)
            )
            if not torch.isfinite(log_lik).any():
                weights = torch.full_like(log_lik, 1.0 / max(1, log_lik.numel()))
                log_weights = torch.log(weights)
            else:
                floor = torch.tensor(-torch.inf, dtype=log_lik.dtype, device=log_lik.device)
                log_lik = torch.where(torch.isfinite(log_lik), log_lik, floor)
                log_weights = log_lik - torch.logsumexp(log_lik, dim=0)
                weights = torch.exp(log_weights)
        self.register_buffer("log_weights", log_weights)
        self.register_buffer("weights", weights)

    def predict_f_samples(self, X: torch.Tensor, n_samples: int, *, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.weights.device)
        generator.manual_seed(int(seed))
        draw_idx = torch.multinomial(self.weights, int(n_samples), replacement=True, generator=generator)
        return self.prior.evaluate(X, self.bank_latents[draw_idx])


class FreshLotkaVolterraSIPPrior(torch.nn.Module):
    """SIP adapter that can draw fresh Lotka-Volterra ODE latents per prior call."""

    def __init__(
        self,
        base_prior: LotkaVolterraPrior,
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
        self.register_buffer("_sample_counter", torch.zeros((), dtype=torch.long, device=base_prior.device))

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
        latents = self.base_prior.sample_latents(sample_count, seed=seed, cache=not self.fresh_prior_samples)
        return self.base_prior.evaluate(X, latents)

    def sample(self, X: torch.Tensor, n: int, seed: int | None = None) -> torch.Tensor:
        if seed is None:
            return self.forward(X, int(n))
        return self.base_prior.sample(X, int(n), seed=int(seed))

    def freeze_parameters(self) -> None:
        self.base_prior.freeze_parameters()


def build_model(method: str, task, config: dict, *, seed: int, device, dtype):
    method = METHOD_ALIASES.get(method, method)
    if method == "ftip":
        config = _ftip_base_config(config)
    if method == "gmvip_empirical":
        config = _gmvip_base_config(config)
    train_cfg = _training_config(config)
    output_dim = int(task.y_train.shape[-1])
    noise_std_norm = task.noise_std.to(dtype=dtype, device=device)
    if method == "map":
        model = DeterministicMAP(
            input_dim=1,
            output_dim=output_dim,
            structure=list(train_cfg.hidden_dims),
            activation=_activation(train_cfg.activation),
            num_data=int(task.X_train.shape[0]),
            l2=float(config.get("training", {}).get("weight_decay", 0.0)),
            y_mean=np.zeros((1, output_dim), dtype=np.float64),
            y_std=np.ones((1, output_dim), dtype=np.float64),
            log_variance_init=_fixed_log_variance(noise_std_norm),
            device=device,
            dtype=dtype,
            seed=seed,
        )
        model.log_variance.requires_grad_(False)
        return model

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
        )
        prior_adapter = FreshLotkaVolterraSIPPrior(
            prior,
            num_samples=num_prior_samples,
            seed=seed + 30,
            fresh_prior_samples=bool(sip_cfg.get("fresh_prior_samples", True)),
        )
        model = SIP(
            generative_function=prior_adapter,
            inducing_inputs=_inducing_grid(int(sip_cfg.get("num_inducing", 32)), device=device, dtype=dtype),
            output_dim=output_dim,
            likelihood="regression",
            num_data=int(task.X_train.shape[0]),
            num_prior_samples=num_prior_samples,
            num_train_samples=sip_cfg.get("num_train_samples", None),
            num_eval_samples=int(sip_cfg.get("num_eval_samples", train_cfg.n_mc_eval)),
            bb_alpha=0.0,
            beta=float(sip_cfg.get("beta", 1.0)),
            beta_warmup_steps=int(train_cfg.kl_warmup_steps),
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
            jitter=float(sip_cfg.get("jitter", 1e-5)),
            log_variance_init=float(sip_cfg.get("log_variance_init", -5.0)),
            min_log_variance=sip_cfg.get("min_log_variance", None),
            device=device,
            dtype=dtype,
            seed=seed + 31,
        )
        _fix_model_noise(model, noise_std_norm)
        return model

    if method in {"gmvip_empirical", "gmvip_rbf"}:
        gmvip_cfg = dict(config.get("gmvip", {}))
        operator = "empirical" if method == "gmvip_empirical" else "rbf"
        num_inducing = int(gmvip_cfg.get("num_inducing", 32))
        bank_size = int(config.get("prior", {}).get("bank_size", 512))
        prior = task.prior.clone_with_normalization(
            y_mean=np.asarray(task.metadata["y_mean"], dtype=np.float64).reshape(1, output_dim),
            y_std=np.asarray(task.metadata["y_std"], dtype=np.float64).reshape(1, output_dim),
            num_samples=max(bank_size, int(train_cfg.n_mc_train), 2),
            seed=seed + 31,
        )
        model = GeneralizedMatheronVIP(
            base_prior=prior,
            inducing_points=_inducing_grid(num_inducing, device=device, dtype=dtype),
            operator_type=operator,
            posterior_type="gaussian",
            likelihood="regression",
            num_operator_bank_samples=bank_size,
            learn_noise=False,
            init_log_noise=torch.log(noise_std_norm.clamp_min(1e-8)),
            min_log_noise=math.log(1e-8),
            freeze_base_prior=True,
            detach_prior_samples=True,
            jitter=float(gmvip_cfg.get("jitter", 1e-5)),
            shrinkage=float(gmvip_cfg.get("shrinkage", 0.02)),
            learn_Z=bool(gmvip_cfg.get("learn_Z", False)),
            learn_kernel=bool(gmvip_cfg.get("learn_kernel", operator == "rbf")),
            ard=bool(gmvip_cfg.get("ard", True)),
            init_lengthscale=gmvip_cfg.get("rbf_lengthscale", 0.25),
            init_outputscale=gmvip_cfg.get("init_outputscale", "prior_marginal"),
            inducing_scale="prior_cholesky",
            mean_mode="prior_sample",
            posterior_init_mean=float(gmvip_cfg.get("posterior_init_mean", 0.0)),
            posterior_init_log_std=float(gmvip_cfg.get("posterior_init_log_std", 0.0)),
            antithetic_samples=True,
            num_data=int(task.X_train.shape[0]),
            num_train_samples=int(train_cfg.n_mc_train),
            beta=float(gmvip_cfg.get("beta", 1.0)),
            beta_warmup_steps=int(train_cfg.kl_warmup_steps),
            data_alpha=float(gmvip_cfg.get("data_alpha", 0.0)),
            max_grad_norm=train_cfg.max_grad_norm,
            output_dim=output_dim,
            operator_bank_seed=seed + 101,
        )
        return model

    if method == "oracle_prior_bank":
        return OraclePriorBankPosterior(
            task,
            bank_size=config.get("oracle_prior_bank", {}).get("bank_size"),
            seed=seed + 401,
            device=device,
            dtype=dtype,
        )

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


def predictive_function_samples(model, method: str, X: torch.Tensor, n_samples: int, seed: int) -> torch.Tensor:
    method = METHOD_ALIASES.get(method, method)
    model.eval()
    with torch.no_grad():
        if method == "map":
            return model.predict_f_samples(X, S=int(n_samples))
        if method == "mfvi":
            return model.predict_f_samples(X, int(n_samples))
        if method == "vip":
            return vip_pathwise_samples(model, X, int(n_samples))
        if method == "ftip":
            if int(n_samples) % 2:
                n_samples += 1
            return model.predict_y(X, int(n_samples))
        if method == "sip":
            return model.predict_f_samples(X, int(n_samples))
        if method in {"gmvip_empirical", "gmvip_rbf"}:
            return model.sample_posterior_values(X, int(n_samples), seed=seed)
        if method == "oracle_prior_bank":
            return model.predict_f_samples(X, int(n_samples), seed=seed)
    raise ValueError(f"Unknown method {method!r}.")


def _validation_loss(model, method: str, task, n_samples: int, seed: int) -> float:
    samples = predictive_function_samples(model, method, task.X_val, n_samples, seed)
    value = gaussian_nll_from_samples(samples, task.y_val, noise_var=task.noise_std.square())
    if not torch.isfinite(value):
        return float("inf")
    return float(value.detach().cpu())


def fit_model(model, method: str, task, config: dict, *, seed: int, device) -> dict:
    train_cfg = _training_config(config)
    if getattr(model, "is_oracle_prior_bank", False):
        return {
            "train_time_sec": 0.0,
            "steps": 0,
            "loss_start": None,
            "loss_end": None,
            "best_val_nll_norm": _validation_loss(model, method, task, min(64, int(train_cfg.n_mc_eval)), seed + 17),
        }
    dataset = TensorDataset(task.X_train, task.y_train)
    full_batch = train_cfg.batch_size == "full"
    batch_size = len(dataset) if full_batch else int(train_cfg.batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not full_batch, num_workers=0)
    if hasattr(model, "prepare_for_training"):
        model.prepare_for_training(loader)
    params = model.vi_parameters() if hasattr(model, "vi_parameters") else model.parameters()
    params = [param for param in params if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(train_cfg.learning_rate))
    best_state = copy.deepcopy(model.state_dict())
    generator_state = model.generator.get_state() if hasattr(model, "generator") else None
    best_val = _validation_loss(model, method, task, min(64, int(train_cfg.n_mc_eval)), seed - 1)
    if generator_state is not None:
        model.generator.set_state(generator_state)
    last_improvement = 0
    losses = []
    start = time.time()
    stream = iter(loader)
    disable = bool(config.get("training", {}).get("disable_tqdm", False))
    loop = tqdm(range(int(train_cfg.max_steps)), desc=f"{method} train", unit=" step", disable=disable)
    for step in loop:
        if full_batch:
            xb, yb = task.X_train, task.y_train
        else:
            try:
                xb, yb = next(stream)
            except StopIteration:
                stream = iter(loader)
                xb, yb = next(stream)
        xb = xb.to(device)
        yb = yb.to(device)
        loss = model._train_step(optimizer, xb, yb)
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step % int(train_cfg.eval_interval) == 0 or step == int(train_cfg.max_steps) - 1:
            val = _validation_loss(model, method, task, min(64, int(train_cfg.n_mc_eval)), seed + step)
            if val < best_val:
                best_val = val
                best_state = copy.deepcopy(model.state_dict())
                last_improvement = step
            if step - last_improvement >= int(train_cfg.early_stopping_patience):
                break
        loop.set_postfix(loss=f"{loss_value:.3f}", val=f"{best_val:.3f}")
    model.load_state_dict(best_state)
    return {
        "train_time_sec": time.time() - start,
        "steps": len(losses),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "best_val_nll_norm": best_val,
    }


def fit_warm_started_ftip(model: FTIP, task, config: dict, *, seed: int, device, dtype) -> dict:
    """Train a VIP source model, initialize FTIP from it, then fine-tune FTIP."""
    ftip_cfg = dict(config.get("ftip", {}))
    base_config = _ftip_base_config(config)
    vip_config = _with_training_overrides(base_config, ftip_cfg.get("warm_start_training", {}))
    vip_model = build_model("vip", task, vip_config, seed=seed, device=device, dtype=dtype)
    vip_info = fit_model(vip_model, "vip", task, vip_config, seed=seed, device=device)
    model.warm_start_from_vip(
        vip_model,
        learnable_affine=bool(ftip_cfg.get("warm_start_learnable_affine", False)),
    )
    ftip_config = _with_training_overrides(base_config, ftip_cfg.get("fine_tune_training", {}))
    ftip_info = fit_model(model, "ftip", task, ftip_config, seed=seed, device=device)
    return {
        "train_time_sec": float(vip_info["train_time_sec"]) + float(ftip_info["train_time_sec"]),
        "steps": int(vip_info["steps"]) + int(ftip_info["steps"]),
        "loss_start": ftip_info["loss_start"],
        "loss_end": ftip_info["loss_end"],
        "best_val_nll_norm": ftip_info["best_val_nll_norm"],
        "warm_start_from_vip": True,
        "vip_warm_start": vip_info,
        "ftip_fine_tune": ftip_info,
    }


def _unnormalize(task, values: torch.Tensor) -> torch.Tensor:
    y_mean = torch.as_tensor(task.metadata["y_mean"], dtype=values.dtype, device=values.device).reshape(1, 2)
    y_std = torch.as_tensor(task.metadata["y_std"], dtype=values.dtype, device=values.device).reshape(1, 2)
    return values * y_std + y_mean


def evaluate_target(model, method: str, task, config: dict, *, seed: int, out_dir: Path) -> dict:
    eval_samples = int(_training_config(config).n_mc_eval)
    start = time.time()
    samples_test_norm = predictive_function_samples(model, method, task.X_test, eval_samples, seed + 501)
    samples_plot_norm = predictive_function_samples(model, method, task.X_plot, eval_samples, seed + 601)
    samples_test = _unnormalize(task, samples_test_norm)
    samples_plot = _unnormalize(task, samples_plot_norm)
    y_test = _unnormalize(task, task.y_test)
    y_plot_true = _unnormalize(task, task.y_plot_true)
    noise_std = torch.as_tensor(task.metadata["noise_std"], dtype=samples_test.dtype, device=samples_test.device)
    coverage = interval_coverage(samples_test, y_test, levels=tuple(config.get("metrics", {}).get("levels", [0.5, 0.8, 0.9, 0.95])))
    widths = interval_width(samples_test, levels=tuple(config.get("metrics", {}).get("levels", [0.5, 0.8, 0.9, 0.95])))
    prior_ids = task.prior.sample_indices(min(512, task.prior.num_paths), seed=seed + 701)
    prior_plot = task.prior.evaluate_raw(task.X_plot, prior_ids).to(samples_plot.device)
    nearest = nearest_prior_mse(samples_plot[: min(eval_samples, 128)], prior_plot, chunk_size=32)
    t_grid = torch.as_tensor(task.metadata["t_grid"], dtype=samples_plot.dtype, device=samples_plot.device)
    residual = lotka_volterra_residual_score(samples_plot[: min(eval_samples, 64)], t_grid)
    mean_test = samples_test.mean(dim=0)
    row = {
        "experiment": "lotka_volterra",
        "method": method,
        "seed": int(seed),
        "target_id": int(task.metadata["target_id"]),
        "rmse": float(rmse(mean_test, y_test).detach().cpu()),
        "nll": float(gaussian_nll_from_samples(samples_test, y_test, noise_var=noise_std.square()).detach().cpu()),
        "crps": float(crps_from_samples(samples_test, y_test).detach().cpu()),
        "nearest_prior_mse": float(nearest["mean"].detach().cpu()),
        "nearest_prior_mse_median": float(nearest["median"].detach().cpu()),
        "ode_residual": float(residual.mean().detach().cpu()),
        "eval_time_sec": time.time() - start,
    }
    for level, value in coverage.items():
        row[f"cov{int(round(100 * level))}"] = float(value.detach().cpu())
    for level, value in widths.items():
        row[f"width{int(round(100 * level))}"] = float(value.detach().cpu())

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    train_idx = np.asarray(task.metadata["train_indices"], dtype=int)
    t_grid_np = np.asarray(task.metadata["t_grid"], dtype=np.float64)
    np.savez_compressed(
        pred_dir / f"target_{task.metadata['target_id']}.npz",
        t_plot=t_grid_np,
        y_true=y_plot_true.detach().cpu().numpy(),
        y_train_x=t_grid_np[train_idx],
        y_train=np.asarray(task.metadata["y_train_physical"], dtype=np.float64),
        samples=samples_plot.detach().cpu().numpy(),
        mean=samples_plot.mean(dim=0).detach().cpu().numpy(),
        std=samples_plot.std(dim=0, unbiased=False).detach().cpu().numpy(),
    )

    if not bool(config.get("plots", {}).get("skip", False)):
        fig_dir = out_dir / "figures"
        y_train = np.asarray(task.metadata["y_train_physical"], dtype=np.float64)
        plot_samples = samples_plot.detach().cpu().numpy()
        plot_lv_posterior_trajectory(
            fig_dir / f"posterior_trajectory_target_{task.metadata['target_id']}",
            t=t_grid_np,
            y_true=y_plot_true.detach().cpu().numpy(),
            train_t=t_grid_np[train_idx],
            train_y=y_train,
            samples=plot_samples,
            method=method,
            n_samples=int(config.get("plots", {}).get("n_posterior_samples", 10)),
        )
        plot_lv_phase_portrait(
            fig_dir / f"phase_portrait_target_{task.metadata['target_id']}",
            y_true=y_plot_true.detach().cpu().numpy(),
            train_y=y_train,
            samples=plot_samples,
            method=method,
            n_samples=int(config.get("plots", {}).get("n_posterior_samples", 10)),
        )
        plot_lv_prior_vs_posterior(
            fig_dir / f"prior_vs_posterior_target_{task.metadata['target_id']}",
            t=t_grid_np,
            prior_samples=prior_plot.detach().cpu().numpy(),
            posterior_samples=plot_samples,
            n_samples=int(config.get("plots", {}).get("n_prior_samples", 20)),
        )
    return row


def _target_ids(config: dict, cli_args, n_tasks: int) -> list[int]:
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
    numeric_keys = [
        key
        for key in rows[0]
        if key not in {"experiment", "method"} and isinstance(rows[0].get(key), (int, float))
    ] if rows else []
    summary = {}
    for key in numeric_keys:
        values = np.array([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
        if values.size:
            summary[key] = {"mean": float(np.nanmean(values)), "stderr": float(np.nanstd(values) / max(1, np.sqrt(values.size)))}
    return summary


def run_method(method: str, config: dict, cli_args) -> dict:
    method = METHOD_ALIASES.get(method, method)
    if method == "ftip":
        config = _ftip_base_config(config)
    if method == "gmvip_empirical":
        config = _gmvip_base_config(config)
    seed = int(cli_args.seed)
    dtype = torch.float64 if str(cli_args.dtype) == "float64" else torch.float32
    device = torch.device(cli_args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = dict(config.get("data", {}))
    prior_cfg = dict(config.get("prior", {}))
    load_prior_bank_size = prior_cfg.get("bank_size")
    if method == "oracle_prior_bank":
        load_prior_bank_size = config.get("oracle_prior_bank", {}).get("bank_size")
    tasks = load_lotka_volterra_tasks(
        data_cfg.get("root", "data/simprior/lotka_volterra"),
        seed=seed,
        n_eval_targets=int(data_cfg.get("n_eval_targets", 20)),
        n_train_times=int(data_cfg.get("n_train_times", 80)),
        noise_scale=float(data_cfg.get("noise_scale", 0.03)),
        prior_bank_size=load_prior_bank_size,
        device=device,
        dtype=dtype,
    )
    ids = _target_ids(config, cli_args, len(tasks))
    out_dir = Path(cli_args.output_dir or "results/simprior") / "lotka_volterra" / method / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    rows = []
    runtimes = []
    for target_idx in ids:
        task = tasks[target_idx]
        target_seed = seed + 1000 * target_idx
        model = build_model(method, task, config, seed=target_seed, device=device, dtype=dtype)
        if method == "ftip" and bool(config.get("ftip", {}).get("warm_start_from_vip", True)):
            train_info = fit_warm_started_ftip(
                model,
                task,
                config,
                seed=target_seed,
                device=device,
                dtype=dtype,
            )
        else:
            train_info = fit_model(model, method, task, config, seed=target_seed, device=device)
        row = evaluate_target(model, method, task, config, seed=target_seed, out_dir=out_dir)
        row["train_time_sec"] = float(train_info["train_time_sec"])
        row["train_steps"] = int(train_info["steps"])
        row["best_val_nll_norm"] = float(train_info["best_val_nll_norm"])
        rows.append(row)
        runtimes.append(train_info)

    metrics = {"method": method, "seed": seed, "targets": ids, "summary": _summarize(rows)}
    (out_dir / "metrics.json").write_text(json.dumps(_tensor_to_json(metrics), indent=2), encoding="utf-8")
    (out_dir / "runtime.json").write_text(json.dumps(_tensor_to_json(runtimes), indent=2), encoding="utf-8")
    _write_csv(out_dir / "metrics_per_target.csv", rows)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulator-prior regression experiments.")
    parser.add_argument("--preset", choices=tuple(CONFIG_PRESETS), default="lotka_volterra")
    parser.add_argument("--config", default=None, help="Optional YAML override. Built-in presets are used by default.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default=None)
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
        config.setdefault("sip", {})["num_inducing"] = int(args.num_inducing)
    if args.prior_bank_size is not None:
        config.setdefault("prior", {})["bank_size"] = int(args.prior_bank_size)
    if args.skip_plots:
        config.setdefault("plots", {})["skip"] = True
    if args.disable_tqdm:
        config.setdefault("training", {})["disable_tqdm"] = True
    if config.get("experiment") != "lotka_volterra":
        raise ValueError("This milestone runner only supports experiment: lotka_volterra.")

    requested = config.get("method", "gmvip_empirical")
    requested = METHOD_ALIASES.get(requested, requested)
    methods = list(METHODS) if requested == "all" else [requested]
    results = {}
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")
        results[method] = run_method(method, config, args)
    return results


if __name__ == "__main__":
    main()
