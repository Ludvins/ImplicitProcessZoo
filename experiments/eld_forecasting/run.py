from __future__ import annotations

import argparse
import copy
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

from experiments.common import (
    build_flow as build_common_flow,
)
from experiments.common import (
    deep_merge,
    fix_gaussian_noise,
    load_yaml,
    write_csv_rows,
)
from experiments.eld_forecasting.datasets import (
    load_electricity_tasks,
    load_synthetic_tasks,
    processed_exists,
)
from experiments.eld_forecasting.metrics import (
    DEFAULT_REGIONS,
    coerce_regions,
    forecast_regions,
    metrics_by_region,
)
from implicit_process_zoo.ftip import FTIP
from implicit_process_zoo.gmvip import GeneralizedMatheronVIP
from implicit_process_zoo.vip import VIP

METHODS = ("analog", "seasonal_naive", "vip", "vip_512", "ftip", "gmvip_empirical")


DEFAULT_CONFIG: dict = {
    "experiment": "eld_forecasting",
    "methodology_version": 2,
    "method": "gmvip_empirical",
    "data": {
        "root": "data/electricity_load_diagrams",
        "split": "test",
        "n_targets": 200,
        "window_length": 192,
        "prefix_length": 32,
        "noise_std_norm": 0.05,
        "min_nonzero_fraction": 0.9,
        "min_prefix_std": 1.0e-3,
    },
    "prior": {
        "bank_size": 2048,
        "selection": "calendar",
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
        "max_steps": 1500,
        "n_mc_train": 16,
        "n_mc_eval": 512,
        "batch_size": "full",
        "regression_coeffs": 256,
        "max_grad_norm": 10.0,
        "disable_tqdm": False,
    },
    "metrics": {
        "levels": [0.9, 0.95],
        "regions": DEFAULT_REGIONS,
    },
    "plots": {
        "skip": True,
    },
}


VALIDATION_CONFIG: dict = {
    **copy.deepcopy(DEFAULT_CONFIG),
    "data": {
        **copy.deepcopy(DEFAULT_CONFIG["data"]),
        "split": "validation",
        "n_targets": 100,
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
}


PILOT_CONFIG: dict = {
    **copy.deepcopy(DEFAULT_CONFIG),
    "data": {
        **copy.deepcopy(DEFAULT_CONFIG["data"]),
        "n_targets": 40,
    },
    "prior": {"bank_size": 512, "selection": "calendar"},
    "gmvip": {
        **copy.deepcopy(DEFAULT_CONFIG["gmvip"]),
        "num_inducing": 64,
    },
    "training": {
        **copy.deepcopy(DEFAULT_CONFIG["training"]),
        "max_steps": 500,
        "n_mc_eval": 256,
    },
}


CONFIG_PRESETS = {
    "eld_smoke": SMOKE_CONFIG,
    "eld_pilot": PILOT_CONFIG,
    "eld_validation": VALIDATION_CONFIG,
    "eld_paper": DEFAULT_CONFIG,
}


def _deep_update(base: dict, override: dict) -> dict:
    return deep_merge(base, override or {})


def _load_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(CONFIG_PRESETS[args.preset])
    if args.config:
        config = _deep_update(config, load_yaml(args.config))
    data_cfg = config.setdefault("data", {})
    config.setdefault("metrics", {})["regions"] = forecast_regions(
        int(data_cfg.get("window_length", 192)),
        int(data_cfg.get("prefix_length", 32)),
    )
    return config


def _training_config(config: dict) -> SimpleNamespace:
    return SimpleNamespace(**dict(config.get("training", {})))


def _fix_model_noise(model, noise_std: torch.Tensor) -> None:
    fix_gaussian_noise(model, noise_std)


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
    if method == "analog":
        return AnalogPriorPredictive(task.prior, seed=seed + 11)
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
        _fix_model_noise(model, noise_std)
        return model
    if method == "ftip":
        prior = task.prior.clone_with_normalization(num_samples=regression_coeffs, seed=seed + 25)
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
        _fix_model_noise(model, noise_std)
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
            learn_noise=False,
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
        if method in {"analog", "seasonal_naive"}:
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
        return {"train_time_sec": 0.0, "steps": 0, "loss_start": None, "loss_end": None}
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
        levels=tuple(config.get("metrics", {}).get("levels", [0.9, 0.95])),
        regions=regions,
    )

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    samples_np = samples.detach().cpu().numpy()
    np.savez_compressed(
        pred_dir / f"target_{task.metadata['target_id']}.npz",
        methodology_version=np.asarray(2, dtype=np.int64),
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
    )

    rows = []
    stress = bool(task.metadata.get("stress", False))
    for region, values in metric_rows.items():
        rows.append(
            {
                "experiment": "eld_forecasting",
                "methodology_version": 2,
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
                "seed": int(seed),
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


def _summarize(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["region"])), []).append(row)
    metrics = (
        "rmse",
        "nll",
        "crps",
        "cqm",
        "cov90",
        "cov95",
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
                    "mean": float(np.nanmean(values)),
                    "stderr": float(np.nanstd(values) / max(1.0, math.sqrt(values.size))),
                }
    return summary


def _requested_methods(value: str) -> list[str]:
    if str(value) == "all":
        return list(METHODS)
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


def _requested_target_ids(args: argparse.Namespace, n_targets: int) -> list[int]:
    if args.target_ids:
        ids = [int(value) for value in args.target_ids.split(",") if value.strip()]
    else:
        start = 0 if args.target_start is None else int(args.target_start)
        stop = n_targets if args.target_stop is None else min(int(args.target_stop), n_targets)
        ids = list(range(start, stop))
    resolved = list(dict.fromkeys(idx for idx in ids if 0 <= idx < n_targets))
    if not resolved:
        raise ValueError("Target selection did not resolve to any valid target ids.")
    return resolved


def _load_tasks(config: dict, args: argparse.Namespace, *, seed: int, device, dtype):
    data_cfg = dict(config.get("data", {}))
    prior_cfg = dict(config.get("prior", {}))
    bank_size = int(prior_cfg.get("bank_size", 512))
    n_targets = int(data_cfg.get("n_targets", 1 if args.synthetic_smoke else 200))
    requested_ids = _requested_target_ids(args, n_targets)
    if args.synthetic_smoke:
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
            "`python -m experiments.eld_forecasting.prepare --download` or pass `--raw-path`."
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
        device=device,
        dtype=dtype,
    )
    _mark_stress_tasks(tasks)
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
    dtype = torch.float64 if str(args.dtype) == "float64" else torch.float32
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = tasks or _load_tasks(config, args, seed=seed, device=device, dtype=dtype)
    ids = [int(task.metadata["target_id"]) for task in tasks]
    out_dir = Path(args.output_dir or "results/eld_forecasting_v2") / method / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    rows = []
    runtimes = []
    for task in tasks:
        target_idx = int(task.metadata["target_id"])
        target_seed = seed + 1000 * target_idx
        model = build_model(method, task, config, seed=target_seed, device=device, dtype=dtype)
        train_info = fit_model(model, task, config, device=device)
        target_rows = evaluate_target(
            model, method, task, config, seed=target_seed, out_dir=out_dir
        )
        for row in target_rows:
            row["train_time_sec"] = float(train_info["train_time_sec"])
            row["train_steps"] = int(train_info["steps"])
            row["loss_start"] = train_info["loss_start"]
            row["loss_end"] = train_info["loss_end"]
        rows.extend(target_rows)
        runtimes.append({"target_id": target_idx, **train_info})

    metrics = {"method": method, "seed": seed, "targets": ids, "summary": _summarize(rows)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "runtime.json").write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
    _write_csv(out_dir / "metrics_per_target_region.csv", rows)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ELD empirical-prior forecasting experiments.")
    parser.add_argument("--preset", choices=tuple(CONFIG_PRESETS), default="eld_smoke")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default=None)
    parser.add_argument("--split", choices=["validation", "test"], default=None)
    parser.add_argument(
        "--prior-years", default=None, help="Comma-separated empirical-prior years, e.g. 2011,2012."
    )
    parser.add_argument(
        "--target-years", default=None, help="Comma-separated held-out target years, e.g. 2014."
    )
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-mc-train", type=int, default=None)
    parser.add_argument("--n-mc-eval", type=int, default=None)
    parser.add_argument("--regression-coeffs", type=int, default=None)
    parser.add_argument("--prior-bank-size", type=int, default=None)
    parser.add_argument(
        "--prior-selection",
        choices=[
            "calendar",
            "prefix_nn",
            "calendar_prefix_nn",
            "same_client_prefix_nn",
            "same_client_calendar_prefix_nn",
            "other_client_prefix_nn",
            "other_client_calendar_prefix_nn",
            "other_client_calendar",
        ],
        default=None,
    )
    parser.add_argument("--num-inducing", type=int, default=None)
    parser.add_argument("--noise-std-norm", type=float, default=None)
    parser.add_argument("--gmvip-beta", type=float, default=None)
    parser.add_argument("--gmvip-posterior-init-log-std", type=float, default=None)
    parser.add_argument("--target-start", type=int, default=None)
    parser.add_argument("--target-stop", type=int, default=None)
    parser.add_argument("--target-ids", default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--output-dir", default="results/eld_forecasting_v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    config = _load_config(args)
    if args.method is not None:
        config["method"] = args.method
    if args.split is not None:
        config.setdefault("data", {})["split"] = args.split
    if args.prior_years is not None:
        config.setdefault("data", {})["prior_years"] = _parse_years(args.prior_years)
    if args.target_years is not None:
        config.setdefault("data", {})["target_years"] = _parse_years(args.target_years)
    if args.n_targets is not None:
        config.setdefault("data", {})["n_targets"] = int(args.n_targets)
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = int(args.max_steps)
    if args.n_mc_train is not None:
        config.setdefault("training", {})["n_mc_train"] = int(args.n_mc_train)
    if args.n_mc_eval is not None:
        config.setdefault("training", {})["n_mc_eval"] = int(args.n_mc_eval)
    if args.regression_coeffs is not None:
        config.setdefault("training", {})["regression_coeffs"] = int(args.regression_coeffs)
    if args.noise_std_norm is not None:
        config.setdefault("data", {})["noise_std_norm"] = float(args.noise_std_norm)
    if args.gmvip_beta is not None:
        config.setdefault("gmvip", {})["beta"] = float(args.gmvip_beta)
    if args.gmvip_posterior_init_log_std is not None:
        config.setdefault("gmvip", {})["posterior_init_log_std"] = float(
            args.gmvip_posterior_init_log_std
        )
    if args.prior_bank_size is not None:
        config.setdefault("prior", {})["bank_size"] = int(args.prior_bank_size)
    if args.prior_selection is not None:
        config.setdefault("prior", {})["selection"] = args.prior_selection
    if args.num_inducing is not None:
        config.setdefault("gmvip", {})["num_inducing"] = int(args.num_inducing)
    if args.disable_tqdm:
        config.setdefault("training", {})["disable_tqdm"] = True
    if config.get("experiment") != "eld_forecasting":
        raise ValueError("This runner only supports experiment: eld_forecasting.")
    methods = _requested_methods(config.get("method", "gmvip_empirical"))
    dtype = torch.float64 if str(args.dtype) == "float64" else torch.float32
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
