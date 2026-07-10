from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .priors import HistoricalLoadWindowPrior


@dataclass(frozen=True)
class WindowSpec:
    client_idx: int
    start_idx: int
    start_time: str
    year: int
    month: int
    is_weekend: bool
    prefix_volatility: float
    prefix_ramp: float


@dataclass
class ElectricityForecastingTask:
    name: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    X_plot: torch.Tensor
    y_plot_true: torch.Tensor
    X_context_observed: torch.Tensor
    X_context_full: torch.Tensor
    noise_std: torch.Tensor
    prior: HistoricalLoadWindowPrior
    metadata: dict


@dataclass
class ElectricityData:
    values: np.ndarray
    timestamps: pd.DatetimeIndex
    clients: list[str]


def processed_exists(root: str | Path) -> bool:
    root = Path(root)
    processed = root / "processed"
    return (
        (processed / "values_float32.npy").exists()
        and (processed / "timestamps_ns.npy").exists()
        and (processed / "clients.json").exists()
    )


def load_processed(root: str | Path, *, mmap: bool = True) -> ElectricityData:
    root = Path(root)
    processed = root / "processed"
    if not processed_exists(root):
        raise FileNotFoundError(
            "Processed ELD data not found. Run "
            "`python -m experiments.eld_forecasting.prepare --root data/electricity_load_diagrams "
            "--raw-path path/to/LD2011_2014.txt` or add `--download`."
        )
    values = np.load(processed / "values_float32.npy", mmap_mode="r" if mmap else None)
    timestamps_ns = np.load(processed / "timestamps_ns.npy")
    unit = "s" if np.nanmax(np.abs(timestamps_ns)) < 10**12 else "ns"
    clients = json.loads((processed / "clients.json").read_text(encoding="utf-8"))
    return ElectricityData(
        values=values, timestamps=pd.to_datetime(timestamps_ns, unit=unit), clients=clients
    )


def _date_starts(timestamps: pd.DatetimeIndex, window_length: int) -> list[int]:
    starts = []
    seen = set()
    for idx, stamp in enumerate(timestamps):
        key = stamp.date()
        if key in seen:
            continue
        seen.add(key)
        if idx + window_length <= len(timestamps):
            starts.append(idx)
    return starts


def _window_is_active(
    window: np.ndarray, prefix_len: int, min_nonzero_fraction: float, min_prefix_std: float
) -> bool:
    if not np.isfinite(window).all():
        return False
    if float(np.count_nonzero(window)) / float(window.size) < min_nonzero_fraction:
        return False
    prefix = window[:prefix_len]
    return float(np.std(prefix)) >= min_prefix_std


def build_window_specs(
    data: ElectricityData,
    *,
    window_length: int,
    prefix_length: int,
    min_nonzero_fraction: float,
    min_prefix_std: float,
) -> list[WindowSpec]:
    starts = _date_starts(data.timestamps, window_length)
    specs: list[WindowSpec] = []
    for start_idx in starts:
        stamp = data.timestamps[start_idx]
        end_idx = start_idx + window_length
        block = np.asarray(data.values[start_idx:end_idx, :], dtype=np.float32)
        for client_idx in range(block.shape[1]):
            window = block[:, client_idx]
            if not _window_is_active(window, prefix_length, min_nonzero_fraction, min_prefix_std):
                continue
            prefix = window[:prefix_length].astype(np.float64)
            prefix_std = max(float(prefix.std()), min_prefix_std)
            specs.append(
                WindowSpec(
                    client_idx=int(client_idx),
                    start_idx=int(start_idx),
                    start_time=str(stamp),
                    year=int(stamp.year),
                    month=int(stamp.month),
                    is_weekend=bool(stamp.weekday() >= 5),
                    prefix_volatility=float(prefix.std()),
                    prefix_ramp=float(abs(prefix[-1] - prefix[0]) / prefix_std),
                )
            )
    return specs


def _normalize_window(
    window: np.ndarray, prefix_length: int, eps: float
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(window, dtype=np.float64)
    prefix = values[:prefix_length]
    mean = float(prefix.mean())
    std = max(float(prefix.std()), float(eps))
    return ((values - mean) / std).astype(np.float32), mean, std


def _sample_prior_specs(
    data: ElectricityData,
    specs: list[WindowSpec],
    target: WindowSpec,
    *,
    years: set[int],
    bank_size: int,
    seed: int,
    window_length: int,
    prefix_length: int,
    prefix_eps: float,
    target_prefix_norm: np.ndarray,
    selection: str = "calendar",
) -> tuple[list[WindowSpec], dict]:
    rng = np.random.default_rng(seed)
    selection = str(selection)
    months = {target.month, 1 + ((target.month - 2) % 12), 1 + (target.month % 12)}
    use_calendar = selection in {
        "calendar",
        "calendar_prefix_nn",
        "same_client_calendar_prefix_nn",
        "other_client_calendar",
        "other_client_calendar_prefix_nn",
    }
    use_same_client = selection in {"same_client_prefix_nn", "same_client_calendar_prefix_nn"}
    use_other_client = selection in {
        "other_client_prefix_nn",
        "other_client_calendar",
        "other_client_calendar_prefix_nn",
    }
    candidates = [
        spec
        for spec in specs
        if spec.year in years
        and (not use_calendar or (spec.month in months and spec.is_weekend == target.is_weekend))
        and (not use_same_client or spec.client_idx == target.client_idx)
        and (not use_other_client or spec.client_idx != target.client_idx)
        and not (spec.client_idx == target.client_idx and spec.start_idx == target.start_idx)
    ]
    if not candidates:
        candidates = [
            spec
            for spec in specs
            if spec.year in years
            and (not use_same_client or spec.client_idx == target.client_idx)
            and (not use_other_client or spec.client_idx != target.client_idx)
            and not (spec.client_idx == target.client_idx and spec.start_idx == target.start_idx)
        ]
    if not candidates and use_same_client:
        candidates = [
            spec
            for spec in specs
            if spec.year in years
            and not (spec.client_idx == target.client_idx and spec.start_idx == target.start_idx)
        ]
    if not candidates:
        raise ValueError(f"No historical prior windows available for target {target}.")

    diagnostics = {
        "prior_selection": selection,
        "prior_candidate_count": len(candidates),
    }
    if selection in {
        "prefix_nn",
        "calendar_prefix_nn",
        "same_client_prefix_nn",
        "same_client_calendar_prefix_nn",
        "other_client_prefix_nn",
        "other_client_calendar_prefix_nn",
    }:
        target_prefix_norm = np.asarray(target_prefix_norm[:prefix_length], dtype=np.float32)
        distances = np.empty(len(candidates), dtype=np.float64)
        for index, spec in enumerate(candidates):
            window = np.asarray(
                data.values[spec.start_idx : spec.start_idx + window_length, spec.client_idx],
                dtype=np.float64,
            )
            candidate_norm = _normalize_window(window, prefix_length, prefix_eps)[0][:prefix_length]
            distances[index] = float(np.mean(np.square(candidate_norm - target_prefix_norm)))
        order = np.argsort(distances, kind="stable")
        nearest_count = min(int(bank_size), len(order))
        nearest = [candidates[int(index)] for index in order[:nearest_count]]
        if nearest_count < int(bank_size):
            extra_indices = rng.integers(0, nearest_count, size=int(bank_size) - nearest_count)
            nearest.extend(nearest[int(index)] for index in extra_indices)
        chosen_distances = distances[order[:nearest_count]]
        diagnostics.update(
            {
                "prior_neighbor_distance_mean": float(np.mean(chosen_distances)),
                "prior_neighbor_distance_max": float(np.max(chosen_distances)),
            }
        )
        return nearest, diagnostics

    indices = rng.integers(0, len(candidates), size=int(bank_size))
    return [candidates[int(idx)] for idx in indices], diagnostics


def _make_tensors(
    y_norm: np.ndarray,
    *,
    prefix_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    window_length = int(y_norm.shape[0])
    x_plot = torch.linspace(-1.0, 1.0, window_length, dtype=dtype, device=device).unsqueeze(-1)
    y_plot = torch.as_tensor(y_norm, dtype=dtype, device=device).reshape(window_length, 1)
    return (
        x_plot[:prefix_length],
        y_plot[:prefix_length],
        x_plot[prefix_length:],
        y_plot[prefix_length:],
        x_plot,
        y_plot,
    )


def _seasonal_previous_day_norm(
    data: ElectricityData,
    target: WindowSpec,
    *,
    window_length: int,
    prefix_length: int,
    target_mean: float,
    target_std: float,
) -> np.ndarray | None:
    previous_start = target.start_idx - 96
    if previous_start < 0 or previous_start + window_length > data.values.shape[0]:
        return None
    previous = np.asarray(
        data.values[previous_start : previous_start + window_length, target.client_idx],
        dtype=np.float64,
    )
    if not np.isfinite(previous).all():
        return None
    return ((previous - target_mean) / target_std).astype(np.float32)


def make_task_from_spec(
    data: ElectricityData,
    specs: list[WindowSpec],
    target: WindowSpec,
    *,
    target_id: int,
    prior_years: set[int],
    target_years: set[int],
    split_name: str,
    bank_size: int,
    window_length: int,
    prefix_length: int,
    noise_std_norm: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    prefix_eps: float,
    prior_selection: str = "calendar",
) -> ElectricityForecastingTask:
    raw = np.asarray(
        data.values[target.start_idx : target.start_idx + window_length, target.client_idx],
        dtype=np.float64,
    )
    y_norm, y_mean, y_std = _normalize_window(raw, prefix_length, prefix_eps)
    prior_specs, prior_diagnostics = _sample_prior_specs(
        data,
        specs,
        target,
        years=prior_years,
        bank_size=bank_size,
        seed=seed,
        window_length=window_length,
        prefix_length=prefix_length,
        prefix_eps=prefix_eps,
        target_prefix_norm=y_norm[:prefix_length],
        selection=prior_selection,
    )
    prior_windows = []
    for spec in prior_specs:
        window = np.asarray(
            data.values[spec.start_idx : spec.start_idx + window_length, spec.client_idx],
            dtype=np.float64,
        )
        prior_windows.append(_normalize_window(window, prefix_length, prefix_eps)[0])
    prior = HistoricalLoadWindowPrior(
        np.stack(prior_windows, axis=0),
        num_samples=bank_size,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    x_train, y_train, x_test, y_test, x_plot, y_plot = _make_tensors(
        y_norm,
        prefix_length=prefix_length,
        device=device,
        dtype=dtype,
    )
    hours = np.arange(window_length, dtype=np.float64) * 0.25
    seasonal = _seasonal_previous_day_norm(
        data,
        target,
        window_length=window_length,
        prefix_length=prefix_length,
        target_mean=y_mean,
        target_std=y_std,
    )
    metadata = {
        "protocol": "historical_prior_online_forecasting",
        "target_id": int(target_id),
        "client_idx": int(target.client_idx),
        "client_id": data.clients[target.client_idx],
        "start_idx": int(target.start_idx),
        "start_time": target.start_time,
        "split": str(split_name),
        "target_role": "heldout_forecast_window",
        "context_role": "observed_prefix_for_online_posterior_conditioning",
        "prior_role": "historical_empirical_trajectory_pool",
        "year": int(target.year),
        "month": int(target.month),
        "is_weekend": bool(target.is_weekend),
        "prefix_volatility": float(target.prefix_volatility),
        "prefix_ramp": float(target.prefix_ramp),
        "stress_score": float(max(target.prefix_volatility, target.prefix_ramp)),
        "t_grid": hours,
        "t_obs": float(prefix_length * 0.25),
        "context_hours": float(prefix_length * 0.25),
        "context_points": int(prefix_length),
        "forecast_points": int(window_length - prefix_length),
        "forecast_hours": float((window_length - prefix_length) * 0.25),
        "window_hours": float(window_length * 0.25),
        "prefix_length": int(prefix_length),
        "window_length": int(window_length),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "sigma_y": float(noise_std_norm * y_std),
        "noise_std_norm": float(noise_std_norm),
        "y_context_physical": raw[:prefix_length].astype(np.float64),
        "y_train_physical": raw[:prefix_length].astype(np.float64),
        "seasonal_window_norm": seasonal,
        "prior_years": sorted(int(year) for year in prior_years),
        "target_years": sorted(int(year) for year in target_years),
        "prior_bank_size": int(bank_size),
        **prior_diagnostics,
    }
    return ElectricityForecastingTask(
        name=f"eld_client{target.client_idx}_start{target.start_idx}",
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        X_plot=x_plot,
        y_plot_true=y_plot,
        X_context_observed=x_train.detach().clone(),
        X_context_full=x_plot.detach().clone(),
        noise_std=torch.tensor([noise_std_norm], dtype=dtype, device=device),
        prior=prior,
        metadata=metadata,
    )


def _select_targets(
    specs: list[WindowSpec], *, years: set[int], n_targets: int, seed: int
) -> list[WindowSpec]:
    candidates = [spec for spec in specs if spec.year in years]
    if not candidates:
        raise ValueError(f"No target windows found for years {sorted(years)}.")
    rng = np.random.default_rng(seed)
    candidates = sorted(candidates, key=lambda item: (item.start_time, item.client_idx))
    if len(candidates) <= n_targets:
        selected = candidates
    else:
        selected = [
            candidates[int(idx)]
            for idx in rng.choice(len(candidates), size=n_targets, replace=False)
        ]
        selected = sorted(selected, key=lambda item: (item.start_time, item.client_idx))
    return selected


def _year_set(values: Iterable[int] | None) -> set[int] | None:
    if values is None:
        return None
    return {int(value) for value in values}


def load_electricity_tasks(
    root: str | Path,
    *,
    seed: int,
    n_targets: int,
    split: str,
    bank_size: int,
    prior_years: Iterable[int] | None = None,
    target_years: Iterable[int] | None = None,
    window_length: int = 192,
    prefix_length: int = 32,
    noise_std_norm: float = 0.05,
    min_nonzero_fraction: float = 0.9,
    min_prefix_std: float = 1.0e-3,
    prior_selection: str = "calendar",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> list[ElectricityForecastingTask]:
    device = torch.device(device or "cpu")
    data = load_processed(root)
    specs = build_window_specs(
        data,
        window_length=window_length,
        prefix_length=prefix_length,
        min_nonzero_fraction=min_nonzero_fraction,
        min_prefix_std=min_prefix_std,
    )
    explicit_prior_years = _year_set(prior_years)
    explicit_target_years = _year_set(target_years)
    if explicit_prior_years is not None and explicit_target_years is not None:
        resolved_prior_years = explicit_prior_years
        resolved_target_years = explicit_target_years
    elif split == "validation":
        resolved_target_years = {2013}
        resolved_prior_years = {2011, 2012}
    elif split == "test":
        resolved_target_years = {2014}
        resolved_prior_years = {2011, 2012, 2013}
    else:
        raise ValueError(
            "split must be 'validation' or 'test' unless both prior_years and target_years are provided."
        )
    if not resolved_prior_years.isdisjoint(resolved_target_years):
        raise ValueError(
            "prior_years and target_years must be disjoint to avoid leakage; "
            f"got prior_years={sorted(resolved_prior_years)}, target_years={sorted(resolved_target_years)}."
        )
    targets = _select_targets(specs, years=resolved_target_years, n_targets=n_targets, seed=seed)
    return [
        make_task_from_spec(
            data,
            specs,
            target,
            target_id=idx,
            prior_years=resolved_prior_years,
            target_years=resolved_target_years,
            split_name=split,
            bank_size=bank_size,
            window_length=window_length,
            prefix_length=prefix_length,
            noise_std_norm=noise_std_norm,
            seed=seed + 7919 * idx,
            device=device,
            dtype=dtype,
            prefix_eps=min_prefix_std,
            prior_selection=prior_selection,
        )
        for idx, target in enumerate(targets)
    ]


def load_synthetic_tasks(
    *,
    seed: int,
    n_targets: int,
    bank_size: int,
    window_length: int = 48,
    prefix_length: int = 8,
    noise_std_norm: float = 0.05,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> list[ElectricityForecastingTask]:
    device = torch.device(device or "cpu")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0, window_length, dtype=np.float64)

    def draw_window(local_rng: np.random.Generator) -> np.ndarray:
        base = local_rng.uniform(20.0, 80.0)
        amp = local_rng.uniform(4.0, 18.0)
        phase = local_rng.uniform(-0.4, 0.4)
        ramp = local_rng.normal(0.0, 2.0) * t
        peak = local_rng.uniform(0.0, 8.0) * np.exp(
            -0.5 * ((t - local_rng.uniform(1.0, 1.7)) / 0.12) ** 2
        )
        noise = local_rng.normal(0.0, 0.7, size=window_length)
        return base + amp * np.sin(2.0 * np.pi * (t + phase)) + ramp + peak + noise

    tasks = []
    for target_id in range(n_targets):
        target_raw = draw_window(rng)
        y_norm, y_mean, y_std = _normalize_window(target_raw, prefix_length, 1.0e-3)
        prior_windows = []
        for _ in range(bank_size):
            prior_windows.append(_normalize_window(draw_window(rng), prefix_length, 1.0e-3)[0])
        prior = HistoricalLoadWindowPrior(
            np.stack(prior_windows, axis=0),
            num_samples=bank_size,
            seed=seed + target_id,
            device=device,
            dtype=dtype,
        )
        x_train, y_train, x_test, y_test, x_plot, y_plot = _make_tensors(
            y_norm,
            prefix_length=prefix_length,
            device=device,
            dtype=dtype,
        )
        hours = np.arange(window_length, dtype=np.float64) * 0.25
        metadata = {
            "protocol": "synthetic_historical_prior_online_forecasting",
            "target_id": int(target_id),
            "client_idx": int(target_id),
            "client_id": f"synthetic_{target_id}",
            "start_idx": int(target_id * window_length),
            "start_time": f"synthetic_{target_id}",
            "split": "synthetic_smoke",
            "target_role": "heldout_forecast_window",
            "context_role": "observed_prefix_for_online_posterior_conditioning",
            "prior_role": "synthetic_empirical_trajectory_pool",
            "year": 2014,
            "month": 1,
            "is_weekend": False,
            "prefix_volatility": float(np.std(target_raw[:prefix_length])),
            "prefix_ramp": float(abs(y_norm[prefix_length - 1] - y_norm[0])),
            "stress_score": float(abs(y_norm[prefix_length - 1] - y_norm[0])),
            "t_grid": hours,
            "t_obs": float(prefix_length * 0.25),
            "context_hours": float(prefix_length * 0.25),
            "context_points": int(prefix_length),
            "forecast_points": int(window_length - prefix_length),
            "forecast_hours": float((window_length - prefix_length) * 0.25),
            "window_hours": float(window_length * 0.25),
            "prefix_length": int(prefix_length),
            "window_length": int(window_length),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "sigma_y": float(noise_std_norm * y_std),
            "noise_std_norm": float(noise_std_norm),
            "y_context_physical": target_raw[:prefix_length].astype(np.float64),
            "y_train_physical": target_raw[:prefix_length].astype(np.float64),
            "seasonal_window_norm": None,
            "prior_years": ["synthetic"],
            "target_years": ["synthetic"],
            "prior_bank_size": int(bank_size),
            "prior_selection": "synthetic",
            "prior_candidate_count": int(bank_size),
        }
        tasks.append(
            ElectricityForecastingTask(
                name=f"synthetic_eld_{target_id}",
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                X_plot=x_plot,
                y_plot_true=y_plot,
                X_context_observed=x_train.detach().clone(),
                X_context_full=x_plot.detach().clone(),
                noise_std=torch.tensor([noise_std_norm], dtype=dtype, device=device),
                prior=prior,
                metadata=metadata,
            )
        )
    return tasks
