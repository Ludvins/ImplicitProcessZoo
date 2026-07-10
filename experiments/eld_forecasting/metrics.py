from __future__ import annotations

import math

import torch

from experiments.common.metrics import empirical_crps

DAY_POINTS = 96


def forecast_regions(window_points: int, prefix_points: int) -> dict[str, dict[str, int]]:
    """Build half-open observed/forecast regions for any ELD preset."""
    window_points = int(window_points)
    prefix_points = int(prefix_points)
    if not 0 < prefix_points < window_points:
        raise ValueError("prefix_points must leave nonempty observed and forecast partitions.")
    regions = {
        "observed_prefix": {"start": 0, "stop": prefix_points},
        "full_forecast": {"start": prefix_points, "stop": window_points},
    }
    same_day_stop = min(DAY_POINTS, window_points)
    if prefix_points < same_day_stop:
        regions["same_day_forecast"] = {"start": prefix_points, "stop": same_day_stop}
    next_day_start = max(DAY_POINTS, prefix_points)
    if next_day_start < window_points:
        regions["next_day_forecast"] = {"start": next_day_start, "stop": window_points}
    validate_region_partition(
        regions,
        window_points,
        partition_names=("observed_prefix", "full_forecast"),
        cover=(0, window_points),
    )
    return regions


def validation_test_regions(
    window_points: int,
    train_points: int,
    context_points: int,
) -> dict[str, dict[str, int]]:
    """Build train/validation/test half-open regions for validation-bank runs."""
    window_points = int(window_points)
    train_points = int(train_points)
    context_points = int(context_points)
    if not 0 < train_points < context_points < window_points:
        raise ValueError("train, validation, and final-test partitions must all be nonempty.")
    regions = {
        "training_context": {"start": 0, "stop": train_points},
        "validation": {"start": train_points, "stop": context_points},
        "observed_context": {"start": 0, "stop": context_points},
        "final_test": {"start": context_points, "stop": window_points},
    }
    same_day_stop = min(DAY_POINTS, window_points)
    if context_points < same_day_stop:
        regions["same_day_test"] = {"start": context_points, "stop": same_day_stop}
    next_day_start = max(DAY_POINTS, context_points)
    if next_day_start < window_points:
        regions["next_day_test"] = {"start": next_day_start, "stop": window_points}
    validate_region_partition(
        regions,
        window_points,
        partition_names=("training_context", "validation", "final_test"),
        cover=(0, window_points),
    )
    return regions


def validate_region_partition(
    regions: dict[str, dict[str, int]],
    window_points: int,
    *,
    partition_names: tuple[str, ...],
    cover: tuple[int, int],
) -> None:
    intervals = []
    for name in partition_names:
        if name not in regions:
            raise ValueError(f"Required metric region {name!r} is missing.")
        start = int(regions[name]["start"])
        stop = int(regions[name]["stop"])
        if not 0 <= start < stop <= int(window_points):
            raise ValueError(
                f"Metric region {name!r} is empty or out of bounds: [{start}, {stop})."
            )
        intervals.append((start, stop, name))
    intervals.sort()
    if intervals[0][0] != int(cover[0]) or intervals[-1][1] != int(cover[1]):
        raise ValueError(f"Metric regions do not cover intended interval {cover}.")
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] != current[0]:
            raise ValueError(
                f"Metric regions {previous[2]!r} and {current[2]!r} overlap or leave a gap."
            )


DEFAULT_REGIONS = forecast_regions(192, 32)


def _as_tensor(value, *, like: torch.Tensor | None = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if like is not None:
        tensor = tensor.to(dtype=like.dtype, device=like.device)
    return tensor


def rmse(pred_mean, y_true):
    pred_mean = _as_tensor(pred_mean)
    y_true = _as_tensor(y_true, like=pred_mean)
    return torch.sqrt(torch.mean((pred_mean - y_true).square()))


def crps_from_samples(samples, y_true):
    return empirical_crps(samples, y_true)


def mixture_gaussian_nlpd(samples, y_true, noise_var, eps: float = 1e-12):
    samples = _as_tensor(samples)
    y_true = _as_tensor(y_true, like=samples)
    noise_var = _as_tensor(noise_var, like=samples).clamp_min(eps)
    while noise_var.ndim < samples.ndim:
        noise_var = noise_var.unsqueeze(0)
    log_probs = -0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(noise_var)
        + (y_true.unsqueeze(0) - samples).square() / noise_var
    )
    return -(torch.logsumexp(log_probs, dim=0) - math.log(samples.shape[0])).mean()


def interval_coverage(samples, y_true, levels=(0.9, 0.95)):
    samples = _as_tensor(samples)
    y_true = _as_tensor(y_true, like=samples)
    result = {}
    for level in levels:
        alpha = 0.5 * (1.0 - float(level))
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1.0 - alpha, dim=0)
        result[float(level)] = ((y_true >= lower) & (y_true <= upper)).to(samples.dtype).mean()
    return result


def interval_width(samples, levels=(0.9, 0.95)):
    samples = _as_tensor(samples)
    result = {}
    for level in levels:
        alpha = 0.5 * (1.0 - float(level))
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1.0 - alpha, dim=0)
        result[float(level)] = (upper - lower).mean()
    return result


def cqm_from_samples(samples, y_true, levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    samples = _as_tensor(samples)
    y_true = _as_tensor(y_true, like=samples)
    errors = []
    for level in levels:
        alpha = 0.5 * (1.0 - float(level))
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1.0 - alpha, dim=0)
        observed = ((y_true >= lower) & (y_true <= upper)).to(samples.dtype).mean()
        errors.append(torch.abs(observed - float(level)))
    return torch.stack(errors).mean()


def coerce_regions(regions: dict | None = None) -> dict[str, tuple[int, int]]:
    if regions is None:
        return dict(DEFAULT_REGIONS)
    result: dict[str, tuple[int, int]] = {}
    for name, spec in regions.items():
        if isinstance(spec, dict):
            start = spec["start"]
            stop = spec["stop"]
        else:
            values = list(spec)
            start, stop = values[:2]
        start = int(start)
        stop = int(stop)
        if start < 0 or stop <= start:
            raise ValueError(f"Invalid half-open region {name!r}: [{start}, {stop}).")
        result[str(name)] = (start, stop)
    return result


def region_masks(t, regions: dict | None = None):
    t = _as_tensor(t).reshape(-1)
    masks = {}
    indices = torch.arange(t.numel(), device=t.device)
    for name, (start, stop) in coerce_regions(regions).items():
        if stop > t.numel():
            raise ValueError(
                f"Metric region {name!r} stops at {stop}, beyond the {t.numel()} samples."
            )
        mask = (indices >= start) & (indices < stop)
        masks[name] = mask
    return masks


def peak_errors(samples, y_true, t):
    mean = samples.mean(dim=0)
    y_true = _as_tensor(y_true, like=mean)
    t = _as_tensor(t, like=mean).reshape(-1)
    pred_idx = int(torch.argmax(mean.reshape(mean.shape[0], -1).mean(dim=-1)).detach().cpu())
    true_idx = int(torch.argmax(y_true.reshape(y_true.shape[0], -1).mean(dim=-1)).detach().cpu())
    return {
        "peak_magnitude_error": float(
            torch.abs(mean[pred_idx] - y_true[true_idx]).mean().detach().cpu()
        ),
        "peak_timing_error_hours": float(torch.abs(t[pred_idx] - t[true_idx]).detach().cpu()),
    }


def metrics_by_region(
    samples, y_true, t, noise_std, levels=(0.9, 0.95), regions: dict | None = None
):
    samples = _as_tensor(samples)
    if samples.ndim == 2:
        samples = samples.unsqueeze(-1)
    y_true = _as_tensor(y_true, like=samples)
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
    t = _as_tensor(t, like=samples).reshape(-1)
    noise_var = _as_tensor(noise_std, like=samples).square()
    rows = {}
    coerced = coerce_regions(regions)
    for region, mask in region_masks(t, regions=regions).items():
        if not bool(mask.any()):
            continue
        region_samples = samples[:, mask, :]
        region_y = y_true[mask, :]
        region_t = t[mask]
        coverage = interval_coverage(region_samples, region_y, levels=levels)
        widths = interval_width(region_samples, levels=levels)
        row = {
            "region_start_idx": int(coerced[region][0]),
            "region_stop_idx": int(coerced[region][1]),
            "region_first_time": float(region_t[0].detach().cpu()),
            "region_last_time": float(region_t[-1].detach().cpu()),
            "rmse": float(rmse(region_samples.mean(dim=0), region_y).detach().cpu()),
            "nll": float(mixture_gaussian_nlpd(region_samples, region_y, noise_var).detach().cpu()),
            "crps": float(crps_from_samples(region_samples, region_y).detach().cpu()),
            "cqm": float(cqm_from_samples(region_samples, region_y).detach().cpu()),
            **peak_errors(region_samples, region_y, region_t),
        }
        for level, value in coverage.items():
            row[f"cov{int(round(100 * level))}"] = float(value.detach().cpu())
        for level, value in widths.items():
            row[f"width{int(round(100 * level))}"] = float(value.detach().cpu())
        rows[region] = row
    return rows
