from __future__ import annotations

import math

import torch


DEFAULT_REGIONS = {
    "interpolation": (0.0, 8.0, True),
    "near_extrapolation": (8.0, 12.0, False),
    "medium_extrapolation": (12.0, 20.0, False),
    "far_extrapolation": (20.0, 30.0, False),
}


def coerce_regions(regions: dict | None = None) -> dict[str, tuple[float, float, bool]]:
    if regions is None:
        return dict(DEFAULT_REGIONS)
    result: dict[str, tuple[float, float, bool]] = {}
    for name, spec in regions.items():
        if isinstance(spec, dict):
            lo = spec["lo"]
            hi = spec["hi"]
            include_left = spec.get("include_left", False)
        else:
            values = list(spec)
            if len(values) not in (2, 3):
                raise ValueError(f"Region {name!r} must be (lo, hi) or (lo, hi, include_left).")
            lo, hi = values[:2]
            include_left = values[2] if len(values) == 3 else False
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            raise ValueError(f"Region {name!r} has invalid bounds ({lo}, {hi}).")
        result[str(name)] = (lo, hi, bool(include_left))
    return result


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
    samples = _as_tensor(samples)
    y_true = _as_tensor(y_true, like=samples)
    term1 = torch.mean(torch.abs(samples - y_true.unsqueeze(0)))
    term2 = 0.5 * torch.mean(torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0)))
    return term1 - term2


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


def region_masks(t, regions: dict[str, tuple[float, float, bool]] | None = None) -> dict[str, torch.Tensor]:
    t = _as_tensor(t).reshape(-1)
    regions = coerce_regions(regions)
    masks = {}
    for name, (lo, hi, include_left) in regions.items():
        if include_left:
            mask = (t >= float(lo)) & (t <= float(hi))
        else:
            mask = (t > float(lo)) & (t <= float(hi))
        masks[name] = mask
    return masks


def metrics_by_region(samples, y_true, t, noise_std, levels=(0.9, 0.95), regions: dict | None = None):
    samples = _as_tensor(samples)
    if samples.ndim == 2:
        samples = samples.unsqueeze(-1)
    y_true = _as_tensor(y_true, like=samples)
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
    t = _as_tensor(t, like=samples).reshape(-1)
    noise_var = _as_tensor(noise_std, like=samples).square()
    rows = {}
    for region, mask in region_masks(t, regions=regions).items():
        if not bool(mask.any()):
            continue
        region_samples = samples[:, mask, :]
        region_y = y_true[mask, :]
        coverage = interval_coverage(region_samples, region_y, levels=levels)
        widths = interval_width(region_samples, levels=levels)
        row = {
            "rmse": float(rmse(region_samples.mean(dim=0), region_y).detach().cpu()),
            "nlpd": float(mixture_gaussian_nlpd(region_samples, region_y, noise_var).detach().cpu()),
            "crps": float(crps_from_samples(region_samples, region_y).detach().cpu()),
        }
        for level, value in coverage.items():
            row[f"cov{int(round(100 * level))}"] = float(value.detach().cpu())
        for level, value in widths.items():
            row[f"width{int(round(100 * level))}"] = float(value.detach().cpu())
        rows[region] = row
    return rows
