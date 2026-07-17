"""Dynamics-aware metrics for ODE trajectory forecasts."""

from __future__ import annotations

import torch


def _as_trajectory(value, *, like: torch.Tensor | None = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if like is not None:
        tensor = tensor.to(dtype=like.dtype, device=like.device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError("trajectory values must have shape [T] or [T, D].")
    return tensor


def _posterior_mean(samples) -> torch.Tensor:
    tensor = samples if torch.is_tensor(samples) else torch.as_tensor(samples)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3 or tensor.shape[0] <= 0:
        raise ValueError("samples must have shape [S, T] or [S, T, D].")
    return tensor.mean(dim=0)


def peak_time_error(samples, target, t, *, channel: int = 0) -> torch.Tensor:
    """Absolute error between first held-out posterior-mean and true peak times."""
    prediction = _posterior_mean(samples)
    target = _as_trajectory(target, like=prediction)
    t = torch.as_tensor(t, dtype=prediction.dtype, device=prediction.device).reshape(-1)
    if prediction.shape != target.shape or prediction.shape[0] != t.numel():
        raise ValueError("samples, target, and time grid must describe the same trajectory.")
    channel = int(channel)
    if channel < 0 or channel >= prediction.shape[1]:
        raise ValueError(f"channel {channel} is outside trajectory dimension {prediction.shape[1]}.")
    predicted_peaks = _local_peak_times(prediction[:, channel], t)
    target_peaks = _local_peak_times(target[:, channel], t)
    predicted_time = (
        predicted_peaks[0]
        if predicted_peaks.numel()
        else t[torch.argmax(prediction[:, channel])]
    )
    target_time = target_peaks[0] if target_peaks.numel() else t[torch.argmax(target[:, channel])]
    return torch.abs(predicted_time - target_time)


def estimate_oscillation_period(values, t) -> torch.Tensor:
    """Estimate a dominant period from upward mean crossings, with an FFT fallback."""
    values = torch.as_tensor(values)
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional trajectory.")
    t = torch.as_tensor(t, dtype=values.dtype, device=values.device).reshape(-1)
    if values.numel() != t.numel() or values.numel() < 4:
        raise ValueError("period estimation needs matching trajectories with at least four points.")
    if not bool(torch.all(t[1:] > t[:-1])):
        raise ValueError("time points must be strictly increasing.")

    centered = values - values.mean()
    left = centered[:-1]
    right = centered[1:]
    crossing_indices = torch.nonzero((left <= 0.0) & (right > 0.0), as_tuple=False).reshape(-1)
    if crossing_indices.numel() >= 2:
        denominator = (right[crossing_indices] - left[crossing_indices]).clamp_min(
            torch.finfo(values.dtype).eps
        )
        fraction = (-left[crossing_indices] / denominator).clamp(0.0, 1.0)
        crossing_times = t[crossing_indices] + fraction * (
            t[crossing_indices + 1] - t[crossing_indices]
        )
        return torch.median(crossing_times[1:] - crossing_times[:-1])

    centered_t = t - t.mean()
    slope = (centered_t * centered).sum() / centered_t.square().sum().clamp_min(
        torch.finfo(values.dtype).eps
    )
    detrended = centered - slope * centered_t
    step = torch.median(t[1:] - t[:-1])
    frequencies = torch.fft.rfftfreq(values.numel(), d=float(step.detach().cpu())).to(
        dtype=values.dtype, device=values.device
    )
    power = torch.fft.rfft(detrended).abs().square()
    if power.numel() <= 1 or not bool(torch.any(power[1:] > 0.0)):
        return torch.full((), float("nan"), dtype=values.dtype, device=values.device)
    index = torch.argmax(power[1:]) + 1
    return frequencies[index].reciprocal()


def oscillation_period_error(samples, target, t, *, channels: tuple[int, ...] = (0,)):
    """Mean absolute dominant-period error across selected output channels."""
    prediction = _posterior_mean(samples)
    target = _as_trajectory(target, like=prediction)
    t = torch.as_tensor(t, dtype=prediction.dtype, device=prediction.device).reshape(-1)
    errors = []
    for channel in channels:
        predicted_period = estimate_oscillation_period(prediction[:, int(channel)], t)
        target_period = estimate_oscillation_period(target[:, int(channel)], t)
        errors.append(torch.abs(predicted_period - target_period))
    return torch.stack(errors).mean()


def _local_peak_times(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if values.numel() < 3:
        return t.new_empty((0,))
    middle = values[1:-1]
    mask = (middle > values[:-2]) & (middle >= values[2:])
    return t[1:-1][mask]


def _phase_lag(values: torch.Tensor, t: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
    prey_peaks = _local_peak_times(values[:, 0], t)
    predator_peaks = _local_peak_times(values[:, 1], t)
    lags = []
    for prey_peak in prey_peaks:
        candidates = predator_peaks[predator_peaks >= prey_peak]
        if candidates.numel():
            lag = candidates[0] - prey_peak
            if not torch.isfinite(period) or lag <= period:
                lags.append(lag)
    if lags:
        return torch.median(torch.stack(lags))
    lag = t[torch.argmax(values[:, 1])] - t[torch.argmax(values[:, 0])]
    if torch.isfinite(period) and period > 0.0:
        lag = torch.remainder(lag, period)
    return lag


def phase_lag_error(samples, target, t) -> torch.Tensor:
    """Absolute prey-to-predator phase-lag error for two-output trajectories."""
    prediction = _posterior_mean(samples)
    target = _as_trajectory(target, like=prediction)
    if prediction.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("phase-lag error requires exactly two trajectory channels.")
    t = torch.as_tensor(t, dtype=prediction.dtype, device=prediction.device).reshape(-1)
    predicted_period = torch.stack(
        [estimate_oscillation_period(prediction[:, channel], t) for channel in (0, 1)]
    ).mean()
    target_period = torch.stack(
        [estimate_oscillation_period(target[:, channel], t) for channel in (0, 1)]
    ).mean()
    predicted_lag = _phase_lag(prediction, t, predicted_period)
    target_lag = _phase_lag(target, t, target_period)
    return torch.abs(predicted_lag - target_lag)


def positivity_violation_rate(samples) -> torch.Tensor:
    """Fraction of posterior sample values below zero."""
    tensor = samples if torch.is_tensor(samples) else torch.as_tensor(samples)
    if tensor.ndim not in (2, 3) or tensor.numel() == 0:
        raise ValueError("samples must have shape [S, T] or [S, T, D].")
    return (tensor < 0.0).to(tensor.dtype).mean()
