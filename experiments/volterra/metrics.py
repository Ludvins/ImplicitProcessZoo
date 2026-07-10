from __future__ import annotations

import math

import torch


def _as_tensor(value, *, like: torch.Tensor | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if like is not None:
        tensor = tensor.to(dtype=like.dtype, device=like.device)
    return tensor


def rmse(pred_mean, y_true, dim=None):
    pred_mean = _as_tensor(pred_mean)
    y_true = _as_tensor(y_true, like=pred_mean)
    return torch.sqrt(torch.mean((pred_mean - y_true).square(), dim=dim))


def nrmse(pred_mean, y_true, scale=None):
    pred_mean = _as_tensor(pred_mean)
    y_true = _as_tensor(y_true, like=pred_mean)
    if scale is None:
        scale_t = y_true.std(unbiased=False).clamp_min(1e-8)
    else:
        scale_t = _as_tensor(scale, like=pred_mean).clamp_min(1e-8)
    return rmse(pred_mean, y_true) / scale_t


def gaussian_nll_from_samples(samples, y_true, noise_var=0.0, eps=1e-6):
    samples = _as_tensor(samples)
    y_true = _as_tensor(y_true, like=samples)
    mu = samples.mean(dim=0)
    var = samples.var(dim=0, unbiased=False) + _as_tensor(noise_var, like=samples) + float(eps)
    nll = 0.5 * (math.log(2.0 * math.pi) + torch.log(var) + (y_true - mu).square() / var)
    return nll.mean()


def crps_from_samples(samples, y_true):
    samples = _as_tensor(samples)
    y_true = _as_tensor(y_true, like=samples)
    term1 = torch.mean(torch.abs(samples - y_true.unsqueeze(0)))
    term2 = 0.5 * torch.mean(torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0)))
    return term1 - term2


def interval_coverage(samples, y_true, levels=(0.5, 0.8, 0.9, 0.95)):
    samples = _as_tensor(samples)
    y_true = _as_tensor(y_true, like=samples)
    result = {}
    for level in levels:
        alpha = 0.5 * (1.0 - float(level))
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1.0 - alpha, dim=0)
        result[float(level)] = ((y_true >= lower) & (y_true <= upper)).to(samples.dtype).mean()
    return result


def interval_width(samples, levels=(0.5, 0.8, 0.9, 0.95)):
    samples = _as_tensor(samples)
    result = {}
    for level in levels:
        alpha = 0.5 * (1.0 - float(level))
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1.0 - alpha, dim=0)
        result[float(level)] = (upper - lower).mean()
    return result


def calibration_error(samples, y_true, levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    coverage = interval_coverage(samples, y_true, levels)
    errors = [torch.abs(value - float(level)) for level, value in coverage.items()]
    return torch.stack(errors).mean()


def nearest_prior_mse(samples, prior_values, chunk_size=128):
    samples = _as_tensor(samples)
    prior_values = _as_tensor(prior_values, like=samples)
    distances = []
    nearest = []
    flat_prior = prior_values.reshape(prior_values.shape[0], -1)
    for start in range(0, samples.shape[0], int(chunk_size)):
        chunk = samples[start : start + int(chunk_size)].reshape(
            samples[start : start + int(chunk_size)].shape[0], -1
        )
        dist = torch.cdist(chunk, flat_prior).square() / float(flat_prior.shape[1])
        values, indices = dist.min(dim=1)
        distances.append(values)
        nearest.append(indices)
    return {
        "mse": torch.cat(distances, dim=0),
        "index": torch.cat(nearest, dim=0),
        "mean": torch.cat(distances, dim=0).mean(),
        "median": torch.cat(distances, dim=0).median(),
    }


def fit_lotka_volterra_theta_least_squares(sample, t_grid):
    sample = _as_tensor(sample)
    t_grid = _as_tensor(t_grid, like=sample)
    if sample.ndim != 2 or sample.shape[-1] != 2:
        raise ValueError("sample must have shape [T, 2].")
    dt = (t_grid[2:] - t_grid[:-2]).clamp_min(1e-12)
    deriv = (sample[2:] - sample[:-2]) / dt.unsqueeze(-1)
    state = sample[1:-1].clamp_min(1e-8)
    prey = state[:, 0]
    predator = state[:, 1]
    interaction = prey * predator
    A_prey = torch.stack([prey, -interaction], dim=-1)
    A_predator = torch.stack([interaction, -predator], dim=-1)
    theta_prey = torch.linalg.lstsq(A_prey, deriv[:, 0]).solution
    theta_predator = torch.linalg.lstsq(A_predator, deriv[:, 1]).solution
    return torch.stack([theta_prey[0], theta_prey[1], theta_predator[0], theta_predator[1]])


def lotka_volterra_residual_score(samples, t_grid):
    samples = _as_tensor(samples)
    t_grid = _as_tensor(t_grid, like=samples)
    if samples.ndim == 2:
        samples = samples.unsqueeze(0)
    scores = []
    for sample in samples:
        theta = fit_lotka_volterra_theta_least_squares(sample, t_grid)
        alpha, beta, delta, gamma = theta
        dt = (t_grid[2:] - t_grid[:-2]).clamp_min(1e-12)
        deriv = (sample[2:] - sample[:-2]) / dt.unsqueeze(-1)
        state = sample[1:-1].clamp_min(1e-8)
        prey = state[:, 0]
        predator = state[:, 1]
        rhs = torch.stack(
            [
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            ],
            dim=-1,
        )
        denom = deriv.square().mean().clamp_min(1e-12)
        scores.append((deriv - rhs).square().mean() / denom)
    return torch.stack(scores)
