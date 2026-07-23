from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.common import (
    build_flow as build_common_flow,
)
from experiments.common import (
    peak_time_error,
    phase_lag_error,
    positivity_violation_rate,
    write_csv_rows,
)
from experiments.common.metrics import empirical_crps
from implicit_process_zoo.ftip import FTIP
from implicit_process_zoo.gmvip import GeneralizedMatheronVIP
from implicit_process_zoo.map_baseline import DeterministicMAP
from implicit_process_zoo.mfvi import MFVI
from implicit_process_zoo.priors.generative_functions import BayesianNN, BayesLinear
from implicit_process_zoo.sip import SIP
from implicit_process_zoo.vip import VIP

DEFAULT_THETA_NAMES = ("alpha", "beta", "delta", "gamma", "x0", "y0")
EVALUATION_SAMPLES = 1024
LATENT_DRAW_BATCH = 512
FIXED_NOISE_NLL = "equal_weight_gaussian_mixture_with_fixed_observation_variance"
LEARNED_NOISE_NLL = "equal_weight_gaussian_mixture_with_learned_observation_variance"


class SimulatorPrior(Protocol):
    input_dim: int
    output_dim: int

    def sample_indices(self, n: int, seed: int | None = None) -> torch.Tensor: ...

    def evaluate(self, X: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor: ...

    def sample(self, X: torch.Tensor, n: int, seed: int | None = None) -> torch.Tensor: ...


@dataclass
class SimPriorTask:
    name: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    X_plot: torch.Tensor
    y_plot_true: torch.Tensor
    noise_std: torch.Tensor
    prior: SimulatorPrior
    metadata: dict


class LotkaVolterraPrior(torch.nn.Module):
    """Deterministic, live Lotka--Volterra prior using one unclipped RK4 solver."""

    input_dim = 1
    output_dim = 2
    theta_names = DEFAULT_THETA_NAMES

    def __init__(
        self,
        t: np.ndarray | torch.Tensor,
        *,
        y_mean: np.ndarray | torch.Tensor | float = 0.0,
        y_std: np.ndarray | torch.Tensor | float = 1.0,
        num_samples: int = 512,
        reference_bank_size: int = 4096,
        seed: int = 0,
        integration_dt: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        device = torch.device(device or "cpu")
        if dtype != torch.float64:
            raise ValueError("The Lotka--Volterra solver is intentionally restricted to float64.")
        t_tensor = torch.as_tensor(t, dtype=dtype, device=device)
        if t_tensor.ndim != 1 or t_tensor.numel() < 2:
            raise ValueError("t must be a one-dimensional grid with at least two values.")
        if torch.any(t_tensor[1:] <= t_tensor[:-1]):
            raise ValueError("t must be strictly increasing.")

        self.num_samples = int(num_samples)
        self.reference_bank_size = int(reference_bank_size)
        self.seed = int(seed)
        self.dtype = dtype
        self.device = device
        self.t_max = float(t_tensor[-1].detach().cpu())
        grid_dt = float((t_tensor[1:] - t_tensor[:-1]).min().detach().cpu())
        self.integration_dt = float(integration_dt or min(grid_dt, 0.05))
        if self.integration_dt <= 0.0:
            raise ValueError("integration_dt must be positive.")

        self.register_buffer("t", t_tensor)
        self.register_buffer(
            "theta_log_means",
            torch.tensor(
                [math.log(1.50), math.log(1.00), math.log(0.75), math.log(1.00)],
                dtype=dtype,
                device=device,
            ),
        )
        self.register_buffer("theta_log_stds", torch.full((4,), 0.15, dtype=dtype, device=device))
        self.register_buffer("initial_low", torch.tensor([0.8, 0.8], dtype=dtype, device=device))
        self.register_buffer("initial_high", torch.tensor([1.2, 1.2], dtype=dtype, device=device))

        y_mean_tensor = torch.as_tensor(y_mean, dtype=dtype, device=device)
        y_std_tensor = torch.as_tensor(y_std, dtype=dtype, device=device)
        if y_mean_tensor.numel() == 1:
            y_mean_tensor = y_mean_tensor.expand(2)
        if y_std_tensor.numel() == 1:
            y_std_tensor = y_std_tensor.expand(2)
        self.register_buffer("y_mean", y_mean_tensor.reshape(1, 2))
        self.register_buffer("y_std", y_std_tensor.reshape(1, 2).clamp_min(1e-8))

        self._latent_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._master_latent_cache: dict[int, torch.Tensor] = {}
        self._grid_cache_key: tuple | None = None
        self._grid_cache_values: torch.Tensor | None = None
        self._fixed_latents: torch.Tensor | None = None

    @property
    def num_paths(self) -> int:
        return int(self.reference_bank_size)

    def KL(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype, device=self.device)

    def freeze_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def clone_with_normalization(
        self,
        *,
        y_mean: np.ndarray | torch.Tensor | float,
        y_std: np.ndarray | torch.Tensor | float,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> LotkaVolterraPrior:
        return LotkaVolterraPrior(
            self.t.detach(),
            y_mean=y_mean,
            y_std=y_std,
            num_samples=self.num_samples if num_samples is None else int(num_samples),
            reference_bank_size=self.reference_bank_size,
            seed=self.seed if seed is None else int(seed),
            integration_dt=self.integration_dt,
            device=self.device,
            dtype=self.dtype,
        )

    def _generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    @staticmethod
    def _rhs(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        alpha, beta, delta, gamma = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
        prey, predator = state[:, 0], state[:, 1]
        return torch.stack(
            (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            ),
            dim=-1,
        )

    def _rk4_step(self, state: torch.Tensor, theta: torch.Tensor, dt: float) -> torch.Tensor:
        h = torch.as_tensor(dt, dtype=state.dtype, device=state.device)
        k1 = self._rhs(state, theta)
        k2 = self._rhs(state + 0.5 * h * k1, theta)
        k3 = self._rhs(state + 0.5 * h * k2, theta)
        k4 = self._rhs(state + h * k3, theta)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _integrate_sorted(self, theta: torch.Tensor, t_sorted: torch.Tensor) -> torch.Tensor:
        if t_sorted.numel() == 0:
            return torch.empty(theta.shape[0], 0, 2, dtype=self.dtype, device=self.device)
        state = theta[:, 4:6].to(dtype=self.dtype, device=self.device)
        outputs: list[torch.Tensor] = []
        current_t = 0.0
        for target_t_tensor in t_sorted:
            target_t = float(target_t_tensor.detach().cpu())
            while current_t + 1e-12 < target_t:
                step = min(self.integration_dt, target_t - current_t)
                state = self._rk4_step(state, theta, step)
                current_t += step
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    @staticmethod
    def _valid_trajectories(values: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(values).all(dim=(1, 2))
        nonnegative = (values >= 0.0).all(dim=(1, 2))
        bounded = (values <= 20.0).all(dim=(1, 2))
        return finite & nonnegative & bounded

    def sample_latents(
        self, num_samples: int, seed: int | None = None, *, cache: bool = True
    ) -> torch.Tensor:
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        resolved_seed = self.seed if seed is None else int(seed)
        cache_key = (num_samples, resolved_seed)
        if cache and cache_key in self._latent_cache:
            return self._latent_cache[cache_key]
        master = self._master_latent_cache.get(resolved_seed) if cache else None
        required = max(num_samples, LATENT_DRAW_BATCH)
        if master is None or master.shape[0] < required:
            generator = self._generator(resolved_seed)
            accepted: list[torch.Tensor] = []
            accepted_count = 0
            attempts = 0
            max_attempts = 80 * required
            while accepted_count < required and attempts < max_attempts:
                batch = LATENT_DRAW_BATCH
                attempts += batch
                log_params = self.theta_log_means.reshape(1, 4) + self.theta_log_stds.reshape(
                    1, 4
                ) * torch.randn(
                    batch,
                    4,
                    generator=generator,
                    dtype=self.dtype,
                    device=self.device,
                )
                initials = self.initial_low.reshape(1, 2) + (
                    self.initial_high - self.initial_low
                ).reshape(1, 2) * torch.rand(
                    batch,
                    2,
                    generator=generator,
                    dtype=self.dtype,
                    device=self.device,
                )
                candidates = torch.cat((torch.exp(log_params), initials), dim=-1)
                candidate_paths = self._integrate_sorted(candidates, self.t)
                valid = self._valid_trajectories(candidate_paths)
                if bool(valid.any()):
                    valid_candidates = candidates[valid]
                    accepted.append(valid_candidates)
                    accepted_count += int(valid_candidates.shape[0])
            if accepted_count < required:
                raise RuntimeError(
                    f"Only generated {accepted_count} valid trajectories after {attempts} draws."
                )
            master = torch.cat(accepted, dim=0)[:required]
            if cache:
                self._master_latent_cache[resolved_seed] = master
        latents = master[:num_samples]
        if cache:
            self._latent_cache[cache_key] = latents
        return latents

    def sample_indices(self, n: int, seed: int | None = None) -> torch.Tensor:
        return self.sample_latents(int(n), seed=seed)

    def _normalized_time_to_physical(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        if X.ndim == 1:
            x = X
        elif X.ndim == 2 and X.shape[-1] == 1:
            x = X[:, 0]
        else:
            raise ValueError("X must have shape [N] or [N, 1].")
        return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5 * self.t_max).clamp(float(self.t[0]), self.t_max)

    def _cache_key(self, theta: torch.Tensor) -> tuple:
        return (
            int(theta.data_ptr()),
            tuple(theta.shape),
            str(theta.dtype),
            str(theta.device),
            float(self.integration_dt),
        )

    def _trajectory_grid(self, theta: torch.Tensor) -> torch.Tensor:
        key = self._cache_key(theta)
        if self._grid_cache_key == key and self._grid_cache_values is not None:
            return self._grid_cache_values
        values = self._integrate_sorted(theta, self.t)
        if not bool(self._valid_trajectories(values).all()):
            raise RuntimeError("An invalid Lotka--Volterra trajectory reached evaluation.")
        self._grid_cache_key = key
        self._grid_cache_values = values
        return values

    def evaluate_raw(self, X: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        theta = latents.to(device=self.device, dtype=self.dtype)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        if theta.ndim != 2 or theta.shape[-1] != 6:
            raise ValueError("Lotka--Volterra latents must have shape [S, 6].")
        t_query = self._normalized_time_to_physical(X)
        grid = self._trajectory_grid(theta)
        idx_hi = torch.searchsorted(self.t.contiguous(), t_query.contiguous(), right=False)
        idx_hi = idx_hi.clamp(1, self.t.shape[0] - 1)
        idx_lo = idx_hi - 1
        t_lo = self.t[idx_lo]
        t_hi = self.t[idx_hi]
        weight = ((t_query - t_lo) / (t_hi - t_lo).clamp_min(1e-12)).reshape(1, -1, 1)
        return grid[:, idx_lo, :] * (1.0 - weight) + grid[:, idx_hi, :] * weight

    def evaluate(self, X: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return (self.evaluate_raw(X, latents) - self.y_mean) / self.y_std

    def evaluate_latents(self, latents: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return self.evaluate(X, latents)

    def sample(self, X: torch.Tensor, n: int, seed: int | None = None) -> torch.Tensor:
        return self.evaluate(X, self.sample_latents(int(n), seed=seed))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self._fixed_latents is None or self._fixed_latents.shape[0] != self.num_samples:
            self._fixed_latents = self.sample_latents(self.num_samples, seed=self.seed)
        return self.evaluate(X, self._fixed_latents)

    def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y.to(dtype=self.dtype, device=self.device) * self.y_std + self.y_mean


def normalize_time(t: np.ndarray | torch.Tensor, t_max: float = 30.0) -> np.ndarray:
    return 2.0 * (np.asarray(t, dtype=np.float64) / float(t_max)) - 1.0


def _select_train_indices(t: np.ndarray, n_train_times: int, seed: int) -> np.ndarray:
    pool = np.flatnonzero((t >= 0.0) & (t <= 15.0))
    first, last = int(pool[0]), int(pool[-1])
    interior = pool[(pool != first) & (pool != last)]
    needed = max(0, int(n_train_times) - 2)
    if needed > interior.size:
        raise ValueError("n_train_times exceeds the available training grid.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(interior, size=needed, replace=False) if needed else np.array([], dtype=int)
    return np.sort(np.concatenate(([first, last], chosen)).astype(int))


def generate_bank(
    n_paths: int,
    *,
    t_grid: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    prior = LotkaVolterraPrior(
        t_grid,
        num_samples=n_paths,
        reference_bank_size=max(int(n_paths), LATENT_DRAW_BATCH),
        seed=seed,
        device="cpu",
        dtype=torch.float64,
    )
    theta = prior.sample_latents(int(n_paths), seed=seed)
    X = torch.as_tensor(
        normalize_time(t_grid, float(t_grid[-1])).reshape(-1, 1),
        dtype=torch.float64,
    )
    paths = prior.evaluate_raw(X, theta)
    return paths.detach().cpu().numpy(), theta.detach().cpu().numpy()


def generate_dataset(
    root: str | Path,
    *,
    n_targets: int = 100,
    dt: float = 0.05,
    t_max: float = 30.0,
    seed: int = 0,
    reference_bank_size: int = 4096,
) -> dict[str, str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, float(t_max) + 0.5 * float(dt), float(dt), dtype=np.float64)
    paths, theta = generate_bank(
        n_targets,
        t_grid=t,
        seed=seed + 1_000_003,
    )
    target_path = root / "target_paths.npz"
    metadata_path = root / "metadata.json"
    np.savez_compressed(
        target_path,
        t=t,
        y=paths,
        theta=theta,
    )
    metadata = {
        "experiment": "lotka_volterra",
        "solver": "fixed_step_rk4_float64",
        "integration_dt": float(min(dt, 0.05)),
        "trajectory_clipping": False,
        "rejection": {"negative": True, "nonfinite": True, "maximum": 20.0},
        "dt": float(dt),
        "t_max": float(t_max),
        "n_prior": int(reference_bank_size),
        "n_targets": int(n_targets),
        "seed_targets": int(seed + 1_000_003),
        "theta_names": list(DEFAULT_THETA_NAMES),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"target_paths": str(target_path), "metadata": str(metadata_path)}


def load_lotka_volterra_tasks(
    root: str | Path,
    *,
    seed: int = 0,
    n_eval_targets: int = 20,
    n_train_times: int = 80,
    noise_scale: float = 0.03,
    prior_bank_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> list[SimPriorTask]:
    root = Path(root)
    with np.load(root / "target_paths.npz") as target_npz:
        t = target_npz["t"].astype(np.float64)
        target_y = target_npz["y"].astype(np.float64)
        target_theta = target_npz["theta"].astype(np.float64)
    metadata_path = root / "metadata.json"
    base_meta = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    t_max = float(t[-1])
    target_count = min(int(n_eval_targets), int(target_y.shape[0]))
    reference_bank_size = int(prior_bank_size or base_meta.get("n_prior", 4096))
    device = torch.device(device or "cpu")
    test_idx = np.flatnonzero((t > 20.0) & (t <= 30.0)).astype(int)
    plot_idx = np.arange(t.shape[0], dtype=int)
    X_all = normalize_time(t, t_max).reshape(-1, 1)

    tasks: list[SimPriorTask] = []
    for target_id in range(target_count):
        clean = target_y[target_id]
        train_idx = _select_train_indices(t, n_train_times, seed + 10_000 * target_id)
        y_train_clean = clean[train_idx]
        noise_std = np.maximum(
            float(noise_scale) * np.std(y_train_clean, axis=0, keepdims=True), 1e-8
        )
        rng = np.random.default_rng(seed + 20_000 + target_id)
        y_train_noisy = y_train_clean + rng.normal(0.0, noise_std, y_train_clean.shape)
        y_mean = y_train_noisy.mean(axis=0, keepdims=True)
        y_std = np.maximum(y_train_noisy.std(axis=0, keepdims=True), 1e-6)
        noise_std_norm = (noise_std / y_std).reshape(2)

        def norm_y(
            values: np.ndarray,
            mean: np.ndarray = y_mean,
            scale: np.ndarray = y_std,
        ) -> np.ndarray:
            return (values - mean) / scale

        prior = LotkaVolterraPrior(
            t,
            y_mean=y_mean,
            y_std=y_std,
            num_samples=reference_bank_size,
            reference_bank_size=reference_bank_size,
            seed=seed + 30_000 + target_id,
            device=device,
            dtype=dtype,
        )
        metadata = {
            **base_meta,
            "target_id": int(target_id),
            "theta": target_theta[target_id].tolist(),
            "t_grid": t.tolist(),
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "plot_indices": plot_idx.tolist(),
            "y_mean": y_mean.reshape(2).tolist(),
            "y_std": y_std.reshape(2).tolist(),
            "noise_std": noise_std.reshape(2).tolist(),
            "noise_std_norm": noise_std_norm.tolist(),
            "y_train_physical": y_train_noisy.tolist(),
            "y_train_clean_physical": y_train_clean.tolist(),
            "y_test_physical": clean[test_idx].tolist(),
            "y_plot_true_physical": clean.tolist(),
        }
        tasks.append(
            SimPriorTask(
                name=f"lotka_volterra_target_{target_id}",
                X_train=torch.as_tensor(X_all[train_idx], dtype=dtype, device=device),
                y_train=torch.as_tensor(norm_y(y_train_noisy), dtype=dtype, device=device),
                X_test=torch.as_tensor(X_all[test_idx], dtype=dtype, device=device),
                y_test=torch.as_tensor(norm_y(clean[test_idx]), dtype=dtype, device=device),
                X_plot=torch.as_tensor(X_all[plot_idx], dtype=dtype, device=device),
                y_plot_true=torch.as_tensor(norm_y(clean[plot_idx]), dtype=dtype, device=device),
                noise_std=torch.as_tensor(noise_std_norm, dtype=dtype, device=device),
                prior=prior,
                metadata=metadata,
            )
        )
    return tasks


def _as_tensor(value, *, like: torch.Tensor | None = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if like is not None:
        tensor = tensor.to(dtype=like.dtype, device=like.device)
    return tensor


def rmse(pred_mean, y_true, dim=None):
    pred_mean = _as_tensor(pred_mean)
    y_true = _as_tensor(y_true, like=pred_mean)
    return torch.sqrt(torch.mean((pred_mean - y_true).square(), dim=dim))


def crps_from_samples(samples, y_true):
    return empirical_crps(samples, y_true)


def mixture_gaussian_nll(samples, y_true, noise_var, eps: float = 1e-12):
    """NLL of the equally weighted observation mixture induced by function draws."""

    samples = _as_tensor(samples)
    y_true = _as_tensor(y_true, like=samples)
    if samples.ndim != y_true.ndim + 1:
        raise ValueError("samples must have shape [S, ...] relative to y_true.")
    if samples.shape[0] <= 0:
        raise ValueError("At least one predictive sample is required.")
    variance = _as_tensor(noise_var, like=samples).clamp_min(float(eps))
    while variance.ndim < samples.ndim:
        variance = variance.unsqueeze(0)
    log_components = -0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(variance)
        + (y_true.unsqueeze(0) - samples).square() / variance
    )
    log_density = torch.logsumexp(log_components, dim=0) - math.log(samples.shape[0])
    return -log_density.mean()


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
    all_distances = torch.cat(distances, dim=0)
    return {
        "mse": all_distances,
        "index": torch.cat(nearest, dim=0),
        "mean": all_distances.mean(),
        "median": all_distances.median(),
    }


def fit_lotka_volterra_theta_least_squares(sample, t_grid):
    sample = _as_tensor(sample)
    t_grid = _as_tensor(t_grid, like=sample)
    if sample.ndim != 2 or sample.shape[-1] != 2:
        raise ValueError("sample must have shape [T, 2].")
    dt = (t_grid[2:] - t_grid[:-2]).clamp_min(1e-12)
    derivative = (sample[2:] - sample[:-2]) / dt.unsqueeze(-1)
    state = sample[1:-1].clamp_min(1e-8)
    prey, predator = state[:, 0], state[:, 1]
    interaction = prey * predator
    prey_design = torch.stack((prey, -interaction), dim=-1)
    predator_design = torch.stack((interaction, -predator), dim=-1)
    prey_theta = torch.linalg.lstsq(prey_design, derivative[:, 0]).solution
    predator_theta = torch.linalg.lstsq(predator_design, derivative[:, 1]).solution
    return torch.stack((prey_theta[0], prey_theta[1], predator_theta[0], predator_theta[1]))


def lotka_volterra_residual_score(samples, t_grid):
    samples = _as_tensor(samples)
    t_grid = _as_tensor(t_grid, like=samples)
    if samples.ndim == 2:
        samples = samples.unsqueeze(0)
    scores = []
    for sample in samples:
        alpha, beta, delta, gamma = fit_lotka_volterra_theta_least_squares(sample, t_grid)
        dt = (t_grid[2:] - t_grid[:-2]).clamp_min(1e-12)
        derivative = (sample[2:] - sample[:-2]) / dt.unsqueeze(-1)
        state = sample[1:-1].clamp_min(1e-8)
        prey, predator = state[:, 0], state[:, 1]
        rhs = torch.stack(
            (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            ),
            dim=-1,
        )
        denominator = derivative.square().mean().clamp_min(1e-12)
        scores.append((derivative - rhs).square().mean() / denominator)
    return torch.stack(scores)


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: dict) -> str:
    payload = json.dumps(_tensor_to_json(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_metadata() -> dict:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    status = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


METHODS = (
    "analog_prior",
    "gmvip_surrogate_prior",
    "empirical_gp",
    "map",
    "mfvi",
    "vip",
    "ftip",
    "sip",
    "gmvip_empirical",
    "gmvip_rbf",
    "oracle_prior_bank",
)
LEARNABLE_NOISE_METHODS = {
    "map",
    "mfvi",
    "vip",
    "ftip",
    "sip",
    "gmvip_empirical",
    "gmvip_rbf",
}
METHOD_ALIASES = {
    "analog": "analog_prior",
    "prior_predictive": "analog_prior",
    "surrogate_prior": "gmvip_surrogate_prior",
    "empirical_gaussian": "empirical_gp",
    "oracle": "oracle_prior_bank",
}


DEFAULT_LOTKA_VOLTERRA_CONFIG: dict = {
    "experiment": "lotka_volterra",
    "likelihood": {
        "learn_observation_noise": True,
    },
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
    "empirical_gp": {
        "bank_size": 512,
        "jitter": 1.0e-8,
    },
    "gmvip": {
        "operator": "empirical",
        "num_inducing": 96,
        "joint_output_covariance": True,
        "prior_bank_size": 512,
        "rbf_lengthscale": 0.25,
        "jitter": 1.0e-5,
        "shrinkage": 0.02,
        "learn_kernel": False,
        "beta": 1.0,
        "training_overrides": {
            "max_steps": 800,
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
            "regression_coeffs": 20,
            "n_mc_train": 8,
        },
        "warm_start_training": {
            "n_mc_train": 4,
            "max_steps": 400,
        },
        "fine_tune_training": {
            "learning_rate": 2.0e-4,
            "max_steps": 400,
        },
    },
    "sip": {
        "num_inducing": 32,
        "num_prior_samples": 128,
        "num_train_samples": 16,
        "num_eval_samples": EVALUATION_SAMPLES,
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
        "max_steps": 800,
        "kl_warmup_steps": 200,
        "n_mc_train": 4,
        "n_mc_eval": EVALUATION_SAMPLES,
        "batch_size": "full",
        "hidden_dims": [32, 32],
        "activation": "tanh",
        "regression_coeffs": 20,
        "disable_tqdm": False,
    },
    "metrics": {
        "levels": [0.5, 0.8, 0.9, 0.95],
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
        "joint_output_covariance": True,
        "rbf_lengthscale": 0.25,
        "jitter": 1.0e-5,
        "shrinkage": 0.02,
        "learn_kernel": False,
    },
    "ftip": {
        **copy.deepcopy(DEFAULT_LOTKA_VOLTERRA_CONFIG["ftip"]),
        "training_overrides": {
            "regression_coeffs": 8,
            "n_mc_train": 2,
        },
        "warm_start_training": {
            "n_mc_train": 2,
            "max_steps": 2,
        },
        "fine_tune_training": {
            "learning_rate": 2.0e-4,
            "max_steps": 2,
        },
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
        "kl_warmup_steps": 0,
        "n_mc_train": 2,
        "n_mc_eval": 16,
        "batch_size": "full",
        "hidden_dims": [8],
        "activation": "tanh",
        "regression_coeffs": 8,
        "disable_tqdm": True,
    },
}


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
            "n_mc_train": training.get("n_mc_train", 8),
            "n_mc_eval": training.get("n_mc_eval", EVALUATION_SAMPLES),
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


def _configure_model_noise(
    model: torch.nn.Module,
    noise_std_norm: torch.Tensor,
    *,
    learn: bool,
) -> None:
    log_var = _fixed_log_variance(noise_std_norm).detach().clone()
    if hasattr(model, "log_variance"):
        param = torch.nn.Parameter(
            log_var.to(dtype=model.log_variance.dtype, device=model.log_variance.device),
            requires_grad=bool(learn),
        )
        model.log_variance = param


def _learn_observation_noise(config: dict) -> bool:
    return bool(config.get("likelihood", {}).get("learn_observation_noise", False))


def _model_noise_std_norm(
    model: torch.nn.Module,
    task,
    config: dict,
) -> torch.Tensor:
    known_noise = task.noise_std
    if not _learn_observation_noise(config):
        return known_noise
    if getattr(model, "likelihood", None) is not None and hasattr(model.likelihood, "noise_std"):
        noise_std = model.likelihood.noise_std
    elif hasattr(model, "effective_log_variance"):
        noise_std = torch.exp(0.5 * model.effective_log_variance())
    elif hasattr(model, "log_variance"):
        noise_std = torch.exp(0.5 * model.log_variance)
    else:
        raise RuntimeError(
            f"{type(model).__name__} does not expose a learnable Gaussian noise parameter."
        )
    noise_std = noise_std.reshape(-1)
    output_dim = int(task.y_train.shape[-1])
    if noise_std.numel() == 1:
        noise_std = noise_std.expand(output_dim)
    if noise_std.numel() != output_dim:
        raise RuntimeError(
            "Learned observation noise must be scalar or have one value per output; "
            f"got {noise_std.numel()} values for {output_dim} outputs."
        )
    if not torch.isfinite(noise_std).all() or torch.any(noise_std <= 0):
        raise RuntimeError(f"Invalid learned observation noise: {noise_std}.")
    return noise_std


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
    flow_type = str(ftip_cfg.get("flow_type", "affine")).lower()
    if flow_type in {"spline_1x1", "spline-1x1", "glow"}:
        flow_type = "spline_1x1"
    return build_common_flow(
        flow_type,
        depth=int(ftip_cfg.get("flow_depth", 2)),
        input_dim=input_dim,
        device=device,
        dtype=dtype,
        seed=seed,
        num_bins=int(ftip_cfg.get("flow_num_bins", 8)),
        domain=float(ftip_cfg.get("flow_domain", 5.0)),
    )


class OraclePriorBankPosterior(torch.nn.Module):
    """Discrete posterior over a Monte Carlo bank sampled from the ODE prior."""

    def __init__(self, task, *, bank_size: int | None, seed: int, device, dtype):
        super().__init__()
        self.is_oracle_prior_bank = True
        self.prior = task.prior
        self.seed = int(seed)
        bank_size = task.prior.num_paths if bank_size is None else int(bank_size)
        bank_latents = task.prior.sample_latents(bank_size, seed=seed).to(
            device=device, dtype=dtype
        )
        self.register_buffer("bank_latents", bank_latents)

        with torch.no_grad():
            X_train = task.X_train.to(device=device, dtype=dtype)
            y_train = task.y_train.to(device=device, dtype=dtype)
            noise_var = (
                task.noise_std.to(device=device, dtype=dtype)
                .square()
                .clamp_min(1e-12)
                .reshape(1, 1, -1)
            )
            bank_train = self.prior.evaluate(X_train, self.bank_latents)
            log_lik = -0.5 * (
                (bank_train - y_train.unsqueeze(0)).square() / noise_var
                + torch.log(2.0 * math.pi * noise_var)
            ).sum(dim=(1, 2))
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
        draw_idx = torch.multinomial(
            self.weights, int(n_samples), replacement=True, generator=generator
        )
        return self.prior.evaluate(X, self.bank_latents[draw_idx])


class AnalogPriorPredictive(torch.nn.Module):
    """Unconditional trajectory draws from the normalized ODE prior."""

    is_fixed_predictive = True

    def __init__(self, prior: LotkaVolterraPrior, *, seed: int):
        super().__init__()
        self.prior = prior
        self.seed = int(seed)

    def predict_f_samples(
        self, X: torch.Tensor, num_samples: int, *, seed: int | None = None
    ) -> torch.Tensor:
        return self.prior.sample(
            X,
            int(num_samples),
            seed=self.seed if seed is None else int(seed),
        )


class EmpiricalGPPredictive(torch.nn.Module):
    """Joint finite-grid empirical GP conditioned on both observed species."""

    is_fixed_predictive = True

    def __init__(
        self,
        task,
        *,
        bank_size: int,
        seed: int,
        jitter: float = 1.0e-8,
    ):
        super().__init__()
        bank_size = int(bank_size)
        if bank_size < 2:
            raise ValueError("The empirical GP needs at least two prior trajectories.")

        with torch.no_grad():
            latents = task.prior.sample_latents(bank_size, seed=int(seed))
            bank = task.prior.evaluate(task.X_plot, latents)
            if bank.ndim != 3 or bank.shape[-1] != 2:
                raise ValueError("Empirical GP prior values must have shape [B, T, 2].")

            grid_size = int(bank.shape[1])
            flat_bank = bank.reshape(bank_size, 2 * grid_size)
            mean = flat_bank.mean(dim=0)
            features = (flat_bank - mean).T / math.sqrt(float(bank_size - 1))

            train_idx = torch.as_tensor(
                task.metadata["train_indices"], dtype=torch.long, device=bank.device
            )
            output_offsets = torch.arange(2, dtype=torch.long, device=bank.device)
            observed_idx = (2 * train_idx.unsqueeze(-1) + output_offsets).reshape(-1)
            observed_features = features[observed_idx]
            observed_mean = mean[observed_idx]
            observed_targets = task.y_train.to(dtype=bank.dtype, device=bank.device).reshape(-1)
            observed_noise = (
                task.noise_std.to(dtype=bank.dtype, device=bank.device)
                .reshape(1, 2)
                .expand(train_idx.numel(), 2)
                .reshape(-1)
            )

            observed_covariance = observed_features @ observed_features.T
            marginal_scale = (
                features.square().sum(dim=1).mean().clamp_min(torch.finfo(bank.dtype).eps)
            )
            numerical_jitter = max(float(jitter), float(torch.finfo(bank.dtype).eps))
            observed_system = observed_covariance + torch.diag(
                observed_noise.square() + numerical_jitter * marginal_scale
            )
            observed_cholesky = torch.linalg.cholesky(observed_system)
            cross_covariance = features @ observed_features.T

        self.grid_size = grid_size
        self.bank_size = bank_size
        self.register_buffer("mean", mean)
        self.register_buffer("features", features)
        self.register_buffer("observed_indices", observed_idx)
        self.register_buffer("observed_mean", observed_mean)
        self.register_buffer("observed_features", observed_features)
        self.register_buffer("observed_targets", observed_targets)
        self.register_buffer("observed_noise", observed_noise)
        self.register_buffer("observed_cholesky", observed_cholesky)
        self.register_buffer("cross_covariance", cross_covariance)

    def _query_indices(self, X: torch.Tensor) -> torch.Tensor:
        positions = (
            (X[:, 0].to(dtype=self.mean.dtype, device=self.mean.device).clamp(-1.0, 1.0) + 1.0)
            * 0.5
            * float(self.grid_size - 1)
        )
        time_idx = positions.round().long().clamp(0, self.grid_size - 1)
        offsets = torch.arange(2, dtype=torch.long, device=self.mean.device)
        return (2 * time_idx.unsqueeze(-1) + offsets).reshape(-1)

    def predict_f_samples(
        self, X: torch.Tensor, num_samples: int, *, seed: int | None = None
    ) -> torch.Tensor:
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        generator = torch.Generator(device=self.mean.device)
        generator.manual_seed(0 if seed is None else int(seed))
        latent_noise = torch.randn(
            num_samples,
            self.bank_size,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        observation_noise = (
            torch.randn(
                num_samples,
                self.observed_targets.numel(),
                dtype=self.mean.dtype,
                device=self.mean.device,
                generator=generator,
            )
            * self.observed_noise
        )

        query_idx = self._query_indices(X)
        query_features = self.features[query_idx]
        prior_query = self.mean[query_idx] + latent_noise @ query_features.T
        prior_observed = self.observed_mean + latent_noise @ self.observed_features.T
        residual = self.observed_targets - prior_observed - observation_noise
        solved = torch.cholesky_solve(residual.T, self.observed_cholesky).T
        correction = solved @ self.cross_covariance[query_idx].T
        return (prior_query + correction).reshape(num_samples, X.shape[0], 2)


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
        self.generator = torch.Generator(device=base_prior.device)
        self.generator.manual_seed(self.seed)

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
            seed = int(
                torch.randint(
                    0,
                    torch.iinfo(torch.int32).max,
                    (),
                    generator=self.generator,
                    dtype=torch.int64,
                    device=self.device,
                ).item()
            )
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


def build_model(method: str, task, config: dict, *, seed: int, device, dtype):
    method = METHOD_ALIASES.get(method, method)
    if method == "ftip":
        config = _ftip_base_config(config)
    if method in {"gmvip_empirical", "gmvip_surrogate_prior"}:
        config = _gmvip_base_config(config)
    train_cfg = _training_config(config)
    output_dim = int(task.y_train.shape[-1])
    noise_std_norm = task.noise_std.to(dtype=dtype, device=device)
    learn_noise = _learn_observation_noise(config)
    if method == "analog_prior":
        return AnalogPriorPredictive(task.prior, seed=seed + 11)

    if method == "empirical_gp":
        empirical_cfg = dict(config.get("empirical_gp", {}))
        return EmpiricalGPPredictive(
            task,
            bank_size=int(empirical_cfg.get("bank_size", 512)),
            seed=seed + 17,
            jitter=float(empirical_cfg.get("jitter", 1.0e-8)),
        )

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
        model.log_variance.requires_grad_(learn_noise)
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
        _configure_model_noise(model, noise_std_norm, learn=learn_noise)
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
        _configure_model_noise(model, noise_std_norm, learn=learn_noise)
        return model

    if method == "ftip":
        prior = task.prior.clone_with_normalization(
            y_mean=np.asarray(task.metadata["y_mean"], dtype=np.float64).reshape(1, output_dim),
            y_std=np.asarray(task.metadata["y_std"], dtype=np.float64).reshape(1, output_dim),
            num_samples=int(train_cfg.regression_coeffs),
            # Match the VIP warm-start basis exactly.  FTIP transfers VIP's
            # coefficient law, so changing the basis would invalidate it.
            seed=seed + 21,
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
        _configure_model_noise(model, noise_std_norm, learn=learn_noise)
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
            inducing_inputs=_inducing_grid(
                int(sip_cfg.get("num_inducing", 32)), device=device, dtype=dtype
            ),
            output_dim=output_dim,
            likelihood="regression",
            num_data=int(task.X_train.shape[0]),
            num_prior_samples=num_prior_samples,
            num_train_samples=sip_cfg.get("num_train_samples"),
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
            min_log_variance=sip_cfg.get("min_log_variance"),
            device=device,
            dtype=dtype,
            seed=seed + 31,
        )
        _configure_model_noise(model, noise_std_norm, learn=learn_noise)
        return model

    if method in {"gmvip_empirical", "gmvip_surrogate_prior", "gmvip_rbf"}:
        gmvip_cfg = dict(config.get("gmvip", {}))
        operator = "rbf" if method == "gmvip_rbf" else "empirical"
        num_inducing = int(gmvip_cfg.get("num_inducing", 32))
        bank_size = int(
            gmvip_cfg.get(
                "prior_bank_size",
                config.get("prior", {}).get("bank_size", 512),
            )
        )
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
            learn_noise=learn_noise,
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
            joint_output_covariance=bool(gmvip_cfg.get("joint_output_covariance", False)),
            operator_bank_seed=seed + 101,
        )
        if int(model.operator.num_bank_samples) != bank_size:
            raise RuntimeError(
                f"GMVIP operator-bank construction did not honor the resolved size B={bank_size}."
            )
        if method == "gmvip_surrogate_prior":
            # Leave the freshly initialized coefficient law at q(a) = N(0, I)
            # and bypass optimization. This isolates the predictive effect of
            # the finite-rank GMVIP surrogate from posterior adaptation.
            model.is_fixed_predictive = True
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


def _capture_sampling_state(model) -> dict:
    generators = []
    counters = []
    seen = set()
    for module in model.modules():
        generator = getattr(module, "generator", None)
        if isinstance(generator, torch.Generator) and id(generator) not in seen:
            seen.add(id(generator))
            generators.append((generator, generator.get_state()))
        counter = getattr(module, "_sample_counter", None)
        if torch.is_tensor(counter):
            counters.append((counter, counter.detach().clone()))
    return {
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generators": generators,
        "counters": counters,
    }


def _restore_sampling_state(state: dict) -> None:
    torch.random.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    for generator, generator_state in state["generators"]:
        generator.set_state(generator_state)
    for counter, counter_state in state["counters"]:
        counter.copy_(counter_state)


def predictive_function_samples(
    model, method: str, X: torch.Tensor, n_samples: int, seed: int
) -> torch.Tensor:
    method = METHOD_ALIASES.get(method, method)
    model.eval()
    state = _capture_sampling_state(model)
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        for index, (generator, _) in enumerate(state["generators"]):
            generator.manual_seed(int(seed) + index)
        with torch.no_grad():
            if method == "map":
                return model.predict_f_samples(X, num_samples=int(n_samples))
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
            if method in {"analog_prior", "empirical_gp"}:
                return model.predict_f_samples(X, int(n_samples), seed=seed)
            if method in {"gmvip_empirical", "gmvip_surrogate_prior", "gmvip_rbf"}:
                return model.sample_posterior_values(X, int(n_samples), seed=seed)
            if method == "oracle_prior_bank":
                return model.predict_f_samples(X, int(n_samples), seed=seed)
        raise ValueError(f"Unknown method {method!r}.")
    finally:
        _restore_sampling_state(state)


def fit_model(model, method: str, task, config: dict, *, device) -> dict:
    train_cfg = _training_config(config)
    if getattr(model, "is_fixed_predictive", False) or getattr(
        model, "is_oracle_prior_bank", False
    ):
        return {
            "train_time_sec": 0.0,
            "steps": 0,
            "loss_start": None,
            "loss_end": None,
            "checkpoint": "fixed_predictive",
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
    losses = []
    start = time.time()
    stream = iter(loader)
    disable = bool(config.get("training", {}).get("disable_tqdm", False))
    loop = tqdm(
        range(int(train_cfg.max_steps)), desc=f"{method} train", unit=" step", disable=disable
    )
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
        if not math.isfinite(loss_value):
            raise RuntimeError(f"{method} produced a non-finite training loss at step {step}.")
        losses.append(loss_value)
        loop.set_postfix(loss=f"{loss_value:.3f}")
    return {
        "train_time_sec": time.time() - start,
        "steps": len(losses),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "checkpoint": "final_step",
    }


def fit_warm_started_ftip(model: FTIP, task, config: dict, *, seed: int, device, dtype) -> dict:
    """Train a VIP source model, initialize FTIP from it, then fine-tune FTIP."""
    ftip_cfg = dict(config.get("ftip", {}))
    ftip_overrides = dict(ftip_cfg.get("training_overrides", {}))
    vip_config = _with_training_overrides(
        config,
        {
            "regression_coeffs": ftip_overrides.get(
                "regression_coeffs",
                config.get("training", {}).get("regression_coeffs", 20),
            ),
            **dict(ftip_cfg.get("warm_start_training", {})),
        },
    )
    vip_model = build_model("vip", task, vip_config, seed=seed, device=device, dtype=dtype)
    vip_info = fit_model(vip_model, "vip", task, vip_config, device=device)
    model.warm_start_from_vip(
        vip_model,
        learnable_affine=bool(ftip_cfg.get("warm_start_learnable_affine", False)),
    )
    ftip_config = _with_training_overrides(
        _ftip_base_config(config), ftip_cfg.get("fine_tune_training", {})
    )
    ftip_info = fit_model(model, "ftip", task, ftip_config, device=device)
    return {
        "train_time_sec": float(vip_info["train_time_sec"]) + float(ftip_info["train_time_sec"]),
        "steps": int(vip_info["steps"]) + int(ftip_info["steps"]),
        "loss_start": ftip_info["loss_start"],
        "loss_end": ftip_info["loss_end"],
        "checkpoint": "final_step",
        "warm_start_from_vip": True,
        "vip_warm_start": vip_info,
        "ftip_fine_tune": ftip_info,
    }


def _unnormalize(task, values: torch.Tensor) -> torch.Tensor:
    y_mean = torch.as_tensor(
        task.metadata["y_mean"], dtype=values.dtype, device=values.device
    ).reshape(1, 2)
    y_std = torch.as_tensor(
        task.metadata["y_std"], dtype=values.dtype, device=values.device
    ).reshape(1, 2)
    return values * y_std + y_mean


def evaluate_target(model, method: str, task, config: dict, *, seed: int, out_dir: Path) -> dict:
    eval_samples = int(_training_config(config).n_mc_eval)
    evaluation_seed = int(seed) + 501
    start = time.time()
    samples_plot_norm = predictive_function_samples(
        model, method, task.X_plot, eval_samples, evaluation_seed
    )
    samples_plot = _unnormalize(task, samples_plot_norm)
    y_plot_true = _unnormalize(task, task.y_plot_true)
    prior_ids = task.prior.sample_indices(min(512, task.prior.num_paths), seed=seed + 701)
    prior_plot = task.prior.evaluate_raw(task.X_plot, prior_ids).to(samples_plot.device)
    t_grid = torch.as_tensor(
        task.metadata["t_grid"], dtype=samples_plot.dtype, device=samples_plot.device
    )
    test_idx = torch.as_tensor(
        task.metadata["test_indices"], dtype=torch.long, device=t_grid.device
    )
    train_idx = np.asarray(task.metadata["train_indices"], dtype=int)
    test_samples_plot = samples_plot[:, test_idx]
    test_truth_plot = y_plot_true[test_idx]
    test_t = t_grid[test_idx]
    noise_std_norm = _model_noise_std_norm(model, task, config).to(
        dtype=test_samples_plot.dtype,
        device=test_samples_plot.device,
    )
    physical_scale = torch.as_tensor(
        task.metadata["y_std"],
        dtype=test_samples_plot.dtype,
        device=test_samples_plot.device,
    ).reshape(-1)
    noise_std = noise_std_norm * physical_scale
    levels = tuple(config.get("metrics", {}).get("levels", [0.5, 0.8, 0.9, 0.95]))
    coverage = interval_coverage(test_samples_plot, test_truth_plot, levels=levels)
    widths = interval_width(test_samples_plot, levels=levels)
    prior_test = prior_plot[:, test_idx]
    nearest = nearest_prior_mse(
        test_samples_plot[: min(eval_samples, 128)], prior_test, chunk_size=32
    )
    residual = lotka_volterra_residual_score(test_samples_plot[: min(eval_samples, 64)], test_t)
    mean_test = test_samples_plot.mean(dim=0)
    row = {
        "experiment": "lotka_volterra",
        "method": method,
        "metric_partition": "test_(20,30]",
        "seed": int(seed),
        "target_id": int(task.metadata["target_id"]),
        "evaluation_seed": evaluation_seed,
        "evaluation_samples": eval_samples,
        "observation_noise_std_norm_prey": float(noise_std_norm[0].detach().cpu()),
        "observation_noise_std_norm_predator": float(noise_std_norm[1].detach().cpu()),
        "observation_noise_std_prey": float(noise_std[0].detach().cpu()),
        "observation_noise_std_predator": float(noise_std[1].detach().cpu()),
        "rmse": float(rmse(mean_test, test_truth_plot).detach().cpu()),
        "nll": float(
            mixture_gaussian_nll(
                test_samples_plot,
                test_truth_plot,
                noise_var=noise_std.square(),
            )
            .detach()
            .cpu()
        ),
        "crps": float(crps_from_samples(test_samples_plot, test_truth_plot).detach().cpu()),
        "nearest_prior_mse": float(nearest["mean"].detach().cpu()),
        "nearest_prior_mse_median": float(nearest["median"].detach().cpu()),
        "ode_residual": float(residual.mean().detach().cpu()),
        "prey_peak_time_error": float(
            peak_time_error(test_samples_plot, test_truth_plot, test_t, channel=0).detach().cpu()
        ),
        "predator_peak_time_error": float(
            peak_time_error(test_samples_plot, test_truth_plot, test_t, channel=1).detach().cpu()
        ),
        "prey_predator_phase_lag_error": float(
            phase_lag_error(test_samples_plot, test_truth_plot, test_t).detach().cpu()
        ),
        "positivity_violation_rate": float(
            positivity_violation_rate(test_samples_plot).detach().cpu()
        ),
        "eval_time_sec": time.time() - start,
    }
    for level, value in coverage.items():
        row[f"cov{int(round(100 * level))}"] = float(value.detach().cpu())
    for level, value in widths.items():
        row[f"width{int(round(100 * level))}"] = float(value.detach().cpu())

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
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
        observation_noise_std=noise_std.detach().cpu().numpy(),
        observation_noise_std_norm=noise_std_norm.detach().cpu().numpy(),
        evaluation_seed=np.asarray(evaluation_seed, dtype=np.int64),
        evaluation_samples=np.asarray(eval_samples, dtype=np.int64),
    )
    return row


def _parse_target_ids(spec: str, n_tasks: int) -> list[int]:
    ids: list[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid target range {token!r}.")
            start = int(parts[0] or 0)
            stop = int(parts[1] or n_tasks)
            ids.extend(range(start, stop))
        else:
            ids.append(int(token))
    ids = list(dict.fromkeys(ids))
    invalid = [target_id for target_id in ids if not 0 <= target_id < n_tasks]
    if invalid:
        raise ValueError(f"Target IDs out of range 0..{n_tasks - 1}: {invalid}")
    if not ids:
        raise ValueError("No target IDs were selected.")
    return ids


def _write_csv(path: Path, rows: list[dict]) -> None:
    write_csv_rows(path, rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    integer_keys = {"seed", "target_id", "evaluation_seed", "evaluation_samples", "train_steps"}
    string_keys = {"experiment", "method", "metric_partition"}
    for row in rows:
        item = {}
        for key, value in row.items():
            if key in string_keys:
                item[key] = value
            elif value in {"", None}:
                item[key] = None
            elif key in integer_keys:
                item[key] = int(float(value))
            else:
                try:
                    item[key] = float(value)
                except ValueError:
                    item[key] = value
        converted.append(item)
    return converted


def _summarize(rows: list[dict]) -> dict:
    identifiers = {
        "experiment",
        "method",
        "metric_partition",
        "seed",
        "target_id",
        "evaluation_seed",
        "evaluation_samples",
    }
    numeric_keys = (
        [
            key
            for key in rows[0]
            if key not in identifiers and isinstance(rows[0].get(key), (int, float))
        ]
        if rows
        else []
    )
    summary = {}
    for key in numeric_keys:
        values = np.array([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
        if values.size:
            finite = values[np.isfinite(values)]
            stderr = (
                float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0
            )
            summary[key] = {
                "mean": float(np.nanmean(values)),
                "stderr": stderr,
                "count": int(finite.size),
            }
    return summary


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_tensor_to_json(value), indent=2), encoding="utf-8")


def _method_manifest(
    *,
    method: str,
    config: dict,
    seed: int,
    basis_size: int,
    dataset_path: Path,
) -> dict:
    learn_noise = _learn_observation_noise(config)
    protocol = {
        "schema_version": 2,
        "experiment": "lotka_volterra",
        "method": method,
        "seed": int(seed),
        "vip_basis_size": int(basis_size),
        "nll": LEARNED_NOISE_NLL if learn_noise else FIXED_NOISE_NLL,
        "observation_noise": {
            "mode": "learned_per_output" if learn_noise else "fixed_per_output",
            "initialization": "known_simulation_noise",
            "reported_units": "physical_output_units",
        },
        "checkpoint_selection": "none_final_step_only",
        "data_usage": {
            "training": "t<=15",
            "unused_gap": "15<t<=20",
            "test": "20<t<=30",
        },
        "config": config,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _sha256_file(dataset_path),
        },
    }
    return {
        **protocol,
        "protocol_hash": _stable_hash(protocol),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": None,
        },
        "git": _git_metadata(),
        "status": "running",
        "completed_targets": [],
    }


def run_method(method: str, config: dict, cli_args) -> dict:
    method = METHOD_ALIASES.get(method, method)
    requested_learn_noise = _learn_observation_noise(config)
    effective_learn_noise = requested_learn_noise and method in LEARNABLE_NOISE_METHODS
    if effective_learn_noise != requested_learn_noise:
        config = copy.deepcopy(config)
        config.setdefault("likelihood", {})["learn_observation_noise"] = effective_learn_noise
    seed = int(cli_args.seed)
    dtype = torch.float64
    device = torch.device(cli_args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = dict(config.get("data", {}))
    prior_cfg = dict(config.get("prior", {}))
    load_prior_bank_size = prior_cfg.get("bank_size")
    if method in {"gmvip_empirical", "gmvip_surrogate_prior"}:
        load_prior_bank_size = config.get("gmvip", {}).get("prior_bank_size", 512)
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
    ids = _parse_target_ids(cli_args.target_ids, len(tasks))
    out_dir = (
        Path(cli_args.output_root) / f"seed_{seed}" / f"S_{int(cli_args.vip_basis_size)}" / method
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    expected_manifest = _method_manifest(
        method=method,
        config=config,
        seed=seed,
        basis_size=int(cli_args.vip_basis_size),
        dataset_path=Path(data_cfg["root"]) / "target_paths.npz",
    )
    expected_manifest["environment"]["device"] = str(device)
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("protocol_hash") != expected_manifest["protocol_hash"]:
            raise RuntimeError(
                f"{out_dir} contains a different protocol; choose a clean output root."
            )
        if not cli_args.resume:
            raise RuntimeError(f"{out_dir} already exists; pass --resume to continue it.")
        manifest = existing_manifest
    else:
        manifest = expected_manifest
        _write_json(manifest_path, manifest)

    rows = _read_csv(out_dir / "metrics_per_target.csv") if cli_args.resume else []
    rows_by_target = {int(row["target_id"]): row for row in rows}
    runtime_path = out_dir / "runtime.json"
    runtimes = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if cli_args.resume and runtime_path.exists()
        else []
    )
    for target_idx in ids:
        prediction_path = out_dir / "predictions" / f"target_{target_idx}.npz"
        checkpoint_path = out_dir / "checkpoints" / f"target_{target_idx}.pt"
        if (
            cli_args.resume
            and target_idx in rows_by_target
            and prediction_path.exists()
            and checkpoint_path.exists()
        ):
            continue
        task = tasks[target_idx]
        target_seed = seed + 1000 * target_idx
        seed_everything(target_seed)
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
            train_info = fit_model(model, method, task, config, device=device)
        row = evaluate_target(model, method, task, config, seed=target_seed, out_dir=out_dir)
        row["train_time_sec"] = float(train_info["train_time_sec"])
        row["train_steps"] = int(train_info["steps"])
        rows_by_target[target_idx] = row
        runtime_row = {"target_id": target_idx, **train_info}
        runtimes = [
            existing for existing in runtimes if int(existing.get("target_id", -1)) != target_idx
        ]
        runtimes.append(runtime_row)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": method,
                "seed": target_seed,
                "target_id": target_idx,
                "protocol_hash": manifest["protocol_hash"],
                "model_state_dict": model.state_dict(),
                "training": train_info,
            },
            checkpoint_path,
        )
        rows = [rows_by_target[key] for key in sorted(rows_by_target)]
        _write_csv(out_dir / "metrics_per_target.csv", rows)
        _write_json(runtime_path, sorted(runtimes, key=lambda item: int(item["target_id"])))
        manifest["completed_targets"] = sorted(rows_by_target)
        _write_json(manifest_path, manifest)

    rows = [rows_by_target[key] for key in sorted(rows_by_target)]
    metrics = {"method": method, "seed": seed, "targets": ids, "summary": _summarize(rows)}
    _write_json(out_dir / "metrics.json", metrics)
    _write_csv(out_dir / "metrics_per_target.csv", rows)
    manifest["status"] = "complete" if set(ids).issubset(rows_by_target) else "partial"
    manifest["completed_targets"] = sorted(rows_by_target)
    _write_json(manifest_path, manifest)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Lotka--Volterra benchmark.")
    parser.add_argument(
        "--methods",
        required=True,
        help=f"Comma-separated method names. Choices: {','.join(METHODS)}",
    )
    parser.add_argument("--vip-basis-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inducing", type=int, default=None)
    parser.add_argument("--prior-bank-size", type=int, default=None)
    parser.add_argument(
        "--learn-observation-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Learn one Gaussian observation-noise scale per output, initialized "
            "at the simulator noise (default). Use --no-learn-observation-noise "
            "for the fixed-noise sensitivity."
        ),
    )
    parser.add_argument("--target-ids", default="0:20")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--output-root", default="results/volterra")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--regenerate-targets", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if int(args.vip_basis_size) <= 1:
        raise ValueError("--vip-basis-size must be greater than one.")
    seed_everything(args.seed)
    config = copy.deepcopy(
        SMOKE_LOTKA_VOLTERRA_CONFIG if args.smoke else DEFAULT_LOTKA_VOLTERRA_CONFIG
    )
    data_root = Path(args.data_root or config["data"]["root"])
    config["data"]["root"] = str(data_root)
    config["training"]["regression_coeffs"] = int(args.vip_basis_size)
    config["ftip"]["training_overrides"]["regression_coeffs"] = int(args.vip_basis_size)
    config.setdefault("likelihood", {})["learn_observation_noise"] = bool(
        args.learn_observation_noise
    )
    if args.num_inducing is not None:
        config.setdefault("gmvip", {})["num_inducing"] = int(args.num_inducing)
        config.setdefault("sip", {})["num_inducing"] = int(args.num_inducing)
    if args.prior_bank_size is not None:
        config.setdefault("prior", {})["bank_size"] = int(args.prior_bank_size)
        config.setdefault("gmvip", {})["prior_bank_size"] = int(args.prior_bank_size)
    if args.disable_tqdm:
        config.setdefault("training", {})["disable_tqdm"] = True

    target_file = data_root / "target_paths.npz"
    if args.regenerate_targets or not target_file.exists():
        generate_dataset(
            data_root,
            n_targets=3 if args.smoke else 100,
            dt=0.5 if args.smoke else 0.05,
            seed=int(args.seed),
            reference_bank_size=32 if args.smoke else 4096,
        )

    methods = [
        METHOD_ALIASES.get(value.strip(), value.strip())
        for value in str(args.methods).split(",")
        if value.strip()
    ]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    results = {}
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")
        results[method] = run_method(method, config, args)
    return results


if __name__ == "__main__":
    main()
