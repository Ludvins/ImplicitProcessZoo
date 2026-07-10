from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class LotkaVolterraPrior(nn.Module):
    """Live Lotka-Volterra ODE prior over trajectory functions.

    A latent sample is the physical parameter vector

        theta = [alpha, beta, delta, gamma, x0, y0].

    Evaluating a latent integrates the Lotka-Volterra ODE from t=0 to the
    requested normalized times.  No prior trajectory bank is stored or
    interpolated; finite banks used by VIP/GM-VIP are Monte Carlo samples from
    this ODE prior.
    """

    input_dim = 1
    output_dim = 2
    theta_names = ("alpha", "beta", "delta", "gamma", "x0", "y0")

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
        t_tensor = torch.as_tensor(t, dtype=dtype, device=device)
        if t_tensor.ndim != 1:
            raise ValueError("t must have shape [T].")
        if t_tensor.numel() < 2:
            raise ValueError("t must contain at least two time points.")
        if torch.any(t_tensor[1:] <= t_tensor[:-1]):
            raise ValueError("t must be strictly increasing.")

        self.num_samples = int(num_samples)
        self.reference_bank_size = int(reference_bank_size)
        self.seed = int(seed)
        self.dtype = dtype
        self.device = device
        self.t_max = float(t_tensor[-1].detach().cpu())
        diffs = t_tensor[1:] - t_tensor[:-1]
        default_dt = float(diffs.min().detach().cpu())
        self.integration_dt = float(integration_dt or default_dt)
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
        self._fixed_latents: torch.Tensor | None = None
        self._latent_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._grid_cache_key: tuple | None = None
        self._grid_cache_values: torch.Tensor | None = None

    @property
    def num_paths(self) -> int:
        """Compatibility value for finite-bank diagnostics."""

        return int(self.reference_bank_size)

    def KL(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype, device=self.device)

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

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

    def _generator(self, seed: int | None = None) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed if seed is None else int(seed))
        return generator

    def sample_latents(
        self, num_samples: int, seed: int | None = None, *, cache: bool = True
    ) -> torch.Tensor:
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        cache_key = (num_samples, self.seed if seed is None else int(seed))
        if cache:
            cached = self._latent_cache.get(cache_key)
            if cached is not None:
                return cached
        generator = self._generator(seed)
        log_params = self.theta_log_means.reshape(1, 4) + self.theta_log_stds.reshape(
            1, 4
        ) * torch.randn(num_samples, 4, generator=generator, dtype=self.dtype, device=self.device)
        ode_params = torch.exp(log_params)
        initials = self.initial_low.reshape(1, 2) + (self.initial_high - self.initial_low).reshape(
            1, 2
        ) * torch.rand(num_samples, 2, generator=generator, dtype=self.dtype, device=self.device)
        latents = torch.cat([ode_params, initials], dim=-1)
        if cache:
            self._latent_cache[cache_key] = latents
        return latents

    def sample_indices(self, n: int, seed: int | None = None) -> torch.Tensor:
        """Compatibility alias returning sampled ODE latents."""

        return self.sample_latents(int(n), seed=seed)

    def _normalized_time_to_physical(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        if X.ndim == 1:
            x = X
        elif X.ndim == 2 and X.shape[-1] == 1:
            x = X[:, 0]
        else:
            raise ValueError("LotkaVolterraPrior expects X with shape [N] or [N, 1].")
        return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5 * self.t_max).clamp(
            float(self.t[0].detach().cpu()),
            self.t_max,
        )

    @staticmethod
    def _rhs(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        alpha, beta, delta, gamma = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
        prey, predator = state[:, 0], state[:, 1]
        return torch.stack(
            [
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            ],
            dim=-1,
        )

    def _rk4_step(self, state: torch.Tensor, theta: torch.Tensor, dt: float) -> torch.Tensor:
        h = torch.as_tensor(dt, dtype=state.dtype, device=state.device)
        k1 = self._rhs(state, theta)
        k2 = self._rhs(state + 0.5 * h * k1, theta)
        k3 = self._rhs(state + 0.5 * h * k2, theta)
        k4 = self._rhs(state + h * k3, theta)
        next_state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.nan_to_num(next_state, nan=0.0, posinf=20.0, neginf=0.0).clamp(0.0, 20.0)

    def _integrate_sorted(self, theta: torch.Tensor, t_sorted: torch.Tensor) -> torch.Tensor:
        if t_sorted.numel() == 0:
            return torch.empty(theta.shape[0], 0, 2, dtype=self.dtype, device=self.device)
        state = theta[:, 4:6].to(dtype=self.dtype, device=self.device)
        outputs = []
        current_t = 0.0
        max_dt = float(self.integration_dt)
        eps = 1.0e-12
        for target_t_tensor in t_sorted:
            target_t = float(target_t_tensor.detach().cpu())
            while current_t + eps < target_t:
                step = min(max_dt, target_t - current_t)
                state = self._rk4_step(state, theta, step)
                current_t += step
            outputs.append(state)
        return torch.stack(outputs, dim=1)

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
        self._grid_cache_key = key
        self._grid_cache_values = values
        return values

    def evaluate_raw(self, X: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        theta = latents.to(device=self.device, dtype=self.dtype)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        if theta.ndim != 2 or theta.shape[-1] != 6:
            raise ValueError("Lotka-Volterra latents must have shape [S, 6].")
        t_query = self._normalized_time_to_physical(X)
        grid = self._trajectory_grid(theta)
        idx_hi = torch.searchsorted(self.t.contiguous(), t_query.contiguous(), right=False)
        idx_hi = idx_hi.clamp(1, self.t.shape[0] - 1)
        idx_lo = idx_hi - 1
        t_lo = self.t[idx_lo]
        t_hi = self.t[idx_hi]
        weight = ((t_query - t_lo) / (t_hi - t_lo).clamp_min(1e-12)).reshape(1, -1, 1)
        y_lo = grid[:, idx_lo, :]
        y_hi = grid[:, idx_hi, :]
        return y_lo * (1.0 - weight) + y_hi * weight

    def evaluate(self, X: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        raw = self.evaluate_raw(X, latents)
        return (raw - self.y_mean) / self.y_std

    def evaluate_latents(self, latents: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return self.evaluate(X, latents)

    def sample(self, X: torch.Tensor, n: int, seed: int | None = None) -> torch.Tensor:
        return self.evaluate(X, self.sample_latents(int(n), seed=seed))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self._fixed_latents is None or int(self._fixed_latents.shape[0]) != int(
            self.num_samples
        ):
            self._fixed_latents = self.sample_latents(self.num_samples, seed=self.seed)
        return self.evaluate(X, self._fixed_latents)

    def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y.to(dtype=self.dtype, device=self.device) * self.y_std + self.y_mean
