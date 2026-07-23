from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class DampedOscillatorPrior(nn.Module):
    """Live randomly forced damped-oscillator simulator prior.

    Latents have schema

        [omega, zeta, A, Omega, phi, x0, v0, drag_c, u_0, ..., u_K].

    The model prior uses ``sample_drag=False`` so ``drag_c`` is always zero.
    Misspecified target generation can use ``sample_drag=True`` while the
    runner still trains against the matched, no-drag prior.
    """

    input_dim = 1
    output_dim = 1
    theta_names = ("omega", "zeta", "A", "Omega", "phi", "x0", "v0", "drag_c")

    def __init__(
        self,
        t: np.ndarray | torch.Tensor,
        *,
        y_mean: np.ndarray | torch.Tensor | float = 0.0,
        y_std: np.ndarray | torch.Tensor | float = 1.0,
        num_samples: int = 512,
        reference_bank_size: int = 4096,
        seed: int = 0,
        forcing_delta: float = 0.1,
        rho: float = 0.98,
        sigma_u: float = 0.05,
        sample_drag: bool = False,
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
            raise ValueError("t must contain at least two points.")
        if torch.any(t_tensor[1:] <= t_tensor[:-1]):
            raise ValueError("t must be strictly increasing.")
        if abs(float(t_tensor[0].detach().cpu())) > 1e-12:
            raise ValueError("DampedOscillatorPrior expects t to start at 0.")
        if forcing_delta <= 0.0:
            raise ValueError("forcing_delta must be positive.")

        self.num_samples = int(num_samples)
        self.reference_bank_size = int(reference_bank_size)
        self.seed = int(seed)
        self.dtype = dtype
        self.device = device
        self.t_max = float(t_tensor[-1].detach().cpu())
        self.forcing_delta = float(forcing_delta)
        self.rho = float(rho)
        self.sigma_u = float(sigma_u)
        self.sample_drag = bool(sample_drag)
        self.forcing_count = int(math.ceil(self.t_max / self.forcing_delta)) + 1
        diffs = t_tensor[1:] - t_tensor[:-1]
        self.integration_dt = float(
            integration_dt or min(float(diffs.min().detach().cpu()), self.forcing_delta)
        )
        if self.integration_dt <= 0.0:
            raise ValueError("integration_dt must be positive.")

        self.register_buffer("t", t_tensor)
        y_mean_tensor = torch.as_tensor(y_mean, dtype=dtype, device=device)
        y_std_tensor = torch.as_tensor(y_std, dtype=dtype, device=device)
        self.register_buffer("y_mean", y_mean_tensor.reshape(1, 1))
        self.register_buffer("y_std", y_std_tensor.reshape(1, 1).clamp_min(1e-8))
        self._fixed_latents: torch.Tensor | None = None
        self._latent_cache: dict[tuple[int, int, bool], torch.Tensor] = {}
        self._grid_cache_key: tuple | None = None
        self._grid_cache_source: torch.Tensor | None = None
        self._grid_cache_values: torch.Tensor | None = None

    @property
    def num_paths(self) -> int:
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
        sample_drag: bool | None = None,
    ) -> DampedOscillatorPrior:
        return DampedOscillatorPrior(
            self.t.detach(),
            y_mean=y_mean,
            y_std=y_std,
            num_samples=self.num_samples if num_samples is None else int(num_samples),
            reference_bank_size=self.reference_bank_size,
            seed=self.seed if seed is None else int(seed),
            forcing_delta=self.forcing_delta,
            rho=self.rho,
            sigma_u=self.sigma_u,
            sample_drag=self.sample_drag if sample_drag is None else bool(sample_drag),
            integration_dt=self.integration_dt,
            device=self.device,
            dtype=self.dtype,
        )

    def _generator(self, seed: int | None = None) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed if seed is None else int(seed))
        return generator

    def sample_latents(
        self,
        num_samples: int,
        seed: int | None = None,
        *,
        cache: bool = True,
        sample_drag: bool | None = None,
    ) -> torch.Tensor:
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        use_drag = self.sample_drag if sample_drag is None else bool(sample_drag)
        cache_key = (num_samples, self.seed if seed is None else int(seed), use_drag)
        if cache:
            cached = self._latent_cache.get(cache_key)
            if cached is not None:
                return cached
        generator = self._generator(seed)
        omega = torch.exp(
            torch.full((num_samples, 1), math.log(1.0), dtype=self.dtype, device=self.device)
            + 0.25
            * torch.randn(num_samples, 1, generator=generator, dtype=self.dtype, device=self.device)
        )
        zeta = 0.03 + 0.12 * torch.rand(
            num_samples, 1, generator=generator, dtype=self.dtype, device=self.device
        )
        amp = 0.2 + 0.8 * torch.rand(
            num_samples, 1, generator=generator, dtype=self.dtype, device=self.device
        )
        drive_omega = 0.6 + 0.8 * torch.rand(
            num_samples, 1, generator=generator, dtype=self.dtype, device=self.device
        )
        phi = (
            2.0
            * math.pi
            * torch.rand(num_samples, 1, generator=generator, dtype=self.dtype, device=self.device)
        )
        x0 = 0.2 * torch.randn(
            num_samples, 1, generator=generator, dtype=self.dtype, device=self.device
        )
        v0 = 0.2 * torch.randn(
            num_samples, 1, generator=generator, dtype=self.dtype, device=self.device
        )
        if use_drag:
            drag = 0.02 + 0.06 * torch.rand(
                num_samples, 1, generator=generator, dtype=self.dtype, device=self.device
            )
        else:
            drag = torch.zeros(num_samples, 1, dtype=self.dtype, device=self.device)

        stationary_std = self.sigma_u / math.sqrt(max(1.0 - self.rho**2, 1e-12))
        forcing = torch.empty(num_samples, self.forcing_count, dtype=self.dtype, device=self.device)
        forcing[:, 0] = stationary_std * torch.randn(
            num_samples, generator=generator, dtype=self.dtype, device=self.device
        )
        for index in range(1, self.forcing_count):
            eps = torch.randn(
                num_samples, generator=generator, dtype=self.dtype, device=self.device
            )
            forcing[:, index] = self.rho * forcing[:, index - 1] + self.sigma_u * eps

        latents = torch.cat([omega, zeta, amp, drive_omega, phi, x0, v0, drag, forcing], dim=-1)
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
            raise ValueError("DampedOscillatorPrior expects X with shape [N] or [N, 1].")
        return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5 * self.t_max).clamp(0.0, self.t_max)

    def _forcing_at(self, theta: torch.Tensor, t_value: float) -> torch.Tensor:
        index = min(max(int(math.floor(t_value / self.forcing_delta)), 0), self.forcing_count - 1)
        return theta[:, 8 + index]

    def _rhs(self, state: torch.Tensor, theta: torch.Tensor, t_value: float) -> torch.Tensor:
        omega = theta[:, 0]
        zeta = theta[:, 1]
        amp = theta[:, 2]
        drive_omega = theta[:, 3]
        phi = theta[:, 4]
        drag = theta[:, 7]
        x = state[:, 0]
        v = state[:, 1]
        forcing = self._forcing_at(theta, t_value)
        dx = v
        dv = (
            -2.0 * zeta * omega * v
            - omega.square() * x
            - drag * v * torch.abs(v)
            + amp * torch.sin(drive_omega * float(t_value) + phi)
            + forcing
        )
        return torch.stack([dx, dv], dim=-1)

    def _rk4_step(
        self, state: torch.Tensor, theta: torch.Tensor, t_value: float, dt: float
    ) -> torch.Tensor:
        h = float(dt)
        k1 = self._rhs(state, theta, t_value)
        k2 = self._rhs(state + 0.5 * h * k1, theta, t_value + 0.5 * h)
        k3 = self._rhs(state + 0.5 * h * k2, theta, t_value + 0.5 * h)
        k4 = self._rhs(state + h * k3, theta, t_value + h)
        next_state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

    def _integrate_sorted(self, theta: torch.Tensor, t_sorted: torch.Tensor) -> torch.Tensor:
        if t_sorted.numel() == 0:
            return torch.empty(theta.shape[0], 0, 1, dtype=self.dtype, device=self.device)
        state = torch.stack([theta[:, 5], theta[:, 6]], dim=-1).to(
            dtype=self.dtype, device=self.device
        )
        outputs = []
        current_t = 0.0
        max_dt = float(self.integration_dt)
        eps = 1.0e-12
        for target_t_tensor in t_sorted:
            target_t = float(target_t_tensor.detach().cpu())
            while current_t + eps < target_t:
                step = min(max_dt, target_t - current_t)
                state = self._rk4_step(state, theta, current_t, step)
                current_t += step
            outputs.append(state[:, 0:1])
        return torch.stack(outputs, dim=1)

    def _cache_key(self, theta: torch.Tensor) -> tuple:
        return (
            int(theta._version),
            tuple(theta.shape),
            str(theta.dtype),
            str(theta.device),
            float(self.integration_dt),
            float(self.forcing_delta),
            float(self.rho),
            float(self.sigma_u),
        )

    def _trajectory_grid(self, theta: torch.Tensor) -> torch.Tensor:
        key = self._cache_key(theta)
        if (
            self._grid_cache_source is theta
            and self._grid_cache_key == key
            and self._grid_cache_values is not None
        ):
            return self._grid_cache_values
        values = self._integrate_sorted(theta, self.t)
        # Keep the source alive and compare object identity. A data pointer alone
        # is not a stable tensor identity because PyTorch's allocator may reuse
        # freed storage for the next fresh latent draw.
        self._grid_cache_source = theta
        self._grid_cache_key = key
        self._grid_cache_values = values
        return values

    def evaluate_raw(self, X: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        theta = latents.to(device=self.device, dtype=self.dtype)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        expected = 8 + self.forcing_count
        if theta.ndim != 2 or theta.shape[-1] != expected:
            raise ValueError(f"Damped oscillator latents must have shape [S, {expected}].")
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
