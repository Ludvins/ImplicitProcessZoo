from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class LotkaVolterraPrior(nn.Module):
    """Fixed bank of coherent Lotka-Volterra simulator trajectories.

    Stored paths are in physical units. Forward/evaluate return normalized
    values using the task-specific y_mean/y_std supplied at construction time.
    """

    input_dim = 1
    output_dim = 2

    def __init__(
        self,
        t: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        theta: np.ndarray | torch.Tensor | None = None,
        *,
        y_mean: np.ndarray | torch.Tensor | float = 0.0,
        y_std: np.ndarray | torch.Tensor | float = 1.0,
        num_samples: int = 512,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        device = torch.device(device or "cpu")
        t_tensor = torch.as_tensor(t, dtype=dtype, device=device)
        y_tensor = torch.as_tensor(y, dtype=dtype, device=device)
        if t_tensor.ndim != 1:
            raise ValueError("t must have shape [T].")
        if y_tensor.ndim != 3 or y_tensor.shape[1] != t_tensor.shape[0] or y_tensor.shape[-1] != 2:
            raise ValueError("y must have shape [N, T, 2].")
        if torch.any(t_tensor[1:] <= t_tensor[:-1]):
            raise ValueError("t must be strictly increasing.")

        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.dtype = dtype
        self.device = device
        self.t_max = float(t_tensor[-1].detach().cpu())
        self.register_buffer("t", t_tensor)
        self.register_buffer("y_bank", y_tensor)
        if theta is None:
            theta_tensor = torch.empty(y_tensor.shape[0], 0, dtype=dtype, device=device)
        else:
            theta_tensor = torch.as_tensor(theta, dtype=dtype, device=device)
        self.register_buffer("theta", theta_tensor)
        y_mean_tensor = torch.as_tensor(y_mean, dtype=dtype, device=device)
        y_std_tensor = torch.as_tensor(y_std, dtype=dtype, device=device)
        if y_mean_tensor.numel() == 1:
            y_mean_tensor = y_mean_tensor.expand(2)
        if y_std_tensor.numel() == 1:
            y_std_tensor = y_std_tensor.expand(2)
        self.register_buffer("y_mean", y_mean_tensor.reshape(1, 2))
        self.register_buffer("y_std", y_std_tensor.reshape(1, 2).clamp_min(1e-8))
        self._fixed_sample_ids: torch.Tensor | None = None

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        y_mean: np.ndarray | torch.Tensor | float = 0.0,
        y_std: np.ndarray | torch.Tensor | float = 1.0,
        num_samples: int = 512,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "LotkaVolterraPrior":
        data = np.load(path)
        theta = data["theta"] if "theta" in data.files else None
        return cls(
            data["t"],
            data["y"],
            theta,
            y_mean=y_mean,
            y_std=y_std,
            num_samples=num_samples,
            seed=seed,
            device=device,
            dtype=dtype,
        )

    def clone_with_normalization(
        self,
        *,
        y_mean: np.ndarray | torch.Tensor | float,
        y_std: np.ndarray | torch.Tensor | float,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> "LotkaVolterraPrior":
        return LotkaVolterraPrior(
            self.t.detach(),
            self.y_bank.detach(),
            self.theta.detach(),
            y_mean=y_mean,
            y_std=y_std,
            num_samples=self.num_samples if num_samples is None else int(num_samples),
            seed=self.seed if seed is None else int(seed),
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def num_paths(self) -> int:
        return int(self.y_bank.shape[0])

    def KL(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype, device=self.device)

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def sample_indices(self, n: int, seed: int | None = None) -> torch.LongTensor:
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive.")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed if seed is None else int(seed))
        if n <= self.num_paths:
            return torch.randperm(self.num_paths, generator=generator, device=self.device)[:n]
        return torch.randint(self.num_paths, (n,), generator=generator, device=self.device)

    def sample_latents(self, num_samples: int, seed: int | None = None) -> torch.LongTensor:
        return self.sample_indices(num_samples, seed=seed)

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

    def evaluate_raw(self, X: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        ids = sample_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        t_query = self._normalized_time_to_physical(X)
        idx_hi = torch.searchsorted(self.t.contiguous(), t_query.contiguous(), right=False)
        idx_hi = idx_hi.clamp(1, self.t.shape[0] - 1)
        idx_lo = idx_hi - 1
        t_lo = self.t[idx_lo]
        t_hi = self.t[idx_hi]
        weight = ((t_query - t_lo) / (t_hi - t_lo).clamp_min(1e-12)).reshape(1, -1, 1)
        y_lo = self.y_bank[ids][:, idx_lo, :]
        y_hi = self.y_bank[ids][:, idx_hi, :]
        return y_lo * (1.0 - weight) + y_hi * weight

    def evaluate(self, X: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        raw = self.evaluate_raw(X, sample_ids)
        return (raw - self.y_mean) / self.y_std

    def evaluate_latents(self, latents: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return self.evaluate(X, latents)

    def sample(self, X: torch.Tensor, n: int, seed: int | None = None) -> torch.Tensor:
        return self.evaluate(X, self.sample_indices(n, seed=seed))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self._fixed_sample_ids is None or int(self._fixed_sample_ids.numel()) != int(self.num_samples):
            self._fixed_sample_ids = self.sample_indices(self.num_samples, seed=self.seed)
        return self.evaluate(X, self._fixed_sample_ids)

    def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y.to(dtype=self.dtype, device=self.device) * self.y_std + self.y_mean
