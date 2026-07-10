from __future__ import annotations

import numpy as np
import torch
from torch import nn


class HistoricalLoadWindowPrior(nn.Module):
    """Empirical prior over prefix-normalized historical load windows.

    The latent variable is just an integer row index into ``windows``.  Each
    row is a full historical trajectory normalized by its own observed prefix,
    so samples live in the same normalized coordinate system as the target
    window.
    """

    input_dim = 1
    output_dim = 1

    def __init__(
        self,
        windows: np.ndarray | torch.Tensor,
        *,
        num_samples: int = 512,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        device = torch.device(device or "cpu")
        values = torch.as_tensor(windows, dtype=dtype, device=device)
        if values.ndim == 2:
            values = values.unsqueeze(-1)
        if values.ndim != 3 or values.shape[-1] != 1:
            raise ValueError("windows must have shape [P, T] or [P, T, 1].")
        if values.shape[0] < 2:
            raise ValueError("HistoricalLoadWindowPrior needs at least two windows.")
        if values.shape[1] < 2:
            raise ValueError("Historical windows must contain at least two time points.")

        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self.register_buffer("windows", values.contiguous())
        self._fixed_latents: torch.Tensor | None = None

    @property
    def num_paths(self) -> int:
        return int(self.windows.shape[0])

    @property
    def window_length(self) -> int:
        return int(self.windows.shape[1])

    def KL(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype, device=self.device)

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def clone_with_normalization(
        self,
        *,
        y_mean=None,
        y_std=None,
        num_samples: int | None = None,
        seed: int | None = None,
        **_: object,
    ) -> HistoricalLoadWindowPrior:
        # Windows are already prefix-normalized. y_mean/y_std are accepted only
        # for compatibility with the simulator-prior constructors.
        return HistoricalLoadWindowPrior(
            self.windows.detach(),
            num_samples=self.num_samples if num_samples is None else int(num_samples),
            seed=self.seed if seed is None else int(seed),
            device=self.device,
            dtype=self.dtype,
        )

    def _generator(self, seed: int | None = None) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed if seed is None else int(seed))
        return generator

    def sample_latents(
        self, num_samples: int, seed: int | None = None, **_: object
    ) -> torch.Tensor:
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        generator = self._generator(seed)
        return torch.randint(
            low=0,
            high=self.num_paths,
            size=(num_samples,),
            generator=generator,
            dtype=torch.long,
            device=self.device,
        )

    def sample_indices(self, n: int, seed: int | None = None) -> torch.Tensor:
        return self.sample_latents(int(n), seed=seed)

    def _coerce_x(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device, dtype=self.dtype)
        if X.ndim == 2 and X.shape[-1] == 1:
            X = X[:, 0]
        if X.ndim != 1:
            raise ValueError("X must have shape [N] or [N, 1].")
        return X.clamp(-1.0, 1.0)

    def evaluate_raw(self, X: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        idx = latents.to(device=self.device, dtype=torch.long).reshape(-1)
        x = self._coerce_x(X)
        pos = (x + 1.0) * 0.5 * float(self.window_length - 1)
        idx_lo = torch.floor(pos).to(torch.long).clamp(0, self.window_length - 1)
        idx_hi = torch.ceil(pos).to(torch.long).clamp(0, self.window_length - 1)
        weight = (pos - idx_lo.to(self.dtype)).reshape(1, -1, 1)
        selected = self.windows[idx]
        y_lo = selected[:, idx_lo, :]
        y_hi = selected[:, idx_hi, :]
        return y_lo * (1.0 - weight) + y_hi * weight

    def evaluate(self, X: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return self.evaluate_raw(X, latents)

    def evaluate_latents(self, latents: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return self.evaluate_raw(X, latents)

    def sample(self, X: torch.Tensor, n: int, seed: int | None = None) -> torch.Tensor:
        return self.evaluate_raw(X, self.sample_latents(int(n), seed=seed))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self._fixed_latents is None or int(self._fixed_latents.shape[0]) != int(
            self.num_samples
        ):
            self._fixed_latents = self.sample_latents(self.num_samples, seed=self.seed)
        return self.evaluate_raw(X, self._fixed_latents)

    def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y.to(dtype=self.dtype, device=self.device)
