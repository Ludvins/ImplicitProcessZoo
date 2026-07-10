"""Shared experiment-side model setup helpers."""

from __future__ import annotations

import torch


def fix_gaussian_noise(model: torch.nn.Module, noise_std: torch.Tensor) -> None:
    if not hasattr(model, "log_variance"):
        return
    value = 2.0 * torch.log(noise_std.clamp_min(torch.finfo(noise_std.dtype).eps))
    value = value.to(dtype=model.log_variance.dtype, device=model.log_variance.device)
    if model.log_variance.ndim == 0:
        value = value.reshape(-1)[0]
    else:
        value = value.expand_as(model.log_variance)
    with torch.no_grad():
        model.log_variance.copy_(value)
    model.log_variance.requires_grad_(False)
