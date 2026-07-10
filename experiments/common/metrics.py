"""Shared probabilistic metrics."""

from __future__ import annotations

import torch


def empirical_crps(samples, targets) -> torch.Tensor:
    """Empirical CRPS using the O(S log S) sorted-sample identity."""
    samples = torch.as_tensor(samples)
    targets = torch.as_tensor(targets, dtype=samples.dtype, device=samples.device)
    if samples.ndim < 2 or samples.shape[0] <= 0:
        raise ValueError("samples must have a nonempty leading sample dimension.")
    term1 = torch.abs(samples - targets.unsqueeze(0)).mean(dim=0)
    count = samples.shape[0]
    if count == 1:
        return term1.mean()
    ordered = samples.sort(dim=0).values
    coefficients = 2 * torch.arange(count, dtype=samples.dtype, device=samples.device) + 1 - count
    shape = (count,) + (1,) * (samples.ndim - 1)
    pairwise_term = (ordered * coefficients.reshape(shape)).sum(dim=0) / (count * count)
    return (term1 - pairwise_term).mean()


def empirical_crps_pairwise(samples, targets) -> torch.Tensor:
    """Reference O(S²) implementation used only for regression tests."""
    samples = torch.as_tensor(samples)
    targets = torch.as_tensor(targets, dtype=samples.dtype, device=samples.device)
    term1 = torch.mean(torch.abs(samples - targets.unsqueeze(0)))
    term2 = 0.5 * torch.mean(torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0)))
    return term1 - term2
