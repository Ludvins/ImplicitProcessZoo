"""Common prediction helpers for the public sample-first model API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch


@torch.no_grad()
def batched_predict_samples(
    model: torch.nn.Module,
    batches: Iterable,
    num_samples: int,
    *,
    kind: Literal["f", "y"] = "y",
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Predict batches and concatenate on the observation axis.

    Models follow the ``[samples, observations, outputs]`` contract. The
    same optional seed is passed to every batch so pathwise models can reuse
    the same sampled function across uneven minibatches.

    Parameters
    ----------
    model : torch.nn.Module
        Model implementing ``predict_f_samples`` and/or ``predict_y_samples``.
    batches : collections.abc.Iterable
        Iterable yielding input tensors or ``(inputs, targets)`` batches.
    num_samples : int
        Number of samples to request from every batch.
    kind : {"f", "y"}, default="y"
        Select latent-function samples (``"f"``) or observation samples
        (``"y"``).
    device : torch.device or str or None, default=None
        Device to which each input batch is moved. If ``None``, preserve the
        batch device.
    seed : int or None, default=None
        Temporary prediction seed passed unchanged to every batch.

    Returns
    -------
    torch.Tensor
        CPU tensor with shape ``[num_samples, observations, outputs]``.

    Raises
    ------
    ValueError
        If the sample count is nonpositive, no batches are supplied, or a
        model returns a tensor that violates the sample-first contract.
    TypeError
        If the selected prediction method is not implemented by ``model``.
    """
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive.")
    method = getattr(model, f"predict_{kind}_samples", None)
    if method is None:
        raise TypeError(f"{type(model).__name__} does not implement predict_{kind}_samples().")

    model.eval()
    outputs = []
    for batch in batches:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        if device is not None:
            inputs = inputs.to(device)
        result = method(inputs, int(num_samples), seed=seed)
        if result.ndim != 3 or result.shape[0] != int(num_samples):
            raise ValueError(
                f"{type(model).__name__}.predict_{kind}_samples() returned "
                f"{tuple(result.shape)}; expected [S, N, D] with S={num_samples}."
            )
        outputs.append(result.detach().cpu())
    if not outputs:
        raise ValueError("batches must contain at least one batch.")
    return torch.cat(outputs, dim=1)
