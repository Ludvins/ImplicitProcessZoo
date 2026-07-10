"""Versioned, resumable training checkpoints."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .random import capture_generator_states, restore_generator_states

CHECKPOINT_SCHEMA_VERSION = 1


def capture_rng_state(model: torch.nn.Module | None = None) -> dict[str, Any]:
    """Capture process-wide and optional model-owned random state.

    Parameters
    ----------
    model : torch.nn.Module or None, default=None
        Model whose owned generator states are included.

    Returns
    -------
    dict of str to object
        Python, NumPy, CPU/CUDA PyTorch, and model-generator states.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if model is not None:
        state["model_generators"] = capture_generator_states(model)
    return state


def restore_rng_state(state: Mapping[str, Any], model: torch.nn.Module | None = None) -> None:
    """Restore process-wide and optional model-owned random state.

    Parameters
    ----------
    state : collections.abc.Mapping
        State produced by :func:`capture_rng_state`.
    model : torch.nn.Module or None, default=None
        Model whose owned generators are restored.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    if model is not None:
        restore_generator_states(model, state.get("model_generators", {}))


def build_training_checkpoint(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    global_step: int,
    arguments: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Build a complete schema-version 1 training bundle.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters, buffers, and owned RNG states are saved.
    optimizer : torch.optim.Optimizer
        Optimizer associated with ``model``.
    scheduler : torch.optim.lr_scheduler.LRScheduler or None
        Optional learning-rate scheduler.
    global_step : int
        Number of completed optimizer updates.
    arguments : collections.abc.Mapping or object
        Experiment arguments, or an object accepted by :func:`vars`.

    Returns
    -------
    dict of str to object
        Complete versioned checkpoint bundle.
    """
    if not isinstance(arguments, Mapping):
        arguments = vars(arguments)
    normalization_state = {
        key: value.detach().cpu()
        for key, value in model.named_buffers()
        if any(token in key.lower() for token in ("mean", "std", "scale", "offset"))
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": int(global_step),
        "arguments": dict(arguments),
        "normalization_state": normalization_state,
        "rng_state": capture_rng_state(model),
    }


def save_training_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> None:
    """Atomically save a versioned training checkpoint.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination checkpoint path.
    checkpoint : collections.abc.Mapping
        Bundle returned by :func:`build_training_checkpoint`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(checkpoint), temporary)
    temporary.replace(path)


def load_training_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load and validate a complete checkpoint for training resumption.

    Parameters
    ----------
    path : str or pathlib.Path
        Checkpoint path to load.
    map_location : str or torch.device or None, default=None
        Device mapping forwarded to :func:`torch.load`.

    Returns
    -------
    dict of str to object
        Validated schema-version 1 checkpoint.

    Raises
    ------
    ValueError
        If the file is model-only, has an unsupported schema, or omits a
        required field.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "schema_version" not in checkpoint:
        raise ValueError(
            "Resume requires a full versioned training checkpoint. This file looks like a "
            "legacy model-only state dict; pass it to --warm-start-from instead."
        )
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {checkpoint['schema_version']!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    required = {"model", "optimizer", "scheduler", "global_step", "arguments", "rng_state"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    return checkpoint


def restore_training_checkpoint(
    checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    """Restore model, optimization, scheduler, and random state.

    Parameters
    ----------
    checkpoint : collections.abc.Mapping
        Validated versioned checkpoint.
    model : torch.nn.Module
        Model receiving parameters and generator state.
    optimizer : torch.optim.Optimizer
        Optimizer receiving its saved state.
    scheduler : torch.optim.lr_scheduler.LRScheduler or None, default=None
        Scheduler receiving saved state when the checkpoint contains it.

    Returns
    -------
    int
        Restored global optimizer step.

    Raises
    ------
    ValueError
        If scheduler state exists but no scheduler is supplied.
    """
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    saved_scheduler = checkpoint.get("scheduler")
    if saved_scheduler is not None:
        if scheduler is None:
            raise ValueError("Checkpoint contains scheduler state but no scheduler was provided.")
        scheduler.load_state_dict(saved_scheduler)
    restore_rng_state(checkpoint["rng_state"], model)
    global_step = int(checkpoint["global_step"])
    model._fit_global_step = global_step
    return global_step


def load_warm_start_state(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> Mapping[str, torch.Tensor]:
    """Read a legacy state dictionary or versioned model component.

    Parameters
    ----------
    path : str or pathlib.Path
        Warm-start file to load.
    map_location : str or torch.device or None, default=None
        Device mapping forwarded to :func:`torch.load`.

    Returns
    -------
    collections.abc.Mapping of str to torch.Tensor
        Model state dictionary suitable for :meth:`torch.nn.Module.load_state_dict`.

    Raises
    ------
    ValueError
        If the file is neither a compatible versioned checkpoint nor a raw
        state dictionary.
    """
    value = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(value, dict) and "schema_version" in value:
        if value["schema_version"] != CHECKPOINT_SCHEMA_VERSION or "model" not in value:
            raise ValueError("Warm-start checkpoint has an unsupported schema.")
        return value["model"]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return value
    raise ValueError("Warm-start file is neither a model state dict nor a versioned checkpoint.")
