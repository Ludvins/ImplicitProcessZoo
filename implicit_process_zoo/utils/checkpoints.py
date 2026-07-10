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
    """Build a complete schema-v1 training bundle."""
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
    """Restore training, optimization, scheduler, and random state."""
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
    """Read either a legacy raw state dict or a versioned model component."""
    value = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(value, dict) and "schema_version" in value:
        if value["schema_version"] != CHECKPOINT_SCHEMA_VERSION or "model" not in value:
            raise ValueError("Warm-start checkpoint has an unsupported schema.")
        return value["model"]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return value
    raise ValueError("Warm-start file is neither a model state dict nor a versioned checkpoint.")
