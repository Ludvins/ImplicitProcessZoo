"""Shared model fitting loop with lightweight model hooks."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from tqdm import tqdm

from .random import ensure_module_generators
from .utils import infinite_loader


def validate_fit_mode(*, epochs: int | None, iterations: int | None) -> None:
    """Require one, and only one, positive fit duration."""
    if (epochs is None) == (iterations is None):
        raise ValueError("Exactly one of epochs or iterations must be set.")
    value = epochs if epochs is not None else iterations
    if int(value) <= 0:
        raise ValueError("The selected fit duration must be positive.")


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable,
    *,
    epochs: int | None,
    iterations: int | None,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Create the existing epoch-stepped cosine schedule safely."""
    validate_fit_mode(epochs=epochs, iterations=iterations)
    period = epochs if epochs is not None else max(1, int(iterations) // len(train_loader))
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(period)),
        eta_min=optimizer.param_groups[0]["lr"] / 100,
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def fit_loop(
    model: torch.nn.Module,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int | None,
    iterations: int | None,
    use_tqdm: bool,
    return_loss: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> list[float]:
    """Run a common fit loop around a model's ``_train_step`` hook."""
    validate_fit_mode(epochs=epochs, iterations=iterations)
    ensure_module_generators(model)
    prepare = getattr(model, "_prepare_fit", None)
    if prepare is not None:
        prepare(train_loader)

    device = _model_device(model)
    model.train()
    losses: list[float] = []
    global_step = int(getattr(model, "_fit_global_step", 0))

    def step(batch) -> None:
        nonlocal global_step
        inputs, target = batch
        inputs = inputs.to(device)
        target = target.to(device)
        before_step = getattr(model, "_before_fit_step", None)
        if before_step is not None:
            before_step(inputs, target)
        loss = model._train_step(optimizer, inputs, target)
        global_step += 1
        if return_loss:
            losses.append(float(loss.detach().cpu()))

    if epochs is not None:
        loop = tqdm(range(epochs), unit=" epoch", desc="Training") if use_tqdm else range(epochs)
        for _ in loop:
            for batch in train_loader:
                step(batch)
            if scheduler is not None:
                scheduler.step()
            if use_tqdm:
                loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    else:
        loop = (
            tqdm(range(iterations), unit=" iter", desc="Training")
            if use_tqdm
            else range(iterations)
        )
        stream = infinite_loader(train_loader)
        for _ in range(global_step % len(train_loader)):
            next(stream)
        for index in loop:
            step(next(stream))
            if scheduler is not None and (index + 1) % len(train_loader) == 0:
                scheduler.step()
            if use_tqdm:
                loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    model._fit_optimizer = optimizer
    model._fit_scheduler = scheduler
    model._fit_global_step = global_step
    return losses
