"""Shared model fitting loop with lightweight model hooks."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from tqdm import tqdm

from .random import ensure_module_generators
from .utils import infinite_loader


def validate_fit_mode(*, epochs: int | None, iterations: int | None) -> None:
    """Require exactly one positive fit duration.

    Parameters
    ----------
    epochs : int or None
        Number of complete loader passes, or ``None`` for iteration mode.
    iterations : int or None
        Number of optimizer updates, or ``None`` for epoch mode.

    Raises
    ------
    ValueError
        If neither duration, both durations, or a nonpositive duration is
        supplied.
    """
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
    """Create an epoch-stepped cosine learning-rate schedule.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate is scheduled.
    train_loader : collections.abc.Iterable
        Finite training loader used to convert iterations into epochs.
    epochs : int or None
        Epoch duration when training in epoch mode.
    iterations : int or None
        Update duration when training in iteration mode.

    Returns
    -------
    torch.optim.lr_scheduler.CosineAnnealingLR
        Scheduler with a period of at least one effective epoch.
    """
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


def prepare_model_for_fit(model: torch.nn.Module, train_loader: Iterable) -> None:
    """Run a model's optional whole-loader preparation hook.

    Custom experiment loops that call ``_train_step`` directly should invoke
    this helper once before their first optimizer step, matching
    :func:`fit_loop`.
    """
    prepare = getattr(model, "_prepare_fit", None)
    if prepare is not None:
        prepare(train_loader)


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
    """Run the shared fit loop around a model training hook.

    Parameters
    ----------
    model : torch.nn.Module
        Model providing a ``_train_step(optimizer, inputs, target)`` method.
    train_loader : collections.abc.Iterable
        Finite iterable of ``(inputs, targets)`` batches.
    optimizer : torch.optim.Optimizer
        Optimizer updated by the model training hook.
    epochs : int or None
        Number of complete loader passes.
    iterations : int or None
        Number of optimizer updates.
    use_tqdm : bool
        Whether to display a progress bar.
    return_loss : bool
        Whether to retain the scalar loss from every update.
    scheduler : torch.optim.lr_scheduler.LRScheduler or None, default=None
        Optional scheduler stepped once per effective epoch.

    Returns
    -------
    list of float
        Per-update losses when ``return_loss`` is true; otherwise an empty
        list.
    """
    validate_fit_mode(epochs=epochs, iterations=iterations)
    ensure_module_generators(model)
    prepare_model_for_fit(model, train_loader)

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
