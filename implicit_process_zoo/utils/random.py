"""Random-number helpers that avoid hidden global state changes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from inspect import signature

import torch


def preserve_constructor_rng(cls):
    """Class decorator that isolates seeded module initialization."""
    original = cls.__init__
    init_signature = signature(original)

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        bound = init_signature.bind_partial(self, *args, **kwargs)
        seed = bound.arguments.get("seed")
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(int(seed))
            original(self, *args, **kwargs)

    cls.__init__ = wrapped
    return cls


def standard_normal_samples(
    num_samples: int,
    *sample_shape: int,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    antithetic: bool = False,
) -> torch.Tensor:
    """Draw exactly ``num_samples`` standard-normal samples.

    With antithetic sampling, paired draws are emitted first and a final
    independent draw is appended when the requested count is odd.

    Parameters
    ----------
    num_samples : int
        Exact number of samples to return.
    *sample_shape : int
        Dimensions following the leading sample axis.
    dtype : torch.dtype
        Floating-point dtype of the returned tensor.
    device : torch.device or str
        Device on which samples are generated.
    generator : torch.Generator or None, default=None
        Optional generator controlling the draws.
    antithetic : bool, default=False
        Whether to emit positive/negative sample pairs.

    Returns
    -------
    torch.Tensor
        Standard-normal tensor with shape ``[num_samples, *sample_shape]``.

    Raises
    ------
    ValueError
        If ``num_samples`` is not positive.
    """
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    shape = tuple(int(value) for value in sample_shape)
    if not antithetic or num_samples == 1:
        return torch.randn(
            (num_samples, *shape),
            dtype=dtype,
            device=device,
            generator=generator,
        )

    num_pairs = num_samples // 2
    paired = torch.randn(
        (num_pairs, *shape),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    parts = [paired, -paired]
    if num_samples % 2:
        parts.append(
            torch.randn(
                (1, *shape),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
    return torch.cat(parts, dim=0)


def ensure_generator_device(
    module: torch.nn.Module,
    device: torch.device | str,
    *,
    attribute: str = "generator",
) -> torch.Generator | None:
    """Recreate an owned generator after the module moves between devices."""
    generator = getattr(module, attribute, None)
    if not isinstance(generator, torch.Generator):
        return None
    target = torch.device(device)
    if generator.device == target:
        return generator

    state = generator.get_state()
    replacement = torch.Generator(device=target)
    try:
        replacement.set_state(state)
    except RuntimeError:
        # Generator state layouts may differ between CPU and accelerator
        # backends. A deterministic seed is safer than retaining a generator
        # bound to the wrong device.
        replacement.manual_seed(int(state[:8].to(torch.int64).sum().item()))
    setattr(module, attribute, replacement)
    return replacement


def capture_generator_states(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Capture all generators owned directly by modules in a hierarchy."""
    states: dict[str, torch.Tensor] = {}
    for module_name, child in module.named_modules():
        generator = getattr(child, "generator", None)
        if isinstance(generator, torch.Generator):
            key = f"{module_name}.generator" if module_name else "generator"
            states[key] = generator.get_state().cpu()
    return states


def ensure_module_generators(module: torch.nn.Module) -> None:
    """Align every owned generator with its module's parameter/buffer device."""
    for child in module.modules():
        generator = getattr(child, "generator", None)
        if not isinstance(generator, torch.Generator):
            continue
        reference = next(child.parameters(recurse=False), None)
        if reference is None:
            reference = next(child.buffers(recurse=False), None)
        if reference is None:
            reference = next(child.parameters(), None)
        if reference is None:
            reference = next(child.buffers(), None)
        if reference is not None:
            ensure_generator_device(child, reference.device)


def restore_generator_states(
    module: torch.nn.Module,
    states: dict[str, torch.Tensor],
) -> None:
    """Restore generator states captured by :func:`capture_generator_states`."""
    modules = dict(module.named_modules())
    modules[""] = module
    for key, state in states.items():
        module_name, _, attribute = key.rpartition(".")
        child = modules.get(module_name)
        if child is None or attribute not in {"generator", ""}:
            continue
        attribute = attribute or "generator"
        generator = getattr(child, attribute, None)
        if isinstance(generator, torch.Generator):
            generator.set_state(state)


@contextmanager
def temporary_generator_seed(
    module: torch.nn.Module,
    seed: int | None,
) -> Iterator[None]:
    """Temporarily seed model-owned generators without advancing callers."""
    ensure_module_generators(module)
    if seed is None:
        yield
        return
    saved = capture_generator_states(module)
    for child in module.modules():
        generator = getattr(child, "generator", None)
        if isinstance(generator, torch.Generator):
            generator.manual_seed(int(seed))
    try:
        yield
    finally:
        restore_generator_states(module, saved)


@contextmanager
def fork_torch_rng(seed: int | None = None) -> Iterator[None]:
    """Isolate constructor initialization from the process-wide RNG."""
    if seed is None:
        yield
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(int(seed))
        yield
