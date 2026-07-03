from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


def _clone_noise(noise):
    if torch.is_tensor(noise):
        return noise.detach().clone()
    if isinstance(noise, tuple):
        return tuple(_clone_noise(item) for item in noise)
    if isinstance(noise, list):
        return [_clone_noise(item) for item in noise]
    return noise


def _noise_to(noise, *, device, dtype):
    if torch.is_tensor(noise):
        return noise.to(device=device, dtype=dtype)
    if isinstance(noise, tuple):
        return tuple(_noise_to(item, device=device, dtype=dtype) for item in noise)
    if isinstance(noise, list):
        return [_noise_to(item, device=device, dtype=dtype) for item in noise]
    return noise


def _squeeze_scalar_output(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(
            "Scalar prior samples must have shape [S, N] or [S, N, 1], "
            f"got {tuple(values.shape)}."
        )
    return values


def _has_first_call_get_noise(module: nn.Module) -> bool:
    if not hasattr(module, "get_noise"):
        return False
    try:
        signature = inspect.signature(module.get_noise)
    except (TypeError, ValueError):
        return False
    return "first_call" in signature.parameters


@contextmanager
def _temporarily_disable_parameter_grads(module: nn.Module):
    states = [(param, param.requires_grad) for param in module.parameters()]
    try:
        for param, _ in states:
            param.requires_grad_(False)
        yield
    finally:
        for param, requires_grad in states:
            param.requires_grad_(requires_grad)


@dataclass
class ModuleNoiseLatents:
    num_samples: int
    module_noises: list[tuple[nn.Module, object]]


class CoherentPriorFunctionSampler:
    """Thin adapter for coherent finite-dimensional prior function samples.

    The adapter first delegates to a prior that already exposes
    ``sample_latents/evaluate_latents`` or ``sample_functions``. Otherwise it
    supports this repository's BayesianNN-style priors by snapshotting each
    stochastic layer's weight-noise tensors and replaying them for every input
    set evaluation.
    """

    def __init__(self, prior: nn.Module):
        self.prior = prior
        if hasattr(prior, "sample_latents") and hasattr(prior, "evaluate_latents"):
            self.mode = "latents"
            self.stochastic_modules = []
        elif hasattr(prior, "sample_functions"):
            self.mode = "functions"
            self.stochastic_modules = []
        else:
            modules = [
                module
                for module in prior.modules()
                if module is not prior and hasattr(module, "num_samples") and _has_first_call_get_noise(module)
            ]
            if not modules:
                raise ValueError(
                    "Prior must support coherent function samples. Need "
                    "sample_latents/evaluate_latents, sample_functions, or "
                    "BayesianNN-style stochastic layers with get_noise(first_call=...)."
                )
            self.mode = "module_noise"
            self.stochastic_modules = modules

    def sample_latents(self, num_samples: int, seed: int | None = None):
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        if self.mode == "latents":
            try:
                return self.prior.sample_latents(num_samples, seed=seed)
            except TypeError:
                return self.prior.sample_latents(num_samples)

        if self.mode == "functions":
            try:
                return self.prior.sample_functions(num_samples, seed=seed)
            except TypeError:
                return self.prior.sample_functions(num_samples)

        if seed is None:
            base_seed = int(torch.randint(0, 2**31 - 1, (), device="cpu").item())
        else:
            base_seed = int(seed)
        module_noises = []
        for index, module in enumerate(self.stochastic_modules):
            old_num_samples = module.num_samples
            old_noise = getattr(module, "noise", None)
            sampler_states = []
            for sampler_name in ("gaussian_sampler", "uniform_sampler"):
                sampler = getattr(module, sampler_name, None)
                generator = getattr(sampler, "generator", None)
                if generator is not None:
                    sampler_states.append((generator, generator.get_state()))
                    generator.manual_seed(base_seed + 1009 * index + len(sampler_states))
            try:
                module.num_samples = num_samples
                noise = module.get_noise(first_call=True)
                module_noises.append((module, _clone_noise(noise)))
            finally:
                module.num_samples = old_num_samples
                if hasattr(module, "noise"):
                    module.noise = old_noise
                for generator, state in sampler_states:
                    generator.set_state(state)
        return ModuleNoiseLatents(num_samples=num_samples, module_noises=module_noises)

    def evaluate_latents(self, latents, X: torch.Tensor) -> torch.Tensor:
        if self.mode == "latents":
            values = self.prior.evaluate_latents(latents, X)
            return _squeeze_scalar_output(values)

        if self.mode == "functions":
            values = torch.stack([fn(X) for fn in latents], dim=0)
            return _squeeze_scalar_output(values)

        old_states = []
        was_training = self.prior.training
        try:
            self.prior.eval()
            for module in self.prior.modules():
                if hasattr(module, "num_samples"):
                    old_states.append(
                        (
                            module,
                            module.num_samples,
                            getattr(module, "noise", None),
                            getattr(module, "fix_random_noise", None),
                        )
                    )
                    module.num_samples = latents.num_samples

            for module, noise in latents.module_noises:
                module.noise = _noise_to(noise, device=X.device, dtype=X.dtype)
                if hasattr(module, "fix_random_noise"):
                    module.fix_random_noise = True

            try:
                values = self.prior(X)
            except TypeError:
                values = self.prior(X, latents.num_samples)
        finally:
            self.prior.train(was_training)
            for module, num_samples, noise, fix_random_noise in reversed(old_states):
                module.num_samples = num_samples
                if hasattr(module, "noise"):
                    module.noise = noise
                if fix_random_noise is not None and hasattr(module, "fix_random_noise"):
                    module.fix_random_noise = fix_random_noise

        return _squeeze_scalar_output(values)

    def sample_values(
        self,
        X: torch.Tensor,
        num_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        latents = self.sample_latents(num_samples, seed=seed)
        return self.evaluate_latents(latents, X)


class PriorFunctionBank:
    """Fixed bank of coherent prior functions used for empirical statistics."""

    def __init__(
        self,
        prior: nn.Module,
        num_bank_samples: int,
        seed: int | None = None,
        detach: bool = True,
        detach_prior_grad: bool = False,
    ):
        self.prior = prior
        self.num_bank_samples = int(num_bank_samples)
        self.detach = bool(detach)
        self.detach_prior_grad = bool(detach_prior_grad)
        self.sampler = CoherentPriorFunctionSampler(prior)
        self.latents = self.sampler.sample_latents(self.num_bank_samples, seed=seed)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        if self.detach_prior_grad:
            with _temporarily_disable_parameter_grads(self.prior):
                values = self.sampler.evaluate_latents(self.latents, X)
        else:
            values = self.sampler.evaluate_latents(self.latents, X)
        if self.detach:
            values = values.detach()
        return values
