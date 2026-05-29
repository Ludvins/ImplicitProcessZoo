"""Density normalizing flows for finite context function values."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..flows.conditional_flows import rq_spline_forward


class RQSplineCouplingLayer(nn.Module):
    """Rational-quadratic spline coupling layer with explicit inverse."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_bins=8,
        bound=3.0,
        device=None,
        dtype=None,
        init_scale=1e-3,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        self.input_dim = input_dim
        self.d_half = input_dim // 2
        self.d_out = input_dim - self.d_half
        self.num_bins = num_bins
        self.bound = bound

        per_dim = 3 * num_bins - 1
        self.net = nn.Sequential(
            nn.Linear(self.d_half, hidden_dim, dtype=dtype, device=device),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.d_out * per_dim, dtype=dtype, device=device),
        )
        self.net[-1].weight.data.normal_(0.0, init_scale)
        self.net[-1].bias.data.zero_()

    def _params(self, context):
        raw = self.net(context).reshape(
            *context.shape[:-1], self.d_out, 3 * self.num_bins - 1
        )
        k = self.num_bins
        widths_raw = raw[..., :k]
        heights_raw = raw[..., k : 2 * k]
        derivs_raw = raw[..., 2 * k :]

        widths = F.softmax(widths_raw, dim=-1)
        widths = 1e-3 + (1.0 - 1e-3 * k) * widths
        heights = F.softmax(heights_raw, dim=-1)
        heights = 1e-3 + (1.0 - 1e-3 * k) * heights

        inner = F.softplus(derivs_raw) + 1e-3
        ones = torch.ones(
            *inner.shape[:-1], 1, dtype=inner.dtype, device=inner.device
        )
        derivatives = torch.cat([ones, inner, ones], dim=-1)
        return widths, heights, derivatives

    def forward(self, x):
        x1 = x[..., : self.d_half]
        x2 = x[..., self.d_half :]
        widths, heights, derivatives = self._params(x1)
        y2, ldj = self._apply_spline(
            x2, widths, heights, derivatives, inverse=False
        )
        return torch.cat([x1, y2], dim=-1), ldj

    def inverse(self, y):
        y1 = y[..., : self.d_half]
        y2 = y[..., self.d_half :]
        widths, heights, derivatives = self._params(y1)
        x2, ldj = self._apply_spline(
            y2, widths, heights, derivatives, inverse=True
        )
        return torch.cat([y1, x2], dim=-1), ldj

    def _apply_spline(self, values, widths, heights, derivatives, inverse):
        original_shape = values.shape
        flat_values = values.reshape(-1)
        flat_widths = widths.reshape(-1, self.num_bins)
        flat_heights = heights.reshape(-1, self.num_bins)
        flat_derivatives = derivatives.reshape(-1, self.num_bins + 1)
        out, log_det = rq_spline_forward(
            flat_values,
            flat_widths,
            flat_heights,
            flat_derivatives,
            self.bound,
            inverse=inverse,
        )
        out = out.reshape(original_shape)
        log_det = log_det.reshape(original_shape).sum(dim=-1)
        return out, log_det


class ConditionalRQSplineCouplingLayer(nn.Module):
    """RQ-spline coupling layer whose parameters also depend on context inputs."""

    def __init__(
        self,
        input_dim,
        condition_dim,
        hidden_dim=128,
        num_bins=8,
        bound=3.0,
        device=None,
        dtype=None,
        init_scale=1e-3,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive.")
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.d_half = input_dim // 2
        self.d_out = input_dim - self.d_half
        self.num_bins = num_bins
        self.bound = bound

        per_dim = 3 * num_bins - 1
        self.net = nn.Sequential(
            nn.Linear(self.d_half + condition_dim, hidden_dim, dtype=dtype, device=device),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.d_out * per_dim, dtype=dtype, device=device),
        )
        self.net[-1].weight.data.normal_(0.0, init_scale)
        self.net[-1].bias.data.zero_()

    def _params(self, context, condition):
        if condition.shape[:-1] != context.shape[:-1]:
            raise ValueError(
                "condition batch shape must match context batch shape, got "
                f"{tuple(condition.shape[:-1])} and {tuple(context.shape[:-1])}."
            )
        raw = self.net(torch.cat([context, condition], dim=-1)).reshape(
            *context.shape[:-1], self.d_out, 3 * self.num_bins - 1
        )
        k = self.num_bins
        widths_raw = raw[..., :k]
        heights_raw = raw[..., k : 2 * k]
        derivs_raw = raw[..., 2 * k :]

        widths = F.softmax(widths_raw, dim=-1)
        widths = 1e-3 + (1.0 - 1e-3 * k) * widths
        heights = F.softmax(heights_raw, dim=-1)
        heights = 1e-3 + (1.0 - 1e-3 * k) * heights

        inner = F.softplus(derivs_raw) + 1e-3
        ones = torch.ones(
            *inner.shape[:-1], 1, dtype=inner.dtype, device=inner.device
        )
        derivatives = torch.cat([ones, inner, ones], dim=-1)
        return widths, heights, derivatives

    def forward(self, x, condition):
        x1 = x[..., : self.d_half]
        x2 = x[..., self.d_half :]
        widths, heights, derivatives = self._params(x1, condition)
        y2, ldj = self._apply_spline(
            x2, widths, heights, derivatives, inverse=False
        )
        return torch.cat([x1, y2], dim=-1), ldj

    def inverse(self, y, condition):
        y1 = y[..., : self.d_half]
        y2 = y[..., self.d_half :]
        widths, heights, derivatives = self._params(y1, condition)
        x2, ldj = self._apply_spline(
            y2, widths, heights, derivatives, inverse=True
        )
        return torch.cat([y1, x2], dim=-1), ldj

    def _apply_spline(self, values, widths, heights, derivatives, inverse):
        original_shape = values.shape
        flat_values = values.reshape(-1)
        flat_widths = widths.reshape(-1, self.num_bins)
        flat_heights = heights.reshape(-1, self.num_bins)
        flat_derivatives = derivatives.reshape(-1, self.num_bins + 1)
        out, log_det = rq_spline_forward(
            flat_values,
            flat_widths,
            flat_heights,
            flat_derivatives,
            self.bound,
            inverse=inverse,
        )
        out = out.reshape(original_shape)
        log_det = log_det.reshape(original_shape).sum(dim=-1)
        return out, log_det


class ContextDensityFlow(nn.Module):
    """Unconditional density flow over flattened context function values."""

    def __init__(
        self,
        input_dim,
        depth=4,
        hidden_dim=128,
        num_bins=8,
        bound=3.0,
        min_scale=1e-4,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__()
        if input_dim < 2:
            raise ValueError("ContextDensityFlow requires input_dim >= 2.")
        self.input_dim = input_dim
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.bound = bound
        self.min_scale = min_scale
        self.dtype = dtype
        self.device = device

        generator = torch.Generator(device if device is not None else "cpu")
        generator.manual_seed(seed)
        self.generator = generator

        self.layers = nn.ModuleList(
            [
                RQSplineCouplingLayer(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_bins=num_bins,
                    bound=bound,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )
        self.register_buffer("_flip_idx", torch.arange(input_dim - 1, -1, -1, device=device))
        self.register_buffer("loc", torch.zeros(input_dim, dtype=dtype, device=device))
        self.register_buffer("scale", torch.ones(input_dim, dtype=dtype, device=device))
        self.register_buffer(
            "_log2pi", torch.tensor(math.log(2.0 * math.pi), dtype=dtype, device=device)
        )

    @torch.no_grad()
    def set_standardization(self, samples):
        if samples.ndim != 2 or samples.shape[-1] != self.input_dim:
            raise ValueError(
                "standardization samples must have shape "
                f"[N, {self.input_dim}], got {tuple(samples.shape)}."
            )
        loc = samples.mean(dim=0)
        scale = samples.std(dim=0, unbiased=False).clamp_min(self.min_scale)
        self.loc.copy_(loc)
        self.scale.copy_(scale)

    def forward(self, z):
        x = z
        log_det = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in self.layers:
            x, ldj = layer(x)
            log_det = log_det + ldj
            x = x[..., self._flip_idx]
        return x, log_det

    def inverse(self, x):
        z = x
        log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for layer in reversed(self.layers):
            z = z[..., self._flip_idx]
            z, ldj = layer.inverse(z)
            log_det = log_det + ldj
        return z, log_det

    def log_prob(self, x):
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x must have shape [N, {self.input_dim}], got {tuple(x.shape)}."
            )
        x_std = (x - self.loc.view(1, -1)) / self.scale.view(1, -1)
        z, inverse_log_det = self.inverse(x_std)
        base_log_prob = -0.5 * (
            z.square().sum(dim=-1) + self.input_dim * self._log2pi
        )
        standardization_log_det = -self.scale.log().sum()
        return base_log_prob + inverse_log_det + standardization_log_det

    def nll(self, x):
        return -self.log_prob(x).mean()

    def sample(self, num_samples):
        z = torch.randn(
            num_samples,
            self.input_dim,
            generator=self.generator,
            dtype=self.dtype,
            device=self.loc.device,
        )
        x_std, _ = self.forward(z)
        return x_std * self.scale.view(1, -1) + self.loc.view(1, -1)


class ConditionalContextDensityFlow(nn.Module):
    """Conditional density flow over flattened context function values.

    The density input is a flattened vector ``f_C``.  The condition is the
    corresponding context input set ``C`` flattened and embedded by an MLP.
    This models finite marginals ``p(f_C | C)`` for a fixed context size.
    """

    def __init__(
        self,
        input_dim,
        condition_dim,
        depth=4,
        hidden_dim=128,
        condition_hidden_dim=None,
        condition_embedding_dim=None,
        num_bins=8,
        bound=3.0,
        min_scale=1e-4,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__()
        if input_dim < 2:
            raise ValueError("ConditionalContextDensityFlow requires input_dim >= 2.")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive.")
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.condition_hidden_dim = condition_hidden_dim or hidden_dim
        self.condition_embedding_dim = condition_embedding_dim or hidden_dim
        self.num_bins = num_bins
        self.bound = bound
        self.min_scale = min_scale
        self.dtype = dtype
        self.device = device

        generator = torch.Generator(device if device is not None else "cpu")
        generator.manual_seed(seed)
        self.generator = generator

        self.condition_net = nn.Sequential(
            nn.Linear(
                condition_dim,
                self.condition_hidden_dim,
                dtype=dtype,
                device=device,
            ),
            nn.Tanh(),
            nn.Linear(
                self.condition_hidden_dim,
                self.condition_embedding_dim,
                dtype=dtype,
                device=device,
            ),
            nn.Tanh(),
        )
        self.layers = nn.ModuleList(
            [
                ConditionalRQSplineCouplingLayer(
                    input_dim=input_dim,
                    condition_dim=self.condition_embedding_dim,
                    hidden_dim=hidden_dim,
                    num_bins=num_bins,
                    bound=bound,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )
        self.register_buffer("_flip_idx", torch.arange(input_dim - 1, -1, -1, device=device))
        self.register_buffer("loc", torch.zeros(input_dim, dtype=dtype, device=device))
        self.register_buffer("scale", torch.ones(input_dim, dtype=dtype, device=device))
        self.register_buffer(
            "_log2pi", torch.tensor(math.log(2.0 * math.pi), dtype=dtype, device=device)
        )

    @torch.no_grad()
    def set_standardization(self, samples):
        if samples.ndim != 2 or samples.shape[-1] != self.input_dim:
            raise ValueError(
                "standardization samples must have shape "
                f"[N, {self.input_dim}], got {tuple(samples.shape)}."
            )
        loc = samples.mean(dim=0)
        scale = samples.std(dim=0, unbiased=False).clamp_min(self.min_scale)
        self.loc.copy_(loc)
        self.scale.copy_(scale)

    def forward(self, z, condition):
        condition_features = self._condition_features(condition, z.shape[0])
        x = z
        log_det = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in self.layers:
            x, ldj = layer(x, condition_features)
            log_det = log_det + ldj
            x = x[..., self._flip_idx]
        return x, log_det

    def inverse(self, x, condition):
        condition_features = self._condition_features(condition, x.shape[0])
        z = x
        log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for layer in reversed(self.layers):
            z = z[..., self._flip_idx]
            z, ldj = layer.inverse(z, condition_features)
            log_det = log_det + ldj
        return z, log_det

    def log_prob(self, x, condition):
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x must have shape [N, {self.input_dim}], got {tuple(x.shape)}."
            )
        x_std = (x - self.loc.view(1, -1)) / self.scale.view(1, -1)
        z, inverse_log_det = self.inverse(x_std, condition)
        base_log_prob = -0.5 * (
            z.square().sum(dim=-1) + self.input_dim * self._log2pi
        )
        standardization_log_det = -self.scale.log().sum()
        return base_log_prob + inverse_log_det + standardization_log_det

    def nll(self, x, condition):
        return -self.log_prob(x, condition).mean()

    def sample(self, num_samples, condition):
        z = torch.randn(
            num_samples,
            self.input_dim,
            generator=self.generator,
            dtype=self.dtype,
            device=self.loc.device,
        )
        x_std, _ = self.forward(z, condition)
        return x_std * self.scale.view(1, -1) + self.loc.view(1, -1)

    def _condition_features(self, condition, batch_size):
        condition = condition.to(dtype=self.dtype, device=self.loc.device)
        if condition.ndim == 1:
            flat = condition.view(1, -1)
        elif condition.ndim == 2:
            if condition.shape[-1] == self.condition_dim:
                flat = condition
            else:
                flat = condition.reshape(1, -1)
        elif condition.ndim == 3:
            flat = condition.reshape(condition.shape[0], -1)
        else:
            raise ValueError(
                "condition must have shape [condition_dim], [M, input_dim], "
                f"or [N, M, input_dim], got {tuple(condition.shape)}."
            )
        if flat.shape[-1] != self.condition_dim:
            raise ValueError(
                "condition flattens to the wrong dimension: expected "
                f"{self.condition_dim}, got {flat.shape[-1]}."
            )
        if flat.shape[0] == 1 and batch_size != 1:
            flat = flat.expand(batch_size, -1)
        elif flat.shape[0] != batch_size:
            raise ValueError(
                f"condition batch size must be 1 or {batch_size}, got {flat.shape[0]}."
            )
        return self.condition_net(flat)
