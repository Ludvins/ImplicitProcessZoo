from __future__ import annotations

import math

import torch
from torch import nn


def _scale_tril_from_raw(
    raw_tril: torch.Tensor,
    num_inducing: int,
    min_log_diag: float | None,
    max_log_diag: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a lower Cholesky factor with positive diagonal from raw entries."""

    idx = torch.tril_indices(num_inducing, num_inducing, device=raw_tril.device)
    scale_tril = raw_tril.new_zeros(*raw_tril.shape[:-1], num_inducing, num_inducing)
    scale_tril[..., idx[0], idx[1]] = raw_tril
    diag = torch.diagonal(scale_tril, dim1=-2, dim2=-1)
    log_diag = diag.clone()
    if min_log_diag is not None or max_log_diag is not None:
        log_diag = torch.clamp(log_diag, min=min_log_diag, max=max_log_diag)
    diag.copy_(log_diag.exp())
    return scale_tril, log_diag


def _standard_normal_sample(
    num_samples: int,
    dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
    antithetic: bool = False,
) -> torch.Tensor:
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not antithetic or num_samples == 1:
        return torch.randn(
            num_samples,
            int(dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )

    num_pairs = num_samples // 2
    eps = torch.randn(
        num_pairs,
        int(dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    samples = [eps, -eps]
    if num_samples % 2:
        samples.append(
            torch.randn(
                1,
                int(dim),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
    return torch.cat(samples, dim=0)


class CholeskyGaussianCoefficientPosterior(nn.Module):
    """Full-covariance Gaussian posterior over whitened inducing coefficients."""

    def __init__(
        self,
        num_inducing: int,
        output_dim: int = 1,
        init_mean: float = 0.0,
        init_log_std: float = 0.0,
        min_log_std: float | None = -8.0,
        max_log_std: float | None = 4.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if num_inducing <= 0:
            raise ValueError("num_inducing must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        self.num_inducing = int(num_inducing)
        self.output_dim = int(output_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.num_tril_entries = self.num_inducing * (self.num_inducing + 1) // 2
        loc_shape = (
            (self.num_inducing,)
            if self.output_dim == 1
            else (self.num_inducing, self.output_dim)
        )
        raw_shape = (
            (self.num_tril_entries,)
            if self.output_dim == 1
            else (self.output_dim, self.num_tril_entries)
        )
        self.loc = nn.Parameter(torch.full(loc_shape, float(init_mean), device=device, dtype=dtype))
        raw = torch.zeros(raw_shape, device=device, dtype=dtype)
        idx = torch.tril_indices(self.num_inducing, self.num_inducing, device=raw.device)
        raw[..., idx[0] == idx[1]] = float(init_log_std)
        self.raw_scale_tril = nn.Parameter(raw)

    @property
    def mean(self) -> torch.Tensor:
        return self.loc

    @property
    def log_std(self) -> torch.Tensor:
        log_diag = self._scale_tril_and_log_diag()[1]
        if self.output_dim == 1:
            return log_diag
        return log_diag.T

    @property
    def clamped_log_std(self) -> torch.Tensor:
        return self.log_std

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    @property
    def scale_tril(self) -> torch.Tensor:
        return self._scale_tril_and_log_diag()[0]

    def _scale_tril_and_log_diag(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _scale_tril_from_raw(
            self.raw_scale_tril,
            self.num_inducing,
            self.min_log_std,
            self.max_log_std,
        )

    def rsample(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
    ) -> torch.Tensor:
        eps = _standard_normal_sample(
            num_samples,
            self.num_inducing * self.output_dim,
            dtype=self.loc.dtype,
            device=self.loc.device,
            generator=generator,
            antithetic=antithetic,
        )
        if self.output_dim == 1:
            return self.loc.unsqueeze(0) + eps.matmul(self.scale_tril.T)
        eps = eps.reshape(int(num_samples), self.output_dim, self.num_inducing)
        samples = self.loc.T.unsqueeze(0) + torch.einsum("skm,kjm->skj", eps, self.scale_tril)
        return samples.transpose(1, 2)

    def rsample_with_kl(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
    ):
        samples = self.rsample(num_samples, generator=generator, antithetic=antithetic)
        kl = self.kl_to_standard_normal()
        diagnostics = {
            "q_std_mean": self.std.mean(),
            "coefficient_displacement": self.loc.square().mean(),
        }
        return samples, kl.expand(int(num_samples)), diagnostics

    def sample_prior(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
    ) -> torch.Tensor:
        samples = _standard_normal_sample(
            num_samples,
            self.num_inducing * self.output_dim,
            dtype=self.loc.dtype,
            device=self.loc.device,
            generator=generator,
            antithetic=antithetic,
        )
        if self.output_dim == 1:
            return samples.reshape(int(num_samples), self.num_inducing)
        return samples.reshape(int(num_samples), self.output_dim, self.num_inducing).transpose(1, 2)

    def kl_to_standard_normal(self) -> torch.Tensor:
        scale_tril, log_diag = self._scale_tril_and_log_diag()
        trace_cov = scale_tril.square().sum()
        logdet_cov = 2.0 * log_diag.sum()
        return 0.5 * (
            trace_cov
            + self.loc.square().sum()
            - float(self.num_inducing * self.output_dim)
            - logdet_cov
        )


def _standard_normal_log_prob(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * (math.log(2.0 * math.pi) + x.square())


def _zero_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


class AffineCouplingLayer(nn.Module):
    """RealNVP affine coupling layer for coefficient vectors."""

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        scale_bound: float = 2.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if dim <= 1:
            raise ValueError("RealNVP coupling needs dim > 1.")
        if scale_bound <= 0:
            raise ValueError("scale_bound must be positive.")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.dim = int(dim)
        self.scale_bound = float(scale_bound)
        self.register_buffer(
            "mask",
            mask.to(device=device, dtype=dtype).reshape(1, self.dim),
        )
        self.net = _make_mlp(
            input_dim=self.dim,
            hidden_dim=int(hidden_dim),
            output_dim=2 * self.dim,
            num_layers=int(num_layers),
            dropout=float(dropout),
        ).to(**factory_kwargs)
        _zero_last_linear(self.net)

    def _shift_log_scale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_mask = 1.0 - self.mask
        shift, raw_log_scale = self.net(x * self.mask).chunk(2, dim=-1)
        log_scale = torch.tanh(raw_log_scale) * self.scale_bound
        return shift * inv_mask, log_scale * inv_mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self._shift_log_scale(x)
        inv_mask = 1.0 - self.mask
        y = x * self.mask + inv_mask * (x * torch.exp(log_scale) + shift)
        log_abs_det = log_scale.sum(dim=-1)
        return y, log_abs_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self._shift_log_scale(y)
        inv_mask = 1.0 - self.mask
        x = y * self.mask + inv_mask * ((y - shift) * torch.exp(-log_scale))
        log_abs_det = -log_scale.sum(dim=-1)
        return x, log_abs_det


class RealNVPCoefficientPosterior(nn.Module):
    """RealNVP posterior over whitened inducing coefficients ``a``."""

    def __init__(
        self,
        num_inducing: int,
        num_flows: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        scale_bound: float = 2.0,
        kl_num_samples: int = 256,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if num_inducing <= 1:
            raise ValueError("RealNVP posterior requires num_inducing > 1.")
        if num_flows <= 0:
            raise ValueError("num_flows must be positive.")
        self.num_inducing = int(num_inducing)
        self.num_flows = int(num_flows)
        self.kl_num_samples = int(kl_num_samples)
        factory_kwargs = {"device": device, "dtype": dtype}

        base_mask = (torch.arange(self.num_inducing, device=device) % 2).to(dtype=dtype)
        layers = []
        for idx in range(self.num_flows):
            mask = base_mask if idx % 2 == 0 else 1.0 - base_mask
            layers.append(
                AffineCouplingLayer(
                    dim=self.num_inducing,
                    mask=mask,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    scale_bound=scale_bound,
                    **factory_kwargs,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.register_buffer("_dummy", torch.empty((), **factory_kwargs), persistent=False)

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def _base_sample(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
    ) -> torch.Tensor:
        return _standard_normal_sample(
            num_samples,
            self.num_inducing,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
            antithetic=antithetic,
        )

    def forward_transform(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z.to(dtype=self.dtype, device=self.device)
        log_abs_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for layer in self.layers:
            x, layer_log_det = layer(x)
            log_abs_det = log_abs_det + layer_log_det
        return x, log_abs_det

    def inverse_transform(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = a.to(dtype=self.dtype, device=self.device)
        log_abs_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for layer in reversed(self.layers):
            x, layer_log_det = layer.inverse(x)
            log_abs_det = log_abs_det + layer_log_det
        return x, log_abs_det

    def rsample(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
    ) -> torch.Tensor:
        z = self._base_sample(num_samples, generator=generator, antithetic=antithetic)
        a, _ = self.forward_transform(z)
        return a

    def rsample_with_kl(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
    ):
        z = self._base_sample(num_samples, generator=generator, antithetic=antithetic)
        a, forward_log_det = self.forward_transform(z)
        log_q = _standard_normal_log_prob(z).sum(dim=-1) - forward_log_det
        log_p = _standard_normal_log_prob(a).sum(dim=-1)
        kl_terms = log_q - log_p
        diagnostics = {
            "q_std_mean": a.std(dim=0, unbiased=False).mean(),
            "coefficient_displacement": a.mean(dim=0).square().mean(),
            "flow_logdet_mean": forward_log_det.mean(),
            "flow_kl_std": kl_terms.std(unbiased=False),
        }
        return a, kl_terms, diagnostics

    def sample_prior(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
    ) -> torch.Tensor:
        return self._base_sample(num_samples, generator=generator, antithetic=antithetic)

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        z, inverse_log_det = self.inverse_transform(a)
        return _standard_normal_log_prob(z).sum(dim=-1) + inverse_log_det

    def kl_to_standard_normal(
        self,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
    ) -> torch.Tensor:
        _, kl_terms, _ = self.rsample_with_kl(
            int(num_samples or self.kl_num_samples),
            generator=generator,
            antithetic=antithetic,
        )
        return kl_terms.mean()


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for _ in range(max(0, int(num_layers) - 1)):
        layers.extend(
            [
                nn.Linear(current_dim, int(hidden_dim)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            ]
        )
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, int(output_dim)))
    return nn.Sequential(*layers)
