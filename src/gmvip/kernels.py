from __future__ import annotations

import math

import torch
from torch import nn


def median_pairwise_distance(X: torch.Tensor) -> torch.Tensor:
    if X.ndim != 2:
        raise ValueError("X must have shape [N, D].")
    if X.shape[0] < 2:
        return torch.ones((), dtype=X.dtype, device=X.device)
    distances = torch.pdist(X)
    positive = distances[distances > 0]
    if positive.numel() == 0:
        return torch.ones((), dtype=X.dtype, device=X.device)
    return positive.median().clamp_min(torch.finfo(X.dtype).eps)


class RBFKernel(nn.Module):
    """Squared-exponential kernel with scalar or ARD lengthscale."""

    def __init__(
        self,
        input_dim: int,
        lengthscale: float | torch.Tensor = 1.0,
        outputscale: float | torch.Tensor = 1.0,
        ard: bool = True,
        learn_kernel: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.ard = bool(ard)
        factory_kwargs = {"device": device, "dtype": dtype}

        lengthscale_tensor = torch.as_tensor(lengthscale, **factory_kwargs).clone().detach()
        if self.ard:
            if lengthscale_tensor.ndim == 0:
                lengthscale_tensor = lengthscale_tensor.expand(self.input_dim).clone()
            if lengthscale_tensor.shape != (self.input_dim,):
                raise ValueError("ARD lengthscale must be scalar or shape [input_dim].")
        else:
            lengthscale_tensor = lengthscale_tensor.reshape(()).clone()

        outputscale_tensor = torch.as_tensor(outputscale, **factory_kwargs).reshape(()).clone().detach()
        self.log_lengthscale = nn.Parameter(lengthscale_tensor.clamp_min(1e-12).log())
        self.log_outputscale = nn.Parameter(outputscale_tensor.clamp_min(1e-12).log())

        if not learn_kernel:
            for param in self.parameters():
                param.requires_grad_(False)

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.log_lengthscale.exp().clamp_min(1e-12)

    @property
    def outputscale(self) -> torch.Tensor:
        return self.log_outputscale.exp().clamp_min(1e-12)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        X1_scaled = X1 / self.lengthscale
        X2_scaled = X2 / self.lengthscale
        sqdist = torch.cdist(X1_scaled, X2_scaled).square()
        return self.outputscale * torch.exp(-0.5 * sqdist)

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        return self.outputscale.expand(X.shape[0])


def initialize_rbf_lengthscale(
    inducing_points: torch.Tensor,
    ard: bool = True,
    value: float | torch.Tensor | str = "median",
) -> torch.Tensor:
    if isinstance(value, str):
        if value != "median":
            raise ValueError("Only init_lengthscale='median' is supported.")
        scalar = median_pairwise_distance(inducing_points) / math.sqrt(inducing_points.shape[1])
    else:
        scalar = torch.as_tensor(value, dtype=inducing_points.dtype, device=inducing_points.device)
    if ard:
        if scalar.ndim == 0:
            return scalar.expand(inducing_points.shape[1]).clone()
        return scalar.to(dtype=inducing_points.dtype, device=inducing_points.device)
    return scalar.reshape(())
