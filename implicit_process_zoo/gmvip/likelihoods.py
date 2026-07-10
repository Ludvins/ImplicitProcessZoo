import math

import torch
from torch import nn


class GaussianRegressionLikelihood(nn.Module):
    """Gaussian observation likelihood with optional learned noise scale.

    Parameters
    ----------
    init_log_noise : float, default=-2.0
        Initial log standard deviation.
    learn_noise : bool, default=True
        Whether the noise scale is trainable.
    min_log_noise : float, optional
        Lower bound applied to the log standard deviation.
    max_log_noise : float, optional
        Upper bound applied to the log standard deviation.
    device : torch.device or str, optional
        Parameter device.
    dtype : torch.dtype, optional
        Parameter data type.
    """

    def __init__(
        self,
        init_log_noise: float = -2.0,
        learn_noise: bool = True,
        min_log_noise: float | None = None,
        max_log_noise: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.min_log_noise = min_log_noise
        self.max_log_noise = max_log_noise
        value = torch.as_tensor(init_log_noise, device=device, dtype=dtype)
        if value.ndim > 1:
            raise ValueError("init_log_noise must be a scalar or one-dimensional tensor.")
        if learn_noise:
            self.log_noise = nn.Parameter(value.clone())
        else:
            self.register_buffer("log_noise", value.clone())

    @property
    def clamped_log_noise(self) -> torch.Tensor:
        log_noise = self.log_noise
        if self.min_log_noise is not None or self.max_log_noise is not None:
            log_noise = torch.clamp(log_noise, min=self.min_log_noise, max=self.max_log_noise)
        return log_noise

    @property
    def noise_std(self) -> torch.Tensor:
        return self.clamped_log_noise.exp()

    def log_prob(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Evaluate elementwise Gaussian log probabilities.

        Parameters
        ----------
        y : torch.Tensor
            Observed targets broadcastable to ``f``.
        f : torch.Tensor
            Latent function values.

        Returns
        -------
        torch.Tensor
            Elementwise log probabilities with the broadcast shape.
        """
        if y.ndim == 2 and y.shape[-1] == 1:
            y = y[..., 0]
        noise_var = torch.exp(2.0 * self.clamped_log_noise).clamp_min(1e-12)
        return -0.5 * (
            math.log(2.0 * math.pi) + torch.log(noise_var) + (y - f).square() / noise_var
        )
