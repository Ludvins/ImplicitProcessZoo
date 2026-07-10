"""Glow-style invertible 1x1 mixing (LU-parameterized) + combined spline flow.

Provides:
  - :class:`InvertibleConv1x1LU` — learnable invertible linear layer in LU form.
  - :class:`SplineCoupling1x1Flow` — spline coupling layers interleaved with
    invertible 1x1 LU layers, replacing the fixed bit-reversal permutation
    used by :class:`SplineCouplingFlow`.

Same constructor signature + ``(out, -LDJ)`` forward convention as the other
flows in this module, so FTIP can consume it interchangeably.
"""

import torch
import torch.nn as nn

from ..utils.random import fork_torch_rng, preserve_constructor_rng
from .flows import AffineLayer
from .spline_coupling import SplineCouplingLayer


class InvertibleConv1x1LU(nn.Module):
    """Glow-style invertible linear layer with LU parameterization.

    W = P @ L @ (U + diag(sign_s * exp(log_s)))

    - ``P``: fixed permutation (from LU decomposition at init)
    - ``L``: lower-triangular with unit diagonal (below-diagonal entries are learnable)
    - ``U``: upper-triangular with zero diagonal (above-diagonal entries are learnable)
    - ``sign_s``: fixed sign of the diagonal (prevents det W from switching sign)
    - ``log_s``: learnable log-magnitude of the diagonal of U

    log|det W| = sum(log_s), independent of batch.

    Parameters
    ----------
    input_dim : int
        Number of coefficient dimensions.
    device : torch.device or str or None, default=None
        Device used for parameters and buffers.
    dtype : torch.dtype or None, default=None
        Floating-point dtype used for parameters and buffers.
    seed : int or None, default=None
        Optional deterministic initialization seed.
    """

    def __init__(self, input_dim, device=None, dtype=None, seed=None):
        super().__init__()
        self.input_dim = input_dim

        # Random rotation → decompose as P L U.
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.seed()
        W_init = torch.linalg.qr(
            torch.randn(
                input_dim,
                input_dim,
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )[0]
        LU, pivots = torch.linalg.lu_factor(W_init)
        P, L, U = torch.lu_unpack(LU, pivots)
        s = torch.diagonal(U).clone()
        U = U - torch.diag(s)  # zero the U diagonal

        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s) + 1e-8)

        # Fixed buffers.
        self.register_buffer("P", P)
        self.register_buffer("sign_s", sign_s)
        self.register_buffer("l_mask", torch.tril(torch.ones_like(W_init), diagonal=-1))
        self.register_buffer("u_mask", torch.triu(torch.ones_like(W_init), diagonal=1))
        self.register_buffer("eye", torch.eye(input_dim, dtype=dtype, device=device))

        # Trainable params.
        self.L = nn.Parameter(L.contiguous())
        self.U = nn.Parameter(U.contiguous())
        self.log_s = nn.Parameter(log_s.contiguous())

    def _W(self):
        L_full = self.L * self.l_mask + self.eye
        U_full = self.U * self.u_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.P @ L_full @ U_full

    def forward(self, a):
        """Mix coefficients with the learned invertible matrix.

        Parameters
        ----------
        a : torch.Tensor
            Coefficients with shape ``[S, input_dim]``.

        Returns
        -------
        transformed : torch.Tensor
            Linearly mixed coefficients with the same shape as ``a``.
        log_abs_det : torch.Tensor
            Per-sample log absolute determinant of the mixing matrix.
        """
        W = self._W()
        out = a @ W.t()
        ldj = self.log_s.sum().expand(a.shape[0])
        return out, ldj


@preserve_constructor_rng
class SplineCoupling1x1Flow(nn.Module):
    """Spline coupling interleaved with invertible 1x1 LU mixing.

    Replaces :class:`SplineCouplingFlow`'s fixed bit-reversal permutation
    with a learnable invertible linear layer between every coupling.

    The constructor matches :class:`CouplingFlow` with spline-specific options.

    Parameters
    ----------
    depth : int
        Number of spline/mixing blocks.
    input_dim : int
        Number of coefficient dimensions.
    device : torch.device or str
        Device used for parameters and the owned generator.
    dtype : torch.dtype
        Floating-point dtype used for parameters.
    seed : int
        Seed for deterministic initialization and sampling state.
    init_scale : float, default=1e-3
        Standard deviation of final-layer weights.
    num_bins : int, default=8
        Number of rational-quadratic spline bins.
    B : float, default=3.0
        Half-width of the spline interval.
    """

    def __init__(self, depth, input_dim, device, dtype, seed, init_scale=1e-3, num_bins=8, B=3.0):
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        with fork_torch_rng(seed):
            self.biyections = nn.ModuleList(
                [
                    SplineCouplingLayer(
                        input_dim=input_dim,
                        num_bins=num_bins,
                        B=B,
                        device=device,
                        dtype=dtype,
                        init_scale=init_scale,
                    )
                    for _ in range(depth)
                ]
            )
            self.conv1x1s = nn.ModuleList(
                [
                    InvertibleConv1x1LU(
                        input_dim,
                        device=device,
                        dtype=dtype,
                        seed=int(seed) + index,
                    )
                    for index in range(depth)
                ]
            )
        self._modules["affine"] = None

    def set_affine(self, shift, L_flat, learnable=False):
        """Attach the affine pre-transform used by VIP warm starts.

        Parameters
        ----------
        shift : torch.Tensor
            Shift vector with shape ``[input_dim]``.
        L_flat : torch.Tensor
            Flattened lower triangle of the affine factor.
        learnable : bool, default=False
            Whether to optimize the affine parameters.
        """
        self.affine = AffineLayer(
            self.input_dim,
            self.device,
            self.dtype,
            learnable=learnable,
        )
        if learnable:
            self.affine.shift.data.copy_(shift.detach())
            self.affine.L_flat.data.copy_(L_flat.detach())
        else:
            self.affine.shift.copy_(shift.detach())
            self.affine.L_flat.copy_(L_flat.detach())

    def forward(self, a):
        """Transform coefficients through spline and mixing blocks.

        Parameters
        ----------
        a : torch.Tensor
            Base coefficients with shape ``[S, input_dim]``.

        Returns
        -------
        transformed : torch.Tensor
            Flow-transformed coefficients with shape ``[S, input_dim]``.
        negative_log_det : torch.Tensor
            Negative forward log-Jacobian determinant expected by FTIP.
        """
        LDJ = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
        if self.affine is not None:
            a, ldj = self.affine(a)
            LDJ += ldj
        for coupling, conv in zip(self.biyections, self.conv1x1s):
            a, ldj = coupling(a)
            LDJ += ldj
            a, ldj = conv(a)
            LDJ += ldj
        return a, -LDJ
