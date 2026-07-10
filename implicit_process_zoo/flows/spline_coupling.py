"""Rational-quadratic spline coupling flow (Durkan et al. 2019, NSF).

Drop-in replacement for :class:`implicit_process_zoo.flows.flows.CouplingFlow` that uses
RQ-spline coupling layers instead of affine coupling. The constructor
signature and forward-return convention match ``CouplingFlow`` so FTIP can
consume either flow interchangeably.

Forward contract (same as CouplingFlow):
    flow(eps_flat) -> (a_flat, -LDJ)
where ``LDJ`` is log|det(Jacobian)|. The negative sign is what FTIP's KL
computation expects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.random import preserve_constructor_rng
from .conditional_flows import rq_spline_forward
from .flows import AffineLayer


class SplineCouplingLayer(nn.Module):
    """Single RQ-spline coupling layer.

    Identity on the first half ``z1``; the second half ``z2`` is transformed
    element-wise by a rational-quadratic spline whose parameters are predicted
    from ``z1`` by a small MLP.
    """

    def __init__(
        self,
        input_dim,
        num_bins=8,
        B=3.0,
        device=None,
        dtype=None,
        init_scale=1e-3,
        hidden_dim=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_half = input_dim // 2
        self.d_out = input_dim - self.d_half
        self.num_bins = num_bins
        self.B = B

        # Params per output dim: K widths + K heights + (K-1) inner derivs = 3K-1.
        per_dim = 3 * num_bins - 1
        hidden = hidden_dim if hidden_dim is not None else input_dim * 2
        self.net = nn.Sequential(
            nn.Linear(self.d_half, hidden, dtype=dtype, device=device),
            nn.Tanh(),
            nn.Linear(hidden, self.d_out * per_dim, dtype=dtype, device=device),
        )
        # Near-identity init so the flow starts close to the identity transform.
        self.net[-1].weight.data.normal_(0.0, init_scale)
        self.net[-1].bias.data.zero_()

    def _spline_params(self, context):
        # context: (N, d_half) -> (N, d_out, 3K - 1)
        raw = self.net(context).reshape(
            *context.shape[:-1],
            self.d_out,
            3 * self.num_bins - 1,
        )
        K = self.num_bins
        widths_raw = raw[..., :K]
        heights_raw = raw[..., K : 2 * K]
        derivs_raw = raw[..., 2 * K :]  # (N, d_out, K - 1)

        widths = F.softmax(widths_raw, dim=-1)
        widths = 1e-3 + (1 - 1e-3 * K) * widths
        heights = F.softmax(heights_raw, dim=-1)
        heights = 1e-3 + (1 - 1e-3 * K) * heights

        inner = F.softplus(derivs_raw) + 1e-3
        ones = torch.ones(*inner.shape[:-1], 1, dtype=inner.dtype, device=inner.device)
        derivatives = torch.cat([ones, inner, ones], dim=-1)  # (N, d_out, K + 1)
        return widths, heights, derivatives

    def forward(self, a):
        z1 = a[..., : self.d_half]
        z2 = a[..., self.d_half :]
        w, h, d = self._spline_params(z1)  # (N, d_out, *)

        # Flatten spline-batch dim so rq_spline_forward sees a 1-D stream.
        N = z2.shape[0]
        z2_flat = z2.reshape(-1)  # (N * d_out,)
        w_flat = w.reshape(-1, self.num_bins)
        h_flat = h.reshape(-1, self.num_bins)
        d_flat = d.reshape(-1, self.num_bins + 1)
        out_flat, ld_flat = rq_spline_forward(
            z2_flat,
            w_flat,
            h_flat,
            d_flat,
            self.B,
            inverse=False,
        )
        out = out_flat.reshape(N, self.d_out)
        ldj = ld_flat.reshape(N, self.d_out).sum(dim=-1)  # (N,)
        return torch.cat([z1, out], dim=-1), ldj


@preserve_constructor_rng
class SplineCouplingFlow(nn.Module):
    """Stack of RQ-spline coupling layers with bit-reversal between them.

    Signature matches :class:`implicit_process_zoo.flows.flows.CouplingFlow` (plus optional
    ``num_bins`` / ``B``). Returns ``(out, -LDJ)``.
    """

    def __init__(
        self,
        depth,
        input_dim,
        device,
        dtype,
        seed,
        init_scale=1e-3,
        num_bins=8,
        B=3.0,
        hidden_dim=None,
    ):
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        self.biyections = nn.ModuleList(
            [
                SplineCouplingLayer(
                    input_dim=input_dim,
                    num_bins=num_bins,
                    B=B,
                    device=device,
                    dtype=dtype,
                    init_scale=init_scale,
                    hidden_dim=hidden_dim,
                )
                for _ in range(depth)
            ]
        )
        self.register_buffer(
            "_flip_idx",
            torch.arange(input_dim - 1, -1, -1, device=device),
        )
        # Match CouplingFlow's affine-attribute pattern so FTIP warm-start works.
        self._modules["affine"] = None

    def set_affine(self, shift, L_flat, learnable=False):
        """Attach an affine pre-transform (used by FTIP's warm-start from VIP)."""
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
        LDJ = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
        if self.affine is not None:
            a, ldj = self.affine(a)
            LDJ += ldj
        for b in self.biyections:
            a, ldj = b(a)
            a = a[..., self._flip_idx]
            LDJ += ldj
        return a, -LDJ  # FTIP sign convention
