import torch

from ..utils.random import preserve_constructor_rng


class AffineLayer(torch.nn.Module):
    """Apply a learnable or fixed lower-triangular affine transform.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input.
    device : torch.device or str or None
        Device used for parameters and buffers.
    dtype : torch.dtype
        Floating-point dtype used for parameters and buffers.
    learnable : bool, default=True
        If false, register the shift and factor as buffers rather than
        parameters.
    """

    def __init__(self, input_dim, device, dtype, learnable=True):
        super().__init__()
        self.input_dim = input_dim
        self.learnable = learnable

        shift = torch.zeros(input_dim, dtype=dtype, device=device)
        # Store lower-triangular elements; initialise to identity
        li, lj = torch.tril_indices(input_dim, input_dim)
        L_init = torch.eye(input_dim, dtype=dtype, device=device)
        L_flat = L_init[li, lj].contiguous()

        if learnable:
            self.shift = torch.nn.Parameter(shift)
            self.L_flat = torch.nn.Parameter(L_flat)
        else:
            self.register_buffer("shift", shift)
            self.register_buffer("L_flat", L_flat)

        self.register_buffer("_tril_row", li)
        self.register_buffer("_tril_col", lj)
        # Indices of diagonal elements inside L_flat
        import numpy as np

        self.register_buffer(
            "_diag_idx",
            torch.tensor(
                np.cumsum(np.arange(1, input_dim + 1)) - 1,
                dtype=torch.long,
                device=device,
            ),
        )

    def forward(self, eps):
        """Transform base samples and evaluate the log-Jacobian determinant.

        Parameters
        ----------
        eps : torch.Tensor
            Base samples with shape ``[..., input_dim]``.

        Returns
        -------
        transformed : torch.Tensor
            Affine-transformed samples with the same shape as ``eps``.
        log_abs_det : torch.Tensor
            Log absolute determinant for each leading sample.
        """
        # Reconstruct L from flat triangular elements
        L = torch.zeros(
            self.input_dim,
            self.input_dim,
            dtype=eps.dtype,
            device=eps.device,
        )
        L[self._tril_row, self._tril_col] = self.L_flat

        # a = eps @ L + shift — VIP convention: Cov[a] = L^T @ L
        a = eps @ L + self.shift

        # log|det(L)| = sum(log|diag(L)|)
        log_abs_det = torch.sum(torch.log(torch.abs(self.L_flat[self._diag_idx])))
        ldj = log_abs_det.expand(eps.shape[0])
        return a, ldj


class CouplingLayer(torch.nn.Module):
    """Apply one near-identity affine coupling transform.

    Parameters
    ----------
    input_dim : int
        Number of coefficient dimensions.
    device : torch.device or str or None
        Device used for neural-network parameters.
    dtype : torch.dtype
        Floating-point dtype used for neural-network parameters.
    init_scale : float, default=1e-3
        Standard deviation of the final-layer weight initialization.
    """

    def __init__(self, input_dim, device, dtype, init_scale=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.d_half = input_dim // 2
        self.d_out = input_dim - self.d_half

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.d_half, input_dim * 2, dtype=dtype, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(input_dim * 2, 2 * self.d_out, dtype=dtype, device=device),
        )
        # Near-identity init: small random weights so gradients flow to all
        # layers from step 1, while the transform stays close to identity.
        self.nn[-1].weight.data.normal_(0, init_scale)
        self.nn[-1].bias.data.fill_(0)

    def forward(self, a):
        """Transform the second coefficient block conditionally on the first.

        Parameters
        ----------
        a : torch.Tensor
            Coefficients with shape ``[..., input_dim]``.

        Returns
        -------
        transformed : torch.Tensor
            Coupling-transformed coefficients with the same shape as ``a``.
        log_abs_det : torch.Tensor
            Per-sample forward log absolute Jacobian determinant.
        """
        z1 = a[..., : self.d_half]
        z2 = a[..., self.d_half :]

        nn = self.nn(z1)
        mu = nn[..., : self.d_out]
        sigma = nn[..., self.d_out :]
        z2 = z2 * torch.exp(sigma) + mu

        ldj = torch.sum(sigma, dim=-1)
        return torch.cat([z1, z2], dim=-1), ldj


@preserve_constructor_rng
class CouplingFlow(torch.nn.Module):
    """Stack affine coupling layers with an optional affine pre-transform.

    Parameters
    ----------
    depth : int
        Number of coupling layers.
    input_dim : int
        Number of coefficient dimensions.
    device : torch.device or str
        Device used for parameters and the owned generator.
    dtype : torch.dtype
        Floating-point dtype used for parameters.
    seed : int
        Seed for deterministic initialization and sampling state.
    init_scale : float, default=1e-3
        Standard deviation of each coupling network's final-layer weights.
    """

    def __init__(self, depth, input_dim, device, dtype, seed, init_scale=1e-3):
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.device = device
        self.dtype = dtype

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        self.biyections = torch.nn.ModuleList(
            [CouplingLayer(input_dim, device, dtype, init_scale=init_scale) for _ in range(depth)]
        )

        # Pre-computed flip index to avoid allocating a new tensor each layer
        self.register_buffer("_flip_idx", torch.arange(input_dim - 1, -1, -1, device=device))

        # Optional affine pre-transform (set via set_affine).
        # Register in _modules so later assignment isn't shadowed by __dict__.
        self._modules["affine"] = None

    def set_affine(self, shift, L_flat, learnable=False):
        """Attach an affine coefficient pre-transform.

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
        """Transform coefficients through the complete coupling flow.

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
        for b in self.biyections:
            a, ldj = b(a)
            a = a[..., self._flip_idx]
            LDJ += ldj
        return a, -LDJ
