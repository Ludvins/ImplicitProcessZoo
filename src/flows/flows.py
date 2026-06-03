import torch


class AffineLayer(torch.nn.Module):
    """Learnable (or fixed) affine transform: a = L @ eps + shift.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input.
    device, dtype : torch device / dtype
    learnable : bool
        If False, shift and L_flat are registered as buffers (not trained).
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
                dtype=torch.long, device=device,
            ),
        )

    def forward(self, eps):
        # Reconstruct L from flat triangular elements
        L = torch.zeros(
            self.input_dim, self.input_dim,
            dtype=eps.dtype, device=eps.device,
        )
        L[self._tril_row, self._tril_col] = self.L_flat

        # a = eps @ L + shift — VIP convention: Cov[a] = L^T @ L
        a = eps @ L + self.shift

        # log|det(L)| = sum(log|diag(L)|)
        log_abs_det = torch.sum(torch.log(torch.abs(self.L_flat[self._diag_idx])))
        ldj = log_abs_det.expand(eps.shape[0])
        return a, ldj


class CouplingLayer(torch.nn.Module):
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
        z1 = a[..., : self.d_half]
        z2 = a[..., self.d_half :]

        nn = self.nn(z1)
        mu = nn[..., : self.d_out]
        sigma = nn[..., self.d_out :]
        z2 = z2 * torch.exp(sigma) + mu

        ldj = torch.sum(sigma, dim=-1)
        return torch.cat([z1, z2], dim=-1), ldj


class CouplingFlow(torch.nn.Module):
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
        self.register_buffer(
            "_flip_idx", torch.arange(input_dim - 1, -1, -1, device=device)
        )

        # Optional affine pre-transform (set via set_affine).
        # Register in _modules so later assignment isn't shadowed by __dict__.
        self._modules['affine'] = None

    def set_affine(self, shift, L_flat, learnable=False):
        """Attach an affine pre-transform: a = L @ eps + shift.

        Parameters
        ----------
        shift : Tensor of shape (input_dim,)
        L_flat : Tensor of shape (n_tril,) — lower-triangular elements of L.
        learnable : bool
            If False, the affine params are fixed buffers.
        """
        self.affine = AffineLayer(
            self.input_dim, self.device, self.dtype, learnable=learnable,
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
        return a, -LDJ
