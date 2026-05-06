"""Normalizing-Flow Variational Inference (NF-VI) for Bayesian neural networks.

Replaces the mean-field Gaussian posterior ``q(w) = N(mu, diag(sigma^2))``
used by :class:`MFVI` with a normalizing flow:

    w = T_phi(eps),  eps ~ N(0, I)

``T_phi`` is any invertible transform that returns ``(w, ldj)`` where ``ldj``
follows the :class:`CouplingFlow` sign convention (``-log|det J_T|``). The KL
term is Monte-Carlo estimated:

    KL(q || p) = E_eps [ log N(eps; 0, I) - log p(w) - log|det J_T| ]

The public API mirrors :class:`MFVI` so NF-VI drops into the same training
loops, benchmarks, and tests.
"""

import math

import numpy as np
import torch
from tqdm import tqdm

from ..utils.likelihood import (
    gaussian_logp,
    bernoulli_logp,
    multiclass_logp,
    inv_probit,
)
from ..utils.flat_mlp import (
    FlatMLP,
    collect_param_spec,
    forward_with_flat_params,
)
from ..utils.utils import infinite_loader


class NFVI(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        structure,
        activation,
        flow,
        likelihood,
        num_data,
        num_samples=20,
        bb_alpha=0,
        sigma_prior=1.0,
        y_mean=0.0,
        y_std=1.0,
        num_classes=None,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        """
        Normalizing-Flow Variational Inference in weight space.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input.
        output_dim : int
            Dimensionality of the output.
        structure : list of int
            Hidden layer widths of the underlying MLP, e.g. ``[50, 50]``.
        activation : callable
            Activation module (e.g. ``torch.nn.Tanh()``).
        flow : torch.nn.Module
            Normalizing flow with ``flow(eps) -> (w, ldj)``. Its ``input_dim``
            must equal the total number of MLP parameters (weights + biases
            across all layers). ``CouplingFlow`` / ``SplineCouplingFlow`` both
            work.
        likelihood : str
            One of ``"regression"``, ``"binary"``, ``"multiclass"``.
        num_data : int
            Total dataset size (for minibatch scaling).
        num_samples : int
            Number of Monte Carlo weight samples per forward pass.
        bb_alpha : float
            Alpha for BB-alpha energy (0 = standard ELBO).
        sigma_prior : float
            Std of the isotropic Gaussian prior ``p(w) = N(0, sigma_prior^2 I)``.
        y_mean, y_std : float or array-like
            Target statistics for denormalization (regression only).
        num_classes : int or None
            Required when ``likelihood="multiclass"``.
        device : torch.device
        dtype : torch dtype
        seed : int
            Seed for the internal epsilon sampler.
        """
        super().__init__()

        if likelihood not in ("regression", "binary", "multiclass"):
            raise ValueError(
                f"likelihood must be 'regression', 'binary', or 'multiclass', "
                f"got '{likelihood}'"
            )
        if likelihood == "multiclass" and num_classes is None:
            raise ValueError("num_classes is required for multiclass likelihood")

        self.likelihood_type = likelihood
        self.num_data = num_data
        self.num_samples = num_samples
        self.bb_alpha = bb_alpha
        self.sigma_prior = sigma_prior
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype

        self.register_buffer(
            "y_mean", torch.as_tensor(y_mean, dtype=dtype, device=device)
        )
        self.register_buffer(
            "y_std", torch.as_tensor(y_std, dtype=dtype, device=device)
        )
        self.register_buffer(
            "_log2pi", torch.tensor(math.log(2.0 * math.pi), dtype=dtype, device=device)
        )

        # Network output dimension: for multiclass the MLP emits logits of size
        # num_classes; the "output_dim" argument is the target dimensionality
        # and is kept separate for denormalization / API compatibility.
        if likelihood == "multiclass":
            self.num_classes = num_classes
            self.epsilon = 1e-3
            net_output_dim = num_classes
        else:
            net_output_dim = output_dim

        # Architecture template: the actual weights come from the flow, so we
        # freeze the template's parameters (PyTorch still initializes them so
        # e.g. biases start at 0 if you decide to warm-start later).
        self.base_net = FlatMLP(
            input_dim, net_output_dim, structure, activation,
            dtype=dtype, device=device,
        )
        for p in self.base_net.parameters():
            p.requires_grad = False
        self._param_names, self._param_shapes, self._total_params = \
            collect_param_spec(self.base_net)
        self._num_layers = len(self.base_net.layers)

        self.flow = flow
        if getattr(flow, "input_dim", None) != self._total_params:
            raise ValueError(
                f"flow.input_dim={flow.input_dim} does not match total MLP "
                f"parameter count {self._total_params}. Build the flow with "
                f"input_dim=NFVI(...)._total_params, or size it by hand."
            )

        if likelihood == "regression":
            self.log_variance = torch.nn.Parameter(
                torch.tensor(-5.0, dtype=dtype, device=device)
            )

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        # Buffers cached during each forward for KL computation
        self._eps = None
        self._w = None
        self._ldj = None

        # Diagnostics
        self.KLs = []
        self.bb_alphas = []
        self._kl_base_buffer = []
        self._kl_ldj_buffer = []

    # ------------------------------------------------------------------
    # Core model methods
    # ------------------------------------------------------------------

    def _sample_weights(self, S):
        """Draw ``S`` weight samples through the flow. Antithetic pairing."""
        S_half = S // 2
        eps_half = torch.randn(
            S_half, self._total_params,
            generator=self.generator, dtype=self.dtype, device=self.device,
        )
        if 2 * S_half == S:
            eps = torch.cat([eps_half, -eps_half], dim=0)
        else:
            # Odd S: pad with one extra independent draw
            eps_extra = torch.randn(
                1, self._total_params,
                generator=self.generator, dtype=self.dtype, device=self.device,
            )
            eps = torch.cat([eps_half, -eps_half, eps_extra], dim=0)

        w, ldj = self.flow(eps)
        self._eps = eps
        self._w = w
        self._ldj = ldj
        return w

    def predict_f_samples(self, X, S):
        """
        Sample latent function values from the flow-posterior.

        Returns
        -------
        F : [S, N, D]
        """
        w = self._sample_weights(S)
        outs = [forward_with_flat_params(self.base_net, w[s], X) for s in range(S)]
        return torch.stack(outs, dim=0)

    def predict_y_samples(self, X, S):
        """
        Sample from the predictive distribution p(y | x).

        Regression: F + eps, eps ~ N(0, sigma2).
        Binary: inv_probit(F).
        Multiclass: softmax(F).
        """
        F = self.predict_f_samples(X, S)
        if self.likelihood_type == "regression":
            std = torch.sqrt(torch.exp(self.log_variance))
            return F + std * torch.randn_like(F)
        if self.likelihood_type == "binary":
            return inv_probit(F)
        return torch.softmax(F, dim=-1)

    def _logp(self, F, Y):
        if self.likelihood_type == "regression":
            return gaussian_logp(F, Y, self.log_variance)
        if self.likelihood_type == "binary":
            return bernoulli_logp(F, Y)
        return multiclass_logp(F, Y, self.num_classes, self.epsilon)

    def KL(self):
        """Monte-Carlo KL(q(w) || N(0, sigma_prior^2 I)) using cached samples."""
        if self._eps is None:
            raise RuntimeError("Call predict_f_samples / nelbo before KL().")

        eps = self._eps
        w = self._w
        ldj = self._ldj.squeeze(-1) if self._ldj.ndim > 1 else self._ldj

        D = eps.shape[-1]
        log_sigma_p2 = 2.0 * math.log(self.sigma_prior)

        # log N(eps; 0, I)
        logp_eps = -0.5 * (torch.sum(eps * eps, dim=-1) + D * self._log2pi)
        # log N(w; 0, sigma_prior^2 I)
        logp_w = -0.5 * (
            torch.sum(w * w, dim=-1) / (self.sigma_prior ** 2)
            + D * (self._log2pi + log_sigma_p2)
        )

        kl_base = torch.mean(logp_eps - logp_w)
        kl_ldj = torch.mean(ldj)

        self._kl_base_buffer.append(kl_base.detach().item())
        self._kl_ldj_buffer.append(kl_ldj.detach().item())

        return kl_base + kl_ldj

    def nelbo(self, X, y):
        """Negative ELBO (or BB-alpha energy) objective."""
        F = self.predict_f_samples(X, self.num_samples)

        logpdf = self._logp(F, y)
        if self.bb_alpha == 0:
            ve = torch.mean(logpdf, dim=0)
        else:
            ve = (
                torch.logsumexp(self.bb_alpha * logpdf, dim=0)
                - math.log(F.shape[0])
            ) / self.bb_alpha
        ve = torch.sum(ve)

        scale = self.num_data / X.shape[0]
        KL = self.KL()

        self.bb_alphas.append((-scale * ve).detach().cpu().numpy())
        self.KLs.append(KL.detach().cpu().numpy())
        return -scale * ve + KL

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, X):
        """Return denormalized y_samples: [S, N, D]."""
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        Y = self.predict_y_samples(X, self.num_samples)
        if self.likelihood_type == "regression":
            return Y * self.y_std + self.y_mean
        return Y

    def predict(self, X, S):
        """Predict y_samples for evaluation.

        Returns
        -------
        Y : [S, N, D], denormalized for regression, probabilities otherwise.
        """
        self.eval()
        with torch.no_grad():
            if self.dtype != X.dtype:
                X = X.to(self.dtype)
            Y = self.predict_y_samples(X, S)
            if self.likelihood_type == "regression":
                return Y * self.y_std + self.y_mean
            return Y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_loader, optimizer=None, lr=0.001, epochs=None,
            iterations=None, use_tqdm=False, return_loss=False,
            cosine_annealing=False):
        """Train the model. See :meth:`MFVI.fit` for parameter semantics."""
        if optimizer is None:
            optimizer = torch.optim.Adam(
                [p for p in self.parameters() if p.requires_grad], lr=lr
            )

        if epochs is None and iterations is None:
            raise ValueError("Either epochs or iterations must be set.")

        scheduler = None
        if cosine_annealing:
            T_max = (
                epochs if epochs is not None
                else max(1, iterations // len(train_loader))
            )
            eta_min = optimizer.param_groups[0]['lr'] / 100
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )

        self.train()
        losses = []

        if epochs is not None:
            loop = tqdm(range(epochs), unit=" epoch", desc="Training") if use_tqdm else range(epochs)
            for _ in loop:
                for inputs, target in train_loader:
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)
                    loss = self._train_step(optimizer, inputs, target)
                    if return_loss:
                        losses.append(loss.detach().cpu().numpy())
                if scheduler is not None:
                    scheduler.step()
                if use_tqdm:
                    loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if iterations is not None:
            loop = tqdm(range(iterations), unit=" iter", desc="Training") if use_tqdm else range(iterations)
            data_stream = infinite_loader(train_loader)
            iters_per_epoch = len(train_loader)
            for i in loop:
                inputs, target = next(data_stream)
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                loss = self._train_step(optimizer, inputs, target)
                if return_loss:
                    losses.append(loss.detach().cpu().numpy())
                if scheduler is not None and (i + 1) % iters_per_epoch == 0:
                    scheduler.step()
                if use_tqdm:
                    loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        return losses

    def _train_step(self, optimizer, X, y):
        if self.dtype != X.dtype:
            X = X.to(self.dtype)

        if self.likelihood_type == "multiclass":
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            y = y.long()
        else:
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            if self.dtype != y.dtype:
                y = y.to(self.dtype)

        optimizer.zero_grad()
        loss = self.nelbo(X, y)
        loss.backward()
        optimizer.step()
        return loss

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def base_KLs(self):
        return self._kl_base_buffer

    @property
    def flow_ldj(self):
        return self._kl_ldj_buffer

    def print_variables(self):
        """Prints the trainable parameters in a formatted manner."""
        print("\n---- MODEL PARAMETERS ----")
        np.set_printoptions(threshold=3, edgeitems=2)
        sections = []
        pad = "  "
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            name = name.split(".")
            for i in range(len(name) - 1):
                if name[i] not in sections:
                    print(pad * i, name[i].upper())
                    sections = name[: i + 1]
            padding = pad * (len(name) - 1)
            print(
                padding,
                "{}: ({})".format(name[-1], str(list(param.data.size()))[1:-1]),
            )
            print(
                padding + " " * (len(name[-1]) + 2),
                param.data.detach().cpu().numpy().flatten(),
            )
        print("\n---------------------------\n\n")
