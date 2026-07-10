import math

import numpy as np
import torch
from tqdm import tqdm

from ..utils.likelihood import (
    bernoulli_logp,
    compute_gp_log_marginal,
    gaussian_logp,
    multiclass_logp,
    predict_mean_and_var_binary,
    predict_mean_and_var_multiclass,
    predict_mean_and_var_regression,
)
from ..utils.model_info import print_trainable_parameters
from ..utils.random import (
    ensure_generator_device,
    preserve_constructor_rng,
    standard_normal_samples,
    temporary_generator_seed,
)
from ..utils.training import fit_loop, make_cosine_scheduler, validate_fit_mode


@preserve_constructor_rng
class FTIP(torch.nn.Module):
    """Flow-Transformed Implicit Process over sampled prior features."""

    def __init__(
        self,
        generative_function,
        num_regression_coeffs,
        output_dim,
        flow,
        likelihood,
        num_data,
        num_samples,
        bb_alpha=0,
        y_mean=0.0,
        y_std=1.0,
        num_classes=None,
        use_prior_regularizer=False,
        prior_regularizer_scaler=1.0,
        regularizer_mode="evidence",
        max_grad_norm=None,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        """
        Flow-Transformed Implicit Process.

        Parameters
        ----------
        generative_function : torch.nn.Module
            Prior function sampler (e.g. BayesianNN).
        num_regression_coeffs : int
            Number of regression coefficients.
        output_dim : int
            Dimensionality of the output.
        flow : torch.nn.Module
            Normalizing flow for transforming the prior to the posterior.
        likelihood : str
            One of "regression", "binary", or "multiclass".
        num_data : int
            Total dataset size (for minibatch scaling).
        num_samples : int
            Number of Monte Carlo samples.
        bb_alpha : float
            Alpha for BB-alpha energy (0 = ELBO).
        y_mean, y_std : float or array-like
            Original target statistics for denormalization.
        num_classes : int or None
            Required when likelihood="multiclass".
        max_grad_norm : float or None
            If set, clip gradient norms to this value each step.
        device : torch.device
        dtype : torch dtype
        seed : int
        """
        super().__init__()

        if likelihood not in ("regression", "binary", "multiclass"):
            raise ValueError(
                f"likelihood must be 'regression', 'binary', or 'multiclass', got '{likelihood}'"
            )
        if likelihood == "multiclass" and num_classes is None:
            raise ValueError("num_classes is required for multiclass likelihood")

        self.likelihood_type = likelihood
        self.num_data = num_data
        self.bb_alpha = bb_alpha
        self.use_prior_regularizer = use_prior_regularizer
        self.prior_regularizer_scaler = prior_regularizer_scaler
        self.regularizer_mode = regularizer_mode
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=dtype, device=device))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=dtype, device=device))
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.dtype = dtype

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        self.num_samples = num_samples
        self.num_coeffs = num_regression_coeffs
        self.output_dim = output_dim
        self.generative_function = generative_function
        self.flow = flow

        # --- Precomputed constants ---
        self.register_buffer(
            "_sqrt_coeffs_m1",
            torch.tensor(math.sqrt(num_regression_coeffs - 1), dtype=dtype, device=device),
        )
        self.register_buffer(
            "_log2pi",
            torch.tensor(math.log(2.0 * math.pi), dtype=dtype, device=device),
        )
        # --- Likelihood-specific parameters ---
        if likelihood == "regression":
            self.log_variance = torch.nn.Parameter(
                torch.full((output_dim,), -2.0, dtype=dtype, device=device)
            )
        if likelihood in ("binary", "multiclass"):
            self.num_gauss_hermite_points = 20
        if likelihood == "multiclass":
            self.num_classes = num_classes
            self.epsilon = 1e-3
            self.K1 = self.epsilon / (num_classes - 1)

        self._kl_buffer = []
        self._kl_base_buffer = []
        self._kl_ldj_buffer = []
        self._bb_alpha_buffer = []
        self._prior_reg_buffer = []

    def load_state_dict(self, state_dict, **kwargs):
        """Auto-create flow affine layer if checkpoint contains affine keys."""
        has_affine = any(k.startswith("flow.affine.") for k in state_dict)
        if has_affine and self.flow.affine is None:
            from ..flows.flows import AffineLayer

            self.flow.affine = AffineLayer(
                self.flow.input_dim,
                self.flow.device,
                self.flow.dtype,
                learnable=True,
            )
        return torch.nn.Module.load_state_dict(self, state_dict, **kwargs)

    # ------------------------------------------------------------------
    # Core model methods
    # ------------------------------------------------------------------

    def predict_f(self, x, S):
        """
        Sample from the flow-transformed posterior.

        Returns (posterior_samples, f) where posterior_samples has shape
        (S, N, output_dim).
        """
        f = self.generative_function(x)

        generator = ensure_generator_device(self, f.device)
        eps = standard_normal_samples(
            S,
            self.num_coeffs,
            self.output_dim,
            generator=generator,
            dtype=f.dtype,
            device=f.device,
            antithetic=True,
        )

        eps_flat = eps.reshape(S, -1)
        a_flat, ldj = self.flow(eps_flat)

        # Cache for KL computation
        self._eps_flat = eps_flat
        self._a_flat = a_flat
        self._ldj = ldj

        a = a_flat.reshape(S, self.num_coeffs, self.output_dim)

        m = f.mean(dim=0, keepdim=True)
        phi = (f - m) / self._sqrt_coeffs_m1

        # Cache for prior_regularizer (avoids re-calling generative_function)
        self._cached_m = m.squeeze(0)
        self._cached_phi = phi

        posterior_samples = torch.einsum("snd, asd->and", phi, a) + m.squeeze(0)

        return posterior_samples, f

    def predict_f_samples(self, x, num_samples, *, seed=None):
        """Return latent samples with shape ``[samples, observations, outputs]``."""
        with temporary_generator_seed(self, seed):
            return self.predict_f(x, int(num_samples))[0]

    def _logp(self, F, Y):
        """Dispatch to the correct logp function."""
        if self.likelihood_type == "regression":
            return gaussian_logp(F, Y, self.log_variance)
        elif self.likelihood_type == "binary":
            return bernoulli_logp(F, Y)
        else:  # multiclass
            return multiclass_logp(F, Y, self.num_classes, self.epsilon)

    def _predict_mean_and_var(self, Fmu, Fvar):
        """Dispatch to the correct predict_mean_and_var function."""
        if self.likelihood_type == "regression":
            return predict_mean_and_var_regression(Fmu, Fvar, self.log_variance)
        elif self.likelihood_type == "binary":
            return predict_mean_and_var_binary(Fmu, Fvar)
        else:  # multiclass
            return predict_mean_and_var_multiclass(
                Fmu,
                Fvar,
                self.num_classes,
                self.epsilon,
                self.num_gauss_hermite_points,
                self.dtype,
                self.device,
            )

    def nelbo(self, X, y):
        """Negative ELBO (or BB-alpha energy) objective."""
        F, _ = self.predict_f(X, self.num_samples)

        logpdf = self._logp(F, y)
        if self.bb_alpha == 0:
            ve = torch.mean(logpdf, axis=0)
        else:
            ve = (
                torch.logsumexp(self.bb_alpha * logpdf, axis=0) - math.log(self.num_samples)
            ) / self.bb_alpha

        ve = torch.sum(ve)

        scale = self.num_data / X.shape[0]

        KL = self.KL()
        self._bb_alpha_buffer.append((-scale * ve).detach().item())
        self._kl_buffer.append(KL.detach().item())
        return -scale * ve + KL

    def KL(self):
        """KL divergence between flow-transformed posterior and standard normal prior."""
        if not hasattr(self, "_eps_flat"):
            raise RuntimeError("Call predict_f before KL")

        eps = self._eps_flat
        a = self._a_flat
        ldj = self._ldj.squeeze(-1) if self._ldj.ndim > 1 else self._ldj

        D = eps.shape[-1]

        logp_eps = -0.5 * (torch.sum(eps * eps, dim=-1) + D * self._log2pi)
        logp_a = -0.5 * (torch.sum(a * a, dim=-1) + D * self._log2pi)

        kl_base = torch.mean(logp_eps - logp_a)
        kl_ldj = torch.mean(ldj)

        self._kl_base_buffer.append(kl_base.detach().item())
        self._kl_ldj_buffer.append(kl_ldj.detach().item())

        return kl_base + kl_ldj

    @property
    def KLs(self):
        return self._kl_buffer

    @property
    def base_KLs(self):
        return self._kl_base_buffer

    @property
    def flow_ldj(self):
        return self._kl_ldj_buffer

    @property
    def bb_alphas(self):
        return self._bb_alpha_buffer

    @property
    def prior_regularizers(self):
        return self._prior_reg_buffer

    def forward(self, predict_at):
        """Returns (samples, std) for regression or (samples, None) for classification."""
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        posterior_samples = self.predict_y(predict_at, self.num_samples)
        if self.likelihood_type == "regression":
            std = torch.exp(0.5 * self.log_variance)
            std = std.unsqueeze(0).expand(posterior_samples.shape[0], -1)
            return posterior_samples * self.y_std + self.y_mean, std * self.y_std
        return posterior_samples, None

    def sample_flow_coefficients(self, S):
        """Sample flow coefficients once (data-independent).

        Returns a tensor of shape (S, num_coeffs, output_dim) that can be
        reused across multiple batches via ``forward_with_coefficients``.
        """
        reference = self._sqrt_coeffs_m1
        generator = ensure_generator_device(self, reference.device)
        eps = standard_normal_samples(
            S,
            self.num_coeffs,
            self.output_dim,
            generator=generator,
            dtype=reference.dtype,
            device=reference.device,
            antithetic=True,
        )
        eps_flat = eps.reshape(S, -1)
        a_flat, _ = self.flow(eps_flat)
        return a_flat.reshape(S, self.num_coeffs, self.output_dim)

    def forward_with_coefficients(self, predict_at, a):
        """Forward pass reusing pre-computed flow coefficients.

        Parameters
        ----------
        predict_at : Tensor of shape (N, input_dim)
        a : Tensor of shape (S, num_coeffs, output_dim) from
            ``sample_flow_coefficients``.

        Returns (mean, std) in the original target scale.
        """
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        f = self.generative_function(predict_at)
        m = f.mean(dim=0, keepdim=True)
        phi = (f - m) / self._sqrt_coeffs_m1
        posterior_samples = torch.einsum("snd, asd->and", phi, a) + m.squeeze(0)

        if self.likelihood_type == "regression":
            std = torch.exp(0.5 * self.log_variance)
            std = std.unsqueeze(0).expand(posterior_samples.shape[0], -1)
            return posterior_samples * self.y_std + self.y_mean, std * self.y_std
        return posterior_samples, None

    def forward_prior(self, predict_at, num_samples):
        """Sample from the prior (without flow transformation)."""
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        f = self.generative_function(predict_at)

        coeffs = torch.randn(
            [num_samples, self.num_coeffs, self.output_dim],
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )

        m = f.mean(dim=0, keepdim=True)
        phi = (f - m) / self._sqrt_coeffs_m1
        f = torch.einsum("snd, asd->and", phi, coeffs) + m.squeeze(0)
        if self.likelihood_type == "regression":
            return f * self.y_std + self.y_mean
        return f

    def get_samples(self, S):
        """Generate S posterior coefficient samples through the flow."""
        coeffs = torch.randn(
            [S, self.num_coeffs, self.output_dim],
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )

        a, _ = self.flow(coeffs.reshape(S, -1))
        a = a.reshape(S, self.num_coeffs, self.output_dim)
        return coeffs, a

    def predict_y(self, predict_at, S):
        """Predict through the likelihood."""
        return self.predict_f(predict_at, S)[0]

    def predict_y_samples(self, x, num_samples, *, seed=None):
        """Return likelihood samples with shape ``[samples, observations, outputs]``."""
        with temporary_generator_seed(self, seed):
            samples = self.predict_f(x, int(num_samples))[0]
            if self.likelihood_type == "regression":
                generator = ensure_generator_device(self, samples.device)
                noise = standard_normal_samples(
                    int(num_samples),
                    samples.shape[1],
                    samples.shape[2],
                    generator=generator,
                    dtype=samples.dtype,
                    device=samples.device,
                )
                samples = samples + torch.exp(0.5 * self.log_variance) * noise
            return samples

    # ------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        optimizer=None,
        lr=0.001,
        epochs=None,
        iterations=None,
        use_tqdm=False,
        return_loss=False,
        cosine_annealing=False,
    ):
        """
        Train the model.

        Parameters
        ----------
        train_loader : DataLoader
        optimizer : torch.optim.Optimizer or None
            If None, creates Adam with the given lr.
        lr : float
            Learning rate (used only if optimizer is None).
        epochs : int or None
            Number of epochs. Exactly one of epochs/iterations must be set.
        iterations : int or None
            Number of gradient steps.
        use_tqdm : bool
        return_loss : bool
        cosine_annealing : bool
            If True, use CosineAnnealingLR with T_max=epochs, stepped once
            per epoch. When using iterations, T_max is the number of
            effective epochs (iterations // len(train_loader)).

        Returns
        -------
        losses : list of float (if return_loss=True)
        """
        validate_fit_mode(epochs=epochs, iterations=iterations)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = (
            make_cosine_scheduler(optimizer, train_loader, epochs=epochs, iterations=iterations)
            if cosine_annealing
            else None
        )
        return fit_loop(
            self,
            train_loader,
            optimizer,
            epochs=epochs,
            iterations=iterations,
            use_tqdm=use_tqdm,
            return_loss=return_loss,
            scheduler=scheduler,
        )

    def _train_step(self, optimizer, X, y):
        """Single gradient step."""
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        optimizer.zero_grad(set_to_none=True)
        loss = self.nelbo(X, y)
        if self.use_prior_regularizer:
            prior_reg = self.prior_regularizer(X, y)
            self._prior_reg_buffer.append(prior_reg.detach().item())
            loss = loss + prior_reg
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()
        return loss

    def prior_regularizer(self, X, y):
        """
        Prior regularizer.

        mode="evidence": Likelihood regularizer from Appendix C.5
            (Ma et al., 2019). Uses cached m and phi from predict_f to
            compute log q_GP(y | X) and penalizes high prior evidence.
        mode="KL": Weight-space KL of the generative function (if available).
        """
        if self.regularizer_mode == "KL":
            kl = self.generative_function.KL()
            return self.prior_regularizer_scaler * kl

        # mode == "evidence"
        if self.likelihood_type != "regression":
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        sigma2 = torch.exp(self.log_variance)
        log_q = compute_gp_log_marginal(self._cached_m, self._cached_phi, y, sigma2 + 1e-3)
        return self.prior_regularizer_scaler / X.shape[0] * log_q

    def predict(self, X, num_samples, *, seed=None):
        """Return tensor predictions; batching is handled by the shared utility."""
        self.eval()
        with torch.no_grad():
            values = self.predict_y_samples(X, num_samples, seed=seed)
            if self.likelihood_type == "regression":
                values = values * self.y_std + self.y_mean
            return values

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def warm_start_from_vip(self, vip_model, learnable_affine=False):
        """Initialize from a trained VIP model.

        Copies the generative function weights, likelihood parameters, and
        sets up an affine pre-transform in the flow so the initial FTIP
        posterior matches VIP's Gaussian Q(a) = N(q_mu, q_sqrt @ q_sqrt^T).

        Parameters
        ----------
        vip_model : VIP
            A trained VIP model with the same architecture.
        learnable_affine : bool
            If True, the affine layer parameters are trainable.
            If False (default), they are fixed buffers.
        """
        # 1. Copy generative function weights + cached noise (for identical phi)
        self.generative_function.load_state_dict(vip_model.generative_function.state_dict())
        for dst, src in zip(
            self.generative_function.modules(), vip_model.generative_function.modules()
        ):
            if hasattr(src, "noise") and src.noise is not None:
                dst.noise = src.noise
        # Copy log_variance from VIP, with a floor.  VIP's analytical
        # likelihood folds Fvar into the effective variance, so VIP can
        # drive log_variance to extremely negative values (e.g. -30)
        # without issue.  FTIP evaluates the Gaussian likelihood directly
        # at MC samples, so a near-zero variance causes (F-y)²/var to
        # explode.  We clamp to a sensible minimum (log(1e-4) ≈ -9.2).
        if hasattr(vip_model, "log_variance"):
            min_log_var = -3.7  # var >= ~0.025; prevents FTIP NLL explosion
            clamped = torch.clamp(vip_model.log_variance.data, min=min_log_var)
            self.log_variance.data.copy_(clamped)

        # 2. Set affine pre-transform from VIP's variational params
        #    q_mu: (num_coeffs, output_dim), q_sqrt_tri: (n_tril, output_dim)
        #    Flow operates on flattened (num_coeffs * output_dim,) vectors.
        #    For output_dim=1 this is straightforward; for output_dim>1
        #    we build a block-diagonal L from per-output-dim triangular factors.
        q_mu = vip_model.q_mu.data  # (num_coeffs, output_dim)
        q_sqrt_tri = vip_model.q_sqrt_tri.data  # (n_tril, output_dim)
        S = self.num_coeffs
        D = self.output_dim

        if D == 1:
            shift = q_mu.flatten()  # (S,)
            L_flat = q_sqrt_tri.flatten()  # (n_tril,)
        else:
            # eps is (S_samples, num_coeffs, output_dim), flattened row-major to
            # (S_samples, num_coeffs * output_dim). VIP has independent L per
            # output dim, so we build a block-diagonal in the flattened space.
            # Flatten order: [c0_d0, c0_d1, ..., c1_d0, c1_d1, ...] i.e.
            # output_dim varies fastest. We need to permute the blocks accordingly.
            #
            # Simpler approach: permute q_mu and L to match flattened layout.
            # For each output dim d, VIP has L_d of shape (S, S) operating on
            # coefficients. In the flattened vector, indices for dim d are
            # d, d+D, d+2D, ... so L_d acts on strided indices.
            # We use a permuted block-diagonal.
            li, lj = torch.tril_indices(S, S)
            L_full = torch.zeros(S * D, S * D, dtype=self.dtype, device=self.device)
            for d in range(D):
                L_block = torch.zeros(S, S, dtype=self.dtype, device=self.device)
                L_block[li, lj] = q_sqrt_tri[:, d]
                # Indices in flattened vector for output dim d: d, d+D, d+2D, ...
                idx = torch.arange(S, device=self.device) * D + d
                L_full[idx.unsqueeze(1), idx.unsqueeze(0)] = L_block

            shift = q_mu.flatten()  # (S*D,) row-major matches reshape
            big_li, big_lj = torch.tril_indices(S * D, S * D)
            L_flat = L_full[big_li, big_lj]

        self.flow.set_affine(shift, L_flat, learnable=learnable_affine)

    def print_variables(self):
        """Print trainable parameters using the shared formatter."""
        print_trainable_parameters(self)


@preserve_constructor_rng
class UnifiedFTIP(torch.nn.Module):
    """Flow-Transformed Implicit Process with built-in Gaussian warmup.

    Single model that transitions from analytical Gaussian inference (Phase 1)
    to MC-based flow inference (Phase 2). Eliminates the need for separate
    VIP pre-training and warm-start transfer.

    Phase 1 ("warmup"): Posterior is Q(a) = N(q_mu, q_sqrt @ q_sqrt^T).
        Uses analytical variational expectations and KL — zero MC noise.
        Coupling layers are frozen.

    Phase 2 ("flow"): Posterior is a = flow(eps), eps ~ N(0, I).
        The affine layer is initialized from the learned q_mu/q_sqrt.
        Coupling layers are unfrozen. Uses MC ELBO estimation.
    """

    def __init__(
        self,
        generative_function,
        num_regression_coeffs,
        output_dim,
        flow,
        likelihood,
        num_data,
        num_samples,
        bb_alpha=0,
        y_mean=0.0,
        y_std=1.0,
        num_classes=None,
        use_prior_regularizer=False,
        prior_regularizer_scaler=1.0,
        regularizer_mode="evidence",
        max_grad_norm=None,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__()

        if likelihood not in ("regression", "binary", "multiclass"):
            raise ValueError(
                f"likelihood must be 'regression', 'binary', or 'multiclass', got '{likelihood}'"
            )
        if likelihood == "multiclass" and num_classes is None:
            raise ValueError("num_classes is required for multiclass likelihood")

        self.likelihood_type = likelihood
        self.num_data = num_data
        self.bb_alpha = bb_alpha
        self.use_prior_regularizer = use_prior_regularizer
        self.prior_regularizer_scaler = prior_regularizer_scaler
        self.regularizer_mode = regularizer_mode
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=dtype, device=device))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=dtype, device=device))
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.dtype = dtype
        self._seed = seed

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        self.num_samples = num_samples
        self.num_coeffs = num_regression_coeffs
        self.output_dim = output_dim
        self.generative_function = generative_function
        self.flow = flow

        # --- Precomputed constants ---
        self.register_buffer(
            "_sqrt_coeffs_m1",
            torch.tensor(math.sqrt(num_regression_coeffs - 1), dtype=dtype, device=device),
        )
        self.register_buffer(
            "_log2pi",
            torch.tensor(math.log(2.0 * math.pi), dtype=dtype, device=device),
        )

        # --- Likelihood parameters ---
        if likelihood == "regression":
            self.log_variance = torch.nn.Parameter(
                torch.full((output_dim,), -2.0, dtype=dtype, device=device)
            )
        if likelihood in ("binary", "multiclass"):
            self.num_gauss_hermite_points = 20
        if likelihood == "multiclass":
            self.num_classes = num_classes
            self.epsilon = 1e-3
            self.K1 = self.epsilon / (num_classes - 1)

        # --- Gaussian variational parameters (Phase 1) ---
        S = num_regression_coeffs
        li, lj = torch.tril_indices(S, S)
        self.register_buffer("_tril_row", li)
        self.register_buffer("_tril_col", lj)
        self.register_buffer(
            "_diag_idx",
            torch.tensor(np.cumsum(np.arange(1, S + 1)) - 1, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "_q_sqrt_buf",
            torch.zeros(S, S, output_dim, dtype=dtype, device=device),
        )

        self.q_mu = torch.nn.Parameter(torch.zeros(S, output_dim, dtype=dtype, device=device))
        q_sqrt_init = torch.eye(S, dtype=dtype, device=device)
        q_sqrt_init = q_sqrt_init.unsqueeze(-1).expand(-1, -1, output_dim)
        self.q_sqrt_tri = torch.nn.Parameter(q_sqrt_init[li, lj].contiguous())

        # --- Diagnostics ---
        self._kl_buffer = []
        self._kl_base_buffer = []
        self._kl_ldj_buffer = []
        self._bb_alpha_buffer = []
        self._prior_reg_buffer = []

        # Start in warmup phase
        self.phase = None
        self.set_phase("warmup")

    # ------------------------------------------------------------------
    # Phase control
    # ------------------------------------------------------------------

    def set_phase(self, phase):
        """Switch between 'warmup' (analytical Gaussian) and 'flow' (MC) phases."""
        if phase == "warmup":
            self.phase = "warmup"
            # Freeze coupling layers
            for layer in self.flow.biyections:
                for p in layer.parameters():
                    p.requires_grad = False
            # Ensure Gaussian params are trainable
            self.q_mu.requires_grad = True
            self.q_sqrt_tri.requires_grad = True

        elif phase == "flow":
            # Transfer q_mu/q_sqrt into affine pre-transform
            S = self.num_coeffs
            D = self.output_dim
            if D == 1:
                shift = self.q_mu.data.flatten()
                L_flat = self.q_sqrt_tri.data.flatten()
            else:
                li, lj = torch.tril_indices(S, S)
                L_full = torch.zeros(S * D, S * D, dtype=self.dtype, device=self.device)
                for d in range(D):
                    L_block = torch.zeros(S, S, dtype=self.dtype, device=self.device)
                    L_block[li, lj] = self.q_sqrt_tri.data[:, d]
                    idx = torch.arange(S, device=self.device) * D + d
                    L_full[idx.unsqueeze(1), idx.unsqueeze(0)] = L_block
                shift = self.q_mu.data.flatten()
                big_li, big_lj = torch.tril_indices(S * D, S * D)
                L_flat = L_full[big_li, big_lj]

            self.flow.set_affine(shift, L_flat, learnable=True)

            # Freeze Gaussian params (now redundant, affine layer has them)
            self.q_mu.requires_grad = False
            self.q_sqrt_tri.requires_grad = False

            # Unfreeze coupling layers
            for layer in self.flow.biyections:
                for p in layer.parameters():
                    p.requires_grad = True

            self.phase = "flow"
        else:
            raise ValueError(f"phase must be 'warmup' or 'flow', got '{phase}'")

    # ------------------------------------------------------------------
    # Warmup phase (analytical, VIP-style)
    # ------------------------------------------------------------------

    def predict_f_gaussian(self, x):
        """Analytical Gaussian prediction (Phase 1). Returns (mean, var)."""
        f = self.generative_function(x)
        m = torch.mean(f, dim=0, keepdim=True)
        phi = (f - m) / self._sqrt_coeffs_m1

        self._cached_m = m.squeeze(0)
        self._cached_phi = phi

        mean = m.squeeze(0) + torch.einsum("sn...,s...->n...", phi, self.q_mu)

        q_sqrt = torch.zeros_like(self._q_sqrt_buf)
        q_sqrt[self._tril_row, self._tril_col] = self.q_sqrt_tri
        phi_q = torch.einsum("ind, sid -> snd", phi, q_sqrt)
        K = torch.einsum("snd, snd -> nd", phi_q, phi_q)

        return mean, K

    def KL_gaussian(self):
        """Analytical KL(N(q_mu, q_sqrt q_sqrt^T) || N(0, I))."""
        diag = self.q_sqrt_tri[self._diag_idx]
        KL = -0.5 * self.output_dim * self.num_coeffs
        KL -= torch.sum(torch.log(torch.abs(diag)))
        KL += 0.5 * torch.sum(torch.square(self.q_sqrt_tri))
        KL += 0.5 * torch.sum(torch.square(self.q_mu))
        return KL

    # ------------------------------------------------------------------
    # Flow phase (MC, FTIP-style)
    # ------------------------------------------------------------------

    def predict_f(self, x, S):
        """MC prediction through the full flow (Phase 2)."""
        f = self.generative_function(x)

        generator = ensure_generator_device(self, f.device)
        eps = standard_normal_samples(
            S,
            self.num_coeffs,
            self.output_dim,
            generator=generator,
            dtype=f.dtype,
            device=f.device,
            antithetic=True,
        )

        eps_flat = eps.reshape(S, -1)
        a_flat, ldj = self.flow(eps_flat)

        self._eps_flat = eps_flat
        self._a_flat = a_flat
        self._ldj = ldj

        a = a_flat.reshape(S, self.num_coeffs, self.output_dim)

        m = f.mean(dim=0, keepdim=True)
        phi = (f - m) / self._sqrt_coeffs_m1

        self._cached_m = m.squeeze(0)
        self._cached_phi = phi

        posterior_samples = torch.einsum("snd, asd->and", phi, a) + m.squeeze(0)
        return posterior_samples, f

    def predict_f_samples(self, x, num_samples, *, seed=None):
        """Return flow-phase latent samples using the common sample-first API."""
        with temporary_generator_seed(self, seed):
            if self.phase == "warmup":
                mean, variance = self.predict_f_gaussian(x)
                generator = ensure_generator_device(self, mean.device)
                noise = standard_normal_samples(
                    int(num_samples),
                    mean.shape[0],
                    mean.shape[1],
                    generator=generator,
                    dtype=mean.dtype,
                    device=mean.device,
                )
                return mean.unsqueeze(0) + noise * torch.sqrt(torch.clamp(variance, min=1e-10))
            return self.predict_f(x, int(num_samples))[0]

    def predict_y_samples(self, x, num_samples, *, seed=None):
        """Return likelihood samples using the common sample-first API."""
        with temporary_generator_seed(self, seed):
            values = self.predict_f_samples(x, int(num_samples))
            if self.likelihood_type == "regression":
                generator = ensure_generator_device(self, values.device)
                noise = standard_normal_samples(
                    int(num_samples),
                    values.shape[1],
                    values.shape[2],
                    generator=generator,
                    dtype=values.dtype,
                    device=values.device,
                )
                values = values + torch.exp(0.5 * self.log_variance) * noise
            return values

    def predict(self, X, num_samples, *, seed=None):
        self.eval()
        with torch.no_grad():
            values = self.predict_y_samples(X, num_samples, seed=seed)
            if self.likelihood_type == "regression":
                values = values * self.y_std + self.y_mean
            return values

    def KL_flow(self):
        """MC-estimated KL for the flow (Phase 2)."""
        eps = self._eps_flat
        a = self._a_flat
        ldj = self._ldj.squeeze(-1) if self._ldj.ndim > 1 else self._ldj
        D = eps.shape[-1]

        logp_eps = -0.5 * (torch.sum(eps * eps, dim=-1) + D * self._log2pi)
        logp_a = -0.5 * (torch.sum(a * a, dim=-1) + D * self._log2pi)

        kl_base = torch.mean(logp_eps - logp_a)
        kl_ldj = torch.mean(ldj)

        self._kl_base_buffer.append(kl_base.detach().item())
        self._kl_ldj_buffer.append(kl_ldj.detach().item())

        return kl_base + kl_ldj

    # ------------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------------

    def _logp(self, F, Y):
        if self.likelihood_type == "regression":
            return gaussian_logp(F, Y, self.log_variance)
        elif self.likelihood_type == "binary":
            return bernoulli_logp(F, Y)
        else:
            return multiclass_logp(F, Y, self.num_classes, self.epsilon)

    def _predict_mean_and_var(self, Fmu, Fvar):
        if self.likelihood_type == "regression":
            return predict_mean_and_var_regression(Fmu, Fvar, self.log_variance)
        elif self.likelihood_type == "binary":
            return predict_mean_and_var_binary(Fmu, Fvar)
        else:
            return predict_mean_and_var_multiclass(
                Fmu,
                Fvar,
                self.num_classes,
                self.epsilon,
                self.num_gauss_hermite_points,
                self.dtype,
                self.device,
            )

    def nelbo(self, X, y):
        """Negative ELBO — dispatches to analytical or MC depending on phase."""
        if self.phase == "warmup":
            return self._nelbo_gaussian(X, y)
        else:
            return self._nelbo_flow(X, y)

    def _nelbo_gaussian(self, X, y):
        """Analytical NELBO (Phase 1)."""
        from ..utils.likelihood import gaussian_variational_expectations

        F_mean, F_var = self.predict_f_gaussian(X)

        if self.likelihood_type == "regression":
            ve = gaussian_variational_expectations(
                F_mean.unsqueeze(0),
                F_var.unsqueeze(0),
                y,
                self.bb_alpha,
                self.log_variance,
            )
        else:
            raise NotImplementedError("Warmup phase only supports regression likelihood")

        ve = torch.sum(ve)
        scale = self.num_data / X.shape[0]
        KL = self.KL_gaussian()

        self._bb_alpha_buffer.append((-scale * ve).detach().item())
        self._kl_buffer.append(KL.detach().item())
        return -scale * ve + KL

    def _nelbo_flow(self, X, y):
        """MC NELBO (Phase 2)."""
        F, _ = self.predict_f(X, self.num_samples)

        logpdf = self._logp(F, y)
        if self.bb_alpha == 0:
            ve = torch.mean(logpdf, axis=0)
        else:
            ve = (
                torch.logsumexp(self.bb_alpha * logpdf, axis=0) - math.log(self.num_samples)
            ) / self.bb_alpha

        ve = torch.sum(ve)
        scale = self.num_data / X.shape[0]
        KL = self.KL_flow()

        self._bb_alpha_buffer.append((-scale * ve).detach().item())
        self._kl_buffer.append(KL.detach().item())
        return -scale * ve + KL

    def forward(self, predict_at):
        """Returns (mean, std) in original target scale."""
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        if self.phase == "warmup":
            Fmean, Fvar = self.predict_f_gaussian(predict_at)
            mean, var = self._predict_mean_and_var(Fmean.unsqueeze(0), Fvar.unsqueeze(0))
            return mean * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std
        else:
            posterior_samples = self.predict_f(predict_at, self.num_samples)[0]
            std = torch.exp(0.5 * self.log_variance)
            std = std.unsqueeze(0).expand(posterior_samples.shape[0], -1)
            return posterior_samples * self.y_std + self.y_mean, std * self.y_std

    def forward_prior(self, predict_at, num_samples):
        """Sample from the prior (without flow transformation)."""
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)
        f = self.generative_function(predict_at)
        coeffs = torch.randn(
            [num_samples, self.num_coeffs, self.output_dim],
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        m = f.mean(dim=0, keepdim=True)
        phi = (f - m) / self._sqrt_coeffs_m1
        f = torch.einsum("snd, asd->and", phi, coeffs) + m.squeeze(0)
        return f * self.y_std + self.y_mean

    def prior_regularizer(self, X, y):
        """Prior regularizer (same for both phases)."""
        if self.regularizer_mode == "KL":
            kl = self.generative_function.KL()
            return self.prior_regularizer_scaler * kl
        if self.likelihood_type != "regression":
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        sigma2 = torch.exp(self.log_variance)
        log_q = compute_gp_log_marginal(self._cached_m, self._cached_phi, y, sigma2 + 1e-3)
        return self.prior_regularizer_scaler / X.shape[0] * log_q

    def _train_step(self, optimizer, X, y):
        """Single gradient step."""
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        optimizer.zero_grad(set_to_none=True)
        loss = self.nelbo(X, y)
        if self.use_prior_regularizer:
            prior_reg = self.prior_regularizer(X, y)
            self._prior_reg_buffer.append(prior_reg.detach().item())
            loss = loss + prior_reg
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()
        return loss

    def fit(
        self,
        train_loader,
        warmup_epochs=None,
        warmup_lr=1e-3,
        flow_epochs=None,
        flow_lr=1e-4,
        cosine_annealing=True,
        use_tqdm=False,
        return_loss=False,
    ):
        """Two-phase training: warmup (analytical) then flow (MC).

        Parameters
        ----------
        train_loader : DataLoader
        warmup_epochs : int
            Epochs for Phase 1 (Gaussian warmup).
        warmup_lr : float
            Learning rate for Phase 1.
        flow_epochs : int
            Epochs for Phase 2 (flow fine-tuning).
        flow_lr : float
            Learning rate for Phase 2.
        cosine_annealing : bool
            Use cosine annealing in both phases.
        """
        losses = []

        # Phase 1: Warmup
        if warmup_epochs and warmup_epochs > 0:
            self.set_phase("warmup")
            optimizer = torch.optim.Adam(
                [p for p in self.parameters() if p.requires_grad], lr=warmup_lr
            )
            scheduler = None
            if cosine_annealing:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=warmup_epochs, eta_min=warmup_lr / 100
                )

            self.train()
            loop = (
                tqdm(range(warmup_epochs), unit=" epoch", desc="Warmup")
                if use_tqdm
                else range(warmup_epochs)
            )
            for _ in loop:
                for inputs, target in train_loader:
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)
                    loss = self._train_step(optimizer, inputs, target)
                    if return_loss:
                        losses.append(loss.item())
                if scheduler is not None:
                    scheduler.step()
                if use_tqdm:
                    loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}", phase="warmup")

        # Phase 2: Flow
        if flow_epochs and flow_epochs > 0:
            self.set_phase("flow")
            optimizer = torch.optim.Adam(
                [p for p in self.parameters() if p.requires_grad], lr=flow_lr
            )
            scheduler = None
            if cosine_annealing:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=flow_epochs, eta_min=flow_lr / 100
                )

            self.train()
            loop = (
                tqdm(range(flow_epochs), unit=" epoch", desc="Flow")
                if use_tqdm
                else range(flow_epochs)
            )
            for _ in loop:
                for inputs, target in train_loader:
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)
                    loss = self._train_step(optimizer, inputs, target)
                    if return_loss:
                        losses.append(loss.item())
                if scheduler is not None:
                    scheduler.step()
                if use_tqdm:
                    loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}", phase="flow")

        return losses

    # ------------------------------------------------------------------
    # Properties for diagnostics
    # ------------------------------------------------------------------

    @property
    def KLs(self):
        return self._kl_buffer

    @property
    def base_KLs(self):
        return self._kl_base_buffer

    @property
    def flow_ldj(self):
        return self._kl_ldj_buffer

    @property
    def bb_alphas(self):
        return self._bb_alpha_buffer

    @property
    def prior_regularizers(self):
        return self._prior_reg_buffer
