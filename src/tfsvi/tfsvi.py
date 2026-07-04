import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils.utils import infinite_loader
from ..utils.likelihood import (
    gaussian_logp,
    bernoulli_logp,
    multiclass_logp,
    inv_probit,
)
from ..utils.flat_mlp import (
    FlatMLP as _BaseMLP,
    unflatten_params as _unflatten_params_fn,
    forward_with_flat_params as _forward_with_flat_params_fn,
)


# ------------------------------------------------------------------
# Tractable Function-Space Variational Inference
# ------------------------------------------------------------------

class TFSVI(nn.Module):
    """
    Tractable Function-Space Variational Inference.

    Mean-field Gaussian q(theta)=N(mu, diag(sigma^2)) over parameters,
    with an isotropic Gaussian prior p(theta)=N(0, sigma_prior^2 I).
    The function-space KL is made tractable by linearizing the network
    mapping at theta=mu (first-order Taylor approximation), yielding
    Gaussian induced distributions q_f~ and p_f~ whose KL is analytic.

    The supremum over context sets is approximated by sampling S_ctx
    context sets of size K_ctx and taking the max KL.

    Reference: Rudner et al., "Tractable Function-Space Variational
    Inference in Bayesian Neural Networks" (2022).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        structure,
        activation,
        likelihood,
        num_data,
        sigma_prior=1.0,
        num_samples=20,
        bb_alpha=0,
        S_ctx=5,
        K_ctx=20,
        y_mean=0.0,
        y_std=1.0,
        num_classes=None,
        generative_function=None,
        device=None,
        dtype=torch.float64,
    ):
        """
        Parameters
        ----------
        input_dim : int
        output_dim : int
            Target dimensionality (D).
        structure : list of int
            Hidden layer widths, e.g. [50, 50].
        activation : callable
            Activation function (e.g. torch.nn.Tanh()).
        likelihood : str
            One of "regression", "binary", or "multiclass".
        num_data : int
            Total dataset size (for minibatch scaling).
        sigma_prior : float
            Prior std: p(theta) = N(0, sigma_prior^2 I).
        num_samples : int
            MC parameter samples for expected log-likelihood (S_lik
            in the paper).
        bb_alpha : float
            Alpha for BB-alpha energy (0 = ELBO).
        S_ctx : int
            Number of context sets sampled for the max-KL estimator.
        K_ctx : int
            Number of points per context set.
        y_mean, y_std : float or array-like
            Target statistics for denormalization (regression only).
        num_classes : int or None
            Required for multiclass.
        generative_function : torch.nn.Module or None
            If provided, used as the predictive function ``f(X; theta)``
            instead of the default deterministic ``FlatMLP``. The
            variational posterior ``q(theta)`` is placed over **all**
            ``generative_function.named_parameters()``: TFSVI initialises
            ``mu`` from the module's current parameter values and learns
            ``mu`` and ``log_sigma`` over that flat vector. Forward passes
            are done via ``torch.func.functional_call`` so any ``nn.Module``
            is supported.

            For Bayesian-layer generative functions (e.g.
            ``BayesianNN(BayesLinear)``, ``BayesianNN(SimplerBayesLinear)``),
            use ``num_samples=1`` and ``fix_random_noise=True`` so the
            forward becomes deterministic in ``theta`` and the linearised
            KL is well-defined.
        device : torch.device
        dtype : torch dtype
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
        self.sigma_prior = sigma_prior
        self.num_samples = num_samples
        self.bb_alpha = bb_alpha
        self.S_ctx = S_ctx
        self.K_ctx = K_ctx
        # Match log_variance / parameter dtype so std_scalar = sqrt(exp(
        # log_variance)) * y_std doesn't get silently downcast to float32
        # and underflow when log_variance dips below ~-88.
        self.y_mean = torch.as_tensor(y_mean, dtype=dtype, device=device)
        self.y_std = torch.as_tensor(y_std, dtype=dtype, device=device)
        self.device = device
        self.dtype = dtype
        self.output_dim = output_dim

        if likelihood == "multiclass":
            self.num_classes = num_classes
            self.epsilon = 1e-3

        # Network output dimension
        if likelihood == "multiclass":
            net_output_dim = num_classes
        else:
            net_output_dim = output_dim

        # Build base network: either the default FlatMLP or a user-supplied
        # generative function.  In both cases the module is treated as a
        # frozen *architecture template* — its parameters become read-only
        # and TFSVI's own (mu, log_sigma) act as the variational q(theta).
        if generative_function is None:
            self.base_net = _BaseMLP(
                input_dim, net_output_dim, structure, activation,
                dtype=dtype, device=device,
            )
            self._use_functional_call = False
        else:
            self.base_net = generative_function
            # Custom generative functions may produce shapes the FlatMLP
            # fast path can't unflatten, so we route through
            # torch.func.functional_call.
            self._use_functional_call = True
        self._num_layers = (
            len(self.base_net.layers) if hasattr(self.base_net, "layers") else None
        )

        # vmap has no batching rule for `aten::lstm.input` (and friends),
        # and lazy random sampling inside the forward (e.g. ExactGP / GP
        # populating their cached noise on first call) violates vmap's
        # randomness contract. Fall back to Python-loop / vectorize=False
        # for any base_net containing such modules.
        from ..priors.generative_functions import ExactGP, GP
        self._vmap_compatible = not any(
            isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN,
                           ExactGP, GP))
            for m in self.base_net.modules()
        )

        # Collect parameter shapes and freeze base network
        self._param_names = []
        self._param_shapes = []
        init_values = []
        for name, p in self.base_net.named_parameters():
            self._param_names.append(name)
            self._param_shapes.append(p.shape)
            init_values.append(p.data.flatten())
            p.requires_grad = False

        total_params = sum(s.numel() for s in self._param_shapes)
        self._total_params = total_params

        # Variational parameters: q(theta) = N(mu, diag(exp(2*log_sigma)))
        self.mu = nn.Parameter(torch.cat(init_values))
        self.log_sigma = nn.Parameter(
            torch.full((total_params,), -5.0, dtype=dtype, device=device)
        )

        # Likelihood noise (regression only)
        if likelihood == "regression":
            self.log_variance = nn.Parameter(
                torch.tensor(-5.0, dtype=dtype, device=device)
            )

        self.KLs = []
        self.bb_alphas = []
        self._train_inputs = None

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------

    def vi_parameters(self):
        """Trainable variational parameters.

        ``base_net`` is a frozen architecture template.  Keeping it as a
        submodule is useful for ``functional_call``, but optimizers should only
        see TFSVI's flat variational parameters and, for regression, the
        likelihood noise parameter.
        """
        params = [self.mu, self.log_sigma]
        if hasattr(self, "log_variance"):
            params.append(self.log_variance)
        return params

    def _unflatten_params(self, flat):
        """Reshape flat parameter vector to named dict."""
        return _unflatten_params_fn(flat, self._param_names, self._param_shapes)

    def _forward_with_flat_params(self, flat_params, x):
        """Forward pass through the generative function using supplied flat params.

        For the default FlatMLP, dispatches to the dedicated fast path. For
        any other generative function (e.g. ``BayesianNN``), uses
        ``torch.func.functional_call`` to override parameters statelessly,
        and squeezes a leading singleton dim if the generative function
        returns ``[1, N, D]`` (e.g. BayesianNN with ``num_samples=1``).
        """
        if not self._use_functional_call:
            return _forward_with_flat_params_fn(self.base_net, flat_params, x)
        params = self._unflatten_params(flat_params)
        out = torch.func.functional_call(self.base_net, params, (x,))
        if out.ndim >= 3 and out.shape[0] == 1:
            out = out.squeeze(0)
        return out

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    def predict_f_samples(self, X, S):
        """
        Sample latent function values from the variational posterior.

        Samples theta_s ~ q(theta) via reparameterization and computes
        f(X; theta_s) for each sample.  The S parameter samples are
        evaluated in parallel via :func:`torch.vmap` over the leading
        dim of ``theta`` (≈ 12x faster than a Python loop on this
        workload, bit-identical output).

        Parameters
        ----------
        X : [N, D_in]
        S : int, number of MC parameter samples

        Returns
        -------
        F : [S, N, D]
        """
        eps = torch.randn(
            S, self._total_params, dtype=self.dtype, device=self.device
        )
        sigma = torch.exp(self.log_sigma)
        theta = self.mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # [S, P]

        if self._vmap_compatible:
            return torch.vmap(
                lambda flat: self._forward_with_flat_params(flat, X)
            )(theta)  # [S, N, D]
        # Recurrent base_net: vmap has no batching rule for aten::lstm,
        # so loop over the S parameter samples.
        return torch.stack(
            [self._forward_with_flat_params(theta[s], X) for s in range(S)],
            dim=0,
        )

    def predict_y_samples(self, X, S):
        """
        Predictive samples in y-space.

        Regression: f + Gaussian noise.
        Classification: logits (raw).

        Returns
        -------
        Y : [S, N, D]
        """
        F = self.predict_f_samples(X, S)
        if self.likelihood_type == "regression":
            std = torch.sqrt(torch.exp(self.log_variance))
            return F + std * torch.randn_like(F)
        return F  # logits for classification

    def _logp(self, F, Y):
        """Dispatch to the correct logp function."""
        if self.likelihood_type == "regression":
            return gaussian_logp(F, Y, self.log_variance)
        elif self.likelihood_type == "binary":
            return bernoulli_logp(F, Y)
        else:
            return multiclass_logp(F, Y, self.num_classes, self.epsilon)

    # ------------------------------------------------------------------
    # Linearized function-space KL
    # ------------------------------------------------------------------

    def _compute_jacobian(self, X_ctx):
        """
        Jacobian of f(X_ctx; theta) w.r.t. flat theta, evaluated at theta=mu.

        Uses ``torch.autograd.functional.jacobian`` with
        ``vectorize=True``, which is always faster than the default
        loop-based path (≈ 5-9x).  The mode is auto-selected to
        minimise the number of AD passes:

        * **forward-mode** when ``P < K_ctx * D`` — only ``P`` JVPs
          (best for low-dim posteriors like a SimplerBayesLinear backbone,
          where P≈12 vs K_ctx*D≈20).
        * **reverse-mode** otherwise — ``K_ctx * D`` VJPs
          (best for full per-weight posteriors / FlatMLP backbones with
          P≈hundreds).

        Returns
        -------
        J : [K_ctx, D, P]   — un-flattened so the per-output-dim KL can
            slice each output's Jacobian directly (matches the
            FSVI reference, trainer.py L1373).
        """
        mu_data = self.mu.detach()
        K_ctx = X_ctx.shape[0]
        out_dim = (
            self.num_classes if self.likelihood_type == "multiclass"
            else self.output_dim
        )
        # Forward-mode + vectorize=False is unsupported by autograd.functional;
        # force reverse-mode whenever we've disabled vectorize to keep the
        # vmap-incompatible base_net path (LSTM / GP) working.
        if self._vmap_compatible:
            strategy = (
                "forward-mode" if mu_data.numel() < K_ctx * out_dim
                else "reverse-mode"
            )
        else:
            strategy = "reverse-mode"

        def f_fn(flat_params):
            return self._forward_with_flat_params(flat_params, X_ctx)

        return torch.autograd.functional.jacobian(
            f_fn, mu_data,
            vectorize=self._vmap_compatible, strategy=strategy,
        )

    @staticmethod
    def _psd_safe_cholesky(M, base_jitter=1e-6, max_attempts=4):
        """Cholesky with adaptive-jitter retry — gpytorch's psd_safe pattern.

        Symmetrizes the matrix, scales the jitter to the diagonal magnitude,
        and retries with 10x larger jitter on failure (up to ``max_attempts``).
        Falls through with the last raised exception if every attempt fails.
        """
        d = M.shape[-1]
        I = torch.eye(d, dtype=M.dtype, device=M.device)
        # Symmetrize (cheap float-noise hygiene)
        M = 0.5 * (M + M.transpose(-1, -2))
        diag_max = torch.diagonal(M, dim1=-2, dim2=-1).abs().max().clamp(min=1.0)
        last_err = None
        for k in range(max_attempts):
            jitter = base_jitter * (10 ** k) * diag_max
            try:
                return torch.linalg.cholesky(M + jitter * I)
            except torch._C._LinAlgError as e:
                last_err = e
        raise last_err

    @classmethod
    def _gaussian_kl(cls, m0, S0, m1, S1):
        """
        KL( N(m0, S0) || N(m1, S1) ).

        Parameters
        ----------
        m0, m1 : [d]
        S0, S1 : [d, d]  (will be symmetrized + jittered as needed)
        """
        d = m0.shape[0]
        L1 = cls._psd_safe_cholesky(S1)
        L0 = cls._psd_safe_cholesky(S0)

        log_det_S1 = 2.0 * torch.sum(torch.log(torch.diagonal(L1)))
        log_det_S0 = 2.0 * torch.sum(torch.log(torch.diagonal(L0)))

        # tr(S1^{-1} S0)
        S1_inv_S0 = torch.cholesky_solve(S0, L1)
        trace_term = torch.trace(S1_inv_S0)

        # (m1 - m0)^T S1^{-1} (m1 - m0)
        diff = (m1 - m0).unsqueeze(-1)  # [d, 1]
        quad = (diff.T @ torch.cholesky_solve(diff, L1)).squeeze()

        return 0.5 * (trace_term + quad - d + log_det_S1 - log_det_S0)

    def _compute_linearized_kl(self, X_ctx):
        """
        KL( q_f~(X_ctx) || p_f~(X_ctx) ) under first-order linearization
        at theta = mu, summed over output dimensions.

        Linearization:
            f(X; theta) ~ f(X; mu) + J (theta - mu)

        Per-output-dim induced distributions (matches FSVI ref,
        trainer.py L1373-L1394 — sum scalar-output KLs over output dims
        rather than build one full [K*D, K*D] covariance):

            q_f~^{(d)} = N( f_d(X;mu),                    J_d diag(sigma^2) J_d^T )
            p_f~^{(d)} = N( f_d(X;mu) - J_d mu,           sigma_prior^2 J_d J_d^T )
        """
        # Jacobian at mu (detached — no second-order gradients through J).
        # Shape [K_ctx, D, P].
        J = self._compute_jacobian(X_ctx)
        D = J.shape[1]

        # f(X_ctx; mu) — gradient flows through mu.  Shape [K_ctx, D].
        f_mu = self._forward_with_flat_params(self.mu, X_ctx)

        # J @ mu summed over params, per (k, d).  Shape [K_ctx, D].
        J_mu = (J * self.mu).sum(dim=-1)

        sigma_sq = torch.exp(2.0 * self.log_sigma)            # [P]
        sigma_p_sq = self.sigma_prior ** 2

        kl = f_mu.new_zeros(())
        for d in range(D):
            J_d = J[:, d, :]                                  # [K_ctx, P]
            m_q_d = f_mu[:, d]                                # [K_ctx]
            m_p_d = m_q_d - J_mu[:, d]                        # [K_ctx]

            S_q_d = (J_d * sigma_sq) @ J_d.T                  # [K_ctx, K_ctx]
            S_p_d = sigma_p_sq * (J_d @ J_d.T)                # [K_ctx, K_ctx]

            kl = kl + self._gaussian_kl(m_q_d, S_q_d, m_p_d, S_p_d)
        return kl

    def _sample_context(self, K):
        """Sample K context points from stored training data."""
        if self._train_inputs is None:
            raise RuntimeError("No training data stored. Call fit() first.")
        N = self._train_inputs.shape[0]
        idx = torch.randperm(N, device=self._train_inputs.device)[:min(K, N)]
        X_ctx = self._train_inputs[idx].to(self.device)
        if self.dtype != X_ctx.dtype:
            X_ctx = X_ctx.to(self.dtype)
        return X_ctx

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nelbo(self, X_batch, y):
        """
        Function-space variational objective.

        loss = -scale * E_q[log p(y|f)]  +  max_i KL_i

        where KL_i = KL(q_f~(X_C^i) || p_f~(X_C^i)) under linearization,
        and the max is over S_ctx randomly sampled context sets.
        """
        N_batch = X_batch.shape[0]

        # --- Expected log-likelihood (MC over parameter samples) ---
        F = self.predict_f_samples(X_batch, self.num_samples)  # [S, N, D]
        logpdf = self._logp(F, y)
        if self.bb_alpha == 0:
            ve = torch.mean(logpdf, dim=0)
        else:
            ve = (
                torch.logsumexp(self.bb_alpha * logpdf, dim=0)
                - torch.log(torch.tensor(
                    F.shape[0], dtype=self.dtype, device=self.device
                ))
            ) / self.bb_alpha
        ve = torch.sum(ve)
        scale = self.num_data / N_batch
        loss_like = -scale * ve

        # --- Function-space KL (max over context sets) ---
        kls = []
        for _ in range(self.S_ctx):
            X_ctx = self._sample_context(self.K_ctx)
            kl = self._compute_linearized_kl(X_ctx)
            kls.append(kl)
        kl_max = torch.max(torch.stack(kls))

        self.bb_alphas.append(loss_like.detach().cpu().numpy())
        self.KLs.append(kl_max.detach().cpu().numpy())

        return loss_like + kl_max

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
        """
        Predict y_samples for evaluation.

        Parameters
        ----------
        X : [N, D_in]
        S : int, number of predictive samples

        Returns
        -------
        Y : [S, N, D] denormalized for regression, logits for classification
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
        """
        Train the model.

        Parameters
        ----------
        train_loader : DataLoader
        optimizer : torch.optim.Optimizer or None
        lr : float
        epochs : int or None
        iterations : int or None
        use_tqdm : bool
        return_loss : bool
        cosine_annealing : bool
        """
        # Store training inputs for context sampling
        all_X = []
        for inputs, _ in train_loader:
            all_X.append(inputs)
        self._train_inputs = torch.cat(all_X, dim=0)

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
            loop = (
                tqdm(range(epochs), unit=" epoch", desc="Training")
                if use_tqdm else range(epochs)
            )
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
            loop = (
                tqdm(range(iterations), unit=" iter", desc="Training")
                if use_tqdm else range(iterations)
            )
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
        """Single gradient step."""
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        optimizer.zero_grad()
        loss = self.nelbo(X, y)
        loss.backward()
        optimizer.step()
        return loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def print_variables(self):
        """Prints the model parameters in a formatted manner."""
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
