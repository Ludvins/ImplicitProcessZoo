import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils.utils import infinite_loader
from ..utils.likelihood import (
    gaussian_logp,
    bernoulli_logp,
    multiclass_logp,
    predict_mean_and_var_regression,
    predict_mean_and_var_binary,
    predict_mean_and_var_multiclass,
    gaussian_variational_expectations,
    bernoulli_variational_expectations,
    multiclass_variational_expectations,
)


class SIP(nn.Module):
    """
    Sparse Implicit Process.

    Performs scalable approximate inference on implicit-prior stochastic
    processes (BNN, etc.) via inducing variables u = f(Z) at M inducing
    inputs Z.  Prior moments and cross-covariances are estimated by
    Monte-Carlo sampling from the generative function, then a sparse-GP
    conditional is used to relate f(X) to u.

    The variational posterior q(u) = N(m_u, L_u L_u^T) is parameterized
    per output dimension with a Cholesky factor L_u.

    Parameters
    ----------
    generative_function : torch.nn.Module
        Implicit prior sampler (BayesianNN, GP, …).
    inducing_inputs : torch.Tensor  [M, D_in]
        Initial inducing locations.
    output_dim : int
    likelihood : str
        ``"regression"``, ``"binary"``, or ``"multiclass"``.
    num_data : int
        Total dataset size (for minibatch scaling).
    num_prior_samples : int
        S0 — number of MC samples from the prior used to estimate
        moments / covariances.
    bb_alpha : float
        BB-alpha energy parameter (0 = standard ELBO).
    learn_inducing : bool
        If True, inducing inputs Z are learnable parameters.
    detach_covariances : bool
        If True, stop gradients through the estimated covariance
        matrices (K_ZZ, K_XZ) while still allowing them through
        the means.  Reduces gradient variance at the cost of
        ignoring second-order prior-tuning signal.
    y_mean, y_std : float or array-like
        Target statistics for denormalization (regression).
    num_classes : int or None
        Required for ``likelihood="multiclass"``.
    jitter : float
        Diagonal jitter added to K_ZZ before Cholesky.
    device : torch.device
    dtype : torch dtype
    seed : int
    """

    def __init__(
        self,
        generative_function,
        inducing_inputs,
        output_dim,
        likelihood,
        num_data,
        num_prior_samples=50,
        bb_alpha=0,
        learn_inducing=False,
        detach_covariances=False,
        y_mean=0.0,
        y_std=1.0,
        num_classes=None,
        jitter=1e-6,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
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
        self.bb_alpha = bb_alpha
        self.num_prior_samples = num_prior_samples
        self.learn_inducing = learn_inducing
        self.detach_covariances = detach_covariances
        self.jitter = jitter
        self.output_dim = output_dim
        self.register_buffer(
            "y_mean", torch.as_tensor(y_mean, dtype=dtype, device=device)
        )
        self.register_buffer(
            "y_std", torch.as_tensor(y_std, dtype=dtype, device=device)
        )
        self.device = device
        self.dtype = dtype

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        # --- Generative function (implicit prior sampler) ---
        self.generative_function = generative_function

        # --- Inducing inputs ---
        Z = torch.as_tensor(inducing_inputs, dtype=dtype, device=device)
        if Z.ndim == 1:
            Z = Z.unsqueeze(-1)
        self.num_inducing = Z.shape[0]
        if learn_inducing:
            self.Z = nn.Parameter(Z)
        else:
            self.register_buffer("Z", Z)

        # --- Likelihood-specific parameters ---
        if likelihood == "regression":
            self.log_variance = nn.Parameter(
                torch.tensor(-5.0, dtype=dtype, device=device)
            )
        if likelihood in ("binary", "multiclass"):
            self.num_gauss_hermite_points = 20
        if likelihood == "multiclass":
            self.num_classes = num_classes
            self.epsilon = 1e-3
            self.K1 = self.epsilon / (num_classes - 1)

        # --- Variational parameters q(u) = N(m_u, L_u L_u^T) ---
        M = self.num_inducing
        D = output_dim

        self.m_u = nn.Parameter(
            torch.zeros(M, D, dtype=dtype, device=device)
        )

        # Cholesky factor stored as packed lower-triangular per output dim
        L_init = np.eye(M)
        L_init = np.tile(L_init[:, :, None], [1, 1, D])
        li, lj = torch.tril_indices(M, M)
        L_tri = L_init[li, lj]                             # [tri_len, D]
        self.L_u_tri = nn.Parameter(
            torch.tensor(L_tri, dtype=dtype, device=device)
        )

        self.KLs = []
        self.bb_alphas = []

    # ------------------------------------------------------------------
    # Unpack Cholesky factor
    # ------------------------------------------------------------------

    def _unpack_L_u(self):
        """Unpack the stored lower-triangular Cholesky into [M, M, D]."""
        M = self.num_inducing
        D = self.output_dim
        L = torch.zeros(M, M, D, dtype=self.dtype, device=self.device)
        li, lj = torch.tril_indices(M, M)
        L[li, lj] = self.L_u_tri
        return L

    # ------------------------------------------------------------------
    # Prior moment estimation
    # ------------------------------------------------------------------

    def _estimate_prior_moments(self, X):
        """
        Sample from the implicit prior at the joint set (X, Z) and
        return moment-matched Gaussian statistics.

        Parameters
        ----------
        X : [B, D_in]

        Returns
        -------
        mX   : [B, D]       prior mean at X
        mZ   : [M, D]       prior mean at Z
        KZZ  : [D, M, M]    prior covariance at Z  (+ jitter)
        KXZ  : [D, B, M]    prior cross-covariance
        varX : [D, B]       prior marginal variance at X  (diagonal)
        """
        B = X.shape[0]
        M = self.num_inducing
        Z = self.Z

        # Concatenate X and Z, sample from prior
        XZ = torch.cat([X, Z], dim=0)                      # [B+M, D_in]
        f = self.generative_function(XZ)                    # [S0, B+M, D]
        fX = f[:, :B, :]                                    # [S0, B, D]
        fZ = f[:, B:, :]                                    # [S0, M, D]

        S0 = f.shape[0]

        # Means
        mX = fX.mean(dim=0)                                # [B, D]
        mZ = fZ.mean(dim=0)                                # [M, D]

        # Centered samples
        fX_c = fX - mX.unsqueeze(0)                        # [S0, B, D]
        fZ_c = fZ - mZ.unsqueeze(0)                        # [S0, M, D]

        # Covariances per output dim  (index d last → transpose into leading)
        # KZZ[d] = (1/(S0-1)) fZ_c[:,:,d]^T fZ_c[:,:,d]   shape [M, M]
        # KXZ[d] = (1/(S0-1)) fX_c[:,:,d]^T fZ_c[:,:,d]   shape [B, M]
        denom = max(S0 - 1, 1)

        KZZ_list = []
        KXZ_list = []
        varX_list = []
        for d in range(self.output_dim):
            fz = fZ_c[:, :, d]                              # [S0, M]
            fx = fX_c[:, :, d]                              # [S0, B]

            Kzz = (fz.T @ fz) / denom                      # [M, M]
            Kxz = (fx.T @ fz) / denom                      # [B, M]
            vx = (fx * fx).sum(dim=0) / denom               # [B]

            if self.detach_covariances:
                Kzz = Kzz.detach()
                Kxz = Kxz.detach()
                vx = vx.detach()

            # Jitter
            Kzz = Kzz + self.jitter * torch.eye(
                M, dtype=self.dtype, device=self.device
            )

            KZZ_list.append(Kzz)
            KXZ_list.append(Kxz)
            varX_list.append(vx)

        KZZ = torch.stack(KZZ_list)                         # [D, M, M]
        KXZ = torch.stack(KXZ_list)                         # [D, B, M]
        varX = torch.stack(varX_list)                       # [D, B]

        return mX, mZ, KZZ, KXZ, varX

    # ------------------------------------------------------------------
    # Sparse-GP conditional
    # ------------------------------------------------------------------

    def _sparse_conditional(self, mX, mZ, KZZ, KXZ, varX, u):
        """
        Compute the mean and (diagonal) variance of f(X) | u under the
        moment-matched sparse-GP conditional.

        f(X)|u ~ N( mX + KXZ K_ZZ^{-1} (u - mZ),
                     diag(varX - row_quadform(KXZ, KZZ^{-1})) )

        Parameters
        ----------
        mX   : [B, D]
        mZ   : [M, D]
        KZZ  : [D, M, M]
        KXZ  : [D, B, M]
        varX : [D, B]
        u    : [S, M, D]   inducing samples

        Returns
        -------
        f_mean : [S, B, D]
        f_var  : [D, B]       (shared across samples — depends only on prior)
        """
        S = u.shape[0]
        D = self.output_dim

        f_means = []
        f_vars = []

        for d in range(D):
            Lzz = torch.linalg.cholesky(KZZ[d])            # [M, M]
            kxz = KXZ[d]                                    # [B, M]

            # alpha_d = K_ZZ^{-1} (u_d - m_Z_d)   for each sample
            # u[:, :, d] - mZ[:, d]  →  [S, M]
            diff = u[:, :, d] - mZ[:, d].unsqueeze(0)       # [S, M]
            alpha = torch.cholesky_solve(
                diff.T, Lzz                                 # [M, S]
            ).T                                             # [S, M]

            # Conditional mean: mX_d + KXZ_d @ alpha_d^T  →  [S, B]
            cond_mean = mX[:, d].unsqueeze(0) + (alpha @ kxz.T)  # [S, B]
            f_means.append(cond_mean)

            # Conditional variance (diagonal): varX_d - diag(KXZ K_ZZ^{-1} KXZ^T)
            # = varX_d - sum_m (KXZ @ L^{-T})^2
            V = torch.linalg.solve_triangular(Lzz, kxz.T, upper=False)  # [M, B]
            quad = (V * V).sum(dim=0)                       # [B]
            cond_var = torch.clamp(varX[d] - quad, min=1e-10)
            f_vars.append(cond_var)

        # Stack: [D, S, B] → [S, B, D]
        f_mean = torch.stack(f_means, dim=0).permute(1, 2, 0)
        f_var = torch.stack(f_vars, dim=0)                  # [D, B]

        return f_mean, f_var

    # ------------------------------------------------------------------
    # Sampling from q(u)
    # ------------------------------------------------------------------

    def _sample_u(self, S):
        """Sample S inducing-point values from q(u) = N(m_u, L_u L_u^T).

        Returns
        -------
        u : [S, M, D]
        """
        M = self.num_inducing
        D = self.output_dim
        L = self._unpack_L_u()                              # [M, M, D]

        eps = torch.randn(
            S, M, D, generator=self.generator,
            dtype=self.dtype, device=self.device,
        )

        # u_d = m_u_d + L_d @ eps_d   per output dim
        # L is [M, M, D], eps is [S, M, D]
        # torch.einsum("ijd,sjd->sid", L, eps) contracts over j (inner M)
        u = self.m_u.unsqueeze(0) + torch.einsum("ijd,sjd->sid", L, eps)
        return u                                            # [S, M, D]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_f(self, X):
        """Compute marginal predictive mean and variance at X.

        Integrates out u analytically under q(u), using the sparse
        conditional.

        Returns
        -------
        mean : [B, D]
        var  : [B, D]   (diagonal)
        """
        mX, mZ, KZZ, KXZ, varX = self._estimate_prior_moments(X)

        D = self.output_dim
        L_u = self._unpack_L_u()

        means = []
        vars_ = []

        for d in range(D):
            Lzz = torch.linalg.cholesky(KZZ[d])
            kxz = KXZ[d]                                    # [B, M]

            # Mean: mX_d + KXZ K_ZZ^{-1} (m_u_d - mZ_d)
            diff = self.m_u[:, d] - mZ[:, d]                # [M]
            alpha = torch.cholesky_solve(
                diff.unsqueeze(-1), Lzz
            ).squeeze(-1)                                   # [M]
            pred_mean = mX[:, d] + kxz @ alpha              # [B]
            means.append(pred_mean)

            # Variance: varX_d - diag(KXZ K_ZZ^{-1} KXZ^T)
            #         + diag(KXZ K_ZZ^{-1} S_u K_ZZ^{-1} KXZ^T)
            V = torch.linalg.solve_triangular(
                Lzz, kxz.T, upper=False
            )                                               # [M, B]
            prior_quad = (V * V).sum(dim=0)                 # [B]

            # S_u contribution: K_ZZ^{-1} L_u L_u^T K_ZZ^{-1}
            # KXZ K_ZZ^{-1} L_u  =  V^T @ (Lzz^{-1} L_u_d)
            L_d = L_u[:, :, d]                              # [M, M]
            W = torch.linalg.solve_triangular(
                Lzz, L_d, upper=False
            )                                               # [M, M]
            # KXZ K_ZZ^{-1} L_u = (kxz @ K_ZZ^{-1}) @ L_u_d
            # = V^T @ W   →  [B, M]
            VtW = V.T @ W                                   # [B, M]
            posterior_quad = (VtW * VtW).sum(dim=-1)         # [B]

            pred_var = torch.clamp(
                varX[d] - prior_quad + posterior_quad, min=1e-10
            )
            vars_.append(pred_var)

        mean = torch.stack(means, dim=-1)                   # [B, D]
        var = torch.stack(vars_, dim=-1)                    # [B, D]
        return mean, var

    def predict_f_samples(self, X, S):
        """Sample latent function values from the approximate posterior.

        Samples u ~ q(u), then draws f(X) from the sparse conditional
        f(X) | u.

        Parameters
        ----------
        X : [B, D_in]
        S : int

        Returns
        -------
        F : [S, B, D]
        """
        mX, mZ, KZZ, KXZ, varX = self._estimate_prior_moments(X)

        u = self._sample_u(S)                               # [S, M, D]

        # Conditional mean and (diagonal) variance
        f_mean, f_var = self._sparse_conditional(
            mX, mZ, KZZ, KXZ, varX, u
        )
        # f_mean: [S, B, D],  f_var: [D, B]

        # Sample: f = f_mean + sqrt(f_var) * eta
        f_std = torch.sqrt(f_var).permute(1, 0).unsqueeze(0)  # [1, B, D]
        eta = torch.randn(
            S, X.shape[0], self.output_dim,
            generator=self.generator,
            dtype=self.dtype, device=self.device,
        )
        return f_mean + f_std * eta                         # [S, B, D]

    def predict_y_samples(self, X, S):
        """Predictive y samples.

        Regression: f + Gaussian noise.
        Classification: raw logits.
        """
        F = self.predict_f_samples(X, S)
        if self.likelihood_type == "regression":
            std = torch.sqrt(torch.exp(self.log_variance))
            return F + std * torch.randn_like(F)
        return F

    # ------------------------------------------------------------------
    # KL divergence  KL(q(u) || p(u))
    # ------------------------------------------------------------------

    def _KL(self, mZ, KZZ):
        """KL( q(u) || p(u) )  where  p(u) = N(mZ, KZZ).

        Per output dim d:
            KL_d = ½ [ tr(K_ZZ^{-1} S_u) + (mZ - m_u)^T K_ZZ^{-1} (mZ - m_u)
                        - M + log det K_ZZ - log det S_u ]

        Returns scalar.
        """
        M = self.num_inducing
        L_u = self._unpack_L_u()                            # [M, M, D]
        kl = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        for d in range(self.output_dim):
            Lzz = torch.linalg.cholesky(KZZ[d])             # [M, M]
            L_d = L_u[:, :, d]                               # [M, M]
            mu_diff = mZ[:, d] - self.m_u[:, d]              # [M]

            # log det K_ZZ = 2 sum log diag(Lzz)
            log_det_Kzz = 2.0 * torch.sum(torch.log(torch.diagonal(Lzz)))

            # log det S_u = 2 sum log |diag(L_u)|
            log_det_Su = 2.0 * torch.sum(
                torch.log(torch.abs(torch.diagonal(L_d)))
            )

            # tr(K_ZZ^{-1} S_u) = tr(Lzz^{-1} L_d (Lzz^{-1} L_d)^T)
            #                    = ||Lzz^{-1} L_d||_F^2
            W = torch.linalg.solve_triangular(Lzz, L_d, upper=False)
            trace_term = (W * W).sum()

            # Quadratic: mu_diff^T K_ZZ^{-1} mu_diff
            alpha = torch.cholesky_solve(
                mu_diff.unsqueeze(-1), Lzz
            ).squeeze(-1)
            quad = (mu_diff * alpha).sum()

            kl = kl + 0.5 * (
                trace_term + quad - M + log_det_Kzz - log_det_Su
            )

        return kl

    # ------------------------------------------------------------------
    # Likelihood dispatch
    # ------------------------------------------------------------------

    def _logp(self, F, Y):
        if self.likelihood_type == "regression":
            return gaussian_logp(F, Y, self.log_variance)
        elif self.likelihood_type == "binary":
            return bernoulli_logp(F, Y)
        else:
            return multiclass_logp(F, Y, self.num_classes, self.epsilon)

    def _variational_expectations(self, Fmu, Fvar, Y):
        if self.likelihood_type == "regression":
            return gaussian_variational_expectations(
                Fmu, Fvar, Y, self.bb_alpha, self.log_variance
            )
        elif self.likelihood_type == "binary":
            return bernoulli_variational_expectations(
                Fmu, Fvar, Y, self.num_gauss_hermite_points,
                self.dtype, self.device,
            )
        else:
            return multiclass_variational_expectations(
                Fmu, Fvar, Y, self.num_classes, self.epsilon,
                self.num_gauss_hermite_points, self.dtype, self.device,
            )

    def _predict_mean_and_var(self, Fmu, Fvar):
        if self.likelihood_type == "regression":
            return predict_mean_and_var_regression(Fmu, Fvar, self.log_variance)
        elif self.likelihood_type == "binary":
            return predict_mean_and_var_binary(Fmu, Fvar)
        else:
            return predict_mean_and_var_multiclass(
                Fmu, Fvar, self.num_classes, self.epsilon,
                self.num_gauss_hermite_points, self.dtype, self.device,
            )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nelbo(self, X, y):
        """Negative ELBO (or BB-alpha energy) objective.

        loss = -scale * E_q[ log p(y | f(X)) ]  +  KL(q(u) || p(u))
        """
        mX, mZ, KZZ, KXZ, varX = self._estimate_prior_moments(X)

        # --- Expected log-likelihood via variational expectations ---
        # Marginal predictive q(f) at X (integrated over q(u)):
        #   mean, var = sparse conditional moments under q(u)
        D = self.output_dim
        L_u = self._unpack_L_u()

        pred_means = []
        pred_vars = []

        for d in range(D):
            Lzz = torch.linalg.cholesky(KZZ[d])
            kxz = KXZ[d]                                    # [B, M]

            # Predictive mean
            diff = self.m_u[:, d] - mZ[:, d]
            alpha = torch.cholesky_solve(
                diff.unsqueeze(-1), Lzz
            ).squeeze(-1)
            pred_mean = mX[:, d] + kxz @ alpha              # [B]
            pred_means.append(pred_mean)

            # Predictive variance
            V = torch.linalg.solve_triangular(Lzz, kxz.T, upper=False)
            prior_quad = (V * V).sum(dim=0)

            L_d = L_u[:, :, d]
            W = torch.linalg.solve_triangular(Lzz, L_d, upper=False)
            VtW = V.T @ W
            posterior_quad = (VtW * VtW).sum(dim=-1)

            pred_var = torch.clamp(
                varX[d] - prior_quad + posterior_quad, min=1e-10
            )
            pred_vars.append(pred_var)

        F_mean = torch.stack(pred_means, dim=-1)            # [B, D]
        F_var = torch.stack(pred_vars, dim=-1)              # [B, D]

        # Variational expectations E_q(f)[ log p(y | f) ]
        ve = self._variational_expectations(
            F_mean.unsqueeze(0), F_var.unsqueeze(0), y
        )
        ve = torch.sum(ve)
        scale = self.num_data / X.shape[0]

        # --- KL ---
        KL = self._KL(mZ, KZZ)

        self.bb_alphas.append((-scale * ve).detach().cpu().numpy())
        self.KLs.append(KL.detach().cpu().numpy())

        return -scale * ve + KL

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, predict_at):
        """Returns (mean, std) in the original target scale."""
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        Fmean, Fvar = self.predict_f(predict_at)
        mean, var = self._predict_mean_and_var(
            Fmean.unsqueeze(0), Fvar.unsqueeze(0)
        )
        return mean * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std

    def forward_prior(self, predict_at, num_samples):
        """Sample from the prior (without variational conditioning)."""
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)
        f = self.generative_function(predict_at)            # [S0, N, D]
        return f[:num_samples] * self.y_std + self.y_mean

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
            If None, creates Adam with the given lr.
        lr : float
        epochs : int or None
        iterations : int or None
        use_tqdm : bool
        return_loss : bool
        cosine_annealing : bool
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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

    def predict(self, data_loader, device=None):
        """
        Run predictions over a DataLoader.

        Returns
        -------
        all_means : torch.Tensor
        all_stds  : torch.Tensor
        """
        if device is None:
            device = self.device

        self.eval()
        all_means = []
        all_stds = []

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                mean, std = self(inputs)
                all_means.append(mean.cpu())
                all_stds.append(std.cpu())

        return torch.cat(all_means, dim=0), torch.cat(all_stds, dim=0)

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
