import math

import torch

from ..utils.likelihood import (
    bernoulli_logp,
    gaussian_logp,
    inv_probit,
    multiclass_logp,
)
from ..utils.model_info import print_trainable_parameters
from ..utils.random import fork_torch_rng, preserve_constructor_rng, temporary_generator_seed
from ..utils.training import fit_loop, make_cosine_scheduler, validate_fit_mode

# ------------------------------------------------------------------
# Spectral Stein Gradient Estimator (SSGE)
# ------------------------------------------------------------------


class SSGE:
    """
    Spectral Stein Gradient Estimator.

    Estimates the score function nabla_x log q(x) from samples using
    the Stein identity with an RBF kernel and truncated eigendecomposition.

    Reference: Shi et al., "A Spectral Approach to Gradient Estimation
    for Implicit Distributions" (2018).
    """

    def __init__(self, num_eigs=None, nugget=1e-4):
        """
        Parameters
        ----------
        num_eigs : int or None
            Number of eigenvectors to retain (J). If None, uses min(S, 20).
        nugget : float
            Diagonal regularization added to the kernel matrix before
            eigendecomposition, preventing ill-conditioning when pairwise
            distances concentrate (e.g. in high dimensions).
        """
        self.num_eigs = num_eigs
        self.nugget = nugget

    def _rbf_kernel(self, X):
        """
        RBF kernel matrix with median heuristic bandwidth.

        Parameters
        ----------
        X : [S, D] — samples

        Returns
        -------
        K : [S, S] — kernel matrix
        h : scalar — bandwidth
        """
        S, D = X.shape
        sq_dists = torch.cdist(X, X, p=2).square()

        # Use upper-triangle only (excludes diagonal self-distances = 0)
        triu_idx = torch.triu_indices(S, S, offset=1, device=X.device)
        pairwise = sq_dists[triu_idx[0], triu_idx[1]]

        if pairwise.numel() == 0 or pairwise.max() < 1e-10:
            h = torch.tensor(1.0, dtype=X.dtype, device=X.device)
        else:
            median_val = torch.median(pairwise)
            h = median_val / (2.0 * math.log(S + 1))
            h = torch.clamp(h, min=1e-5)
        K = torch.exp(-sq_dists / (2.0 * h))
        return K, h

    def _compute_grad_K(self, X, K, h):
        """
        Compute G[j, d] = sum_i grad_{x_j} k(x_i, x_j).

        For RBF: grad_{x_j} k(x_i, x_j) = k(x_i, x_j) * (x_i - x_j) / h

        Parameters
        ----------
        X : [S, D]
        K : [S, S]
        h : scalar

        Returns
        -------
        G : [S, D]
        """
        # diff[i, j, d] = X[i, d] - X[j, d]
        diff = X.unsqueeze(0) - X.unsqueeze(1)  # [S, S, D]
        # K[i, j] * (X[i] - X[j]) / h, summed over i
        G = torch.einsum("ij,ijd->jd", K, diff) / h
        return G

    def compute_score(self, X):
        """
        Estimate the score nabla_x log q(x) from samples X.

        Uses the Stein identity: K @ s = -G, solved via truncated
        eigendecomposition s ≈ -(U_J Lambda_J^{-1} U_J^T) G.

        Parameters
        ----------
        X : [S, D] — samples from q(x)

        Returns
        -------
        score : [S, D] — estimated score at each sample point
        """
        S, D = X.shape
        J = self.num_eigs if self.num_eigs is not None else min(S, 20)
        J = min(J, S)

        K, h = self._rbf_kernel(X)
        G = self._compute_grad_K(X, K, h)

        # Nugget-regularized eigendecomposition
        K_reg = K + self.nugget * torch.eye(S, dtype=K.dtype, device=K.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(K_reg)
        # eigh returns ascending order; take last J
        eigenvalues = eigenvalues[-J:]
        eigenvectors = eigenvectors[:, -J:]

        # Regularize: clamp small eigenvalues
        eigenvalues = torch.clamp(eigenvalues, min=1e-5)

        # s = -(U Lambda^{-1} U^T) G
        # = -U (Lambda^{-1} (U^T G))
        UtG = eigenvectors.T @ G  # [J, D]
        score = -eigenvectors @ (UtG / eigenvalues.unsqueeze(-1))  # [S, D]

        return score

    def compute_score_nystrom(self, X_ref, X_eval):
        """
        Estimate score at X_eval using Nystrom extension from X_ref samples.

        The eigenvectors at X_ref define the kernel eigenspace. For a new
        point x', the Nystrom extension gives:
            phi_j(x') = (1 / lambda_j) * sum_i k(x_i, x') * u_j[i]

        Then score(x') = -sum_j phi_j(x') * (U_j^T G) / lambda_j

        Parameters
        ----------
        X_ref : [S_ref, D] — reference samples (used to build kernel)
        X_eval : [S_eval, D] — points at which to evaluate the score

        Returns
        -------
        score : [S_eval, D]
        """
        assert X_ref.ndim == 2 and X_eval.ndim == 2, (
            f"Expected 2D inputs, got X_ref {X_ref.shape}, X_eval {X_eval.shape}"
        )
        assert X_ref.shape[-1] == X_eval.shape[-1], (
            f"Dimension mismatch: X_ref[-1]={X_ref.shape[-1]}, X_eval[-1]={X_eval.shape[-1]}"
        )

        S_ref = X_ref.shape[0]
        J = self.num_eigs if self.num_eigs is not None else min(S_ref, 20)
        J = min(J, S_ref)

        K_ref, h = self._rbf_kernel(X_ref)
        G_ref = self._compute_grad_K(X_ref, K_ref, h)

        # Nugget-regularized eigendecomposition
        K_reg = K_ref + self.nugget * torch.eye(S_ref, dtype=K_ref.dtype, device=K_ref.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(K_reg)
        eigenvalues = eigenvalues[-J:]
        eigenvectors = eigenvectors[:, -J:]
        eigenvalues = torch.clamp(eigenvalues, min=1e-5)

        # Cross-kernel: k(x_ref_i, x_eval_j)
        sq_dists = torch.cdist(X_ref, X_eval, p=2).square()
        K_cross = torch.exp(-sq_dists / (2.0 * h))  # [S_ref, S_eval]

        # Nystrom eigenvectors at eval points: [S_eval, J]
        phi_eval = (K_cross.T @ eigenvectors) / eigenvalues.unsqueeze(0)

        # Score at eval points using reference gradients G_ref.
        # phi_eval contains 1/λ (Nystrom eigenvector extension), and the
        # coefficient c_j = (1/λ_j) * (u_j^T G_ref) needs its own 1/λ.
        # Consistency check: at reference points phi_eval[i,j] = u_j[i]
        # (since K u_j = λ_j u_j cancels the 1/λ), so we recover
        # score = -U (Λ^{-1} U^T G_ref), matching compute_score.
        UtG = eigenvectors.T @ G_ref  # [J, D]
        score = -phi_eval @ (UtG / eigenvalues.unsqueeze(-1))  # [S_eval, D]

        return score


# ------------------------------------------------------------------
# Functional Bayesian Neural Network (fBNN)
# ------------------------------------------------------------------


@preserve_constructor_rng
class FBNN(torch.nn.Module):
    """
    Functional Bayesian Neural Network.

    Function-space variational inference using SSGE for score estimation.
    The posterior is a BayesianNN with trainable variational parameters.
    The prior is a GP or BNN with frozen parameters.

    Reference: Sun et al., "Functional Variational Bayesian Neural Networks"
    (2019).
    """

    def __init__(
        self,
        generative_function,
        prior_function,
        output_dim,
        likelihood,
        num_data,
        num_samples=20,
        num_measurement=20,
        num_context=20,
        context_std=2.0,
        bb_alpha=0,
        lambda_kl=1.0,
        num_eigs=None,
        nugget=1e-4,
        gamma2=1e-4,
        reservoir_size=1000,
        y_mean=0.0,
        y_std=1.0,
        num_classes=None,
        freeze_prior=True,
        device=None,
        dtype=torch.float64,
    ):
        """
        Parameters
        ----------
        generative_function : BayesianNN
            Posterior BNN (fix_random_noise=True, trainable parameters).
        prior_function : GP or BayesianNN
            Prior function sampler (parameters will be frozen).
            BNN priors must use fix_random_noise=True for deterministic samples.
        output_dim : int
            Dimensionality of the output.
        likelihood : str
            One of "regression", "binary", or "multiclass".
        num_data : int
            Total dataset size (for minibatch scaling).
        num_samples : int
            Number of MC function samples for training.
        num_measurement : int
            Number of measurement points sampled from training data (X_D).
        num_context : int
            Number of context points sampled from the input domain (X_M).
            These regularize the posterior in OOD regions where no training
            data exists, which is the key contribution of fBNN.
        context_std : float
            Standard deviation of the Gaussian from which context points are
            sampled.  Since inputs are normalized to zero-mean unit-variance,
            values > 1 cover out-of-distribution regions.
        bb_alpha : float
            Alpha for BB-alpha energy (0 = ELBO).
        lambda_kl : float
            Weight on the functional KL term.
        num_eigs : int or None
            Number of eigenvectors for SSGE.
        nugget : float
            Diagonal regularization for SSGE kernel eigendecomposition.
        gamma2 : float
            Jitter for GP analytic score (K + gamma2 * I).
        reservoir_size : int
            Maximum number of training points kept in memory for measurement
            set sampling.  Uses reservoir sampling so every training point has
            equal probability of being retained.
        y_mean, y_std : float or array-like
            Original target statistics for denormalization (regression only).
        num_classes : int or None
            Required when likelihood="multiclass".
        freeze_prior : bool
            If True (default), freeze all parameters of ``prior_function``
            so the prior is fixed. If False, the prior's parameters
            (e.g. GP kernel amplitude / length-scale) are learned jointly
            with the variational posterior, matching the original
            fBNN paper (Sun et al., 2019).
        device : torch.device
        dtype : torch dtype
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
        self.num_samples = num_samples
        self.num_measurement = num_measurement
        self.num_context = num_context
        self.context_std = context_std
        self.bb_alpha = bb_alpha
        self.lambda_kl = lambda_kl
        self.gamma2 = gamma2
        self._reservoir_size = reservoir_size
        self._reservoir = None
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=dtype, device=device))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=dtype, device=device))
        self.device = device
        self.dtype = dtype
        self.output_dim = output_dim

        self.generative_function = generative_function

        # Enforce fix_random_noise=True so that the S weight noise vectors
        # are cached and reused.  This means each of the S samples defines a
        # consistent function across all input locations (measurement + batch),
        # which is required for the SSGE score estimates to be coherent.
        self._ensure_fixed_noise(self.generative_function)

        # Optionally freeze prior parameters. The original fBNN paper learns
        # the GP kernel hyper-parameters jointly with the posterior, so callers
        # can opt out of freezing.
        self.prior_function = prior_function
        if freeze_prior:
            self.prior_function.freeze_parameters()
        self.is_gp_prior = hasattr(prior_function, "forward_mean")

        # BNN priors also need fix_random_noise=True so that the prior
        # score reference distribution is deterministic across iterations.
        if not self.is_gp_prior:
            self._ensure_fixed_noise(self.prior_function)

        self.ssge = SSGE(num_eigs=num_eigs, nugget=nugget)

        if likelihood == "regression":
            self.log_variance = torch.nn.Parameter(torch.tensor(-5.0, dtype=dtype, device=device))
        if likelihood == "multiclass":
            self.num_classes = num_classes
            self.epsilon = 1e-3

        self.KLs = []
        self.bb_alphas = []

    # ------------------------------------------------------------------
    # Sample count management
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_fixed_noise(module):
        """Make Bayesian submodules use coherent cached noise samples."""
        for child in module.modules():
            if not hasattr(child, "fix_random_noise"):
                continue
            child.fix_random_noise = True
            if hasattr(child, "get_noise"):
                child.noise = child.get_noise(first_call=True)

    def _set_num_samples(self, S):
        """Set the posterior sample count on any Bayesian generator.

        The original helper assumed an MLP-style ``.layers`` attribute. CNN
        generators keep their stochastic modules under ``.head`` or as nested
        convolutional layers, so traverse ``modules()`` instead.
        """
        for module in self.generative_function.modules():
            if not hasattr(module, "num_samples"):
                continue
            old = module.num_samples
            if old == S:
                continue
            module.num_samples = S
            if getattr(module, "fix_random_noise", False) and hasattr(module, "get_noise"):
                module.noise = module.get_noise(first_call=True)

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    def predict_f_samples(self, X, num_samples, *, seed=None):
        """
        Sample latent function values from the posterior BNN.

        Returns
        -------
        F : [S, N, D]
        """
        self._set_num_samples(num_samples)
        with temporary_generator_seed(self, seed), fork_torch_rng(seed):
            return self.generative_function(X)

    def predict_y_samples(self, X, num_samples, *, seed=None):
        """
        Sample from the predictive distribution p(y | x).

        Regression: F + eps, eps ~ N(0, sigma2)
        Binary: inv_probit(F) (probability samples)
        Multiclass: F (raw logits — matches TFSVI; downstream metrics softmax)

        Returns
        -------
        Y : [S, N, D]
        """
        with temporary_generator_seed(self, seed), fork_torch_rng(seed):
            F = self.predict_f_samples(X, num_samples)
            if self.likelihood_type == "regression":
                std = torch.sqrt(torch.exp(self.log_variance))
                return F + std * torch.randn_like(F)
            elif self.likelihood_type == "binary":
                return inv_probit(F)
            else:  # multiclass
                return F

    def _logp(self, F, Y):
        if self.likelihood_type == "regression":
            return gaussian_logp(F, Y, self.log_variance)
        elif self.likelihood_type == "binary":
            return bernoulli_logp(F, Y)
        else:
            return multiclass_logp(F, Y, self.num_classes, self.epsilon)

    # ------------------------------------------------------------------
    # Score estimation
    # ------------------------------------------------------------------

    def _gp_analytic_score(self, F_meas, X):
        """
        Analytic prior score for GP prior: -(K + gamma2 * I)^{-1} (f - m(X)).

        Vectorized: solves all S samples x D output dims in a single
        cholesky_solve call on an [N, S*D] right-hand side.

        Parameters
        ----------
        F_meas : [S, M, D] -- posterior function samples (detached)
        X : [M, D_in] -- input locations

        Returns
        -------
        score : [S, M, D]
        """
        M = X.shape[0]
        D = self.output_dim
        S = F_meas.shape[0]
        gp = self.prior_function

        # GP kernel matrix K(X, X)
        x_scaled = X / torch.exp(gp.log_kernel_length)
        sq_dists = torch.cdist(x_scaled, x_scaled, p=2).square()
        amp = torch.exp(gp.log_kernel_amp)
        K = amp * torch.exp(-0.5 * sq_dists)  # [M, M]

        K_reg = K + self.gamma2 * torch.eye(M, dtype=K.dtype, device=K.device)
        L = torch.linalg.cholesky(K_reg)

        # GP mean
        m = gp.forward_mean(X)  # [M, D]

        residual = F_meas - m.unsqueeze(0)  # [S, M, D]

        # Batch all S*D systems into one cholesky_solve:
        # [S, M, D] -> permute -> [M, S, D] -> reshape -> [M, S*D]
        rhs = residual.permute(1, 0, 2).reshape(M, S * D)
        solved = torch.cholesky_solve(rhs, L)  # [M, S*D]

        # [M, S*D] -> [M, S, D] -> [S, M, D]
        score = -solved.reshape(M, S, D).permute(1, 0, 2)
        return score

    def _estimate_prior_score(self, F_meas, X):
        """
        Estimate prior score at posterior sample points, per output dimension.

        For GP priors: analytic score via Cholesky solve.
        For BNN priors: SSGE with Nystrom extension.  The prior BNN uses
        fix_random_noise=True, so it returns the same samples for the same
        X every time — no "moving target" problem.

        Computing the score per output dimension (on [S, M] vectors) instead
        of on the full flattened [S, M*D] vector avoids distance concentration
        in high dimensions, where the median-heuristic bandwidth would fail.

        Parameters
        ----------
        F_meas : [S, M, D] — posterior samples at measurement points (detached)
        X : [M, D_in] — measurement input locations

        Returns
        -------
        scores : [S, M, D]
        """
        if self.is_gp_prior:
            return self._gp_analytic_score(F_meas, X)
        else:
            S, M, D = F_meas.shape
            with torch.no_grad():
                F_prior = self.prior_function(X)  # [S_prior, M, D]
            scores = torch.empty_like(F_meas)
            for d in range(D):
                scores[:, :, d] = self.ssge.compute_score_nystrom(
                    F_prior[:, :, d],  # [S_prior, M]
                    F_meas[:, :, d],  # [S, M]
                )
            return scores

    def _estimate_posterior_score(self, F_meas):
        """
        Estimate posterior score per output dimension using split-half Nystrom.

        Avoids "double-pass" bias (Shi et al., 2018): if the same samples
        are used to build the kernel eigenbasis AND to evaluate the score,
        the estimator overfits to the sample locations, producing collapsed
        scores.  Splitting into two independent halves and cross-evaluating
        via Nystrom removes this bias.

        Computing the score per output dimension (on [S, M] vectors) avoids
        distance concentration in high dimensions that would cause the
        RBF median-heuristic bandwidth to fail on the flattened [S, M*D]
        representation.

        Parameters
        ----------
        F_meas : [S, M, D] — posterior function samples at measurement points

        Returns
        -------
        scores : [S, M, D]
        """
        F = F_meas.detach()
        S, M, D = F.shape
        perm = torch.randperm(S, device=F.device)
        half = S // 2

        idx_a, idx_b = perm[:half], perm[half:]

        scores = torch.empty_like(F)
        for d in range(D):
            F_d = F[:, :, d]  # [S, M]
            F_a, F_b = F_d[idx_a], F_d[idx_b]

            # Each half's score is estimated from the OTHER half's kernel
            score_a = self.ssge.compute_score_nystrom(F_b, F_a)
            score_b = self.ssge.compute_score_nystrom(F_a, F_b)

            scores[idx_a, :, d] = score_a
            scores[idx_b, :, d] = score_b

        return scores

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nelbo(self, X_batch, y):
        """
        Negative ELBO with functional KL.

        Parameters
        ----------
        X_batch : [N_batch, D_in]
        y : [N_batch, D]
        """
        N_batch = X_batch.shape[0]

        # Measurement set = context points (OOD) + training subset + batch
        X_measure = self._sample_measurement_set(X_batch)
        M = X_measure.shape[0] - N_batch  # context + training measurement pts

        # Forward through posterior BNN at measurement points
        F = self.predict_f_samples(X_measure, self.num_samples)  # [S, M+N_batch, D]

        # Likelihood on the batch portion (after M measurement points)
        F_batch = F[:, M:, :]  # [S, N_batch, D]
        if self.likelihood_type == "multiclass":
            assert F_batch.shape[-1] == self.num_classes, (
                f"Expected {self.num_classes} classes in last dim, got {F_batch.shape[-1]}"
            )
        logpdf = self._logp(F_batch, y)
        if self.bb_alpha == 0:
            ve = torch.mean(logpdf, dim=0)
        else:
            ve = (
                torch.logsumexp(self.bb_alpha * logpdf, dim=0) - math.log(F_batch.shape[0])
            ) / self.bb_alpha
        ve = torch.sum(ve)
        scale = self.num_data / N_batch

        # Functional KL via SSGE — only at measurement points (not batch).
        # Score estimation is done per output dimension to avoid distance
        # concentration in high dimensions (the RBF median-heuristic bandwidth
        # would fail on flattened M*D vectors).
        if M > 0:
            F_meas = F[:, :M, :]  # [S, M, D]
            X_M = X_measure[:M]  # [M, D_in]

            with torch.no_grad():
                s_q = self._estimate_posterior_score(F_meas)  # [S, M, D]
                s_p = self._estimate_prior_score(F_meas.detach(), X_M)  # [S, M, D]
                score_diff = s_q - s_p  # [S, M, D]

            # KL gradient surrogate, normalized by M (expectation over locations)
            loss_kl = torch.mean(torch.sum(F_meas * score_diff, dim=(1, 2))) / M
        else:
            loss_kl = torch.tensor(0.0, dtype=X_batch.dtype, device=X_batch.device)

        KL = self.lambda_kl * loss_kl
        self.bb_alphas.append((-scale * ve).detach().cpu().numpy())
        self.KLs.append(KL.detach().cpu().numpy())
        return -scale * ve + KL

    def _sample_measurement_set(self, X_batch):
        """
        Build measurement set per the fBNN paper (Sun et al., 2019):

        (a) X_D — random subset of training data (from reservoir)
        (b) X_M — context points sampled from the input domain

        Both components regularize the posterior via the functional KL.
        The context points (b) are critical: they enforce the prior in
        OOD regions, giving fBNN its extrapolation properties.

        Returns
        -------
        X_measure : [M_context + M_train + N_batch, D_in]
        """
        parts = []
        D_in = X_batch.shape[-1]

        # (b) Context points from the input domain (including OOD regions).
        # Since inputs are normalized to N(0,1), sampling from N(0, σ²)
        # with σ > 1 naturally covers out-of-distribution regions.
        if self.num_context > 0:
            X_context = (
                torch.randn(
                    self.num_context,
                    D_in,
                    dtype=X_batch.dtype,
                    device=X_batch.device,
                )
                * self.context_std
            )
            parts.append(X_context)

        # (a) Random subset of training data
        if self._reservoir is not None and self.num_measurement > 0:
            N_pool = self._reservoir.shape[0]
            M_train = min(self.num_measurement, N_pool)
            idx = torch.randperm(N_pool)[:M_train]
            X_train = self._reservoir[idx].to(dtype=X_batch.dtype, device=X_batch.device)
            parts.append(X_train)

        parts.append(X_batch)
        return torch.cat(parts, dim=0)

    def _fill_reservoir(self, train_loader):
        """
        One-pass reservoir sampling over the training loader.

        Every training point has equal probability of being retained.
        Memory usage is O(reservoir_size) regardless of dataset size.
        """
        R = self._reservoir_size
        if R == 0:
            return

        reservoir = None
        count = 0

        for inputs, _ in train_loader:
            inputs_cpu = inputs.detach()
            B = inputs_cpu.shape[0]

            if reservoir is None:
                D_in = inputs_cpu.shape[-1]
                reservoir = torch.empty(R, D_in, dtype=inputs_cpu.dtype)

            if count + B <= R:
                # Buffer not full: store entire batch
                reservoir[count : count + B] = inputs_cpu
                count += B
                continue

            # Fill remaining capacity
            if count < R:
                take = R - count
                reservoir[count:R] = inputs_cpu[:take]
                inputs_cpu = inputs_cpu[take:]
                B = inputs_cpu.shape[0]
                count = R

            # Vectorized reservoir sampling for remaining items
            stream_pos = torch.arange(B, dtype=torch.float64) + count
            probs = float(R) / (stream_pos + 1.0)
            accept = torch.rand(B, dtype=torch.float64) < probs
            sel = torch.where(accept)[0]
            if sel.numel() > 0:
                positions = torch.randint(0, R, (sel.numel(),))
                reservoir[positions] = inputs_cpu[sel]
            count += B

        if reservoir is None:
            self._reservoir = None
        elif count < R:
            self._reservoir = reservoir[:count]
        else:
            self._reservoir = reservoir

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

    def predict(self, X, num_samples, *, seed=None):
        """
        Predict y_samples for evaluation.

        Returns
        -------
        Y : [S, N, D] denormalized for regression, logits for multiclass,
            inv_probit probabilities for binary
        """
        self.eval()
        with torch.no_grad():
            if self.dtype != X.dtype:
                X = X.to(self.dtype)
            Y = self.predict_y_samples(X, num_samples, seed=seed)
            if self.likelihood_type == "regression":
                return Y * self.y_std + self.y_mean
            return Y

    # ------------------------------------------------------------------
    # Training
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
        lr : float
        epochs : int or None
        iterations : int or None
        use_tqdm : bool
        return_loss : bool
        cosine_annealing : bool
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

    def _prepare_fit(self, train_loader):
        """Build the bounded measurement reservoir before shared fitting."""
        self._fill_reservoir(train_loader)

    def _train_step(self, optimizer, X, y):
        """Single gradient step."""
        if self.dtype != X.dtype:
            X = X.to(self.dtype)

        if self.likelihood_type == "multiclass":
            # Class indices must stay as integers; unsqueeze to [N, 1]
            # for multiclass_logp's broadcasting with argmax(F, -1).
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            y = y.long()
        else:
            # Regression / binary: float targets with shape [N, 1].
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
    # Utilities
    # ------------------------------------------------------------------

    def print_variables(self):
        """Print trainable parameters using the shared formatter."""
        print_trainable_parameters(self)
