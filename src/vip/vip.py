import math

import numpy as np
import torch
from tqdm import tqdm

from ..utils.utils import infinite_loader
from ..utils.likelihood import (
    gaussian_variational_expectations,
    gaussian_variational_expectations_full_cov,
    bernoulli_variational_expectations,
    predict_mean_and_var_regression,
    predict_mean_and_var_binary,
    multiclass_logp,
    bernoulli_logp,
    compute_gp_log_marginal,
)


class VIP(torch.nn.Module):
    def __init__(
        self,
        generative_function,
        num_regression_coeffs,
        output_dim,
        likelihood,
        num_data,
        bb_alpha=0,
        y_mean=0.0,
        y_std=1.0,
        num_classes=None,
        num_mc_samples=200,
        use_prior_regularizer=False,
        prior_regularizer_scaler=1.0,
        regularizer_mode="evidence",
        use_full_cov_elbo=False,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        """
        Variational Implicit Process.

        Parameters
        ----------
        generative_function : torch.nn.Module
            Prior function sampler (e.g. BayesianNN).
        num_regression_coeffs : int
            Number of regression coefficients (S).
        output_dim : int
            Dimensionality of the output.
        likelihood : str
            One of "regression", "binary", or "multiclass".
        num_data : int
            Total dataset size (for minibatch scaling).
        bb_alpha : float
            Alpha for BB-alpha energy (0 = ELBO).
        y_mean, y_std : float or array-like
            Original target statistics for denormalization.
        num_classes : int or None
            Required when likelihood="multiclass".
        num_mc_samples : int
            Number of MC samples for classification forward/training.
        device : torch.device
        dtype : torch dtype
        seed : int
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
        self.bb_alpha = bb_alpha
        self.use_prior_regularizer = use_prior_regularizer
        self.prior_regularizer_scaler = prior_regularizer_scaler
        self.regularizer_mode = regularizer_mode
        self.use_full_cov_elbo = use_full_cov_elbo
        self.y_mean = torch.tensor(y_mean, device=device)
        self.y_std = torch.tensor(y_std, device=device)
        self.device = device
        self.dtype = dtype

        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        self.num_samples = 1
        self.num_coeffs = num_regression_coeffs
        self.output_dim = output_dim
        self.generative_function = generative_function

        # --- Precomputed constants ---
        self.register_buffer(
            "_sqrt_coeffs_m1",
            torch.tensor(math.sqrt(num_regression_coeffs - 1), dtype=dtype, device=device),
        )
        li, lj = torch.tril_indices(num_regression_coeffs, num_regression_coeffs)
        self.register_buffer("_tril_row", li)
        self.register_buffer("_tril_col", lj)
        self.register_buffer(
            "_diag_idx",
            torch.tensor(
                np.cumsum(np.arange(1, num_regression_coeffs + 1)) - 1,
                dtype=torch.long, device=device,
            ),
        )
        # Persistent buffer for q_sqrt to avoid allocation every forward pass
        self.register_buffer(
            "_q_sqrt_buf",
            torch.zeros(num_regression_coeffs, num_regression_coeffs, output_dim,
                        dtype=dtype, device=device),
        )

        # --- Likelihood-specific parameters ---
        self.num_mc_samples = num_mc_samples
        if likelihood == "regression":
            self.log_variance = torch.nn.Parameter(
                torch.full((output_dim,), -5.0, dtype=dtype, device=device)
            )
        if likelihood == "binary":
            self.num_gauss_hermite_points = 20
        if likelihood == "multiclass":
            self.num_classes = num_classes
            self.epsilon = 1e-3
            self.K1 = self.epsilon / (num_classes - 1)

        # --- Variational parameters ---
        self.q_mu = torch.nn.Parameter(
            torch.zeros(num_regression_coeffs, output_dim, dtype=dtype, device=device)
        )

        q_sqrt_init = torch.eye(num_regression_coeffs, dtype=dtype, device=device)
        q_sqrt_init = q_sqrt_init.unsqueeze(-1).expand(-1, -1, output_dim)
        self.q_sqrt_tri = torch.nn.Parameter(q_sqrt_init[li, lj].contiguous())

        self._kl_buffer = []
        self._bb_alpha_buffer = []
        self._prior_reg_buffer = []

    # ------------------------------------------------------------------
    # Core model methods
    # ------------------------------------------------------------------

    def predict_f(self, predict_at, full_covariance=False):
        """
        Computes the marginal Q*(y|x) by marginalizing the regression
        coefficients a from Q(a) = N(q_mu, q_sqrt q_sqrt^T).

        Parameters
        ----------
        predict_at : (N, input_dim)
        full_covariance : bool
            If False (default), returns diagonal variance (N, output_dim).
            If True, returns full covariance (N, output_dim, output_dim).

        Returns
        -------
        mean : (N, output_dim)
        K : (N, output_dim) if full_covariance=False,
            (N, output_dim, output_dim) if full_covariance=True.
        """
        f = self.generative_function(predict_at)
        m = torch.mean(f, dim=0, keepdims=True)

        phi = (f - m) / self._sqrt_coeffs_m1

        # Cache for prior_regularizer (avoids re-calling generative_function)
        self._cached_m = m.squeeze(0)
        self._cached_phi = phi

        mean = m.squeeze(axis=0) + torch.einsum(
            "sn...,s...->n...", phi, self.q_mu
        )

        q_sqrt = torch.zeros_like(self._q_sqrt_buf)
        q_sqrt[self._tril_row, self._tril_col] = self.q_sqrt_tri

        # phi^T @ q_sqrt then sum of squares over S dimension
        # Fuse into: K_nd = sum_s (sum_i phi_ind * q_sqrt_sid)^2
        phi_q = torch.einsum("ind, sid -> snd", phi, q_sqrt)

        if full_covariance:
            K = torch.einsum("snd, sne -> nde", phi_q, phi_q)
        else:
            K = torch.einsum("snd, snd -> nd", phi_q, phi_q)

        return mean, K

    def _sample_posterior(self, Fmean, Fvar, S, Fcov=None):
        """Draw S samples from q(f|x).

        Parameters
        ----------
        Fmean : (N, output_dim)
        Fvar  : (N, output_dim)
            Diagonal variance (used when Fcov is None).
        S     : int
        Fcov  : (N, output_dim, output_dim) or None
            Full covariance. If provided, samples are correlated;
            Fvar is ignored.

        Returns
        -------
        samples : (S, N, output_dim)
        """
        eps = torch.randn(
            S, *Fmean.shape,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        if Fcov is not None:
            # Correlated samples via Cholesky: mean + L @ eps
            L = torch.linalg.cholesky(Fcov + 1e-6 * torch.eye(
                Fcov.shape[-1], dtype=Fcov.dtype, device=Fcov.device))
            # L: (N, D, D), eps: (S, N, D) -> (S, N, D, 1) -> matmul -> (S, N, D)
            return Fmean.unsqueeze(0) + torch.einsum("nde, sne -> snd", L, eps)
        else:
            std = torch.sqrt(Fvar).unsqueeze(0)      # (1, N, D)
            return Fmean.unsqueeze(0) + eps * std     # (S, N, D)

    def _logp(self, F, Y):
        """Log-likelihood for classification (same interface as FTIP)."""
        if self.likelihood_type == "binary":
            return bernoulli_logp(F, Y)
        else:  # multiclass
            return multiclass_logp(F, Y, self.num_classes, self.epsilon)

    def _variational_expectations(self, Fmu, Fvar, Y):
        """Dispatch to the correct variational expectations function."""
        if self.likelihood_type == "regression":
            return gaussian_variational_expectations(
                Fmu, Fvar, Y, self.bb_alpha, self.log_variance
            )
        elif self.likelihood_type == "binary":
            return bernoulli_variational_expectations(
                Fmu, Fvar, Y, self.num_gauss_hermite_points,
                self.dtype, self.device,
            )
        else:  # multiclass — not used; nelbo uses MC sampling instead
            raise RuntimeError("Use nelbo's MC path for multiclass")

    def nelbo(self, X, y):
        """Negative ELBO (or BB-alpha energy) objective."""
        if self.use_full_cov_elbo and self.likelihood_type == "regression":
            F_mean, F_cov = self.predict_f(X, full_covariance=True)
            bb_alpha = gaussian_variational_expectations_full_cov(
                F_mean, F_cov, y, self.bb_alpha, self.log_variance
            )
            bb_alpha = torch.sum(bb_alpha)
        elif self.likelihood_type == "regression" or self.likelihood_type == "binary":
            F_mean, F_var = self.predict_f(X)
            bb_alpha = self._variational_expectations(
                F_mean.unsqueeze(0), F_var.unsqueeze(0), y
            )
            bb_alpha = torch.sum(bb_alpha)
        else:
            F_mean, F_var = self.predict_f(X)
            # Multiclass: MC sampling (same approach as FTIP)
            F = self._sample_posterior(F_mean, F_var, self.num_mc_samples)
            logpdf = self._logp(F, y)
            if self.bb_alpha == 0:
                ve = torch.mean(logpdf, axis=0)
            else:
                ve = (
                    torch.logsumexp(self.bb_alpha * logpdf, axis=0)
                    - math.log(self.num_mc_samples)
                ) / self.bb_alpha
            bb_alpha = torch.sum(ve)

        scale = self.num_data / X.shape[0]

        KL = self.KL()

        self._bb_alpha_buffer.append((-scale * bb_alpha).detach().item())
        self._kl_buffer.append(KL.detach().item())
        return -scale * bb_alpha + KL

    def KL(self):
        """KL(N(q_mu, q_sqrt q_sqrt^T) || N(0, I))."""
        diag = self.q_sqrt_tri[self._diag_idx]

        KL = -0.5 * self.output_dim * self.num_coeffs
        KL -= torch.sum(torch.log(torch.abs(diag)))
        KL += 0.5 * torch.sum(torch.square(self.q_sqrt_tri))
        KL += 0.5 * torch.sum(torch.square(self.q_mu))
        return KL

    @property
    def KLs(self):
        return self._kl_buffer

    @property
    def bb_alphas(self):
        return self._bb_alpha_buffer

    @property
    def prior_regularizers(self):
        return self._prior_reg_buffer

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

    def forward(self, predict_at):
        """Forward pass for prediction.

        For regression: returns (mean, std) denormalized to original scale.
        For classification: returns (S, N, K) posterior samples and None.
        """
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        Fmean, Fvar = self.predict_f(predict_at)
        if self.likelihood_type == "regression":
            mean, var = predict_mean_and_var_regression(
                Fmean.unsqueeze(0), Fvar.unsqueeze(0), self.log_variance
            )
            return mean * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std
        if self.likelihood_type == "binary":
            mean, var = predict_mean_and_var_binary(
                Fmean.unsqueeze(0), Fvar.unsqueeze(0)
            )
            return mean, var
        # Multiclass: return posterior samples (same interface as FTIP)
        samples = self._sample_posterior(Fmean, Fvar, self.num_mc_samples)
        return samples, None

    # ------------------------------------------------------------------
    # Training and prediction
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
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if epochs is None and iterations is None:
            raise ValueError("Either epochs or iterations must be set.")

        scheduler = None
        if cosine_annealing:
            T_max = epochs if epochs is not None else iterations // len(train_loader)
            eta_min = optimizer.param_groups[0]['lr'] / 100
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

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
                        losses.append(loss.item())
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
                    losses.append(loss.item())
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

        optimizer.zero_grad(set_to_none=True)
        loss = self.nelbo(X, y)
        if self.use_prior_regularizer:
            prior_reg = self.prior_regularizer(X, y)
            self._prior_reg_buffer.append(prior_reg.detach().item())
            loss = loss + prior_reg
        loss.backward()
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
        log_q = compute_gp_log_marginal(
            self._cached_m, self._cached_phi, y, sigma2 + 1e-3
        )
        return self.prior_regularizer_scaler / X.shape[0] * log_q

    def predict(self, data_loader, device=None):
        """
        Run predictions over a DataLoader.

        Returns
        -------
        all_means : torch.Tensor
        all_stds : torch.Tensor
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
