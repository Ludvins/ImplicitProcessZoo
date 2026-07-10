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


@preserve_constructor_rng
class MFVI(torch.nn.Module):
    def __init__(
        self,
        generative_function,
        output_dim,
        likelihood,
        num_data,
        num_samples=10,
        bb_alpha=0,
        y_mean=0.0,
        y_std=1.0,
        num_classes=None,
        device=None,
        dtype=torch.float64,
    ):
        """
        Mean-Field Variational Inference baseline in weight space.

        Uses a BayesianNN with per-weight Gaussian variational parameters
        (mu, log_sigma) optimized via ELBO or BB-alpha energy.

        Parameters
        ----------
        generative_function : BayesianNN
            BNN whose variational parameters define the posterior.
            Should use fix_random_noise=False for fresh MC samples each step.
        output_dim : int
            Dimensionality of the output.
        likelihood : str
            One of "regression", "binary", or "multiclass".
        num_data : int
            Total dataset size (for minibatch scaling).
        num_samples : int
            Number of MC weight samples for training.
        bb_alpha : float
            Alpha for BB-alpha energy (0 = ELBO).
        y_mean, y_std : float or array-like
            Original target statistics for denormalization (regression only).
        num_classes : int or None
            Required when likelihood="multiclass".
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
        self.bb_alpha = bb_alpha
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=dtype, device=device))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=dtype, device=device))
        self.device = device
        self.dtype = dtype
        self.output_dim = output_dim

        self.generative_function = generative_function

        if likelihood == "regression":
            self.log_variance = torch.nn.Parameter(torch.tensor(-5.0, dtype=dtype, device=device))
        if likelihood == "multiclass":
            self.num_classes = num_classes
            self.epsilon = 1e-3

        self.KLs = []
        self.bb_alphas = []

    # ------------------------------------------------------------------
    # Core model methods
    # ------------------------------------------------------------------

    def _set_num_samples(self, S):
        """Update any Bayesian generator's sample count.

        Older code assumed an MLP-style ``.layers`` attribute. Image
        classifiers keep stochastic modules under ``.head`` or nested conv
        blocks, so traverse the module tree instead.
        """
        for module in self.generative_function.modules():
            if not hasattr(module, "num_samples"):
                continue
            old = module.num_samples
            module.num_samples = S
            if (
                getattr(module, "fix_random_noise", False)
                and hasattr(module, "get_noise")
                and old != S
            ):
                module.noise = module.get_noise(first_call=True)

    def predict_f_samples(self, X, num_samples, *, seed=None):
        """
        Sample latent function values from the BNN posterior.

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
        """Dispatch to the correct logp function."""
        if self.likelihood_type == "regression":
            return gaussian_logp(F, Y, self.log_variance)
        elif self.likelihood_type == "binary":
            return bernoulli_logp(F, Y)
        else:  # multiclass
            return multiclass_logp(F, Y, self.num_classes, self.epsilon)

    def KL(self):
        """KL divergence of BNN weights to their standard normal prior."""
        return self.generative_function.KL()

    def nelbo(self, X, y):
        """Negative ELBO (or BB-alpha energy) objective."""
        F = self.predict_f_samples(X, self.num_samples)

        logpdf = self._logp(F, y)
        if self.bb_alpha == 0:
            ve = torch.mean(logpdf, dim=0)
        else:
            ve = (
                torch.logsumexp(self.bb_alpha * logpdf, dim=0) - math.log(F.shape[0])
            ) / self.bb_alpha

        ve = torch.sum(ve)

        scale = self.num_data / X.shape[0]

        KL = self.KL()
        self.bb_alphas.append((-scale * ve).detach().cpu().numpy())
        self.KLs.append(KL.detach().cpu().numpy())
        return -scale * ve + KL

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

        Parameters
        ----------
        X : [N, D_in] input tensor
        S : int, number of predictive samples

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
