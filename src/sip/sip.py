from contextlib import contextmanager
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..utils.utils import infinite_loader
from ..utils.linalg import safe_cholesky
from ..utils.likelihood import (
    gaussian_logp,
    bernoulli_logp,
    multiclass_logp,
    predict_mean_and_var_regression,
    predict_mean_and_var_binary,
    predict_mean_and_var_multiclass,
)


class _MLP(nn.Module):
    """Small MLP used for the implicit inducing posterior and critic."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        depth=2,
        activation=nn.LeakyReLU,
        zero_last=False,
        dtype=torch.float64,
        device=None,
    ):
        super().__init__()
        layers = []
        in_dim = int(input_dim)
        for _ in range(max(0, int(depth))):
            layers.append(nn.Linear(in_dim, int(hidden_dim), dtype=dtype, device=device))
            layers.append(activation())
            in_dim = int(hidden_dim)
        final = nn.Linear(in_dim, int(output_dim), dtype=dtype, device=device)
        if zero_last:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@contextmanager
def _module_requires_grad(module, requires_grad):
    states = [(param, param.requires_grad) for param in module.parameters()]
    try:
        for param, _ in states:
            param.requires_grad_(requires_grad)
        yield
    finally:
        for param, state in states:
            param.requires_grad_(state)


class SIP(nn.Module):
    """Sparse Implicit Process with critic-estimated inducing KL.

    SIP introduces inducing values ``u = f(Z)`` and uses the sparse
    factorization

    ``q_phi(f_X, u) = p_theta(f_X | u) q_phi(u)``.

    The posterior ``q_phi(u)`` is an implicit sampler ``u = h_phi(eps)`` with
    Gaussian noise ``eps``. Since neither ``q_phi(u)`` nor the prior
    ``p_theta(u)`` needs a closed-form density, a separate critic estimates the
    inducing-space log density ratio.
    """

    def __init__(
        self,
        generative_function,
        inducing_inputs,
        output_dim,
        likelihood,
        num_data,
        num_prior_samples=50,
        num_train_samples=None,
        num_eval_samples=200,
        bb_alpha=0,
        beta=1.0,
        beta_warmup_steps=0,
        learn_inducing=False,
        detach_covariances=False,
        critic_hidden_dim=50,
        critic_lr=1e-3,
        critic_steps=1,
        posterior_noise_dim=100,
        posterior_hidden_dim=50,
        posterior_depth=2,
        fresh_prior_samples=True,
        y_mean=0.0,
        y_std=1.0,
        num_classes=None,
        jitter=1e-5,
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

        Z = torch.as_tensor(inducing_inputs, dtype=dtype, device=device)
        if Z.ndim == 1:
            Z = Z.unsqueeze(-1)

        self.device = Z.device
        self.dtype = dtype
        self.generative_function = generative_function
        self.likelihood_type = likelihood
        self.num_data = int(num_data)
        self.output_dim = int(output_dim)
        self.num_inducing = int(Z.shape[0])
        self.u_dim = self.num_inducing * self.output_dim
        self.posterior_noise_dim = int(100 if posterior_noise_dim is None else posterior_noise_dim)
        self.num_prior_samples = int(num_prior_samples)
        self.num_train_samples = int(
            self.num_prior_samples if num_train_samples is None else num_train_samples
        )
        self.num_eval_samples = int(num_eval_samples)
        self.bb_alpha = float(bb_alpha)
        self.beta = float(beta)
        self.beta_warmup_steps = int(beta_warmup_steps)
        self._step = 0
        self.learn_inducing = bool(learn_inducing)
        self.detach_covariances = bool(detach_covariances)
        self.critic_lr = float(critic_lr)
        self.critic_steps = int(critic_steps)
        self.fresh_prior_samples = bool(fresh_prior_samples)
        self.jitter = float(jitter)

        self.register_buffer(
            "y_mean", torch.as_tensor(y_mean, dtype=dtype, device=self.device)
        )
        self.register_buffer(
            "y_std", torch.as_tensor(y_std, dtype=dtype, device=self.device)
        )

        if self.learn_inducing:
            self.Z = nn.Parameter(Z)
        else:
            self.register_buffer("Z", Z)

        if likelihood == "regression":
            self.log_variance = nn.Parameter(
                torch.tensor(-5.0, dtype=dtype, device=self.device)
            )
        if likelihood in ("binary", "multiclass"):
            self.num_gauss_hermite_points = 20
        if likelihood == "multiclass":
            self.num_classes = int(num_classes)
            self.epsilon = 1e-3

        generator_device = self.device if self.device.type != "cpu" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(int(seed))

        self.posterior_sampler = _MLP(
            input_dim=self.posterior_noise_dim,
            output_dim=self.u_dim,
            hidden_dim=posterior_hidden_dim,
            depth=posterior_depth,
            activation=nn.LeakyReLU,
            zero_last=False,
            dtype=dtype,
            device=self.device,
        )
        self.posterior_noise_mean = nn.Parameter(
            torch.zeros(1, self.posterior_noise_dim, dtype=dtype, device=self.device)
        )
        self.posterior_noise_log_var = nn.Parameter(
            torch.full(
                (1, self.posterior_noise_dim),
                -5.0,
                dtype=dtype,
                device=self.device,
            )
        )
        self.critic = _MLP(
            input_dim=self.u_dim,
            output_dim=1,
            hidden_dim=critic_hidden_dim,
            depth=2,
            activation=nn.LeakyReLU,
            zero_last=False,
            dtype=dtype,
            device=self.device,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )

        self.KLs = []
        self.function_terms = []
        self.bb_alphas = []
        self.betas = []
        self.critic_losses = []
        self.sip_critic_losses = self.critic_losses
        self.critic_accuracies = []
        self.critic_saturation_fractions = []
        self.kl_forwards = []
        self.kl_reverses = []

    # ------------------------------------------------------------------
    # Parameter groups and prior sampling
    # ------------------------------------------------------------------

    def vi_parameters(self):
        """Parameters optimized by the outer variational optimizer."""
        critic_ids = {id(param) for param in self.critic.parameters()}
        return [
            param for param in self.parameters()
            if param.requires_grad and id(param) not in critic_ids
        ]

    def _sample_prior_values(self, X, num_samples):
        """Draw coherent prior values with the requested sample count."""
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        old_states = []
        modules = list(self.generative_function.modules())
        try:
            for module in modules:
                if hasattr(module, "num_samples"):
                    old_states.append(
                        (
                            module,
                            module.num_samples,
                            getattr(module, "fix_random_noise", None),
                            getattr(module, "noise", None),
                        )
                    )
                    module.num_samples = num_samples
                    if hasattr(module, "fix_random_noise"):
                        module.fix_random_noise = not self.fresh_prior_samples
                    if (
                        not self.fresh_prior_samples
                        and getattr(module, "fix_random_noise", False)
                        and hasattr(module, "get_noise")
                    ):
                        try:
                            module.noise = module.get_noise(first_call=True)
                        except TypeError:
                            pass

            try:
                values = self.generative_function(X, num_samples)
            except TypeError:
                values = self.generative_function(X)
        finally:
            for module, old_num_samples, old_fix_noise, old_noise in reversed(old_states):
                module.num_samples = old_num_samples
                if old_fix_noise is not None and hasattr(module, "fix_random_noise"):
                    module.fix_random_noise = old_fix_noise
                if hasattr(module, "noise"):
                    module.noise = old_noise

        if values.ndim == 2:
            values = values.unsqueeze(-1)
        if values.ndim != 3:
            raise ValueError(
                "SIP prior samples must have shape [S, N, D], "
                f"got {tuple(values.shape)}."
            )
        return values[:num_samples]

    def _sample_prior_u(self, num_samples):
        return self._sample_prior_values(self.Z, num_samples)

    # ------------------------------------------------------------------
    # Prior moment estimation and sparse conditional
    # ------------------------------------------------------------------

    def _estimate_prior_moments(self, X):
        B = X.shape[0]
        M = self.num_inducing
        XZ = torch.cat([X, self.Z], dim=0)
        f = self._sample_prior_values(XZ, self.num_prior_samples)
        if f.shape[-1] != self.output_dim:
            raise ValueError(
                f"Prior output_dim mismatch: expected {self.output_dim}, got {f.shape[-1]}."
            )

        fX = f[:, :B, :]
        fZ = f[:, B:, :]
        mX = fX.mean(dim=0)
        mZ = fZ.mean(dim=0)
        fX_c = fX - mX.unsqueeze(0)
        fZ_c = fZ - mZ.unsqueeze(0)
        denom = max(int(f.shape[0]) - 1, 1)

        KZZ = torch.einsum("smd,snd->dmn", fZ_c, fZ_c) / denom
        KXX = torch.einsum("sbd,snd->dbn", fX_c, fX_c) / denom
        KXZ = torch.einsum("sbd,smd->dbm", fX_c, fZ_c) / denom
        varX = torch.einsum("sbd,sbd->db", fX_c, fX_c) / denom

        if self.detach_covariances:
            KXX = KXX.detach()
            KZZ = KZZ.detach()
            KXZ = KXZ.detach()
            varX = varX.detach()

        eye = torch.eye(M, dtype=self.dtype, device=self.device).unsqueeze(0)
        KZZ = KZZ + self.jitter * eye
        return mX, mZ, KZZ, KXZ, varX, KXX

    def _sparse_conditional(self, mX, mZ, KZZ, KXZ, varX, u):
        S = u.shape[0]
        f_means = []
        f_vars = []

        for d in range(self.output_dim):
            Lzz = safe_cholesky(KZZ[d], initial_jitter=self.jitter)
            kxz = KXZ[d]
            diff = u[:, :, d] - mZ[:, d].unsqueeze(0)
            alpha = torch.cholesky_solve(diff.T, Lzz).T
            cond_mean = mX[:, d].unsqueeze(0) + alpha @ kxz.T
            f_means.append(cond_mean)

            V = torch.linalg.solve_triangular(Lzz, kxz.T, upper=False)
            quad = (V * V).sum(dim=0)
            cond_var = torch.clamp(varX[d] - quad, min=1e-10)
            f_vars.append(cond_var)

        f_mean = torch.stack(f_means, dim=0).permute(1, 2, 0)
        f_var = torch.stack(f_vars, dim=0)
        assert f_mean.shape == (S, mX.shape[0], self.output_dim)
        return f_mean, f_var

    def _sample_f_given_u(self, mX, mZ, KZZ, KXZ, varX, u):
        f_mean, f_var = self._sparse_conditional(mX, mZ, KZZ, KXZ, varX, u)
        f_std = torch.sqrt(f_var).permute(1, 0).unsqueeze(0)
        eta = torch.randn(
            f_mean.shape,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        return f_mean + f_std * eta

    def _sample_f_given_u_full(self, mX, mZ, KZZ, KXZ, KXX, u):
        """Sample from the full sparse conditional covariance.

        This matches the released SIP code's evaluation path. The training
        objective uses the diagonal version above for independent likelihood
        terms on minibatches.
        """
        f_mean, _ = self._sparse_conditional(
            mX,
            mZ,
            KZZ,
            KXZ,
            torch.diagonal(KXX, dim1=-2, dim2=-1),
            u,
        )
        samples = []
        B = mX.shape[0]
        eye = torch.eye(B, dtype=self.dtype, device=self.device)
        for d in range(self.output_dim):
            Lzz = safe_cholesky(KZZ[d], initial_jitter=self.jitter)
            kxz = KXZ[d]
            D = torch.linalg.solve_triangular(Lzz, kxz.T, upper=False)
            cov = KXX[d] - D.T @ D
            cov = 0.5 * (cov + cov.T) + self.jitter * eye
            L = safe_cholesky(cov, initial_jitter=self.jitter)
            eta = torch.randn(
                u.shape[0],
                B,
                generator=self.generator,
                dtype=self.dtype,
                device=self.device,
            )
            samples.append(f_mean[:, :, d] + eta @ L.T)
        return torch.stack(samples, dim=-1)

    # ------------------------------------------------------------------
    # Implicit q_phi(u) and critic KL
    # ------------------------------------------------------------------

    def _sample_u(self, num_samples):
        pre_noise = torch.randn(
            int(num_samples),
            self.posterior_noise_dim,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        eps = self.posterior_noise_mean + torch.exp(
            0.5 * self.posterior_noise_log_var
        ) * pre_noise
        u_flat = self.posterior_sampler(eps)
        return u_flat.reshape(int(num_samples), self.num_inducing, self.output_dim)

    def _flat_u(self, u):
        return u.reshape(u.shape[0], self.u_dim)

    def _critic_loss(self, num_samples=None):
        S = int(self.num_train_samples if num_samples is None else num_samples)
        with torch.no_grad():
            u_q = self._sample_u(S)
            u_p = self._sample_prior_u(S)
        t_q = self.critic(self._flat_u(u_q)).squeeze(-1)
        t_p = self.critic(self._flat_u(u_p)).squeeze(-1)
        loss = -0.5 * (F.logsigmoid(t_q).mean() + F.logsigmoid(-t_p).mean())
        accuracy = 0.5 * ((t_q > 0).double().mean() + (t_p < 0).double().mean())
        saturation = 0.5 * (
            (t_q.abs() > 10.0).double().mean()
            + (t_p.abs() > 10.0).double().mean()
        )
        return loss, accuracy, saturation

    def _train_critic(self):
        if self.critic_steps <= 0:
            return None
        last_loss = None
        last_accuracy = None
        last_saturation = None
        for _ in range(self.critic_steps):
            self.critic_optimizer.zero_grad()
            loss, accuracy, saturation = self._critic_loss()
            loss.backward()
            self.critic_optimizer.step()
            last_loss = loss
            last_accuracy = accuracy
            last_saturation = saturation
        self.critic_losses.append(float(last_loss.detach().cpu()))
        self.critic_accuracies.append(float(last_accuracy.detach().cpu()))
        self.critic_saturation_fractions.append(float(last_saturation.detach().cpu()))
        return last_loss

    def _kl_regularizer(self, posterior_u=None):
        if posterior_u is None:
            posterior_u = self._sample_u(self.num_train_samples)
        prior_u = self._sample_prior_u(posterior_u.shape[0])

        with _module_requires_grad(self.critic, False):
            t_q = self.critic(self._flat_u(posterior_u)).squeeze(-1)
            t_p = self.critic(self._flat_u(prior_u)).squeeze(-1)

        forward_kl = t_q.mean()
        reverse_kl = -t_p.mean()
        sym_kl = 0.5 * (forward_kl + reverse_kl)
        return sym_kl, forward_kl, reverse_kl

    def _scheduled_beta(self):
        if self.beta_warmup_steps <= 0:
            return self.beta
        return self.beta * min(1.0, self._step / float(self.beta_warmup_steps))

    # ------------------------------------------------------------------
    # Likelihood dispatch
    # ------------------------------------------------------------------

    def _logp(self, F_samples, Y):
        if self.likelihood_type == "regression":
            return gaussian_logp(F_samples, Y, self.log_variance)
        if self.likelihood_type == "binary":
            return bernoulli_logp(F_samples, Y)
        return multiclass_logp(F_samples, Y, self.num_classes, self.epsilon)

    def _predict_mean_and_var(self, Fmu, Fvar):
        if self.likelihood_type == "regression":
            return predict_mean_and_var_regression(Fmu, Fvar, self.log_variance)
        if self.likelihood_type == "binary":
            return predict_mean_and_var_binary(Fmu, Fvar)
        return predict_mean_and_var_multiclass(
            Fmu, Fvar, self.num_classes, self.epsilon,
            self.num_gauss_hermite_points, self.dtype, self.device,
        )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def nelbo(self, X, y):
        """Negative SIP functional ELBO with critic-estimated KL."""
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        mX, mZ, KZZ, KXZ, varX, _ = self._estimate_prior_moments(X)
        posterior_u = self._sample_u(self.num_train_samples)
        F_samples = self._sample_f_given_u(mX, mZ, KZZ, KXZ, varX, posterior_u)

        logp = self._logp(F_samples, y).sum(dim=-1)
        if self.bb_alpha == 0:
            data_fit = logp.mean(dim=0).sum()
        else:
            alpha = torch.as_tensor(self.bb_alpha, dtype=self.dtype, device=self.device)
            log_s = math.log(logp.shape[0])
            data_fit = ((torch.logsumexp(alpha * logp, dim=0) - log_s) / alpha).sum()

        scale = self.num_data / X.shape[0]
        kl, forward_kl, reverse_kl = self._kl_regularizer(posterior_u)
        beta = self._scheduled_beta()
        loss = -scale * data_fit + beta * kl

        self.bb_alphas.append(float((-scale * data_fit).detach().cpu()))
        self.KLs.append(float(kl.detach().cpu()))
        self.function_terms.append(float(kl.detach().cpu()))
        self.betas.append(float(beta))
        self.kl_forwards.append(float(forward_kl.detach().cpu()))
        self.kl_reverses.append(float(reverse_kl.detach().cpu()))
        return loss

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_f_samples(self, X, S):
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        mX, mZ, KZZ, KXZ, _, KXX = self._estimate_prior_moments(X)
        u = self._sample_u(int(S))
        return self._sample_f_given_u_full(mX, mZ, KZZ, KXZ, KXX, u)

    def predict_f(self, X):
        samples = self.predict_f_samples(X, self.num_eval_samples)
        return samples.mean(dim=0), samples.var(dim=0, unbiased=False)

    def predict_y_samples(self, X, S):
        F_samples = self.predict_f_samples(X, S)
        if self.likelihood_type == "regression":
            std = torch.sqrt(torch.exp(self.log_variance))
            return F_samples + std * torch.randn_like(F_samples)
        return F_samples

    def forward(self, predict_at):
        """Return predictive mixture components in the original target scale."""
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        if self.likelihood_type == "regression":
            F_samples = self.predict_f_samples(predict_at, self.num_eval_samples)
            means = F_samples * self.y_std + self.y_mean
            std = torch.sqrt(torch.exp(self.log_variance)) * self.y_std
            return means, torch.ones_like(means) * std

        Fmean, Fvar = self.predict_f(predict_at)
        mean, var = self._predict_mean_and_var(Fmean.unsqueeze(0), Fvar.unsqueeze(0))
        return mean * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std

    def forward_prior(self, predict_at, num_samples):
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)
        samples = self._sample_prior_values(predict_at, num_samples)
        return samples * self.y_std + self.y_mean

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
        if optimizer is None:
            optimizer = torch.optim.Adam(self.vi_parameters(), lr=lr)

        if epochs is None and iterations is None:
            raise ValueError("Either epochs or iterations must be set.")

        scheduler = None
        if cosine_annealing:
            T_max = (
                epochs if epochs is not None
                else max(1, iterations // len(train_loader))
            )
            eta_min = optimizer.param_groups[0]["lr"] / 100
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
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        self._train_critic()
        self._step += 1
        optimizer.zero_grad()
        loss = self.nelbo(X, y)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, data_loader, device=None):
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
        return torch.cat(all_means, dim=1), torch.cat(all_stds, dim=1)
