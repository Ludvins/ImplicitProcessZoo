"""Flow-Calibrated FSVI with a flow-estimated finite-context KL."""

import copy
import math

import torch
from tqdm import tqdm

from ..utils.likelihood import (
    bernoulli_logp,
    gaussian_logp,
    inv_probit,
    multiclass_logp,
)
from ..utils.utils import infinite_loader
from .density_flow import (
    AffineContextDensityFlow,
    ConditionalContextDensityFlow,
    ContextDensityFlow,
)


class FCFSVI(torch.nn.Module):
    """Function-space VI with a flow-calibrated finite-context KL."""

    def __init__(
        self,
        generative_function,
        prior_function=None,
        input_dim=None,
        output_dim=1,
        likelihood="regression",
        num_classes=None,
        num_data=1,
        num_context=64,
        num_samples=512,
        num_prior_samples=2048,
        beta=1.0,
        beta_start=1.0,
        beta_warmup_steps=0,
        data_pretrain_steps=0,
        data_loss="expected_nll",
        context_weights=(0.2, 0.2, 0.6),
        near_data_noise=0.1,
        domain_bounds=None,
        domain_std=2.0,
        reservoir_size=1000,
        nf_flow_arch="prior_whitened_affine",
        nf_flow_depth=4,
        nf_flow_hidden_dim=128,
        nf_conditional_flow=False,
        nf_flow_condition_hidden_dim=None,
        nf_flow_condition_embedding_dim=None,
        nf_flow_num_bins=8,
        nf_flow_bound=3.0,
        nf_residual_flow_depth=2,
        nf_residual_flow_hidden_dim=64,
        nf_residual_flow_scale_bound=0.2,
        nf_prior_fit_steps=50000,
        nf_prior_fit_rtol=1e-5,
        nf_prior_fit_patience=10,
        nf_prior_fit_eval_every=200,
        nf_prior_fit_val_samples=16384,
        nf_posterior_flow_steps=600,
        nf_posterior_flow_rtol=1e-4,
        nf_posterior_flow_patience=1,
        nf_posterior_flow_min_steps=1,
        nf_posterior_flow_eval_every=1,
        nf_posterior_flow_val_fraction=0.25,
        nf_posterior_flow_restore_best=False,
        nf_flow_diagnostics_every=50,
        nf_flow_lr=1e-3,
        nf_flow_batch_size=512,
        nf_kl_floor=0.0,
        optimize_context=False,
        context_optimization_steps=0,
        context_optimization_lr=1e-2,
        context_optimization_every=1,
        context_optimization_num_samples=None,
        log_variance_init=-2.0,
        y_mean=0.0,
        y_std=1.0,
        max_grad_norm=None,
        device=None,
        dtype=torch.float64,
        seed=2147483647,
    ):
        super().__init__()
        if likelihood not in ("regression", "binary", "multiclass"):
            raise ValueError(
                "likelihood must be 'regression', 'binary', or 'multiclass', "
                f"got {likelihood!r}."
            )
        if likelihood == "multiclass" and num_classes is None:
            raise ValueError("num_classes is required for multiclass likelihood.")
        if generative_function is None:
            raise ValueError("FCFSVI requires a generative_function.")
        if input_dim is None:
            input_dim = getattr(generative_function, "input_dim", None)
        if input_dim is None:
            raise ValueError("input_dim is required when the generator lacks input_dim.")
        output_dim = getattr(generative_function, "output_dim", output_dim)
        if num_context <= 0:
            raise ValueError("num_context must be positive.")
        if num_context * output_dim < 2:
            raise ValueError("num_context * output_dim must be at least 2.")
        if num_samples <= 0 or num_prior_samples <= 0:
            raise ValueError("num_samples and num_prior_samples must be positive.")
        if nf_prior_fit_steps < 0 or nf_posterior_flow_steps < 0:
            raise ValueError("flow fit step counts must be non-negative.")
        if nf_prior_fit_rtol is not None and nf_prior_fit_rtol < 0:
            raise ValueError("nf_prior_fit_rtol must be non-negative or None.")
        if nf_posterior_flow_rtol is not None and nf_posterior_flow_rtol < 0:
            raise ValueError("nf_posterior_flow_rtol must be non-negative or None.")
        if nf_prior_fit_patience <= 0 or nf_posterior_flow_patience <= 0:
            raise ValueError("flow fit patience values must be positive.")
        if nf_prior_fit_eval_every <= 0 or nf_posterior_flow_eval_every <= 0:
            raise ValueError("flow fit eval_every values must be positive.")
        if nf_prior_fit_val_samples <= 0:
            raise ValueError("nf_prior_fit_val_samples must be positive.")
        if nf_posterior_flow_min_steps < 0:
            raise ValueError("nf_posterior_flow_min_steps must be non-negative.")
        if not (0.0 <= nf_posterior_flow_val_fraction < 1.0):
            raise ValueError("nf_posterior_flow_val_fraction must be in [0, 1).")
        if nf_flow_diagnostics_every <= 0:
            raise ValueError("nf_flow_diagnostics_every must be positive.")
        if nf_flow_batch_size <= 0:
            raise ValueError("nf_flow_batch_size must be positive.")
        if context_optimization_steps < 0:
            raise ValueError("context_optimization_steps must be non-negative.")
        if context_optimization_lr <= 0:
            raise ValueError("context_optimization_lr must be positive.")
        if context_optimization_every <= 0:
            raise ValueError("context_optimization_every must be positive.")
        if (
            context_optimization_num_samples is not None
            and context_optimization_num_samples <= 0
        ):
            raise ValueError("context_optimization_num_samples must be positive.")
        if data_loss not in ("expected_nll", "expected", "elbo", "predictive_nll", "mixture_nll", "log_mean_exp"):
            raise ValueError(f"Unknown data_loss mode: {data_loss!r}.")
        if nf_flow_arch not in ("spline", "prior_whitened_affine"):
            raise ValueError(
                "nf_flow_arch must be 'spline' or 'prior_whitened_affine', "
                f"got {nf_flow_arch!r}."
            )
        if nf_flow_arch == "prior_whitened_affine" and nf_conditional_flow:
            raise ValueError(
                "prior_whitened_affine does not support nf_conditional_flow; "
                "use a fixed context set and the unconditional prior flow."
            )
        if nf_residual_flow_depth <= 0:
            raise ValueError("nf_residual_flow_depth must be positive.")
        if nf_residual_flow_hidden_dim <= 0:
            raise ValueError("nf_residual_flow_hidden_dim must be positive.")
        if nf_residual_flow_scale_bound <= 0:
            raise ValueError("nf_residual_flow_scale_bound must be positive.")

        self.likelihood_type = likelihood
        self.num_classes = num_classes
        self.epsilon = 1e-3
        self.num_data = num_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_context = num_context
        self.num_samples = num_samples
        self.num_prior_samples = num_prior_samples
        self.beta = beta
        self.beta_start = beta_start
        self.beta_warmup_steps = beta_warmup_steps
        self.data_pretrain_steps = data_pretrain_steps
        self.data_loss = data_loss
        self.near_data_noise = near_data_noise
        self.domain_std = domain_std
        self._reservoir_size = reservoir_size
        self._reservoir = None
        self.nf_flow_arch = nf_flow_arch
        self.nf_prior_fit_steps = nf_prior_fit_steps
        self.nf_prior_fit_rtol = nf_prior_fit_rtol
        self.nf_prior_fit_patience = nf_prior_fit_patience
        self.nf_prior_fit_eval_every = nf_prior_fit_eval_every
        self.nf_prior_fit_val_samples = nf_prior_fit_val_samples
        self.nf_posterior_flow_steps = nf_posterior_flow_steps
        self.nf_posterior_flow_rtol = nf_posterior_flow_rtol
        self.nf_posterior_flow_patience = nf_posterior_flow_patience
        self.nf_posterior_flow_min_steps = nf_posterior_flow_min_steps
        self.nf_posterior_flow_eval_every = nf_posterior_flow_eval_every
        self.nf_posterior_flow_val_fraction = nf_posterior_flow_val_fraction
        self.nf_posterior_flow_restore_best = bool(nf_posterior_flow_restore_best)
        self.nf_flow_diagnostics_every = nf_flow_diagnostics_every
        self.nf_flow_lr = nf_flow_lr
        self.nf_flow_batch_size = nf_flow_batch_size
        self.nf_kl_floor = nf_kl_floor
        self.nf_conditional_flow = bool(nf_conditional_flow)
        self.nf_flow_condition_hidden_dim = nf_flow_condition_hidden_dim
        self.nf_flow_condition_embedding_dim = nf_flow_condition_embedding_dim
        self.nf_residual_flow_depth = nf_residual_flow_depth
        self.nf_residual_flow_hidden_dim = nf_residual_flow_hidden_dim
        self.nf_residual_flow_scale_bound = nf_residual_flow_scale_bound
        self.optimize_context = bool(optimize_context)
        self.context_optimization_steps = context_optimization_steps
        self.context_optimization_lr = context_optimization_lr
        self.context_optimization_every = context_optimization_every
        self.context_optimization_num_samples = context_optimization_num_samples
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.dtype = dtype
        self._step = 0
        self._flows_initialized = False

        self.generative_function = generative_function
        if prior_function is None:
            prior_function = copy.deepcopy(generative_function)
        self.prior_function = prior_function
        for param in self.prior_function.parameters():
            param.requires_grad = False

        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=dtype, device=device))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=dtype, device=device))
        self.register_buffer(
            "context_weights",
            _normalize_weights(context_weights, dtype=dtype, device=device),
        )
        self.register_buffer(
            "context_inputs",
            torch.empty(0, input_dim, dtype=dtype, device=device),
        )
        if domain_bounds is not None:
            bounds = torch.as_tensor(domain_bounds, dtype=dtype, device=device)
            if bounds.ndim == 1 and bounds.numel() == 2:
                bounds = bounds.view(1, 2)
            self.register_buffer("domain_bounds", bounds)
        else:
            self.domain_bounds = None

        if likelihood == "regression":
            log_variance = torch.as_tensor(log_variance_init, dtype=dtype, device=device)
            if log_variance.ndim == 0:
                log_variance = log_variance.expand(output_dim).clone()
            elif log_variance.shape != (output_dim,):
                raise ValueError(
                    "log_variance_init must be a scalar or have shape "
                    f"({output_dim},), got {tuple(log_variance.shape)}."
                )
            self.log_variance = torch.nn.Parameter(log_variance.clone())

        flow_dim = num_context * output_dim
        flow_kwargs = dict(
            input_dim=flow_dim,
            depth=nf_flow_depth,
            hidden_dim=nf_flow_hidden_dim,
            num_bins=nf_flow_num_bins,
            bound=nf_flow_bound,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        if self.nf_conditional_flow:
            flow_kwargs.update(
                condition_dim=num_context * input_dim,
                condition_hidden_dim=nf_flow_condition_hidden_dim,
                condition_embedding_dim=nf_flow_condition_embedding_dim,
            )
            self.prior_flow = ConditionalContextDensityFlow(**flow_kwargs)
            self.posterior_flow = ConditionalContextDensityFlow(**flow_kwargs)
        elif self._uses_prior_whitened_residual:
            self.prior_flow = ContextDensityFlow(**flow_kwargs)
            self.posterior_flow = AffineContextDensityFlow(
                input_dim=flow_dim,
                depth=nf_residual_flow_depth,
                hidden_dim=nf_residual_flow_hidden_dim,
                scale_bound=nf_residual_flow_scale_bound,
                device=device,
                dtype=dtype,
                seed=seed,
            )
        else:
            self.prior_flow = ContextDensityFlow(**flow_kwargs)
            self.posterior_flow = ContextDensityFlow(**flow_kwargs)
        _set_requires_grad(self.prior_flow, False)
        _set_requires_grad(self.posterior_flow, False)
        self._posterior_flow_optimizer = None

        gen_device = torch.device(device) if device is not None else torch.device("cpu")
        self.generator = torch.Generator(gen_device)
        self.generator.manual_seed(seed)
        self._cpu_generator = torch.Generator()
        self._cpu_generator.manual_seed(seed)

        self.data_terms = []
        self.KLs = []
        self.function_terms = []
        self.betas = []
        self.nf_kl_raws = []
        self.prior_flow_nlls = []
        self.prior_flow_train_nlls = []
        self.prior_flow_val_nlls = []
        self.prior_flow_relative_improvements = []
        self.prior_flow_update_counts = []
        self.prior_flow_converged_flags = []
        self.prior_flow_best_val_nlls = []
        self.prior_flow_final_val_nlls = []
        self.posterior_flow_nlls = []
        self.posterior_flow_nlls_before = []
        self.posterior_flow_nlls_after = []
        self.posterior_flow_train_nlls = []
        self.posterior_flow_val_nlls = []
        self.posterior_flow_relative_improvements = []
        self.posterior_flow_update_counts = []
        self.posterior_flow_converged_flags = []
        self.posterior_flow_fit_sample_counts = []
        self.posterior_flow_val_sample_counts = []
        self.context_optimization_kls_before = []
        self.context_optimization_kls_after = []
        self.context_optimization_update_counts = []
        self.context_input_norms = []

    def load_state_dict(self, state_dict, strict=True):
        context = state_dict.get("context_inputs")
        if context is not None and tuple(context.shape) != tuple(self.context_inputs.shape):
            self.context_inputs.resize_as_(context)
        result = torch.nn.Module.load_state_dict(self, state_dict, strict=strict)
        self._flows_initialized = self.context_inputs.numel() > 0
        if self._flows_initialized:
            _set_requires_grad(self.prior_flow, False)
            _set_requires_grad(self.posterior_flow, False)
            self._posterior_flow_optimizer = torch.optim.Adam(
                self.posterior_flow.parameters(), lr=self.nf_flow_lr
            )
        return result

    def vi_parameters(self):
        params = [p for p in self.generative_function.parameters() if p.requires_grad]
        if hasattr(self, "log_variance"):
            params.append(self.log_variance)
        return params

    @property
    def _uses_prior_whitened_residual(self):
        return self.nf_flow_arch == "prior_whitened_affine"

    # ------------------------------------------------------------------
    # Sampling and prediction
    # ------------------------------------------------------------------

    def predict_f_samples(self, X, S):
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        return _sample_function(self.generative_function, X, S)

    def predict_y_samples(self, X, S):
        F = self.predict_f_samples(X, S)
        if self.likelihood_type == "binary":
            return inv_probit(F)
        if self.likelihood_type == "multiclass":
            return torch.softmax(F, dim=-1)
        std = torch.sqrt(torch.exp(self.log_variance)).view(1, 1, -1)
        return F + std * torch.randn(
            F.shape, generator=self.generator, dtype=F.dtype, device=F.device
        )

    def forward(self, X):
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.likelihood_type != "regression":
            return self.predict_y_samples(X, self.num_samples), None
        F = self.predict_f_samples(X, self.num_samples)
        mean = F * self.y_std + self.y_mean
        std = torch.sqrt(torch.exp(self.log_variance)).view(1, 1, -1)
        std = std.expand_as(F) * self.y_std
        return mean, std

    def predict(self, X, S):
        self.eval()
        with torch.no_grad():
            if self.dtype != X.dtype:
                X = X.to(self.dtype)
            Y = self.predict_y_samples(X, S)
            if self.likelihood_type == "regression":
                return Y * self.y_std + self.y_mean
            return Y

    def forward_prior(self, X, num_samples):
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        prior = self._sample_prior(X, num_samples)
        if self.likelihood_type == "regression":
            return prior * self.y_std + self.y_mean
        if self.likelihood_type == "binary":
            return inv_probit(prior)
        return torch.softmax(prior, dim=-1)

    def _sample_prior(self, X, S):
        with torch.no_grad():
            return _sample_function(self.prior_function, X, S)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nelbo(self, X, y):
        X = X.to(dtype=self.dtype, device=self.device)
        y = self._prepare_targets(y)
        self._ensure_context_and_flows_initialized(X)
        self._maybe_optimize_context()

        context = self.context_inputs.to(dtype=self.dtype, device=X.device)
        X_joint = torch.cat([context, X], dim=0)
        F_joint = self.predict_f_samples(X_joint, self.num_samples)
        z_q = _flatten_context_values(F_joint[:, : self.num_context, :])
        F_batch = F_joint[:, self.num_context :, :]

        log_flow_after = self._fine_tune_posterior_flow(z_q.detach(), context)
        log_q, log_p = self._context_log_probs(z_q, context)
        kl_raw = (log_q - log_p).mean()
        if log_flow_after:
            if self._uses_prior_whitened_residual:
                fit_z = self._posterior_flow_fit_inputs(z_q.detach(), context)
                with torch.no_grad():
                    self.posterior_flow_nlls_after.append(
                        self._flow_nll(self.posterior_flow, fit_z).detach()
                    )
            else:
                self.posterior_flow_nlls_after.append((-log_q.detach()).mean())
        kl = kl_raw.clamp_min(self.nf_kl_floor)

        data_term = self._data_term(F_batch, y, X.shape[0])
        beta = self._scheduled_beta()
        loss = data_term + beta * kl

        self.data_terms.append(data_term.detach())
        self.KLs.append(kl.detach())
        self.function_terms.append(kl.detach())
        self.betas.append(beta)
        self.nf_kl_raws.append(kl_raw.detach())
        return loss

    def _data_term(self, F, y, batch_size):
        logpdf = self._logp(F, y)
        if self.data_loss in ("expected_nll", "expected", "elbo"):
            ve = torch.mean(logpdf, dim=0)
        else:
            ve = torch.logsumexp(logpdf, dim=0) - math.log(F.shape[0])
        return -(self.num_data / batch_size) * torch.sum(ve)

    def _logp(self, F, y):
        if self.likelihood_type == "regression":
            return gaussian_logp(F, y, self.log_variance)
        if self.likelihood_type == "binary":
            return bernoulli_logp(F, y)
        return multiclass_logp(F, y, self.num_classes, self.epsilon)

    def _context_kl(self, z_q, context_inputs=None):
        log_q, log_p = self._context_log_probs(z_q, context_inputs)
        return (log_q - log_p).mean()

    def _context_log_probs(self, z_q, context_inputs=None):
        _set_requires_grad(self.prior_flow, False)
        _set_requires_grad(self.posterior_flow, False)
        if self._uses_prior_whitened_residual:
            v_q, prior_log_det = self._prior_whiten(z_q, context_inputs)
            log_p = self._standard_normal_log_prob(v_q) + prior_log_det
            log_q = self._flow_log_prob(self.posterior_flow, v_q) + prior_log_det
            return log_q, log_p
        log_q = self._flow_log_prob(self.posterior_flow, z_q, context_inputs)
        log_p = self._flow_log_prob(self.prior_flow, z_q, context_inputs)
        return log_q, log_p

    def _prior_whiten(self, samples, context_inputs=None):
        if self.nf_conditional_flow:
            if context_inputs is None:
                context_inputs = self.context_inputs
            x_std = (
                samples - self.prior_flow.loc.view(1, -1)
            ) / self.prior_flow.scale.view(1, -1)
            v, inverse_log_det = self.prior_flow.inverse(x_std, context_inputs)
        else:
            x_std = (
                samples - self.prior_flow.loc.view(1, -1)
            ) / self.prior_flow.scale.view(1, -1)
            v, inverse_log_det = self.prior_flow.inverse(x_std)
        standardization_log_det = -self.prior_flow.scale.log().sum()
        return v, inverse_log_det + standardization_log_det

    def _standard_normal_log_prob(self, samples):
        log2pi = torch.as_tensor(
            math.log(2.0 * math.pi), dtype=samples.dtype, device=samples.device
        )
        return -0.5 * (samples.square().sum(dim=-1) + samples.shape[-1] * log2pi)

    def _scheduled_beta(self):
        if self._step <= self.data_pretrain_steps:
            return 0.0
        if self.beta_warmup_steps <= 0:
            return self.beta
        warmup_step = min(
            max(self._step - self.data_pretrain_steps, 0), self.beta_warmup_steps
        )
        progress = warmup_step / self.beta_warmup_steps
        return self.beta_start + progress * (self.beta - self.beta_start)

    def _prepare_targets(self, y):
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        target_device = self.device if self.device is not None else y.device
        if self.likelihood_type == "multiclass":
            return y.to(device=target_device)
        return y.to(dtype=self.dtype, device=target_device)

    # ------------------------------------------------------------------
    # Context and flow fitting
    # ------------------------------------------------------------------

    def _ensure_context_and_flows_initialized(self, X_batch):
        if self.context_inputs.numel() == 0:
            context = self._sample_context_inputs(X_batch)
            self.context_inputs.resize_as_(context)
            self.context_inputs.copy_(context)
        if not self._flows_initialized:
            self._initialize_flows()

    def _initialize_flows(self):
        with torch.no_grad():
            init_samples = self._sample_prior(
                self.context_inputs,
                max(self.num_prior_samples, self.nf_flow_batch_size),
            )
            init_flat = _flatten_context_values(init_samples)
        self.prior_flow.set_standardization(init_flat)
        if self._uses_prior_whitened_residual:
            self.posterior_flow.reset_identity()
        else:
            self.posterior_flow.load_state_dict(self.prior_flow.state_dict())

        losses = []
        converged = False
        prior_val_flat = None
        best_val_nll = None
        best_state = None
        bad_checks = 0
        use_prior_early_stop = self.nf_prior_fit_rtol is not None
        if use_prior_early_stop:
            with torch.no_grad():
                prior_val = self._sample_prior(
                    self.context_inputs, self.nf_prior_fit_val_samples
                )
                prior_val_flat = _flatten_context_values(prior_val)
                val_nll = self._flow_nll(
                    self.prior_flow, prior_val_flat, self.context_inputs
                ).detach()
            self.prior_flow_val_nlls.append(val_nll)
            best_val_nll = val_nll
            best_state = _clone_state_dict(self.prior_flow)
        if self.nf_prior_fit_steps > 0:
            _set_requires_grad(self.prior_flow, True)
            optimizer = torch.optim.Adam(self.prior_flow.parameters(), lr=self.nf_flow_lr)
            for step in range(1, self.nf_prior_fit_steps + 1):
                with torch.no_grad():
                    batch = self._sample_prior(self.context_inputs, self.nf_flow_batch_size)
                    batch = _flatten_context_values(batch)
                optimizer.zero_grad(set_to_none=True)
                loss = self._flow_nll(self.prior_flow, batch, self.context_inputs)
                loss.backward()
                optimizer.step()
                loss_detached = loss.detach()
                losses.append(loss_detached)
                self.prior_flow_train_nlls.append(loss_detached)
                if (
                    use_prior_early_stop
                    and step % self.nf_prior_fit_eval_every == 0
                ):
                    with torch.no_grad():
                        val_nll = self._flow_nll(
                            self.prior_flow, prior_val_flat, self.context_inputs
                        ).detach()
                    self.prior_flow_val_nlls.append(val_nll)
                    rel = _relative_improvement(best_val_nll, val_nll)
                    self.prior_flow_relative_improvements.append(rel)
                    if val_nll < best_val_nll:
                        best_val_nll = val_nll
                        best_state = _clone_state_dict(self.prior_flow)
                    if rel.item() > self.nf_prior_fit_rtol:
                        bad_checks = 0
                    else:
                        bad_checks += 1
                    if bad_checks >= self.nf_prior_fit_patience:
                        converged = True
                        break
        if use_prior_early_stop and best_state is not None:
            self.prior_flow.load_state_dict(best_state)
        self.prior_flow_update_counts.append(len(losses))
        with torch.no_grad():
            prior_nll = self._flow_nll(
                self.prior_flow, init_flat, self.context_inputs
            )
            self.prior_flow_nlls.append(prior_nll.detach())
            if use_prior_early_stop:
                final_val_nll = self._flow_nll(
                    self.prior_flow, prior_val_flat, self.context_inputs
                ).detach()
                self.prior_flow_best_val_nlls.append(best_val_nll)
                self.prior_flow_final_val_nlls.append(final_val_nll)
        if use_prior_early_stop:
            self.prior_flow_converged_flags.append(float(converged))

        if self._uses_prior_whitened_residual:
            self.posterior_flow.reset_identity()
        else:
            self.posterior_flow.load_state_dict(self.prior_flow.state_dict())
        _set_requires_grad(self.prior_flow, False)
        _set_requires_grad(self.posterior_flow, False)
        _clear_grads(self.prior_flow)
        _clear_grads(self.posterior_flow)
        self._posterior_flow_optimizer = torch.optim.Adam(
            self.posterior_flow.parameters(), lr=self.nf_flow_lr
        )
        self._flows_initialized = True

    def _maybe_optimize_context(self):
        if (
            not self.optimize_context
            or self.context_optimization_steps <= 0
            or self._step % self.context_optimization_every != 0
        ):
            return
        self._optimize_context_inputs()

    def _optimize_context_inputs(self):
        _set_requires_grad(self.prior_flow, False)
        _set_requires_grad(self.posterior_flow, False)
        param_states = _freeze_params(self.generative_function)
        C = self.context_inputs.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([C], lr=self.context_optimization_lr)
        S = self.context_optimization_num_samples or self.num_samples

        try:
            with torch.enable_grad():
                with torch.no_grad():
                    kl_before = self._context_kl_for_inputs(C.detach(), S).detach()
                for _ in range(self.context_optimization_steps):
                    optimizer.zero_grad(set_to_none=True)
                    kl = self._context_kl_for_inputs(C, S)
                    (-kl).backward()
                    optimizer.step()
        finally:
            _restore_params(param_states)
        with torch.no_grad():
            kl_after = self._context_kl_for_inputs(C.detach(), S).detach()
            self.context_inputs.copy_(C.detach())
            self.context_optimization_kls_before.append(kl_before)
            self.context_optimization_kls_after.append(kl_after)
            self.context_optimization_update_counts.append(
                self.context_optimization_steps
            )
            self.context_input_norms.append(self.context_inputs.norm().detach())

    def _context_kl_for_inputs(self, context_inputs, S):
        F = self.predict_f_samples(context_inputs, S)
        z = _flatten_context_values(F)
        return self._context_kl(z, context_inputs)

    def _should_log_flow_diagnostics(self):
        return self._step % self.nf_flow_diagnostics_every == 0

    def _fine_tune_posterior_flow(self, z_q, context_inputs=None):
        fit_z = self._posterior_flow_fit_inputs(z_q, context_inputs)
        log_diagnostics = self._should_log_flow_diagnostics()
        if log_diagnostics:
            with torch.no_grad():
                nll_before = self._flow_nll(
                    self.posterior_flow, fit_z, context_inputs
                ).detach()
            self.posterior_flow_nlls_before.append(nll_before)
        if self.nf_posterior_flow_steps <= 0:
            if log_diagnostics:
                self.posterior_flow_nlls.append(nll_before)
            self.posterior_flow_update_counts.append(0)
            self.posterior_flow_converged_flags.append(0.0)
            self.posterior_flow_fit_sample_counts.append(0)
            self.posterior_flow_val_sample_counts.append(0)
            return log_diagnostics
        _set_requires_grad(self.posterior_flow, True)
        if self._posterior_flow_optimizer is None:
            self._posterior_flow_optimizer = torch.optim.Adam(
                self.posterior_flow.parameters(), lr=self.nf_flow_lr
            )
        train_z, val_z = self._posterior_flow_train_val_split(fit_z)
        use_posterior_early_stop = (
            self.nf_posterior_flow_rtol is not None and val_z is not None
        )
        best_val_nll = None
        best_state = None
        bad_checks = 0
        converged = False
        if use_posterior_early_stop:
            with torch.no_grad():
                val_nll = self._flow_nll(
                    self.posterior_flow, val_z, context_inputs
                ).detach()
            if log_diagnostics:
                self.posterior_flow_val_nlls.append(val_nll)
            best_val_nll = val_nll
            if self.nf_posterior_flow_restore_best:
                best_state = _clone_state_dict(self.posterior_flow)
        losses = []
        update_count = 0
        fit_sample_counts = []
        for step in range(1, self.nf_posterior_flow_steps + 1):
            batch = self._flow_batch(train_z, train_z.shape[0])
            fit_sample_counts.append(batch.shape[0])
            self._posterior_flow_optimizer.zero_grad(set_to_none=True)
            loss = self._flow_nll(self.posterior_flow, batch, context_inputs)
            loss.backward()
            self._posterior_flow_optimizer.step()
            update_count += 1
            loss_detached = loss.detach()
            if log_diagnostics:
                losses.append(loss_detached)
                self.posterior_flow_train_nlls.append(loss_detached)
            if (
                use_posterior_early_stop
                and step >= self.nf_posterior_flow_min_steps
                and step % self.nf_posterior_flow_eval_every == 0
            ):
                with torch.no_grad():
                    val_nll = self._flow_nll(
                        self.posterior_flow, val_z, context_inputs
                    ).detach()
                if log_diagnostics:
                    self.posterior_flow_val_nlls.append(val_nll)
                rel = _relative_improvement(best_val_nll, val_nll)
                if log_diagnostics:
                    self.posterior_flow_relative_improvements.append(rel)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    if self.nf_posterior_flow_restore_best:
                        best_state = _clone_state_dict(self.posterior_flow)
                if rel.item() > self.nf_posterior_flow_rtol:
                    bad_checks = 0
                else:
                    bad_checks += 1
                if bad_checks >= self.nf_posterior_flow_patience:
                    converged = True
                    break
        if use_posterior_early_stop and best_state is not None:
            self.posterior_flow.load_state_dict(best_state)
        if losses:
            self.posterior_flow_nlls.append(torch.stack(losses).mean().detach())
        self.posterior_flow_update_counts.append(update_count)
        fit_count = fit_sample_counts[-1] if fit_sample_counts else 0
        val_count = 0 if val_z is None else val_z.shape[0]
        self.posterior_flow_converged_flags.append(float(converged))
        self.posterior_flow_fit_sample_counts.append(fit_count)
        self.posterior_flow_val_sample_counts.append(val_count)
        _clear_grads(self.posterior_flow)
        _set_requires_grad(self.posterior_flow, False)
        return log_diagnostics

    def _flow_log_prob(self, flow, samples, context_inputs=None):
        if self.nf_conditional_flow:
            if context_inputs is None:
                context_inputs = self.context_inputs
            return flow.log_prob(samples, context_inputs)
        return flow.log_prob(samples)

    def _flow_nll(self, flow, samples, context_inputs=None):
        if self.nf_conditional_flow:
            if context_inputs is None:
                context_inputs = self.context_inputs
            return flow.nll(samples, context_inputs)
        return flow.nll(samples)

    def _posterior_flow_fit_inputs(self, samples, context_inputs=None):
        if not self._uses_prior_whitened_residual:
            return samples
        with torch.no_grad():
            whitened, _ = self._prior_whiten(samples, context_inputs)
        return whitened.detach()

    def _posterior_flow_train_val_split(self, z_q):
        if (
            self.nf_posterior_flow_rtol is None
            or self.nf_posterior_flow_val_fraction <= 0
            or z_q.shape[0] < 2
        ):
            return z_q, None
        val_count = int(round(z_q.shape[0] * self.nf_posterior_flow_val_fraction))
        val_count = max(1, min(z_q.shape[0] - 1, val_count))
        return z_q[:-val_count], z_q[-val_count:]

    def _flow_batch(self, samples, n):
        if n <= self.nf_flow_batch_size:
            return samples
        idx = torch.randint(
            n,
            (self.nf_flow_batch_size,),
            generator=self.generator,
            device=samples.device,
        )
        return samples[idx]

    def _sample_context_inputs(self, X_batch):
        counts = _allocate_counts(self.context_weights, self.num_context)
        parts = []
        if counts[0] > 0:
            parts.append(self._sample_data_points(X_batch, counts[0]))
        if counts[1] > 0:
            base = self._sample_data_points(X_batch, counts[1])
            near = base + self.near_data_noise * torch.randn(
                base.shape, generator=self.generator, dtype=base.dtype, device=base.device
            )
            parts.append(self._clip_to_domain(near))
        if counts[2] > 0:
            parts.append(self._sample_domain_points(X_batch, counts[2]))
        context = torch.cat(parts, dim=0)
        perm = torch.randperm(context.shape[0], generator=self.generator, device=context.device)
        return context[perm]

    def _sample_data_points(self, X_batch, count):
        source = self._reservoir
        if source is None:
            source = X_batch.detach()
        source = source.to(dtype=X_batch.dtype, device=X_batch.device)
        idx = torch.randint(
            source.shape[0],
            (count,),
            generator=self.generator,
            device=X_batch.device,
        )
        return source[idx]

    def _sample_domain_points(self, X_batch, count):
        d = X_batch.shape[-1]
        if self.domain_bounds is None:
            return torch.randn(
                count,
                d,
                generator=self.generator,
                dtype=X_batch.dtype,
                device=X_batch.device,
            ) * self.domain_std
        bounds = self.domain_bounds.to(dtype=X_batch.dtype, device=X_batch.device)
        if bounds.shape[0] == 1:
            low = bounds[:, 0].expand(1, d)
            high = bounds[:, 1].expand(1, d)
        else:
            low = bounds[:, 0].view(1, d)
            high = bounds[:, 1].view(1, d)
        unit = torch.rand(count, d, generator=self.generator, dtype=X_batch.dtype, device=X_batch.device)
        return low + unit * (high - low)

    def _clip_to_domain(self, X):
        if self.domain_bounds is None:
            return X
        bounds = self.domain_bounds.to(dtype=X.dtype, device=X.device)
        d = X.shape[-1]
        if bounds.shape[0] == 1:
            low = bounds[:, 0].expand(1, d)
            high = bounds[:, 1].expand(1, d)
        else:
            low = bounds[:, 0].view(1, d)
            high = bounds[:, 1].view(1, d)
        return torch.maximum(torch.minimum(X, high), low)

    def _fill_reservoir(self, train_loader):
        if self._reservoir_size == 0:
            return
        seen = 0
        reservoir = None
        for batch_X, _ in train_loader:
            batch_X = batch_X.detach()
            if reservoir is None:
                reservoir = torch.empty(
                    self._reservoir_size, batch_X.shape[-1], dtype=batch_X.dtype
                )
            for row in batch_X:
                if seen < self._reservoir_size:
                    reservoir[seen] = row
                else:
                    j = torch.randint(
                        0, seen + 1, (1,), generator=self._cpu_generator
                    ).item()
                    if j < self._reservoir_size:
                        reservoir[j] = row
                seen += 1
        self._reservoir = None if reservoir is None else reservoir[: min(seen, self._reservoir_size)]

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
        self._fill_reservoir(train_loader)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.vi_parameters(), lr=lr)
        if epochs is None and iterations is None:
            raise ValueError("Either epochs or iterations must be set.")

        scheduler = None
        if cosine_annealing:
            t_max = epochs if epochs is not None else max(1, iterations // len(train_loader))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=lr / 100
            )

        self.train()
        losses = []
        if epochs is not None:
            loop = tqdm(range(epochs), unit=" epoch", desc="Training") if use_tqdm else range(epochs)
            for _ in loop:
                for inputs, target in train_loader:
                    loss = self._train_step(optimizer, inputs, target)
                    if return_loss:
                        losses.append(loss.detach().cpu().numpy())
                if scheduler is not None:
                    scheduler.step()

        if iterations is not None:
            data_stream = infinite_loader(train_loader)
            iters_per_epoch = len(train_loader)
            loop = tqdm(range(iterations), unit=" iter", desc="Training") if use_tqdm else range(iterations)
            for i in loop:
                inputs, target = next(data_stream)
                loss = self._train_step(optimizer, inputs, target)
                if return_loss:
                    losses.append(loss.detach().cpu().numpy())
                if scheduler is not None and (i + 1) % iters_per_epoch == 0:
                    scheduler.step()
        return losses

    def _train_step(self, optimizer, X, y):
        X = X.to(dtype=self.dtype, device=self.device)
        y = self._prepare_targets(y)
        self._step += 1
        optimizer.zero_grad(set_to_none=True)
        loss = self.nelbo(X, y)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()
        return loss


def _sample_function(function, X, S):
    try:
        return function(X, num_samples=S)
    except TypeError:
        pass
    states = _set_num_samples_recursive(function, S)
    try:
        try:
            return function(X)
        except TypeError:
            return function(X, S)
    finally:
        _restore_num_samples_recursive(states)


def _set_num_samples_recursive(module, S):
    states = []
    for submodule in module.modules():
        if hasattr(submodule, "num_samples"):
            state = (submodule, submodule.num_samples, getattr(submodule, "noise", None))
            states.append(state)
            old = submodule.num_samples
            submodule.num_samples = S
            if (
                hasattr(submodule, "fix_random_noise")
                and submodule.fix_random_noise
                and old != S
                and hasattr(submodule, "get_noise")
            ):
                submodule.noise = submodule.get_noise(first_call=True)
    return states


def _restore_num_samples_recursive(states):
    for module, num_samples, noise in states:
        module.num_samples = num_samples
        if hasattr(module, "noise"):
            module.noise = noise


def _flatten_context_values(values):
    if values is None:
        return None
    if values.ndim != 3:
        raise ValueError(f"context values must have shape [S, M, D], got {tuple(values.shape)}.")
    return values.reshape(values.shape[0], -1)


def _set_requires_grad(module, requires_grad):
    requires_grad = bool(requires_grad)
    if getattr(module, "_fcfsvi_requires_grad_state", None) == requires_grad:
        return
    for param in module.parameters():
        param.requires_grad_(requires_grad)
    module._fcfsvi_requires_grad_state = requires_grad


def _freeze_params(module):
    states = []
    for param in module.parameters():
        states.append((param, param.requires_grad))
        param.requires_grad_(False)
    return states


def _restore_params(states):
    for param, requires_grad in states:
        param.requires_grad_(requires_grad)


def _clear_grads(module):
    module.zero_grad(set_to_none=True)


def _clone_state_dict(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _relative_improvement(previous, current, eps=1e-12):
    denom = torch.clamp(torch.abs(previous), min=eps)
    return (previous - current) / denom


def _normalize_weights(weights, dtype, device):
    weights = torch.as_tensor(weights, dtype=dtype, device=device)
    if weights.numel() != 3:
        raise ValueError("context_weights must contain data/near/domain weights.")
    if torch.any(weights < 0) or torch.sum(weights) <= 0:
        raise ValueError("context_weights must be non-negative with positive sum.")
    return weights / torch.sum(weights)


def _allocate_counts(weights, total):
    raw = weights * total
    counts = torch.floor(raw).to(torch.long)
    remainder = int(total - counts.sum().item())
    if remainder > 0:
        fractional = raw - counts.to(raw.dtype)
        order = torch.argsort(fractional, descending=True)
        for idx in order[:remainder]:
            counts[idx] += 1
    return [int(v.item()) for v in counts]
