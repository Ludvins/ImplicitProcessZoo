"""Tests for NF-IP (normalizing-flow context KL for implicit processes)."""

import torch

from src.nf_ip import ConditionalContextDensityFlow, ContextDensityFlow, NFIP
from src.priors.generative_functions import BayesianNN, BayesLinear
from tests.conftest import (
    BATCH_SIZE,
    DEVICE,
    DTYPE,
    INPUT_DIM,
    NUM_DATA,
    NUM_SAMPLES,
    OUTPUT_DIM,
    SEED,
)


def _make_generator(num_samples=NUM_SAMPLES, seed=SEED):
    return BayesianNN(
        structure=[8],
        activation=torch.tanh,
        num_samples=num_samples,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        layer_model=BayesLinear,
        seed=seed,
        fix_random_noise=False,
        device=DEVICE,
        dtype=DTYPE,
    )


def _make_model(**kwargs):
    defaults = dict(
        generative_function=_make_generator(num_samples=8),
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        likelihood="regression",
        num_data=NUM_DATA,
        num_context=4,
        num_samples=8,
        num_prior_samples=16,
        beta=0.1,
        beta_start=0.1,
        beta_warmup_steps=0,
        data_pretrain_steps=0,
        nf_flow_depth=2,
        nf_flow_hidden_dim=16,
        nf_flow_num_bins=4,
        nf_prior_fit_steps=3,
        nf_posterior_flow_steps=1,
        nf_flow_batch_size=8,
        nf_flow_lr=1e-2,
        device=DEVICE,
        dtype=DTYPE,
        seed=SEED,
    )
    defaults.update(kwargs)
    return NFIP(**defaults)


def _state_clone(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _state_equal(left, right):
    return all(torch.allclose(value, right[name]) for name, value in left.items())


def _state_changed(left, module):
    current = module.state_dict()
    return any(not torch.allclose(value, current[name]) for name, value in left.items())


class TestContextDensityFlow:

    def test_log_prob_is_finite_and_input_differentiable(self):
        flow = ContextDensityFlow(
            input_dim=4,
            depth=2,
            hidden_dim=16,
            num_bins=4,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        x = torch.randn(10, 4, dtype=DTYPE, device=DEVICE)
        flow.set_standardization(x)
        x_eval = x.detach().clone().requires_grad_(True)
        log_prob = flow.log_prob(x_eval)

        assert log_prob.shape == (10,)
        assert torch.isfinite(log_prob).all()
        log_prob.sum().backward()
        assert x_eval.grad is not None
        assert torch.isfinite(x_eval.grad).all()

    def test_mle_reduces_nll_on_shifted_gaussian(self):
        torch.manual_seed(SEED)
        samples = torch.randn(128, 4, dtype=DTYPE, device=DEVICE) * 0.5 + 1.0
        flow = ContextDensityFlow(
            input_dim=4,
            depth=2,
            hidden_dim=32,
            num_bins=4,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        flow.set_standardization(samples)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
        before = flow.nll(samples).detach()
        for _ in range(15):
            optimizer.zero_grad(set_to_none=True)
            loss = flow.nll(samples)
            loss.backward()
            optimizer.step()
        after = flow.nll(samples).detach()

        assert torch.isfinite(after)
        assert after < before


class TestConditionalContextDensityFlow:

    def test_log_prob_is_finite_and_condition_differentiable(self):
        flow = ConditionalContextDensityFlow(
            input_dim=4,
            condition_dim=6,
            depth=2,
            hidden_dim=16,
            condition_hidden_dim=12,
            condition_embedding_dim=8,
            num_bins=4,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        x = torch.randn(10, 4, dtype=DTYPE, device=DEVICE)
        condition = torch.randn(3, 2, dtype=DTYPE, device=DEVICE)
        flow.set_standardization(x)
        x_eval = x.detach().clone().requires_grad_(True)
        condition_eval = condition.detach().clone().requires_grad_(True)
        log_prob = flow.log_prob(x_eval, condition_eval)

        assert log_prob.shape == (10,)
        assert torch.isfinite(log_prob).all()
        log_prob.sum().backward()
        assert x_eval.grad is not None
        assert condition_eval.grad is not None
        assert torch.isfinite(x_eval.grad).all()
        assert torch.isfinite(condition_eval.grad).all()

    def test_sample_has_expected_shape(self):
        flow = ConditionalContextDensityFlow(
            input_dim=4,
            condition_dim=6,
            depth=1,
            hidden_dim=8,
            num_bins=4,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        samples = torch.randn(16, 4, dtype=DTYPE, device=DEVICE)
        condition = torch.randn(3, 2, dtype=DTYPE, device=DEVICE)
        flow.set_standardization(samples)

        out = flow.sample(5, condition)

        assert out.shape == (5, 4)
        assert torch.isfinite(out).all()


class TestNFIP:

    def test_prior_and_posterior_flows_are_copied_after_initialization(self, regression_data):
        X, _ = regression_data
        model = _make_model(nf_posterior_flow_steps=0)
        model._ensure_context_and_flows_initialized(X[:BATCH_SIZE])

        prior_state = model.prior_flow.state_dict()
        posterior_state = model.posterior_flow.state_dict()
        assert _state_equal(_state_clone(model.prior_flow), posterior_state)
        assert _state_equal(_state_clone(model.posterior_flow), prior_state)

    def test_train_step_keeps_prior_flow_fixed_and_updates_posterior_flow(
        self, regression_data
    ):
        X, y = regression_data
        model = _make_model(nf_prior_fit_steps=2, nf_posterior_flow_steps=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model._ensure_context_and_flows_initialized(X[:BATCH_SIZE])
        prior_before = _state_clone(model.prior_flow)
        posterior_before = _state_clone(model.posterior_flow)

        loss = model._train_step(optimizer, X[:BATCH_SIZE], y[:BATCH_SIZE])

        assert torch.isfinite(loss)
        assert _state_equal(prior_before, model.prior_flow.state_dict())
        assert _state_changed(posterior_before, model.posterior_flow)

    def test_nelbo_is_finite_scalar_requiring_grad(self, regression_data):
        X, y = regression_data
        model = _make_model(nf_prior_fit_steps=1, nf_posterior_flow_steps=1)
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])

        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_conditional_flow_nelbo_is_finite(self, regression_data):
        X, y = regression_data
        model = _make_model(
            nf_conditional_flow=True,
            nf_flow_condition_hidden_dim=16,
            nf_flow_condition_embedding_dim=8,
            nf_prior_fit_steps=1,
            nf_posterior_flow_steps=1,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])

        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_context_is_fixed_and_diagnostics_are_recorded(self, regression_data):
        X, y = regression_data
        model = _make_model(nf_prior_fit_steps=1, nf_posterior_flow_steps=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model._train_step(optimizer, X[:BATCH_SIZE], y[:BATCH_SIZE])
        context = model.context_inputs.detach().clone()
        model._train_step(optimizer, X[BATCH_SIZE:2 * BATCH_SIZE], y[BATCH_SIZE:2 * BATCH_SIZE])

        assert torch.allclose(context, model.context_inputs)
        assert len(model.KLs) == 2
        assert len(model.data_terms) == 2
        assert len(model.betas) == 2
        assert len(model.nf_kl_raws) == 2
        assert len(model.posterior_flow_nlls) == 2
        assert len(model.prior_flow_train_nlls) == 1
        assert model.prior_flow_update_counts == [1]
        assert len(model.posterior_flow_nlls_before) == 2
        assert len(model.posterior_flow_nlls_after) == 2
        assert len(model.posterior_flow_train_nlls) == 2
        assert model.posterior_flow_update_counts == [1, 1]
        assert model.posterior_flow_fit_sample_counts == [8, 8]

    def test_posterior_flow_fit_samples_are_current_only(self, regression_data):
        X, y = regression_data
        model = _make_model(
            nf_prior_fit_steps=1,
            nf_posterior_flow_steps=1,
            nf_flow_batch_size=5,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model._train_step(optimizer, X[:BATCH_SIZE], y[:BATCH_SIZE])
        model._train_step(optimizer, X[BATCH_SIZE:2 * BATCH_SIZE], y[BATCH_SIZE:2 * BATCH_SIZE])

        assert model.posterior_flow_fit_sample_counts == [5, 5]
        assert not hasattr(model, "_posterior_flow_replay")

    def test_prior_flow_early_stopping_records_actual_steps(self, regression_data):
        X, _ = regression_data
        model = _make_model(
            nf_prior_fit_steps=10,
            nf_prior_fit_rtol=1e9,
            nf_prior_fit_patience=1,
            nf_prior_fit_eval_every=1,
            nf_prior_fit_val_samples=16,
        )

        model._ensure_context_and_flows_initialized(X[:BATCH_SIZE])

        assert model.prior_flow_update_counts == [1]
        assert model.prior_flow_converged_flags == [1.0]
        assert len(model.prior_flow_val_nlls) >= 2
        assert len(model.prior_flow_relative_improvements) == 1
        assert len(model.prior_flow_best_val_nlls) == 1
        assert len(model.prior_flow_final_val_nlls) == 1
        assert torch.isfinite(model.prior_flow_final_val_nlls[-1])

    def test_posterior_flow_early_stopping_splits_current_samples(self, regression_data):
        X, y = regression_data
        model = _make_model(
            nf_prior_fit_steps=1,
            nf_posterior_flow_steps=10,
            nf_posterior_flow_rtol=1e9,
            nf_posterior_flow_patience=1,
            nf_posterior_flow_min_steps=1,
            nf_posterior_flow_eval_every=1,
            nf_posterior_flow_val_fraction=0.25,
            nf_flow_batch_size=8,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model._train_step(optimizer, X[:BATCH_SIZE], y[:BATCH_SIZE])

        assert model.posterior_flow_update_counts == [1]
        assert model.posterior_flow_converged_flags == [1.0]
        assert model.posterior_flow_fit_sample_counts == [6]
        assert model.posterior_flow_val_sample_counts == [2]
        assert len(model.posterior_flow_val_nlls) >= 2
        assert len(model.posterior_flow_relative_improvements) == 1

    def test_unregularized_context_optimization_records_diagnostics(self, regression_data):
        X, _ = regression_data
        model = _make_model(
            nf_prior_fit_steps=1,
            optimize_context=True,
            context_optimization_steps=2,
            context_optimization_lr=5e-2,
        )
        model._ensure_context_and_flows_initialized(X[:BATCH_SIZE])
        context_before = model.context_inputs.detach().clone()

        with torch.no_grad():
            for param in model.posterior_flow.parameters():
                param.add_(0.05 * torch.randn_like(param))
                break

        model._maybe_optimize_context()

        assert len(model.context_optimization_kls_before) == 1
        assert len(model.context_optimization_kls_after) == 1
        assert model.context_optimization_update_counts == [2]
        assert len(model.context_input_norms) == 1
        assert torch.isfinite(model.context_optimization_kls_after[-1])
        assert not torch.allclose(context_before, model.context_inputs)

    def test_shifted_samples_have_larger_estimated_kl(self, regression_data):
        X, _ = regression_data
        model = _make_model(
            num_prior_samples=64,
            nf_prior_fit_steps=20,
            nf_posterior_flow_steps=0,
            nf_flow_hidden_dim=32,
            nf_flow_batch_size=32,
            nf_flow_lr=5e-3,
        )
        model._ensure_context_and_flows_initialized(X[:BATCH_SIZE])

        with torch.no_grad():
            prior = model._sample_prior(model.context_inputs, 128)
            matching = prior.reshape(128, -1)
            shifted = matching + 2.0

        model.posterior_flow.load_state_dict(model.prior_flow.state_dict())
        model.nf_posterior_flow_steps = 20
        model._posterior_flow_optimizer = torch.optim.Adam(
            model.posterior_flow.parameters(), lr=model.nf_flow_lr
        )
        model._fine_tune_posterior_flow(matching)
        matching_kl = model._context_kl(matching)

        model.posterior_flow.load_state_dict(model.prior_flow.state_dict())
        model._posterior_flow_optimizer = torch.optim.Adam(
            model.posterior_flow.parameters(), lr=model.nf_flow_lr
        )
        model._fine_tune_posterior_flow(shifted)
        shifted_kl = model._context_kl(shifted)

        assert torch.isfinite(matching_kl)
        assert torch.isfinite(shifted_kl)
        assert shifted_kl > matching_kl
