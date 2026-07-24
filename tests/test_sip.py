"""Tests for SIP (Sparse Implicit Process)."""

import numpy as np
import pytest
import torch

from implicit_process_zoo.sip import SIP
from implicit_process_zoo.utils import batched_predict_samples
from tests.conftest import (
    BATCH_SIZE,
    DEVICE,
    DTYPE,
    INPUT_DIM,
    NUM_DATA,
    OUTPUT_DIM,
    SEED,
)

NUM_INDUCING = 5


def _make_inducing(M=NUM_INDUCING):
    """Create inducing inputs from a small random grid."""
    rng = np.random.RandomState(SEED)
    Z = rng.randn(M, INPUT_DIM).astype(np.float64)
    return torch.tensor(Z, dtype=DTYPE, device=DEVICE)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSIPConstruction:
    def test_regression(self, bnn):
        Z = _make_inducing()
        model = SIP(
            bnn, Z, OUTPUT_DIM, "regression", NUM_DATA, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        assert model.likelihood_type == "regression"
        assert hasattr(model, "log_variance")

    def test_binary(self, bnn):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "binary", NUM_DATA, device=DEVICE, dtype=DTYPE, seed=SEED)
        assert model.likelihood_type == "binary"

    def test_multiclass(self, bnn_multiclass):
        Z = _make_inducing()
        model = SIP(
            bnn_multiclass,
            Z,
            3,
            "multiclass",
            NUM_DATA,
            num_classes=3,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        assert model.likelihood_type == "multiclass"

    def test_invalid_likelihood(self, bnn):
        Z = _make_inducing()
        with pytest.raises(ValueError):
            SIP(bnn, Z, OUTPUT_DIM, "poisson", NUM_DATA, device=DEVICE, dtype=DTYPE)

    def test_implicit_posterior_and_critic_shapes(self, bnn):
        Z = _make_inducing()
        model = SIP(
            bnn, Z, OUTPUT_DIM, "regression", NUM_DATA, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        assert model.posterior_noise_dim == 100
        assert model.posterior_noise_mean.shape == (1, 100)
        assert model.posterior_noise_log_var.shape == (1, 100)
        u = model._sample_u(4)
        assert u.shape == (4, NUM_INDUCING, OUTPUT_DIM)
        logits = model.critic(model._flat_u(u))
        assert logits.shape == (4, 1)

    def test_vi_parameters_exclude_critic(self, bnn):
        Z = _make_inducing()
        model = SIP(
            bnn, Z, OUTPUT_DIM, "regression", NUM_DATA, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        critic_ids = {id(param) for param in model.critic.parameters()}
        vi_ids = {id(param) for param in model.vi_parameters()}
        assert critic_ids.isdisjoint(vi_ids)

    def test_inducing_fixed_by_default(self, bnn):
        Z = _make_inducing()
        model = SIP(
            bnn, Z, OUTPUT_DIM, "regression", NUM_DATA, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        assert not isinstance(model.Z, torch.nn.Parameter)

    def test_inducing_learnable(self, bnn):
        Z = _make_inducing()
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            learn_inducing=True,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        assert isinstance(model.Z, torch.nn.Parameter)
        assert model.Z.requires_grad


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


class TestSIPShapes:
    @pytest.fixture
    def model(self, bnn):
        Z = _make_inducing()
        return SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

    def test_predict_f(self, model, regression_data):
        X, _ = regression_data
        mean, var = model.predict_f(X)
        N = X.shape[0]
        assert mean.shape == (N, OUTPUT_DIM)
        assert var.shape == (N, OUTPUT_DIM)
        assert (var >= 0).all()

    def test_predict_f_samples(self, model, regression_data):
        X, _ = regression_data
        S = 7
        F = model.predict_f_samples(X, S)
        assert F.shape == (S, X.shape[0], OUTPUT_DIM)

    def test_predict_y_samples(self, model, regression_data):
        X, _ = regression_data
        S = 7
        Y = model.predict_y_samples(X, S)
        assert Y.shape == (S, X.shape[0], OUTPUT_DIM)

    def test_forward(self, model, regression_data):
        X, _ = regression_data
        mean, std = model(X)
        N = X.shape[0]
        # forward returns (1, N, D) due to unsqueeze in predict_mean_and_var
        assert mean.shape[-2:] == (N, OUTPUT_DIM)
        assert std.shape[-2:] == (N, OUTPUT_DIM)

    def test_forward_prior(self, model, regression_data):
        X, _ = regression_data
        S = 3  # must be <= generative_function.num_samples
        samples = model.forward_prior(X, S)
        assert samples.shape == (S, X.shape[0], OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class TestSIPLoss:
    @pytest.fixture
    def model(self, bnn):
        Z = _make_inducing()
        return SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

    def test_nelbo_is_scalar(self, model, regression_data):
        X, y = regression_data
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestSIPTraining:
    def test_direct_train_step_updates_critic_and_schedule(self, bnn, regression_data):
        Z = _make_inducing()
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        optimizer = torch.optim.Adam(model.vi_parameters(), lr=1.0e-3)
        X, y = regression_data

        model._train_step(optimizer, X[:BATCH_SIZE], y[:BATCH_SIZE])

        assert model._step == 1
        assert len(model.critic_losses) == 1

    def test_fit_epochs(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = model.fit(regression_loader, epochs=2, return_loss=True)
        assert len(losses) > 0

    def test_fit_iterations(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = model.fit(regression_loader, iterations=5, return_loss=True)
        assert len(losses) == 5

    def test_fit_cosine_annealing(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = model.fit(regression_loader, epochs=2, return_loss=True, cosine_annealing=True)
        assert len(losses) > 0

    def test_kls_tracked(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model.fit(regression_loader, iterations=3)
        assert len(model.KLs) == 3
        assert len(model.betas) == 3
        assert len(model.critic_losses) == 3
        assert len(model.critic_accuracies) == 3
        assert len(model.critic_saturation_fractions) == 3

    def test_beta_warmup(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            beta=1.0,
            beta_warmup_steps=4,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model.fit(regression_loader, iterations=2)
        assert model.betas == [0.25, 0.5]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestSIPPrediction:
    def test_predict(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model.eval()
        samples = batched_predict_samples(
            model, regression_loader, num_samples=5, kind="f", seed=SEED
        )
        assert samples.shape == (5, NUM_DATA, OUTPUT_DIM)

    def test_predict_no_grad(self, bnn, regression_data):
        Z = _make_inducing()
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model.eval()
        X, _ = regression_data
        with torch.no_grad():
            mean, std = model(X)
        assert not mean.requires_grad


# ---------------------------------------------------------------------------
# Detach covariances
# ---------------------------------------------------------------------------


class TestSIPDetachCov:
    def test_detach_covariances(self, bnn, regression_data):
        Z = _make_inducing()
        X, y = regression_data
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            num_prior_samples=10,
            detach_covariances=True,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# BB-alpha & likelihoods
# ---------------------------------------------------------------------------


class TestSIPLikelihoods:
    def test_bb_alpha(self, bnn, regression_data):
        Z = _make_inducing()
        X, y = regression_data
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            bb_alpha=0.5,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0

    def test_binary_nelbo(self, bnn, binary_data):
        Z = _make_inducing()
        X, y = binary_data
        model = SIP(
            bnn,
            Z,
            OUTPUT_DIM,
            "binary",
            NUM_DATA,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_multiclass_nelbo(self, bnn_multiclass, multiclass_data):
        Z = _make_inducing()
        X, y = multiclass_data
        model = SIP(
            bnn_multiclass,
            Z,
            3,
            "multiclass",
            NUM_DATA,
            num_classes=3,
            num_prior_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
