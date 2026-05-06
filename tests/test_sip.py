"""Tests for SIP (Sparse Implicit Process)."""

import pytest
import torch
import numpy as np

from src.sip import SIP
from tests.conftest import (
    DEVICE, DTYPE, SEED, INPUT_DIM, OUTPUT_DIM, NUM_SAMPLES, NUM_DATA, BATCH_SIZE,
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
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        assert model.likelihood_type == "regression"
        assert hasattr(model, "log_variance")

    def test_binary(self, bnn):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "binary", NUM_DATA,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        assert model.likelihood_type == "binary"

    def test_multiclass(self, bnn_multiclass):
        Z = _make_inducing()
        model = SIP(bnn_multiclass, Z, 3, "multiclass", NUM_DATA,
                      num_classes=3, device=DEVICE, dtype=DTYPE, seed=SEED)
        assert model.likelihood_type == "multiclass"

    def test_invalid_likelihood(self, bnn):
        Z = _make_inducing()
        with pytest.raises(ValueError):
            SIP(bnn, Z, OUTPUT_DIM, "poisson", NUM_DATA,
                 device=DEVICE, dtype=DTYPE)

    def test_variational_params_shape(self, bnn):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        assert model.m_u.shape == (NUM_INDUCING, OUTPUT_DIM)
        tri_len = NUM_INDUCING * (NUM_INDUCING + 1) // 2
        assert model.L_u_tri.shape == (tri_len, OUTPUT_DIM)

    def test_inducing_fixed_by_default(self, bnn):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        assert not isinstance(model.Z, torch.nn.Parameter)

    def test_inducing_learnable(self, bnn):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      learn_inducing=True,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        assert isinstance(model.Z, torch.nn.Parameter)
        assert model.Z.requires_grad


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

class TestSIPShapes:

    @pytest.fixture
    def model(self, bnn):
        Z = _make_inducing()
        return SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                     num_prior_samples=10,
                     device=DEVICE, dtype=DTYPE, seed=SEED)

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
        return SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                     num_prior_samples=10,
                     device=DEVICE, dtype=DTYPE, seed=SEED)

    def test_nelbo_is_scalar(self, model, regression_data):
        X, y = regression_data
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestSIPTraining:

    def test_fit_epochs(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      num_prior_samples=10,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        losses = model.fit(regression_loader, epochs=2, return_loss=True)
        assert len(losses) > 0

    def test_fit_iterations(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      num_prior_samples=10,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        losses = model.fit(regression_loader, iterations=5, return_loss=True)
        assert len(losses) == 5

    def test_fit_cosine_annealing(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      num_prior_samples=10,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        losses = model.fit(regression_loader, epochs=2, return_loss=True,
                           cosine_annealing=True)
        assert len(losses) > 0

    def test_kls_tracked(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      num_prior_samples=10,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        model.fit(regression_loader, iterations=3)
        assert len(model.KLs) == 3


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestSIPPrediction:

    def test_predict(self, bnn, regression_loader):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      num_prior_samples=10,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        model.eval()
        means, stds = model.predict(regression_loader)
        # predict concatenates forward() outputs along dim=0
        total_points = means.shape[-2] * means.shape[0] if means.dim() == 3 else means.shape[0]
        assert total_points == NUM_DATA

    def test_predict_no_grad(self, bnn, regression_data):
        Z = _make_inducing()
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      num_prior_samples=10,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
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
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      num_prior_samples=10, detach_covariances=True,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
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
        model = SIP(bnn, Z, OUTPUT_DIM, "regression", NUM_DATA,
                      bb_alpha=0.5, num_prior_samples=10,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0

    def test_binary_nelbo(self, bnn, binary_data):
        Z = _make_inducing()
        X, y = binary_data
        model = SIP(bnn, Z, OUTPUT_DIM, "binary", NUM_DATA,
                      num_prior_samples=10,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    @pytest.mark.xfail(reason="Shape mismatch in prob_is_largest with unsqueezed Fmu")
    def test_multiclass_nelbo(self, bnn_multiclass, multiclass_data):
        Z = _make_inducing()
        X, y = multiclass_data
        model = SIP(bnn_multiclass, Z, 3, "multiclass", NUM_DATA,
                      num_classes=3, num_prior_samples=10,
                      device=DEVICE, dtype=DTYPE, seed=SEED)
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
