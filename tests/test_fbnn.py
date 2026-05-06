"""Tests for fBNN (Functional Bayesian Neural Network)."""

import pytest
import torch

from src.fbnn import FBNN
from tests.conftest import (
    DEVICE, DTYPE, SEED, INPUT_DIM, OUTPUT_DIM, NUM_SAMPLES, NUM_DATA, BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestFBNNConstruction:

    def test_regression(self, bnn_fbnn_posterior, bnn_fbnn_prior):
        model = FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                       "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        assert model.likelihood_type == "regression"
        assert hasattr(model, "log_variance")

    def test_binary(self, bnn_fbnn_posterior, bnn_fbnn_prior):
        model = FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                       "binary", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        assert model.likelihood_type == "binary"

    def test_invalid_likelihood(self, bnn_fbnn_posterior, bnn_fbnn_prior):
        with pytest.raises(ValueError):
            FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                  "poisson", NUM_DATA, device=DEVICE, dtype=DTYPE)

    def test_prior_frozen(self, bnn_fbnn_posterior, bnn_fbnn_prior):
        model = FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                       "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        for p in model.prior_function.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

class TestFBNNShapes:

    @pytest.fixture
    def model(self, bnn_fbnn_posterior, bnn_fbnn_prior):
        return FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                      "regression", NUM_DATA,
                      device=DEVICE, dtype=DTYPE)

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
        Y = model(X)
        assert Y.shape[1] == X.shape[0]
        assert Y.shape[2] == OUTPUT_DIM


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestFBNNLoss:

    @pytest.fixture
    def model(self, bnn_fbnn_posterior, bnn_fbnn_prior, regression_data):
        X, _ = regression_data
        m = FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                   "regression", NUM_DATA,
                   device=DEVICE, dtype=DTYPE)
        # Store training data so measurement set can be sampled
        m._train_inputs = X
        return m

    def test_nelbo_is_scalar(self, model, regression_data):
        X, y = regression_data
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestFBNNTraining:

    def test_fit_epochs(self, bnn_fbnn_posterior, bnn_fbnn_prior, regression_loader):
        model = FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                       "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        losses = model.fit(regression_loader, epochs=2, return_loss=True)
        assert len(losses) > 0

    def test_fit_iterations(self, bnn_fbnn_posterior, bnn_fbnn_prior, regression_loader):
        model = FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                       "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        losses = model.fit(regression_loader, iterations=5, return_loss=True)
        assert len(losses) == 5

    def test_kls_tracked(self, bnn_fbnn_posterior, bnn_fbnn_prior, regression_loader):
        model = FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                       "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        model.fit(regression_loader, iterations=3)
        assert len(model.KLs) == 3


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestFBNNPredict:

    def test_predict(self, bnn_fbnn_posterior, bnn_fbnn_prior, regression_data):
        model = FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                       "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        model.eval()
        X, _ = regression_data
        S = 4
        Y = model.predict(X, S)
        assert Y.shape == (S, X.shape[0], OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Likelihoods
# ---------------------------------------------------------------------------

class TestFBNNLikelihoods:

    def test_binary_nelbo(self, bnn_fbnn_posterior, bnn_fbnn_prior, binary_data):
        X, y = binary_data
        model = FBNN(bnn_fbnn_posterior, bnn_fbnn_prior, OUTPUT_DIM,
                       "binary", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        model._train_inputs = X
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad
