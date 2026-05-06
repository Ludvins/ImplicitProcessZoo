"""Tests for MFVI (Mean-Field Variational Inference)."""

import pytest
import torch

from src.mfvi import MFVI
from tests.conftest import (
    DEVICE, DTYPE, SEED, INPUT_DIM, OUTPUT_DIM, NUM_SAMPLES, NUM_DATA, BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMFVIConstruction:

    def test_regression(self, bnn_variational):
        model = MFVI(bnn_variational, OUTPUT_DIM, "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        assert model.likelihood_type == "regression"
        assert hasattr(model, "log_variance")

    def test_binary(self, bnn_variational):
        model = MFVI(bnn_variational, OUTPUT_DIM, "binary", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        assert model.likelihood_type == "binary"

    def test_multiclass(self, bnn_variational_multiclass):
        model = MFVI(bnn_variational_multiclass, 3, "multiclass", NUM_DATA,
                       num_classes=3, device=DEVICE, dtype=DTYPE)
        assert model.likelihood_type == "multiclass"

    def test_invalid_likelihood(self, bnn_variational):
        with pytest.raises(ValueError):
            MFVI(bnn_variational, OUTPUT_DIM, "poisson", NUM_DATA,
                  device=DEVICE, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

class TestMFVIShapes:

    @pytest.fixture
    def model(self, bnn_variational):
        return MFVI(bnn_variational, OUTPUT_DIM, "regression", NUM_DATA,
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

class TestMFVILoss:

    @pytest.fixture
    def model(self, bnn_variational):
        return MFVI(bnn_variational, OUTPUT_DIM, "regression", NUM_DATA,
                      device=DEVICE, dtype=DTYPE)

    def test_nelbo_is_scalar(self, model, regression_data):
        X, y = regression_data
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_kl_is_scalar(self, model):
        kl = model.KL()
        assert kl.dim() == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestMFVITraining:

    def test_fit_epochs(self, bnn_variational, regression_loader):
        model = MFVI(bnn_variational, OUTPUT_DIM, "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        losses = model.fit(regression_loader, epochs=2, return_loss=True)
        assert len(losses) > 0

    def test_fit_iterations(self, bnn_variational, regression_loader):
        model = MFVI(bnn_variational, OUTPUT_DIM, "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        losses = model.fit(regression_loader, iterations=5, return_loss=True)
        assert len(losses) == 5

    def test_kls_tracked(self, bnn_variational, regression_loader):
        model = MFVI(bnn_variational, OUTPUT_DIM, "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        model.fit(regression_loader, iterations=3)
        assert len(model.KLs) == 3


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestMFVIPredict:

    def test_predict(self, bnn_variational, regression_data):
        model = MFVI(bnn_variational, OUTPUT_DIM, "regression", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        model.eval()
        X, _ = regression_data
        S = 4
        Y = model.predict(X, S)
        assert Y.shape == (S, X.shape[0], OUTPUT_DIM)


# ---------------------------------------------------------------------------
# BB-alpha & likelihoods
# ---------------------------------------------------------------------------

class TestMFVILikelihoods:

    def test_bb_alpha(self, bnn_variational, regression_data):
        X, y = regression_data
        model = MFVI(bnn_variational, OUTPUT_DIM, "regression", NUM_DATA,
                       bb_alpha=0.5, device=DEVICE, dtype=DTYPE)
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0

    def test_binary_nelbo(self, bnn_variational, binary_data):
        X, y = binary_data
        model = MFVI(bnn_variational, OUTPUT_DIM, "binary", NUM_DATA,
                       device=DEVICE, dtype=DTYPE)
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_multiclass_nelbo(self, bnn_variational_multiclass, multiclass_data):
        X, y = multiclass_data
        model = MFVI(bnn_variational_multiclass, 3, "multiclass", NUM_DATA,
                       num_classes=3, device=DEVICE, dtype=DTYPE)
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
