"""Tests for TFSVI (Task-Free Stochastic Variational Inference)."""

import pytest
import torch

from src.tfsvi import TFSVI
from tests.conftest import (
    DEVICE, DTYPE, SEED, INPUT_DIM, OUTPUT_DIM, NUM_SAMPLES, NUM_DATA, BATCH_SIZE,
    HIDDEN,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTFSVIConstruction:

    def test_regression(self):
        model = TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                        "regression", NUM_DATA,
                        device=DEVICE, dtype=DTYPE)
        assert model.likelihood_type == "regression"
        assert hasattr(model, "log_variance")

    def test_binary(self):
        model = TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                        "binary", NUM_DATA,
                        device=DEVICE, dtype=DTYPE)
        assert model.likelihood_type == "binary"

    def test_multiclass(self):
        model = TFSVI(INPUT_DIM, 3, HIDDEN, torch.nn.Tanh(),
                        "multiclass", NUM_DATA, num_classes=3,
                        device=DEVICE, dtype=DTYPE)
        assert model.likelihood_type == "multiclass"

    def test_invalid_likelihood(self):
        with pytest.raises(ValueError):
            TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                   "poisson", NUM_DATA, device=DEVICE, dtype=DTYPE)

    def test_variational_params_exist(self):
        model = TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                        "regression", NUM_DATA,
                        device=DEVICE, dtype=DTYPE)
        assert hasattr(model, "mu")
        assert hasattr(model, "log_sigma")
        assert model.mu.shape == model.log_sigma.shape


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

class TestTFSVIShapes:

    @pytest.fixture
    def model(self):
        return TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
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
# Loss — TFSVI.nelbo requires _train_inputs to be set
# ---------------------------------------------------------------------------

class TestTFSVILoss:

    @pytest.fixture
    def model(self, regression_data):
        X, _ = regression_data
        m = TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                    "regression", NUM_DATA,
                    device=DEVICE, dtype=DTYPE)
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

class TestTFSVITraining:

    def test_fit_epochs(self, regression_loader):
        model = TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                        "regression", NUM_DATA,
                        device=DEVICE, dtype=DTYPE)
        losses = model.fit(regression_loader, epochs=2, return_loss=True)
        assert len(losses) > 0

    def test_fit_iterations(self, regression_loader):
        model = TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                        "regression", NUM_DATA,
                        device=DEVICE, dtype=DTYPE)
        losses = model.fit(regression_loader, iterations=5, return_loss=True)
        assert len(losses) == 5

    def test_kls_tracked(self, regression_loader):
        model = TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                        "regression", NUM_DATA,
                        device=DEVICE, dtype=DTYPE)
        model.fit(regression_loader, iterations=3)
        assert len(model.KLs) == 3


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestTFSVIPredict:

    def test_predict(self, regression_data):
        model = TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                        "regression", NUM_DATA,
                        device=DEVICE, dtype=DTYPE)
        model.eval()
        X, _ = regression_data
        S = 4
        Y = model.predict(X, S)
        assert Y.shape == (S, X.shape[0], OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Likelihoods — need _train_inputs for nelbo
# ---------------------------------------------------------------------------

class TestTFSVILikelihoods:

    def test_binary_nelbo(self, binary_data):
        X, y = binary_data
        model = TFSVI(INPUT_DIM, OUTPUT_DIM, HIDDEN, torch.nn.Tanh(),
                        "binary", NUM_DATA,
                        device=DEVICE, dtype=DTYPE)
        model._train_inputs = X
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0

    def test_multiclass_nelbo(self, multiclass_data):
        X, y = multiclass_data
        model = TFSVI(INPUT_DIM, 3, HIDDEN, torch.nn.Tanh(),
                        "multiclass", NUM_DATA, num_classes=3,
                        device=DEVICE, dtype=DTYPE)
        model._train_inputs = X
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
