"""Tests for NF-VI (Normalizing-Flow Variational Inference)."""

import pytest
import torch

from src.nf_vi import NFVI
from src.utils.flat_mlp import FlatMLP, collect_param_spec
from tests.conftest import (
    DEVICE, DTYPE, INPUT_DIM, OUTPUT_DIM, HIDDEN, NUM_SAMPLES, NUM_DATA, BATCH_SIZE,
)


def _total_params(input_dim, output_dim, structure):
    """Total trainable parameters of a ``FlatMLP(input_dim, output_dim, structure)``."""
    template = FlatMLP(input_dim, output_dim, structure, torch.nn.Tanh(),
                      dtype=DTYPE, device=DEVICE)
    _, _, total = collect_param_spec(template)
    return total


def _make_model(nfvi_flow_factory, likelihood="regression", num_classes=None):
    net_out = num_classes if likelihood == "multiclass" else OUTPUT_DIM
    total = _total_params(INPUT_DIM, net_out, HIDDEN)
    flow = nfvi_flow_factory(total)
    kwargs = dict(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM if likelihood != "multiclass" else num_classes,
        structure=HIDDEN,
        activation=torch.nn.Tanh(),
        flow=flow,
        likelihood=likelihood,
        num_data=NUM_DATA,
        num_samples=NUM_SAMPLES,
        device=DEVICE,
        dtype=DTYPE,
    )
    if likelihood == "multiclass":
        kwargs["num_classes"] = num_classes
    return NFVI(**kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestNFVIConstruction:

    def test_regression(self, nfvi_flow_factory):
        model = _make_model(nfvi_flow_factory, "regression")
        assert model.likelihood_type == "regression"
        assert hasattr(model, "log_variance")

    def test_binary(self, nfvi_flow_factory):
        model = _make_model(nfvi_flow_factory, "binary")
        assert model.likelihood_type == "binary"

    def test_multiclass(self, nfvi_flow_factory):
        model = _make_model(nfvi_flow_factory, "multiclass", num_classes=3)
        assert model.likelihood_type == "multiclass"
        assert model.num_classes == 3

    def test_invalid_likelihood(self, nfvi_flow_factory):
        total = _total_params(INPUT_DIM, OUTPUT_DIM, HIDDEN)
        flow = nfvi_flow_factory(total)
        with pytest.raises(ValueError):
            NFVI(
                input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                structure=HIDDEN, activation=torch.nn.Tanh(),
                flow=flow, likelihood="poisson", num_data=NUM_DATA,
                device=DEVICE, dtype=DTYPE,
            )

    def test_multiclass_requires_num_classes(self, nfvi_flow_factory):
        total = _total_params(INPUT_DIM, 3, HIDDEN)
        flow = nfvi_flow_factory(total)
        with pytest.raises(ValueError):
            NFVI(
                input_dim=INPUT_DIM, output_dim=3,
                structure=HIDDEN, activation=torch.nn.Tanh(),
                flow=flow, likelihood="multiclass", num_data=NUM_DATA,
                device=DEVICE, dtype=DTYPE,
            )

    def test_flow_dim_mismatch_raises(self, nfvi_flow_factory):
        total = _total_params(INPUT_DIM, OUTPUT_DIM, HIDDEN)
        wrong_flow = nfvi_flow_factory(total + 4)  # deliberately wrong size
        with pytest.raises(ValueError):
            NFVI(
                input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                structure=HIDDEN, activation=torch.nn.Tanh(),
                flow=wrong_flow, likelihood="regression", num_data=NUM_DATA,
                device=DEVICE, dtype=DTYPE,
            )


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

class TestNFVIShapes:

    @pytest.fixture
    def model(self, nfvi_flow_factory):
        return _make_model(nfvi_flow_factory, "regression")

    def test_predict_f_samples(self, model, regression_data):
        X, _ = regression_data
        S = 6
        F = model.predict_f_samples(X, S)
        assert F.shape == (S, X.shape[0], OUTPUT_DIM)

    def test_predict_y_samples(self, model, regression_data):
        X, _ = regression_data
        S = 6
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

class TestNFVILoss:

    @pytest.fixture
    def model(self, nfvi_flow_factory):
        return _make_model(nfvi_flow_factory, "regression")

    def test_nelbo_is_scalar(self, model, regression_data):
        X, y = regression_data
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_kl_is_scalar(self, model, regression_data):
        X, y = regression_data
        model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        kl = model.KL()
        assert kl.dim() == 0

    def test_kl_requires_forward_first(self, nfvi_flow_factory):
        model = _make_model(nfvi_flow_factory, "regression")
        with pytest.raises(RuntimeError):
            model.KL()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestNFVITraining:

    def test_fit_epochs(self, nfvi_flow_factory, regression_loader):
        model = _make_model(nfvi_flow_factory, "regression")
        losses = model.fit(regression_loader, epochs=2, return_loss=True)
        assert len(losses) > 0

    def test_fit_iterations(self, nfvi_flow_factory, regression_loader):
        model = _make_model(nfvi_flow_factory, "regression")
        losses = model.fit(regression_loader, iterations=5, return_loss=True)
        assert len(losses) == 5

    def test_kls_tracked(self, nfvi_flow_factory, regression_loader):
        model = _make_model(nfvi_flow_factory, "regression")
        model.fit(regression_loader, iterations=3)
        assert len(model.KLs) == 3
        assert len(model.bb_alphas) == 3

    def test_cosine_annealing(self, nfvi_flow_factory, regression_loader):
        model = _make_model(nfvi_flow_factory, "regression")
        losses = model.fit(
            regression_loader, epochs=2, return_loss=True,
            cosine_annealing=True,
        )
        assert len(losses) > 0


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestNFVIPredict:

    def test_predict(self, nfvi_flow_factory, regression_data):
        model = _make_model(nfvi_flow_factory, "regression")
        X, _ = regression_data
        S = 4
        Y = model.predict(X, S)
        assert Y.shape == (S, X.shape[0], OUTPUT_DIM)


# ---------------------------------------------------------------------------
# BB-alpha & likelihoods
# ---------------------------------------------------------------------------

class TestNFVILikelihoods:

    def test_bb_alpha(self, nfvi_flow_factory, regression_data):
        X, y = regression_data
        total = _total_params(INPUT_DIM, OUTPUT_DIM, HIDDEN)
        flow = nfvi_flow_factory(total)
        model = NFVI(
            input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, structure=HIDDEN,
            activation=torch.nn.Tanh(), flow=flow, likelihood="regression",
            num_data=NUM_DATA, num_samples=NUM_SAMPLES, bb_alpha=0.5,
            device=DEVICE, dtype=DTYPE,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0

    def test_binary_nelbo(self, nfvi_flow_factory, binary_data):
        X, y = binary_data
        model = _make_model(nfvi_flow_factory, "binary")
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad
