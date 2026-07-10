"""Tests for VIP (Variational Implicit Process)."""

import pytest
import torch

from implicit_process_zoo.utils import batched_predict_samples
from implicit_process_zoo.vip import VIP
from tests.conftest import (
    BATCH_SIZE,
    DEVICE,
    DTYPE,
    NUM_DATA,
    NUM_SAMPLES,
    OUTPUT_DIM,
    SEED,
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestVIPConstruction:
    def test_regression(self, bnn):
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        assert model.likelihood_type == "regression"
        assert hasattr(model, "log_variance")

    def test_binary(self, bnn):
        model = VIP(
            bnn, NUM_SAMPLES, OUTPUT_DIM, "binary", NUM_DATA, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        assert model.likelihood_type == "binary"
        assert not hasattr(model, "log_variance")

    def test_multiclass(self, bnn_multiclass):
        model = VIP(
            bnn_multiclass,
            NUM_SAMPLES,
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
        with pytest.raises(ValueError):
            VIP(bnn, NUM_SAMPLES, OUTPUT_DIM, "poisson", NUM_DATA, device=DEVICE, dtype=DTYPE)

    def test_multiclass_requires_num_classes(self, bnn_multiclass):
        with pytest.raises(ValueError):
            VIP(bnn_multiclass, NUM_SAMPLES, 3, "multiclass", NUM_DATA, device=DEVICE, dtype=DTYPE)

    def test_variational_params_shape(self, bnn):
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        assert model.q_mu.shape == (NUM_SAMPLES, OUTPUT_DIM)
        tri_len = NUM_SAMPLES * (NUM_SAMPLES + 1) // 2
        assert model.q_sqrt_tri.shape == (tri_len, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Forward / predict shapes
# ---------------------------------------------------------------------------


class TestVIPShapes:
    @pytest.fixture
    def model(self, bnn):
        return VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
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

    def test_forward(self, model, regression_data):
        X, _ = regression_data
        out_mean, out_std = model(X)
        N = X.shape[0]
        # forward returns (1, N, D) due to unsqueeze in predict_mean_and_var
        assert out_mean.shape[-2:] == (N, OUTPUT_DIM)
        assert out_std.shape[-2:] == (N, OUTPUT_DIM)

    def test_forward_prior(self, model, regression_data):
        X, _ = regression_data
        S = 3  # must be <= NUM_SAMPLES for BNN with fix_random_noise=True
        samples = model.forward_prior(X, S)
        assert samples.shape == (S, X.shape[0], OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class TestVIPLoss:
    @pytest.fixture
    def model(self, bnn):
        return VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

    def test_nelbo_is_scalar(self, model, regression_data):
        X, y = regression_data
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_kl_is_scalar(self, model):
        kl = model.KL()
        assert kl.dim() == 0

    def test_prior_regularizer_kl_mode(self, bnn, regression_data):
        X, y = regression_data
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            use_prior_regularizer=True,
            regularizer_mode="KL",
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        reg = model.prior_regularizer(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert reg.dim() == 0

    def test_prior_regularizer_evidence_mode(self, bnn, regression_data):
        X, y = regression_data
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            use_prior_regularizer=True,
            regularizer_mode="evidence",
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        # nelbo caches _cached_m and _cached_phi required by evidence mode
        model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        reg = model.prior_regularizer(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert reg.dim() == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestVIPTraining:
    def test_fit_epochs(self, bnn, regression_loader):
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = model.fit(regression_loader, epochs=2, return_loss=True)
        assert len(losses) > 0

    def test_fit_iterations(self, bnn, regression_loader):
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = model.fit(regression_loader, iterations=5, return_loss=True)
        assert len(losses) == 5

    def test_fit_cosine_annealing(self, bnn, regression_loader):
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = model.fit(regression_loader, epochs=2, return_loss=True, cosine_annealing=True)
        assert len(losses) > 0

    def test_kls_tracked(self, bnn, regression_loader):
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model.fit(regression_loader, iterations=3)
        assert len(model.KLs) == 3


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestVIPPrediction:
    def test_predict(self, bnn, regression_loader):
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
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
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
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
# BB-alpha
# ---------------------------------------------------------------------------


class TestVIPBBAlpha:
    def test_bb_alpha_nonzero(self, bnn, regression_data):
        X, y = regression_data
        model = VIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            "regression",
            NUM_DATA,
            bb_alpha=0.5,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# Likelihoods
# ---------------------------------------------------------------------------


class TestVIPLikelihoods:
    def test_binary_nelbo(self, bnn, binary_data):
        X, y = binary_data
        model = VIP(
            bnn, NUM_SAMPLES, OUTPUT_DIM, "binary", NUM_DATA, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_multiclass_nelbo(self, bnn_multiclass, multiclass_data):
        X, y = multiclass_data
        model = VIP(
            bnn_multiclass,
            NUM_SAMPLES,
            3,
            "multiclass",
            NUM_DATA,
            num_classes=3,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad
