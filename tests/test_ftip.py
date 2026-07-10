"""Tests for FTIP (Flow-Transformed Implicit Process)."""

import pytest
import torch

from implicit_process_zoo.flows.flows import CouplingFlow
from implicit_process_zoo.ftip import FTIP
from implicit_process_zoo.vip import VIP
from tests.conftest import (
    BATCH_SIZE,
    DEVICE,
    DTYPE,
    MNIST_INPUT_DIM,
    NUM_CLASSES,
    NUM_DATA,
    NUM_SAMPLES,
    OUTPUT_DIM,
    SEED,
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFTIPConstruction:
    def test_regression(self, bnn, coupling_flow):
        model = FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "regression",
            NUM_DATA,
            NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        assert model.likelihood_type == "regression"
        assert hasattr(model, "log_variance")

    def test_binary(self, bnn, coupling_flow):
        model = FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "binary",
            NUM_DATA,
            NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        assert model.likelihood_type == "binary"

    def test_multiclass(self, bnn_multiclass):
        flow = CouplingFlow(2, NUM_SAMPLES, DEVICE, DTYPE, SEED)
        model = FTIP(
            bnn_multiclass,
            NUM_SAMPLES,
            3,
            flow,
            "multiclass",
            NUM_DATA,
            NUM_SAMPLES,
            num_classes=3,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        assert model.likelihood_type == "multiclass"

    def test_invalid_likelihood(self, bnn, coupling_flow):
        with pytest.raises(ValueError):
            FTIP(
                bnn,
                NUM_SAMPLES,
                OUTPUT_DIM,
                coupling_flow,
                "poisson",
                NUM_DATA,
                NUM_SAMPLES,
                device=DEVICE,
                dtype=DTYPE,
            )


# ---------------------------------------------------------------------------
# Forward / predict shapes
# ---------------------------------------------------------------------------


class TestFTIPShapes:
    @pytest.fixture
    def model(self, bnn, coupling_flow):
        return FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "regression",
            NUM_DATA,
            NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

    def test_predict_f(self, model, regression_data):
        X, _ = regression_data
        S = 3
        posterior_samples, f = model.predict_f(X, S)
        N = X.shape[0]
        # posterior_samples are [S, N, D]; f is prior samples [gen_num_samples, N, D]
        assert posterior_samples.shape == (S, N, OUTPUT_DIM)

    def test_forward(self, model, regression_data):
        X, _ = regression_data
        out_samples, out_std = model(X)
        N = X.shape[0]
        # forward returns [num_samples, N, D] denormalized samples and std
        assert out_samples.shape == (NUM_SAMPLES, N, OUTPUT_DIM)

    def test_predict_y(self, model, regression_data):
        X, _ = regression_data
        S = 3
        y_samples = model.predict_y(X, S)
        assert y_samples.shape == (S, X.shape[0], OUTPUT_DIM)

    def test_get_samples(self, model):
        S = 4
        prior_coeffs, posterior_coeffs = model.get_samples(S)
        assert prior_coeffs.shape == (S, NUM_SAMPLES, OUTPUT_DIM)
        assert posterior_coeffs.shape == (S, NUM_SAMPLES, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class TestFTIPLoss:
    @pytest.fixture
    def model(self, bnn, coupling_flow):
        return FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "regression",
            NUM_DATA,
            NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

    def test_nelbo_is_scalar(self, model, regression_data):
        X, y = regression_data
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_kl_is_scalar(self, model, regression_data):
        X, _ = regression_data
        # KL requires predict_f to be called first (caches _eps_flat)
        model.predict_f(X[:BATCH_SIZE], NUM_SAMPLES)
        kl = model.KL()
        assert kl.dim() == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestFTIPTraining:
    def test_fit_epochs(self, bnn, coupling_flow, regression_loader):
        model = FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "regression",
            NUM_DATA,
            NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = model.fit(regression_loader, epochs=2, return_loss=True)
        assert len(losses) > 0

    def test_fit_iterations(self, bnn, coupling_flow, regression_loader):
        model = FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "regression",
            NUM_DATA,
            NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = model.fit(regression_loader, iterations=5, return_loss=True)
        assert len(losses) == 5

    def test_kls_tracked(self, bnn, coupling_flow, regression_loader):
        model = FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "regression",
            NUM_DATA,
            NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model.fit(regression_loader, iterations=3)
        assert len(model.KLs) == 3

    def test_prior_regularizer_kl_mode(self, bnn, coupling_flow, regression_data):
        X, y = regression_data
        model = FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "regression",
            NUM_DATA,
            NUM_SAMPLES,
            use_prior_regularizer=True,
            regularizer_mode="KL",
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        reg = model.prior_regularizer(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert reg.dim() == 0

    def test_prior_regularizer_evidence_mode(self, bnn, coupling_flow, regression_data):
        X, y = regression_data
        model = FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "regression",
            NUM_DATA,
            NUM_SAMPLES,
            use_prior_regularizer=True,
            regularizer_mode="evidence",
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        # nelbo caches _cached_m and _cached_phi needed by evidence mode
        model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        reg = model.prior_regularizer(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert reg.dim() == 0


# ---------------------------------------------------------------------------
# BB-alpha & likelihoods
# ---------------------------------------------------------------------------


class TestFTIPLikelihoods:
    def test_bb_alpha(self, bnn, coupling_flow, regression_data):
        X, y = regression_data
        model = FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "regression",
            NUM_DATA,
            NUM_SAMPLES,
            bb_alpha=0.5,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0

    def test_binary_nelbo(self, bnn, coupling_flow, binary_data):
        X, y = binary_data
        model = FTIP(
            bnn,
            NUM_SAMPLES,
            OUTPUT_DIM,
            coupling_flow,
            "binary",
            NUM_DATA,
            NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# BayesianCNN (LeNet) warm-start
# ---------------------------------------------------------------------------


class TestFTIPCNNWarmStart:
    """Verify VIP -> FTIP warm-start works with BayesianCNN generative function."""

    def test_warm_start_from_vip(self, cnn_multiclass, mnist_multiclass_data):
        """Train VIP briefly, warm-start FTIP, verify forward + nelbo work."""
        X, y = mnist_multiclass_data

        # Build and train VIP for 1 step
        vip = VIP(
            cnn_multiclass,
            NUM_SAMPLES,
            NUM_CLASSES,
            "multiclass",
            NUM_DATA,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        opt = torch.optim.Adam(vip.parameters(), lr=1e-3)
        vip._train_step(opt, X[:BATCH_SIZE], y[:BATCH_SIZE])

        # Build FTIP with identical CNN
        from implicit_process_zoo.priors.generative_functions import BayesianCNN, SimplerBayesLinear

        cnn_ftip = BayesianCNN(
            num_samples=NUM_SAMPLES,
            input_dim=MNIST_INPUT_DIM,
            output_dim=NUM_CLASSES,
            layer_model=SimplerBayesLinear,
            head_dims=[32],
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        flow = CouplingFlow(2, NUM_SAMPLES * NUM_CLASSES, DEVICE, DTYPE, SEED)
        ftip = FTIP(
            cnn_ftip,
            NUM_SAMPLES,
            NUM_CLASSES,
            flow,
            "multiclass",
            NUM_DATA,
            NUM_SAMPLES,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

        # This is the line that previously raised AttributeError
        ftip.warm_start_from_vip(vip, learnable_affine=True)

        # Verify forward pass works after warm-start
        out, std = ftip(X[:BATCH_SIZE])
        assert out.shape == (NUM_SAMPLES, BATCH_SIZE, NUM_CLASSES)

        # Verify nelbo computes without error
        loss = ftip.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_warm_start_copies_weights(self, cnn_multiclass, mnist_multiclass_data):
        """Verify generative function weights are copied from VIP to FTIP."""
        X, y = mnist_multiclass_data

        vip = VIP(
            cnn_multiclass,
            NUM_SAMPLES,
            NUM_CLASSES,
            "multiclass",
            NUM_DATA,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        opt = torch.optim.Adam(vip.parameters(), lr=1e-3)
        vip._train_step(opt, X[:BATCH_SIZE], y[:BATCH_SIZE])

        from implicit_process_zoo.priors.generative_functions import BayesianCNN, SimplerBayesLinear

        cnn_ftip = BayesianCNN(
            num_samples=NUM_SAMPLES,
            input_dim=MNIST_INPUT_DIM,
            output_dim=NUM_CLASSES,
            layer_model=SimplerBayesLinear,
            head_dims=[32],
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        flow = CouplingFlow(2, NUM_SAMPLES * NUM_CLASSES, DEVICE, DTYPE, SEED)
        ftip = FTIP(
            cnn_ftip,
            NUM_SAMPLES,
            NUM_CLASSES,
            flow,
            "multiclass",
            NUM_DATA,
            NUM_SAMPLES,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

        ftip.warm_start_from_vip(vip, learnable_affine=False)

        # Conv and head weights should match
        for p_ftip, p_vip in zip(
            ftip.generative_function.parameters(), vip.generative_function.parameters()
        ):
            assert torch.allclose(p_ftip, p_vip)

    def test_ftip_cnn_trainable_after_warm_start(self, cnn_multiclass, mnist_multiclass_data):
        """Verify FTIP with CNN can take a training step after warm-start."""
        X, y = mnist_multiclass_data

        vip = VIP(
            cnn_multiclass,
            NUM_SAMPLES,
            NUM_CLASSES,
            "multiclass",
            NUM_DATA,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

        from implicit_process_zoo.priors.generative_functions import BayesianCNN, SimplerBayesLinear

        cnn_ftip = BayesianCNN(
            num_samples=NUM_SAMPLES,
            input_dim=MNIST_INPUT_DIM,
            output_dim=NUM_CLASSES,
            layer_model=SimplerBayesLinear,
            head_dims=[32],
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        flow = CouplingFlow(2, NUM_SAMPLES * NUM_CLASSES, DEVICE, DTYPE, SEED)
        ftip = FTIP(
            cnn_ftip,
            NUM_SAMPLES,
            NUM_CLASSES,
            flow,
            "multiclass",
            NUM_DATA,
            NUM_SAMPLES,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

        ftip.warm_start_from_vip(vip, learnable_affine=True)

        # Take a training step — should not raise
        opt = torch.optim.Adam(ftip.parameters(), lr=1e-4)
        loss_before = ftip.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        loss_before.backward()
        opt.step()
        opt.zero_grad()

        loss_after = ftip.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        # Just verify it ran; loss may go up or down after 1 step
        assert loss_after.dim() == 0
