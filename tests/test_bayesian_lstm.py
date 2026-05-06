"""Tests for BayesianLSTM generative function and its integration with VIP/FTIP."""

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.priors.generative_functions import BayesianLSTM, SimplerBayesLinear
from src.flows.flows import CouplingFlow
from src.vip import VIP
from src.ftip import FTIP


DEVICE = torch.device("cpu")
DTYPE = torch.float64
SEED = 42

T_OBS = 8
FEATURE_DIM = 2
T_PRED = 12
INPUT_DIM = T_OBS * FEATURE_DIM   # 16
OUTPUT_DIM = T_PRED * 2            # 24
NUM_SAMPLES = 6  # must be even for CouplingFlow
NUM_DATA = 30
BATCH_SIZE = 10


@pytest.fixture
def blstm():
    return BayesianLSTM(
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        t_obs=T_OBS,
        feature_dim=FEATURE_DIM,
        lstm_hidden=32,
        lstm_layers=1,
        head_dims=[16],
        layer_model=SimplerBayesLinear,
        device=DEVICE,
        dtype=DTYPE,
        seed=SEED,
    )


@pytest.fixture
def traj_data():
    rng = np.random.RandomState(SEED)
    X = rng.randn(NUM_DATA, INPUT_DIM).astype(np.float64)
    y = rng.randn(NUM_DATA, OUTPUT_DIM).astype(np.float64)
    return (
        torch.tensor(X, dtype=DTYPE, device=DEVICE),
        torch.tensor(y, dtype=DTYPE, device=DEVICE),
    )


@pytest.fixture
def traj_loader(traj_data):
    X, y = traj_data
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# BayesianLSTM unit tests
# ---------------------------------------------------------------------------

class TestBayesianLSTM:
    def test_forward_shape(self, blstm, traj_data):
        X, _ = traj_data
        out = blstm(X)
        assert out.shape == (NUM_SAMPLES, NUM_DATA, OUTPUT_DIM)

    def test_kl_scalar(self, blstm):
        kl = blstm.KL()
        assert kl.shape == ()
        assert kl.item() >= 0

    def test_regularizer(self, blstm):
        assert blstm.regularizer().item() == blstm.KL().item()

    def test_deterministic_encoder(self, blstm, traj_data):
        """LSTM encoder should produce same features for same input."""
        X, _ = traj_data
        x_seq = X[:5].reshape(5, T_OBS, FEATURE_DIM)
        _, (h1, _) = blstm.encoder(x_seq)
        _, (h2, _) = blstm.encoder(x_seq)
        assert torch.allclose(h1, h2)


# ---------------------------------------------------------------------------
# VIP integration
# ---------------------------------------------------------------------------

class TestVIPIntegration:
    def test_construct_and_forward(self, blstm, traj_data):
        X, _ = traj_data
        vip = VIP(
            generative_function=blstm,
            num_regression_coeffs=NUM_SAMPLES,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        mean, std = vip(X)
        # VIP forward for regression returns (1, N, D) via unsqueeze(0)
        assert mean.shape == (1, NUM_DATA, OUTPUT_DIM)
        assert std.shape == (1, NUM_DATA, OUTPUT_DIM)

    def test_nelbo(self, blstm, traj_data):
        X, y = traj_data
        vip = VIP(
            generative_function=blstm,
            num_regression_coeffs=NUM_SAMPLES,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = vip.nelbo(X, y)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_fit(self, blstm, traj_loader):
        vip = VIP(
            generative_function=blstm,
            num_regression_coeffs=NUM_SAMPLES,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = vip.fit(traj_loader, epochs=2, return_loss=True)
        assert len(losses) > 0
        assert all(np.isfinite(l) for l in losses)


# ---------------------------------------------------------------------------
# FTIP integration + warm-start
# ---------------------------------------------------------------------------

class TestFTIPIntegration:
    def _make_ftip(self, blstm):
        flow = CouplingFlow(
            depth=2,
            input_dim=NUM_SAMPLES * OUTPUT_DIM,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        return FTIP(
            generative_function=blstm,
            num_regression_coeffs=NUM_SAMPLES,
            output_dim=OUTPUT_DIM,
            flow=flow,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

    def test_construct_and_forward(self, blstm, traj_data):
        X, _ = traj_data
        ftip = self._make_ftip(blstm)
        samples, std = ftip(X)
        assert samples.shape[1] == NUM_DATA
        assert samples.shape[2] == OUTPUT_DIM

    def test_nelbo(self, blstm, traj_data):
        X, y = traj_data
        ftip = self._make_ftip(blstm)
        loss = ftip.nelbo(X, y)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_warm_start_from_vip(self, traj_data):
        """FTIP warm-started from VIP should produce finite predictions."""
        blstm_vip = BayesianLSTM(
            num_samples=NUM_SAMPLES, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
            t_obs=T_OBS, feature_dim=FEATURE_DIM, lstm_hidden=32, lstm_layers=1,
            head_dims=[16], layer_model=SimplerBayesLinear,
            device=DEVICE, dtype=DTYPE, seed=SEED,
        )
        vip = VIP(
            generative_function=blstm_vip,
            num_regression_coeffs=NUM_SAMPLES,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            device=DEVICE, dtype=DTYPE, seed=SEED,
        )

        blstm_ftip = BayesianLSTM(
            num_samples=NUM_SAMPLES, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
            t_obs=T_OBS, feature_dim=FEATURE_DIM, lstm_hidden=32, lstm_layers=1,
            head_dims=[16], layer_model=SimplerBayesLinear,
            device=DEVICE, dtype=DTYPE, seed=SEED,
        )
        flow = CouplingFlow(depth=2, input_dim=NUM_SAMPLES * OUTPUT_DIM,
                            device=DEVICE, dtype=DTYPE, seed=SEED)
        ftip = FTIP(
            generative_function=blstm_ftip,
            num_regression_coeffs=NUM_SAMPLES,
            output_dim=OUTPUT_DIM,
            flow=flow,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=10,
            device=DEVICE, dtype=DTYPE, seed=SEED,
        )

        ftip.warm_start_from_vip(vip, learnable_affine=True)

        X, _ = traj_data
        samples, std = ftip(X)
        assert torch.isfinite(samples).all()
