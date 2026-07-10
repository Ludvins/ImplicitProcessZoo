"""
Shared fixtures for testing all TVIP models.

Provides small-scale BayesianNN, GP, data loaders, and common
parameters so that each test module can focus on model-specific logic.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from implicit_process_zoo.flows.flows import CouplingFlow
from implicit_process_zoo.priors.generative_functions import (
    GP,
    BayesianCNN,
    BayesianNN,
    BayesLinear,
    SimplerBayesLinear,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
DTYPE = torch.float64
SEED = 42

INPUT_DIM = 2
OUTPUT_DIM = 1
NUM_SAMPLES = 6  # prior function samples (S); must be even for CouplingFlow
HIDDEN = [16]  # small hidden layer for speed
NUM_DATA = 30  # tiny dataset
BATCH_SIZE = 10


# ---------------------------------------------------------------------------
# Generative functions (priors)
# ---------------------------------------------------------------------------


@pytest.fixture
def bnn():
    """BayesianNN with BayesLinear layers, fix_random_noise=True."""
    return BayesianNN(
        structure=HIDDEN,
        activation=torch.nn.Tanh(),
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        layer_model=BayesLinear,
        seed=SEED,
        fix_random_noise=True,
        device=DEVICE,
        dtype=DTYPE,
    )


@pytest.fixture
def bnn_variational():
    """BayesianNN with SimplerBayesLinear, fix_random_noise=False (for MFVI)."""
    return BayesianNN(
        structure=HIDDEN,
        activation=torch.nn.Tanh(),
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        layer_model=SimplerBayesLinear,
        seed=SEED,
        fix_random_noise=False,
        device=DEVICE,
        dtype=DTYPE,
    )


@pytest.fixture
def bnn_fbnn_posterior():
    """BayesianNN for fBNN posterior (trainable, fix_random_noise=False)."""
    return BayesianNN(
        structure=HIDDEN,
        activation=torch.nn.Tanh(),
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        layer_model=SimplerBayesLinear,
        seed=SEED + 1,
        fix_random_noise=False,
        device=DEVICE,
        dtype=DTYPE,
    )


@pytest.fixture
def bnn_fbnn_prior():
    """BayesianNN for fBNN prior (frozen, fix_random_noise=False)."""
    prior = BayesianNN(
        structure=HIDDEN,
        activation=torch.nn.Tanh(),
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        layer_model=SimplerBayesLinear,
        seed=SEED + 2,
        fix_random_noise=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    prior.freeze_parameters()
    return prior


@pytest.fixture
def gp():
    """GP prior with random features."""
    return GP(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        inner_layer_dim=10,
        seed=SEED,
        device=DEVICE,
        dtype=DTYPE,
    )


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


@pytest.fixture
def coupling_flow():
    """CouplingFlow for FTIP."""
    return CouplingFlow(
        depth=2,
        input_dim=NUM_SAMPLES,
        device=DEVICE,
        dtype=DTYPE,
        seed=SEED,
    )


@pytest.fixture
def nfvi_flow_factory():
    """Build a small CouplingFlow sized to the NFVI MLP's flat weight dim.

    NF-VI's flow operates over the entire MLP parameter vector, whose size
    depends on ``input_dim``/``output_dim``/``structure``. Tests should call
    this factory with the MLP's total parameter count to avoid drift.
    """

    def _build(total_params):
        return CouplingFlow(
            depth=2,
            input_dim=total_params,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

    return _build


# ---------------------------------------------------------------------------
# Datasets & data loaders
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_data():
    """(X, y) tensors for regression, normalised to ~N(0,1)."""
    rng = np.random.RandomState(SEED)
    X = rng.randn(NUM_DATA, INPUT_DIM).astype(np.float64)
    y = (X @ rng.randn(INPUT_DIM, OUTPUT_DIM) + 0.1 * rng.randn(NUM_DATA, OUTPUT_DIM)).astype(
        np.float64
    )
    X_t = torch.tensor(X, dtype=DTYPE, device=DEVICE)
    y_t = torch.tensor(y, dtype=DTYPE, device=DEVICE)
    return X_t, y_t


@pytest.fixture
def regression_loader(regression_data):
    """DataLoader yielding (X_batch, y_batch)."""
    X, y = regression_data
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture
def binary_data():
    """(X, y) tensors for binary classification."""
    rng = np.random.RandomState(SEED)
    X = rng.randn(NUM_DATA, INPUT_DIM).astype(np.float64)
    y = (X[:, 0:1] > 0).astype(np.float64)
    X_t = torch.tensor(X, dtype=DTYPE, device=DEVICE)
    y_t = torch.tensor(y, dtype=DTYPE, device=DEVICE)
    return X_t, y_t


@pytest.fixture
def binary_loader(binary_data):
    X, y = binary_data
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture
def multiclass_data():
    """(X, y) tensors for 3-class classification.

    ``y`` is class indices of shape ``(N, 1)`` — the format expected by
    :func:`implicit_process_zoo.utils.likelihood.multiclass_logp` and the ``nelbo`` paths
    of all multiclass models in this repo (matches ``mnist_multiclass_data``).
    """
    rng = np.random.RandomState(SEED)
    X = rng.randn(NUM_DATA, INPUT_DIM).astype(np.float64)
    labels = rng.randint(0, 3, size=(NUM_DATA, 1)).astype(np.float64)
    X_t = torch.tensor(X, dtype=DTYPE, device=DEVICE)
    y_t = torch.tensor(labels, dtype=DTYPE, device=DEVICE)
    return X_t, y_t


@pytest.fixture
def multiclass_loader(multiclass_data):
    X, y = multiclass_data
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# Multi-output versions (OUTPUT_DIM=3 for multiclass models)
# ---------------------------------------------------------------------------


@pytest.fixture
def bnn_multiclass():
    """BayesianNN with output_dim=3 for multiclass."""
    return BayesianNN(
        structure=HIDDEN,
        activation=torch.nn.Tanh(),
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        output_dim=3,
        layer_model=BayesLinear,
        seed=SEED,
        fix_random_noise=True,
        device=DEVICE,
        dtype=DTYPE,
    )


@pytest.fixture
def bnn_variational_multiclass():
    """BayesianNN for MFVI multiclass."""
    return BayesianNN(
        structure=HIDDEN,
        activation=torch.nn.Tanh(),
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        output_dim=3,
        layer_model=SimplerBayesLinear,
        seed=SEED,
        fix_random_noise=False,
        device=DEVICE,
        dtype=DTYPE,
    )


# ---------------------------------------------------------------------------
# BayesianCNN (LeNet) fixtures
# ---------------------------------------------------------------------------

MNIST_INPUT_DIM = 784
NUM_CLASSES = 10


@pytest.fixture
def cnn_multiclass():
    """BayesianCNN (LeNet) for MNIST multiclass classification."""
    return BayesianCNN(
        num_samples=NUM_SAMPLES,
        input_dim=MNIST_INPUT_DIM,
        output_dim=NUM_CLASSES,
        layer_model=SimplerBayesLinear,
        head_dims=[32],
        device=DEVICE,
        dtype=DTYPE,
        seed=SEED,
    )


@pytest.fixture
def mnist_multiclass_data():
    """(X, y) tensors for MNIST-shaped 10-class classification (y as class indices)."""
    rng = np.random.RandomState(SEED)
    X = rng.randn(NUM_DATA, MNIST_INPUT_DIM).astype(np.float64)
    labels = rng.randint(0, NUM_CLASSES, size=(NUM_DATA, 1)).astype(np.float64)
    return (
        torch.tensor(X, dtype=DTYPE, device=DEVICE),
        torch.tensor(labels, dtype=DTYPE, device=DEVICE),
    )


@pytest.fixture
def mnist_multiclass_loader(mnist_multiclass_data):
    X, y = mnist_multiclass_data
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
