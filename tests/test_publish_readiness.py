"""Regression tests for public APIs and publish-readiness correctness fixes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.common.metrics import empirical_crps, empirical_crps_pairwise
from implicit_process_zoo.data import canonical_dataset_name
from implicit_process_zoo.data.base import Training_Dataset
from implicit_process_zoo.flows.glow_mixing import SplineCoupling1x1Flow
from implicit_process_zoo.ftip import FTIP
from implicit_process_zoo.map_baseline import DeterministicMAP
from implicit_process_zoo.utils import (
    batched_predict_samples,
    build_training_checkpoint,
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
    standard_normal_samples,
)
from implicit_process_zoo.utils.likelihood import prob_is_largest
from implicit_process_zoo.vip import VIP
from tests.conftest import (
    DEVICE,
    DTYPE,
    INPUT_DIM,
    NUM_DATA,
    NUM_SAMPLES,
    OUTPUT_DIM,
    SEED,
)


def _make_map(seed: int = SEED) -> DeterministicMAP:
    return DeterministicMAP(
        INPUT_DIM,
        OUTPUT_DIM,
        [8],
        torch.nn.Tanh(),
        NUM_DATA,
        device=DEVICE,
        dtype=DTYPE,
        seed=seed,
    )


def test_dataset_normalization_is_finite_for_constant_targets():
    dataset = Training_Dataset(
        np.arange(12, dtype=np.float64).reshape(6, 2),
        np.ones((6, 1), dtype=np.float64),
        verbose=False,
    )
    assert dataset.targets_std.item() > 0
    assert np.isfinite(dataset.targets).all()


def test_dataset_shape_validation_and_deprecated_aliases():
    with pytest.raises(ValueError, match="rank-2"):
        Training_Dataset(np.ones(3), np.ones((3, 1)), verbose=False)
    with pytest.warns(DeprecationWarning, match="yacht"):
        assert canonical_dataset_name("yatch") == "yacht"
    with pytest.warns(DeprecationWarning, match="heteroscedastic"):
        assert canonical_dataset_name("heterocedastic") == "heteroscedastic"


def test_multiclass_probability_matches_two_class_analytic_result():
    means = torch.tensor([[0.4, -0.2], [-0.5, 0.7]], dtype=DTYPE)
    variances = torch.tensor([[0.25, 1.0], [0.5, 0.3]], dtype=DTYPE)
    labels = torch.tensor([[0], [1]])

    actual = prob_is_largest(labels, means, variances, 2, 80, DTYPE, DEVICE).squeeze(-1)
    selected = torch.tensor([0.6, 1.2], dtype=DTYPE)
    total_variance = torch.tensor([1.25, 0.8], dtype=DTYPE)
    expected = 0.5 * (1.0 + torch.erf(selected / torch.sqrt(2.0 * total_variance)))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_multiclass_probability_matches_monte_carlo():
    means = torch.tensor([[0.3, -0.1]], dtype=DTYPE)
    variances = torch.tensor([[0.16, 1.21]], dtype=DTYPE)
    labels = torch.tensor([[0]])
    actual = prob_is_largest(labels, means, variances, 2, 80, DTYPE, DEVICE).item()

    generator = torch.Generator().manual_seed(SEED)
    draws = means + torch.sqrt(variances) * torch.randn(
        150_000, 2, dtype=DTYPE, generator=generator
    )
    monte_carlo = (draws[:, 0] > draws[:, 1]).to(DTYPE).mean().item()
    assert actual == pytest.approx(monte_carlo, abs=4e-3)


@pytest.mark.parametrize("shape", [(1, 7, 1), (3, 5, 2), (8, 4, 3)])
def test_sorted_crps_matches_pairwise_identity(shape):
    generator = torch.Generator().manual_seed(SEED)
    samples = torch.randn(*shape, dtype=DTYPE, generator=generator)
    targets = torch.randn(*shape[1:], dtype=DTYPE, generator=generator)
    torch.testing.assert_close(
        empirical_crps(samples, targets),
        empirical_crps_pairwise(samples, targets),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("count", [1, 3, 4, 5])
def test_antithetic_sampler_returns_exact_requested_count(count):
    generator = torch.Generator().manual_seed(SEED)
    samples = standard_normal_samples(
        count,
        3,
        dtype=DTYPE,
        device=DEVICE,
        generator=generator,
        antithetic=True,
    )
    assert samples.shape == (count, 3)
    pairs = count // 2
    if pairs:
        torch.testing.assert_close(samples[:pairs], -samples[pairs : 2 * pairs])


@pytest.mark.parametrize("count", [1, 3, 5])
def test_ftip_odd_prediction_counts(bnn, coupling_flow, regression_data, count):
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
    X, _ = regression_data
    assert model.predict_f_samples(X, count, seed=17).shape == (count, NUM_DATA, OUTPUT_DIM)


def test_shared_prediction_handles_uneven_final_batch(bnn, regression_data):
    X, y = regression_data
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)
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
    samples = batched_predict_samples(model, loader, 7, kind="f", seed=99)
    assert samples.shape == (7, NUM_DATA, OUTPUT_DIM)


def test_normalization_buffers_follow_to_and_round_trip(bnn):
    model = VIP(
        bnn,
        NUM_SAMPLES,
        OUTPUT_DIM,
        "regression",
        NUM_DATA,
        y_mean=2.5,
        y_std=0.75,
        device=DEVICE,
        dtype=DTYPE,
        seed=SEED,
    )
    state = model.state_dict()
    assert "y_mean" in state and "y_std" in state
    model.to(dtype=torch.float32)
    assert model.y_mean.dtype == torch.float32
    assert model.y_std.dtype == torch.float32


def test_seeded_constructors_preserve_global_rng():
    torch.manual_seed(12345)
    expected_state = torch.random.get_rng_state().clone()
    _make_map(seed=77)
    assert torch.equal(torch.random.get_rng_state(), expected_state)

    SplineCoupling1x1Flow(
        depth=2,
        input_dim=4,
        device=DEVICE,
        dtype=DTYPE,
        seed=77,
    )
    assert torch.equal(torch.random.get_rng_state(), expected_state)


def test_fit_rejects_ambiguous_modes_and_supports_short_cosine(regression_loader):
    model = _make_map()
    with pytest.raises(ValueError, match="Exactly one"):
        model.fit(regression_loader, epochs=1, iterations=1)
    with pytest.raises(ValueError, match="Exactly one"):
        model.fit(regression_loader)
    losses = model.fit(
        regression_loader,
        iterations=1,
        cosine_annealing=True,
        return_loss=True,
    )
    assert len(losses) == 1


def test_checkpoint_interruption_resume_equivalence(tmp_path, regression_loader):
    uninterrupted = _make_map(seed=19)
    uninterrupted.fit(regression_loader, iterations=4, lr=2e-3)

    interrupted = _make_map(seed=19)
    interrupted.fit(regression_loader, iterations=2, lr=2e-3)
    checkpoint = build_training_checkpoint(
        interrupted,
        optimizer=interrupted._fit_optimizer,
        scheduler=interrupted._fit_scheduler,
        global_step=interrupted._fit_global_step,
        arguments={"lr": 2e-3},
    )
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, checkpoint)

    resumed = _make_map(seed=999)
    optimizer = torch.optim.Adam(resumed.parameters(), lr=2e-3)
    loaded = load_training_checkpoint(path)
    restore_training_checkpoint(loaded, resumed, optimizer)
    resumed.fit(regression_loader, optimizer=optimizer, iterations=2)

    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_legacy_state_dict_is_rejected_for_resume(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(_make_map().state_dict(), path)
    with pytest.raises(ValueError, match="--warm-start-from"):
        load_training_checkpoint(path)
