import numpy as np
import torch

from experiments.common import (
    peak_time_error,
    phase_lag_error,
    positivity_violation_rate,
)
from experiments.volterra.benchmark import (
    _summarize,
    crps_from_samples,
    interval_coverage,
    interval_width,
    lotka_volterra_residual_score,
    mixture_gaussian_nll,
    nearest_prior_mse,
)


def test_crps_is_zero_when_all_samples_equal_target():
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    samples = y.unsqueeze(0).expand(5, -1, -1)

    assert torch.allclose(crps_from_samples(samples, y), torch.zeros(()))


def test_mixture_gaussian_nll_is_finite_with_zero_sample_variance():
    y = torch.ones(3, 2)
    samples = y.unsqueeze(0).expand(4, -1, -1)

    nll = mixture_gaussian_nll(samples, y, noise_var=0.0)

    assert torch.isfinite(nll)


def test_mixture_gaussian_nll_matches_manual_log_mixture():
    samples = torch.tensor([[[0.0]], [[2.0]]], dtype=torch.float64)
    target = torch.tensor([[1.0]], dtype=torch.float64)
    variance = torch.tensor([0.25], dtype=torch.float64)
    component_logs = -0.5 * (
        torch.log(2.0 * torch.pi * variance)
        + (target[0] - samples[:, 0]).square() / variance
    )
    expected = -(
        torch.logsumexp(component_logs, dim=0) - torch.log(torch.tensor(2.0))
    ).mean()

    assert torch.allclose(mixture_gaussian_nll(samples, target, variance), expected)


def test_summary_uses_sample_standard_error():
    rows = [
        {"method": "vip", "target_id": index, "rmse": value}
        for index, value in enumerate([1.0, 2.0, 3.0])
    ]
    summary = _summarize(rows)

    assert summary["rmse"]["mean"] == 2.0
    assert np.isclose(summary["rmse"]["stderr"], 1.0 / np.sqrt(3.0))


def test_interval_coverage_and_width_for_inside_target():
    samples = torch.tensor(
        [
            [[0.0], [0.0]],
            [[1.0], [1.0]],
            [[2.0], [2.0]],
            [[3.0], [3.0]],
            [[4.0], [4.0]],
        ]
    )
    y = torch.tensor([[2.0], [2.0]])

    coverage = interval_coverage(samples, y, levels=(0.8,))
    width = interval_width(samples, levels=(0.8,))

    assert torch.allclose(coverage[0.8], torch.ones(()))
    assert width[0.8] > 0.0


def test_nearest_prior_mse_zero_for_copied_prior_samples():
    prior = torch.randn(6, 5, 2)
    samples = prior[[1, 4]]

    result = nearest_prior_mse(samples, prior)

    assert torch.allclose(result["mse"], torch.zeros(2))
    assert torch.allclose(result["mean"], torch.zeros(()))


def test_lotka_volterra_residual_score_is_finite_for_valid_curve():
    t = torch.linspace(0.0, 4.0, 21)
    prey = 1.0 + 0.1 * torch.sin(t)
    predator = 1.0 + 0.1 * torch.cos(t)
    sample = torch.stack([prey, predator], dim=-1)

    score = lotka_volterra_residual_score(sample, t)

    assert score.shape == (1,)
    assert torch.isfinite(score).all()


def test_lotka_volterra_dynamics_metrics_have_known_values():
    t = torch.linspace(0.0, 10.0, 1001, dtype=torch.float64)
    truth = torch.stack(
        [
            2.0 + torch.cos(2.0 * torch.pi * (t - 2.0) / 4.0),
            2.0 + torch.cos(2.0 * torch.pi * (t - 3.0) / 4.0),
        ],
        dim=-1,
    )
    prediction = torch.stack(
        [
            2.0 + torch.cos(2.0 * torch.pi * (t - 2.5) / 4.0),
            2.0 + torch.cos(2.0 * torch.pi * (t - 4.0) / 4.0),
        ],
        dim=-1,
    ).unsqueeze(0)

    assert torch.allclose(peak_time_error(prediction, truth, t, channel=0), t.new_tensor(0.5))
    assert torch.allclose(phase_lag_error(prediction, truth, t), t.new_tensor(0.5), atol=1e-6)

    prediction[:, :10, 0] = -0.1
    expected_rate = torch.tensor(10 / prediction.numel(), dtype=prediction.dtype)
    assert torch.allclose(positivity_violation_rate(prediction), expected_rate)
