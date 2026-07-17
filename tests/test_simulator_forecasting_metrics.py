import torch

from experiments.common import oscillation_period_error
from experiments.simulator_forecasting.metrics import (
    coerce_regions,
    crps_from_samples,
    interval_coverage,
    metrics_by_region,
    mixture_gaussian_nlpd,
    region_masks,
)


def test_region_masks_partition_forecasting_horizon():
    t = torch.tensor([0.0, 8.0, 8.1, 12.0, 15.0, 20.0, 25.0, 30.0])
    masks = region_masks(t)

    assert masks["interpolation"].tolist() == [True, True, False, False, False, False, False, False]
    assert masks["near_extrapolation"].tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert masks["medium_extrapolation"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert masks["far_extrapolation"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_mixture_nlpd_and_crps_are_finite_for_degenerate_samples():
    y = torch.ones(3, 1)
    samples = y.unsqueeze(0).expand(4, -1, -1)

    nlpd = mixture_gaussian_nlpd(samples, y, noise_var=torch.tensor([0.01]))
    crps = crps_from_samples(samples, y)

    assert torch.isfinite(nlpd)
    assert torch.allclose(crps, torch.zeros(()))


def test_interval_coverage_for_inside_target():
    y = torch.tensor([[2.0], [2.0]])
    samples = torch.tensor(
        [
            [[0.0], [0.0]],
            [[1.0], [1.0]],
            [[2.0], [2.0]],
            [[3.0], [3.0]],
            [[4.0], [4.0]],
        ]
    )

    coverage = interval_coverage(samples, y, levels=(0.8,))

    assert torch.allclose(coverage[0.8], torch.ones(()))


def test_metrics_by_region_outputs_expected_keys():
    t = torch.linspace(0.0, 30.0, 31)
    y = torch.sin(t).reshape(-1, 1)
    samples = y.unsqueeze(0).expand(5, -1, -1)

    rows = metrics_by_region(samples, y, t, noise_std=torch.tensor([0.1]))

    assert set(rows) == {
        "interpolation",
        "near_extrapolation",
        "medium_extrapolation",
        "far_extrapolation",
    }
    assert {"rmse", "nlpd", "crps", "cov90", "cov95", "width90", "width95"} <= set(
        rows["far_extrapolation"]
    )


def test_metrics_by_region_accepts_configurable_t_obs_regions():
    t = torch.linspace(0.0, 30.0, 31)
    y = torch.sin(t).reshape(-1, 1)
    samples = y.unsqueeze(0).expand(5, -1, -1)
    regions = coerce_regions(
        {
            "interpolation": {"lo": 0.0, "hi": 15.0, "include_left": True},
            "near_extrapolation": {"lo": 15.0, "hi": 20.0, "include_left": False},
            "far_extrapolation": {"lo": 20.0, "hi": 30.0, "include_left": False},
        }
    )

    masks = region_masks(t, regions=regions)
    rows = metrics_by_region(samples, y, t, noise_std=torch.tensor([0.1]), regions=regions)

    assert masks["interpolation"].sum().item() == 16
    assert masks["near_extrapolation"].sum().item() == 5
    assert masks["far_extrapolation"].sum().item() == 10
    assert set(rows) == {"interpolation", "near_extrapolation", "far_extrapolation"}


def test_oscillation_period_error_recovers_sine_period_difference():
    t = torch.linspace(0.0, 12.0, 1201, dtype=torch.float64)
    truth = torch.sin(2.0 * torch.pi * t / 2.0).reshape(-1, 1)
    prediction = torch.sin(2.0 * torch.pi * t / 3.0).reshape(1, -1, 1).expand(4, -1, -1)

    error = oscillation_period_error(prediction, truth, t)

    assert torch.allclose(error, torch.tensor(1.0, dtype=error.dtype), atol=1e-3)
