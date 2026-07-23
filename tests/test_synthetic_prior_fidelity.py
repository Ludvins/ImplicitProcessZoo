from __future__ import annotations

import csv

import pytest
import torch

from experiments.synthetic.prior_fidelity import (
    main,
    sample_gmvip_surrogate,
    sample_true_prior,
    sample_vip_surrogate,
)
from experiments.synthetic.prior_fidelity_metrics import (
    estimate_rbf_bandwidth,
    fidelity_metrics,
    projection_directions,
)


def _metric_inputs():
    generator = torch.Generator().manual_seed(7)
    reference = torch.randn(64, 11, generator=generator, dtype=torch.float64)
    directions = projection_directions(
        11,
        24,
        seed=19,
        device="cpu",
        dtype=torch.float64,
    )
    bandwidth = estimate_rbf_bandwidth(reference)
    return reference, directions, bandwidth


def test_prior_fidelity_metrics_are_zero_for_identical_samples():
    reference, directions, bandwidth = _metric_inputs()
    metrics, pointwise = fidelity_metrics(
        reference,
        reference.clone(),
        directions=directions,
        mmd_bandwidth=bandwidth,
        chunk_size=16,
    )

    assert torch.count_nonzero(pointwise) == 0
    for value in metrics.values():
        assert value == pytest.approx(0.0, abs=1e-12)


def test_prior_fidelity_metrics_detect_controlled_shift():
    reference, directions, bandwidth = _metric_inputs()
    candidate = reference + 0.75
    metrics, pointwise = fidelity_metrics(
        reference,
        candidate,
        directions=directions,
        mmd_bandwidth=bandwidth,
        chunk_size=16,
    )

    assert torch.allclose(pointwise, torch.full_like(pointwise, 0.75))
    assert metrics["joint_sw2"] > 0.0
    assert metrics["marginal_w1_mean"] == pytest.approx(0.75)
    assert metrics["mean_rmse"] == pytest.approx(0.75)
    assert metrics["energy_distance"] > 0.0
    assert metrics["rbf_mmd2"] > 0.0


def test_projection_directions_and_true_prior_samples_are_deterministic():
    first_directions = projection_directions(
        9,
        12,
        seed=3,
        device="cpu",
        dtype=torch.float64,
    )
    second_directions = projection_directions(
        9,
        12,
        seed=3,
        device="cpu",
        dtype=torch.float64,
    )
    x_grid = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64).unsqueeze(-1)
    first_samples = sample_true_prior(
        x_grid,
        8,
        prior_seed=11,
        sample_seed=12,
    )
    second_samples = sample_true_prior(
        x_grid,
        8,
        prior_seed=11,
        sample_seed=12,
    )

    assert torch.equal(first_directions, second_directions)
    assert torch.equal(first_samples, second_samples)


def test_vip_surrogate_has_empirical_basis_moments_and_is_deterministic():
    x_grid = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64).unsqueeze(-1)
    first, diagnostics = sample_vip_surrogate(
        x_grid,
        basis_size=6,
        num_samples=10,
        prior_seed=21,
        basis_seed=22,
        coefficient_seed=23,
    )
    second, second_diagnostics = sample_vip_surrogate(
        x_grid,
        basis_size=6,
        num_samples=10,
        prior_seed=21,
        basis_seed=22,
        coefficient_seed=23,
    )
    basis = diagnostics["basis"]
    centered = basis - basis.mean(dim=0, keepdim=True)
    empirical_covariance = centered.T @ centered / float(basis.shape[0] - 1)

    assert torch.equal(first, second)
    assert torch.equal(diagnostics["basis"], second_diagnostics["basis"])
    assert torch.allclose(diagnostics["mean"], basis.mean(dim=0))
    assert torch.allclose(diagnostics["covariance"], empirical_covariance)
    assert diagnostics["coefficient_prior_kl"].item() == 0.0


def test_gmvip_surrogate_is_deterministic_and_uses_standard_normal_prior():
    x_grid = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64).unsqueeze(-1)
    kwargs = {
        "num_inducing": 4,
        "operator_bank_size": 8,
        "num_samples": 10,
        "prior_seed": 31,
        "operator_seed": 32,
        "sample_seed": 33,
        "jitter": 1e-5,
        "shrinkage": 0.02,
    }
    first, diagnostics = sample_gmvip_surrogate(x_grid, **kwargs)
    second, second_diagnostics = sample_gmvip_surrogate(x_grid, **kwargs)

    assert torch.equal(first, second)
    assert first.shape == (10, 9)
    assert diagnostics["coefficient_prior_kl"].item() == pytest.approx(0.0)
    assert second_diagnostics["coefficient_prior_kl"].item() == pytest.approx(0.0)


def test_smoke_runner_writes_label_free_metric_artifacts(tmp_path):
    output_dir = tmp_path / "prior_fidelity"
    result = main(
        [
            "--smoke",
            "--device",
            "cpu",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
        ]
    )

    expected = {
        "metrics.csv",
        "summary.csv",
        "pointwise_w1.csv",
        "run_config.json",
        "default_samples_and_profiles.npz",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    assert result["config"]["label_free"] is True
    assert result["config"]["architecture"]["hidden_dims"] == [10, 10]
    assert {row["method"] for row in result["metrics"]} == {
        "true_null",
        "vip",
        "gmvip",
    }
    assert all(row["coefficient_prior_kl"] == 0.0 for row in result["metrics"][1:])
    with (output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(result["metrics"])
    assert any(row["in_matched_sweep"] == "True" for row in csv_rows)
    assert any(row["in_bank_sweep"] == "True" for row in csv_rows)


def test_smoke_runner_generates_every_plot_when_matplotlib_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "prior_fidelity_plots"
    main(
        [
            "--smoke",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )

    plot_stems = {
        "prior_samples",
        "pointwise_w1_default",
        "distance_vs_dimension",
        "moment_errors_vs_dimension",
        "gmvip_bank_sensitivity",
        "robustness_metrics_vs_dimension",
    }
    for stem in plot_stems:
        assert (output_dir / f"{stem}.png").is_file()
        assert (output_dir / f"{stem}.pdf").is_file()
