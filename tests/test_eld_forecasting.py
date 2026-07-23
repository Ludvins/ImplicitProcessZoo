import csv
import json

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.common.electricity_data import (
    ElectricityData,
    WindowIndex,
    WindowSpec,
    _validate_expected_targets,
    load_synthetic_tasks,
    stress_diagnostics_for_targets,
)
from experiments.common.electricity_metrics import forecast_regions
from experiments.common.electricity_prior import HistoricalLoadWindowPrior
from experiments.eld_forecasting.benchmark import (
    DEFAULT_CONFIG,
    EmpiricalGaussianPredictive,
    ExactEmpiricalMatheronPredictive,
    _load_expected_targets,
    build_model,
    fit_model,
    main,
    parse_args,
    predictive_function_samples,
)


def test_historical_load_window_prior_interpolates_and_samples():
    windows = torch.stack(
        [
            torch.linspace(0.0, 1.0, 5),
            torch.linspace(1.0, 2.0, 5),
        ],
        dim=0,
    )
    prior = HistoricalLoadWindowPrior(windows, num_samples=3, seed=7, dtype=torch.float64)
    X = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    latents = torch.tensor([0, 1], dtype=torch.long)
    values = prior.evaluate_latents(latents, X)

    assert values.shape == (2, 3, 1)
    assert torch.allclose(values[0, :, 0], torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
    assert torch.allclose(values[1, :, 0], torch.tensor([1.0, 1.5, 2.0], dtype=torch.float64))
    assert prior(X).shape == (3, 3, 1)


def test_eld_synthetic_build_train_predict_smoke():
    tasks = load_synthetic_tasks(
        seed=0, n_targets=1, bank_size=16, window_length=48, prefix_length=8
    )
    task = tasks[0]
    assert task.metadata["context_points"] == 8
    assert task.metadata["forecast_points"] == 40
    assert task.metadata["context_role"] == "observed_prefix_for_online_posterior_conditioning"
    assert task.metadata["target_role"] == "heldout_forecast_window"
    config = {
        "prior": {"bank_size": 16},
        "gmvip": {"num_inducing": 8, "jitter": 1e-5, "shrinkage": 0.02, "beta": 1.0},
        "training": {
            "learning_rate": 1e-3,
            "max_steps": 1,
            "n_mc_train": 2,
            "n_mc_eval": 4,
            "batch_size": "full",
            "regression_coeffs": 8,
            "max_grad_norm": 5.0,
            "disable_tqdm": True,
        },
    }

    for method in (
        "analog",
        "seasonal_naive",
        "empirical_gaussian",
        "gmvip_empirical_exact",
        "vip",
        "ftip",
        "gmvip_empirical",
    ):
        model = build_model(
            method, task, config, seed=123, device=torch.device("cpu"), dtype=torch.float64
        )
        info = fit_model(model, task, config, device=torch.device("cpu"))
        samples = predictive_function_samples(model, method, task.X_plot[:5], 3, seed=456)

        assert info["steps"] >= 0
        assert samples.shape == (3, 5, 1)
        assert torch.isfinite(samples).all()


def test_empirical_gaussian_matches_analytic_conditioning():
    task = load_synthetic_tasks(
        seed=3, n_targets=1, bank_size=16, window_length=12, prefix_length=5
    )[0]
    model = EmpiricalGaussianPredictive(task)

    windows = task.prior.windows[..., 0]
    mean = windows.mean(dim=0)
    centered = windows - mean
    covariance = centered.mT @ centered / float(windows.shape[0] - 1)
    prefix_length = int(task.X_train.shape[0])
    noise_variance = task.noise_std[0].square()
    system = covariance[:prefix_length, :prefix_length] + noise_variance * torch.eye(
        prefix_length, dtype=windows.dtype
    )
    cross_covariance = covariance[:, :prefix_length]
    residual = task.y_train[:, 0] - mean[:prefix_length]
    expected_mean = mean + cross_covariance @ torch.linalg.solve(system, residual)
    expected_covariance = covariance - cross_covariance @ torch.linalg.solve(
        system, cross_covariance.mT
    )

    torch.testing.assert_close(model.posterior_mean, expected_mean, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(
        model.posterior_covariance, expected_covariance, atol=1e-8, rtol=1e-8
    )
    samples = model.predict_f_samples(task.X_plot, 7, seed=19)
    assert samples.shape == (7, 12, 1)
    assert torch.isfinite(samples).all()


def test_exact_empirical_matheron_has_full_exact_coefficient_posterior():
    task = load_synthetic_tasks(
        seed=5, n_targets=1, bank_size=24, window_length=16, prefix_length=6
    )[0]
    matheron = ExactEmpiricalMatheronPredictive(task)

    expected_inducing_mean = matheron.prior_mean[: matheron.prefix_length] + (
        matheron.inducing_scale @ matheron.coefficient_posterior_mean
    )
    torch.testing.assert_close(
        matheron.inducing_posterior_mean,
        expected_inducing_mean,
        atol=1e-10,
        rtol=1e-10,
    )
    expected_inducing_covariance = (
        matheron.inducing_scale
        @ matheron.coefficient_posterior_covariance
        @ matheron.inducing_scale.mT
    )
    torch.testing.assert_close(
        matheron.inducing_posterior_covariance,
        expected_inducing_covariance,
        atol=1e-10,
        rtol=1e-10,
    )

    prefix_length = int(task.X_train.shape[0])
    assert matheron.coefficient_posterior_covariance.shape == (
        prefix_length,
        prefix_length,
    )
    off_diagonal = matheron.coefficient_posterior_covariance - torch.diag(
        torch.diagonal(matheron.coefficient_posterior_covariance)
    )
    assert torch.count_nonzero(off_diagonal.abs() > 1e-10) > 0
    samples = matheron.predict_f_samples(task.X_plot, 32, seed=91)
    assert samples.shape == (32, 16, 1)
    assert torch.isfinite(samples).all()


def test_eld_synthetic_runner_writes_metrics(tmp_path):
    out_root = tmp_path / "results"
    main(
        [
            "--methods",
            "analog,vip,gmvip_empirical",
            "--smoke",
            "--seed",
            "0",
            "--target-ids",
            "0",
            "--output-root",
            str(out_root),
            "--disable-tqdm",
        ]
    )

    for method in ("analog", "vip", "gmvip_empirical"):
        method_dir = out_root / "seed_0" / "S_20" / method
        path = method_dir / "metrics_per_target_region.csv"
        assert path.exists()
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["region"] for row in rows} >= {"full_forecast", "same_day_forecast"}
        assert {row["run_seed"] for row in rows} == {"0"}
        assert {row["target_seed"] for row in rows} == {"0"}
        assert all("region_start_idx" in row and "region_stop_idx" in row for row in rows)
        assert all("region_include_left" not in row for row in rows)
        assert all("cov80" in row and "width80" in row for row in rows)
        assert all("evaluation_samples" in row for row in rows)
        assert all("observation_noise_std" in row for row in rows)
        metrics = json.loads((method_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["run_seed"] == 0
        manifest = json.loads((method_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["checkpoint_selection"] == "none_final_step_only"
        expected_noise = "fixed_scalar" if method == "analog" else "learned_scalar"
        assert manifest["observation_noise"]["mode"] == expected_noise
        assert (method_dir / "checkpoints" / "target_0.pt").exists()


def test_eld_defaults_are_the_canonical_experiment():
    config = DEFAULT_CONFIG
    args = parse_args(["--methods", "analog,vip,ftip,gmvip_empirical"])

    assert args.target_ids == "0:25"
    assert args.vip_basis_size == 20
    assert args.learn_observation_noise is True
    assert not hasattr(args, "validation")
    assert config["data"]["n_targets"] == 25
    assert config["data"]["window_length"] == 192
    assert config["data"]["prefix_length"] == 96
    assert config["data"]["prior_years"] == [2011, 2012, 2013]
    assert config["data"]["target_years"] == [2014]
    assert config["prior"] == {"bank_size": 2048, "selection": "calendar_prefix_nn"}
    assert config["gmvip"]["num_inducing"] == 96
    assert config["ftip"]["flow_depth"] == 1
    assert "warm_start_from_vip" not in config["ftip"]
    assert config["training"]["learning_rate"] == 5.0e-3
    assert config["training"]["max_steps"] == 500
    assert config["training"]["n_mc_train"] == 8
    assert config["training"]["n_mc_eval"] == 1024
    assert config["training"]["regression_coeffs"] == 20
    assert config["likelihood"]["learn_observation_noise"] is True
    assert config["metrics"]["levels"] == [0.8, 0.9, 0.95]
    assert config["metrics"]["regions"] == {"test_forecast": {"start": 96, "stop": 192}}
    expected = _load_expected_targets(config["data"]["target_manifest"], run_seed=0)
    assert expected is not None
    assert len(expected) == 25
    assert expected[18] == ("MT_353", "2014-09-23 00:00:00")


def test_frozen_target_validation_rejects_identity_drift():
    targets = [
        WindowSpec(0, 0, "2014-01-01 00:00:00", 2014, 1, False, 0.1, 0.2),
        WindowSpec(1, 1, "2014-01-02 00:00:00", 2014, 1, False, 0.2, 0.3),
    ]
    expected = {
        0: ("MT_001", "2014-01-01 00:00:00"),
        1: ("MT_999", "2014-01-02 00:00:00"),
    }
    with pytest.raises(ValueError, match="target selection drifted"):
        _validate_expected_targets(targets, ["MT_001", "MT_002"], expected)


def test_stress_diagnostics_are_ranked_against_complete_target_set():
    targets = [
        WindowSpec(index, index, str(index), 2014, 1, False, float(index + 1), float(5 - index))
        for index in range(5)
    ]
    diagnostics = stress_diagnostics_for_targets(targets)

    assert len(diagnostics) == 5
    assert [item["stress_threshold"] for item in diagnostics] == pytest.approx([0.6] * 5)
    assert [item["stress_score"] for item in diagnostics] == pytest.approx([0.6] * 5)


def test_eld_artifact_resume_skips_complete_targets_and_rejects_config_changes(tmp_path):
    out_root = tmp_path / "resumable"
    argv = [
        "--methods",
        "analog",
        "--smoke",
        "--seed",
        "0",
        "--target-ids",
        "0",
        "--output-root",
        str(out_root),
        "--disable-tqdm",
    ]
    main(argv)
    prediction = out_root / "seed_0" / "S_20" / "analog" / "predictions" / "target_0.npz"
    original_mtime = prediction.stat().st_mtime_ns

    result = main([*argv, "--resume"])

    assert prediction.stat().st_mtime_ns == original_mtime
    assert result["analog"]["targets"] == [0]
    with pytest.raises(ValueError, match="config|manifest"):
        main([*argv, "--resume", "--no-learn-observation-noise"])


def test_eld_regions_are_half_open_nonoverlapping_and_dynamic():
    default = forecast_regions(window_points=192, prefix_points=32)
    assert default["observed_prefix"] == {"start": 0, "stop": 32}
    assert default["full_forecast"] == {"start": 32, "stop": 192}
    assert default["same_day_forecast"] == {"start": 32, "stop": 96}
    assert default["next_day_forecast"] == {"start": 96, "stop": 192}

    smoke = forecast_regions(window_points=48, prefix_points=8)
    assert smoke["observed_prefix"]["stop"] == smoke["full_forecast"]["start"]
    assert "next_day_forecast" not in smoke


def test_window_index_vectorized_selection_records_fallback_and_caches():
    values = np.asarray(
        [
            [10.0, 30.0],
            [12.0, 29.0],
            [11.0, 28.0],
            [13.0, 27.0],
            [20.0, 40.0],
            [24.0, 39.0],
            [21.0, 38.0],
            [25.0, 37.0],
            [15.0, 10.0],
            [16.0, 12.0],
            [17.0, 11.0],
            [18.0, 13.0],
        ],
        dtype=np.float32,
    )
    data = ElectricityData(
        values=values,
        timestamps=pd.date_range("2014-01-01", periods=len(values), freq="15min"),
        clients=["a", "b"],
    )
    target = WindowSpec(0, 0, "target", 2014, 1, False, 0.1, 2.0)
    same_client_noncalendar = WindowSpec(0, 4, "prior-a", 2011, 6, False, 0.1, 2.0)
    other_client_calendar = WindowSpec(1, 8, "prior-b", 2011, 1, False, 0.1, 2.0)
    index = WindowIndex(
        data,
        [target, same_client_noncalendar, other_client_calendar],
        window_length=4,
        prefix_length=2,
        prefix_eps=1e-3,
    )
    target_prefix = np.asarray([-1.0, 1.0], dtype=np.float32)
    selected, diagnostics = index.select(
        target,
        years={2011},
        bank_size=2,
        seed=9,
        target_prefix_norm=target_prefix,
        selection="same_client_calendar_prefix_nn",
    )
    assert {spec.start_idx for spec in selected} == {4}
    assert diagnostics["prior_requested_candidate_count"] == 0
    assert diagnostics["prior_fallback_tier"] == 1
    assert diagnostics["prior_actual_client_constraint"] == "same"
    cached, cached_diagnostics = index.select(
        target,
        years={2011},
        bank_size=2,
        seed=9,
        target_prefix_norm=target_prefix,
        selection="same_client_calendar_prefix_nn",
    )
    assert [spec.start_idx for spec in cached] == [spec.start_idx for spec in selected]
    assert cached_diagnostics == diagnostics
