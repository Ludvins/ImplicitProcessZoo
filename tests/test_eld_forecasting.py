import csv
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.eld_forecasting import valbank
from experiments.eld_forecasting.datasets import (
    ElectricityData,
    WindowIndex,
    WindowSpec,
    load_synthetic_tasks,
)
from experiments.eld_forecasting.metrics import forecast_regions, validation_test_regions
from experiments.eld_forecasting.priors import HistoricalLoadWindowPrior
from experiments.eld_forecasting.run import (
    build_model,
    fit_model,
    main,
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

    for method in ("analog", "seasonal_naive", "vip", "ftip", "gmvip_empirical"):
        model = build_model(
            method, task, config, seed=123, device=torch.device("cpu"), dtype=torch.float64
        )
        info = fit_model(model, task, config, device=torch.device("cpu"))
        samples = predictive_function_samples(model, method, task.X_plot[:5], 3, seed=456)

        assert info["steps"] >= 0
        assert samples.shape == (3, 5, 1)
        assert torch.isfinite(samples).all()


def test_eld_synthetic_runner_writes_metrics(tmp_path):
    out_root = tmp_path / "results"
    main(
        [
            "--preset",
            "eld_smoke",
            "--method",
            "analog,vip,gmvip_empirical",
            "--synthetic-smoke",
            "--seed",
            "0",
            "--output-dir",
            str(out_root),
            "--disable-tqdm",
        ]
    )

    for method in ("analog", "vip", "gmvip_empirical"):
        path = out_root / method / "seed_0" / "metrics_per_target_region.csv"
        assert path.exists()
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["region"] for row in rows} >= {"full_forecast", "same_day_forecast"}
        assert {row["methodology_version"] for row in rows} == {"2"}
        assert all("region_start_idx" in row and "region_stop_idx" in row for row in rows)
        assert all("region_include_left" not in row for row in rows)
        metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["methodology_version"] == 2


def test_eld_artifact_resume_skips_complete_targets_and_rejects_config_changes(tmp_path):
    out_root = tmp_path / "resumable"
    argv = [
        "--preset",
        "eld_smoke",
        "--method",
        "analog",
        "--synthetic-smoke",
        "--seed",
        "0",
        "--output-dir",
        str(out_root),
        "--disable-tqdm",
    ]
    main(argv)
    prediction = out_root / "analog" / "seed_0" / "predictions" / "target_0.npz"
    original_mtime = prediction.stat().st_mtime_ns

    result = main([*argv, "--resume-artifacts"])

    assert prediction.stat().st_mtime_ns == original_mtime
    assert result["analog"]["targets"] == [0]
    with pytest.raises(ValueError, match="config differs"):
        main([*argv, "--resume-artifacts", "--prior-bank-size", "17"])


def test_eld_regions_are_half_open_nonoverlapping_and_dynamic():
    default = forecast_regions(window_points=192, prefix_points=32)
    assert default["observed_prefix"] == {"start": 0, "stop": 32}
    assert default["full_forecast"] == {"start": 32, "stop": 192}
    assert default["same_day_forecast"] == {"start": 32, "stop": 96}
    assert default["next_day_forecast"] == {"start": 96, "stop": 192}

    smoke = forecast_regions(window_points=48, prefix_points=8)
    assert smoke["observed_prefix"]["stop"] == smoke["full_forecast"]["start"]
    assert "next_day_forecast" not in smoke

    split = validation_test_regions(window_points=192, train_points=60, context_points=80)
    assert split["validation"] == {"start": 60, "stop": 80}
    assert split["final_test"] == {"start": 80, "stop": 192}
    assert split["same_day_test"] == {"start": 80, "stop": 96}
    assert split["next_day_test"] == {"start": 96, "stop": 192}


def test_validation_candidates_share_target_seed(monkeypatch):
    args = valbank.parse_args([])
    args.candidate_prior_selections = "calendar,prefix_nn"
    observed_seeds = []

    def fake_make_task(*_args, prior_selection, seed, **_kwargs):
        observed_seeds.append(seed)
        return SimpleNamespace(
            metadata={
                "client_id": "client",
                "start_time": "time",
                "rule": prior_selection,
                "prior_candidate_count": 4,
                "prior_requested_candidate_count": 4,
                "prior_fallback_tier": 0,
                "prior_actual_calendar_constraint": prior_selection == "calendar",
                "prior_actual_client_constraint": "any",
            }
        )

    monkeypatch.setattr(valbank, "_make_task", fake_make_task)
    monkeypatch.setattr(valbank.base_run, "build_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        valbank.base_run,
        "fit_model",
        lambda *_args, **_kwargs: {"train_time_sec": 0.0, "steps": 0},
    )
    monkeypatch.setattr(
        valbank,
        "_score_validation",
        lambda _model, _method, task, _config, **_kwargs: {
            "crps": 0.0 if task.metadata["rule"] == "prefix_nn" else 1.0
        },
    )

    selected, rows = valbank._select_prior_rules(
        [object()],
        None,
        None,
        args=args,
        device=torch.device("cpu"),
        dtype=torch.float64,
        train_points=60,
        window_points=192,
        window_index=None,
    )
    assert len(set(observed_seeds)) == 1
    assert selected == {0: "prefix_nn"}
    assert all(row["candidate_rule_count"] == 2 for row in rows)


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
