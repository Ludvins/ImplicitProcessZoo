import csv

import torch

from experiments.eld_forecasting.datasets import load_synthetic_tasks
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
