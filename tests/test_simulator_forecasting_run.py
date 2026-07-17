import copy

import torch

from experiments.simulator_forecasting.datasets import load_damped_oscillator_tasks
from experiments.simulator_forecasting.generate import generate_dataset
from experiments.simulator_forecasting.priors import DampedOscillatorPrior
from experiments.simulator_forecasting.run import (
    METHODS,
    SMOKE_SIMULATOR_FORECASTING_CONFIG,
    FreshDampedOscillatorSIPPrior,
    build_model,
    fit_model,
    main,
    predictive_function_samples,
)


def test_sip_prior_adapter_fresh_draws_change_between_calls():
    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    base = DampedOscillatorPrior(
        t, y_mean=0.0, y_std=1.0, num_samples=3, seed=11, integration_dt=0.1
    )
    X = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)

    fresh = FreshDampedOscillatorSIPPrior(base, num_samples=3, seed=5, fresh_prior_samples=True)
    first = fresh(X, 3)
    second = fresh(X, 3)

    fixed = FreshDampedOscillatorSIPPrior(base, num_samples=3, seed=5, fresh_prior_samples=False)
    fixed_first = fixed(X, 3)
    fixed_second = fixed(X, 3)

    assert not torch.allclose(first, second)
    assert torch.allclose(fixed_first, fixed_second)


def test_damped_oscillator_trajectory_cache_uses_tensor_identity(monkeypatch):
    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    base = DampedOscillatorPrior(
        t, y_mean=0.0, y_std=1.0, num_samples=3, seed=11, integration_dt=0.1
    )
    first_latents = base.sample_latents(3, seed=5, cache=False)
    second_latents = base.sample_latents(3, seed=6, cache=False)

    # Model allocator reuse deterministically: the old implementation treated
    # an equal cache key as proof that two distinct tensors were the same draw.
    monkeypatch.setattr(base, "_cache_key", lambda _theta: ("reused-storage",))
    first = base._trajectory_grid(first_latents)
    second = base._trajectory_grid(second_latents)

    assert base._grid_cache_source is second_latents
    assert not torch.allclose(first, second)


def test_build_train_predict_smoke_methods(tmp_path):
    generate_dataset(tmp_path, n_targets=1, n_prior=8, n_test=31, t_max=30.0, seed=0)
    config = copy.deepcopy(SMOKE_SIMULATOR_FORECASTING_CONFIG)
    config["data"]["root"] = str(tmp_path)
    config["training"]["max_steps"] = 1
    task = load_damped_oscillator_tasks(
        tmp_path, seed=0, n_eval_targets=1, n_train=4, prior_bank_size=8, context_points=8
    )[0]

    for method in METHODS:
        model = build_model(
            method, task, config, seed=123, device=torch.device("cpu"), dtype=torch.float64
        )
        info = fit_model(model, method, task, config, device=torch.device("cpu"))
        samples = predictive_function_samples(model, method, task.X_plot[:5], 3, seed=456)

        assert info["steps"] >= 1
        assert samples.shape == (3, 5, 1)
        assert torch.isfinite(samples).all()


def test_run_smoke_all_writes_region_metrics(tmp_path):
    config_path = tmp_path / "override.yaml"
    data_root = tmp_path / "data"
    out_root = tmp_path / "results"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {data_root.as_posix()}",
                "  n_eval_targets: 1",
                "  n_train: [4]",
                "  n_test: 31",
                "  context_points: 8",
                "prior:",
                "  bank_size: 8",
                "training:",
                "  max_steps: 1",
                "  n_mc_eval: 3",
                "  n_mc_train: 2",
                "plots:",
                "  skip: true",
            ]
        ),
        encoding="utf-8",
    )

    main(
        [
            "--preset",
            "simulator_forecasting_smoke",
            "--method",
            "all",
            "--seed",
            "0",
            "--config",
            str(config_path),
            "--output-dir",
            str(out_root),
            "--skip-plots",
            "--disable-tqdm",
        ]
    )

    for method in METHODS:
        metrics_path = (
            out_root / "simulator_forecasting" / method / "seed_0" / "metrics_per_target_region.csv"
        )
        assert metrics_path.exists()
        metrics_text = metrics_path.read_text(encoding="utf-8")
        assert "far_extrapolation" in metrics_text
        assert "oscillation_period_error" in metrics_text
