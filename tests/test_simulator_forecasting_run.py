import copy

import torch

from experiments.common.oscillator_data import load_damped_oscillator_tasks
from experiments.common.oscillator_generate import generate_dataset
from experiments.common.oscillator_prior import DampedOscillatorPrior
from experiments.simulator_forecasting.benchmark import (
    METHODS,
    SMOKE_SIMULATOR_FORECASTING_CONFIG,
    TOBS15_VIP_FTIP_GMVIP_20TARGET_CONFIG,
    FreshDampedOscillatorSIPPrior,
    build_model,
    fit_model,
    main,
    parse_args,
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


def test_canonical_oscillator_defaults():
    args = parse_args(["--methods", "vip,ftip,gmvip"])
    config = TOBS15_VIP_FTIP_GMVIP_20TARGET_CONFIG

    assert args.vip_basis_size == 256
    assert args.target_ids == "0:20"
    assert args.learn_observation_noise is True
    assert not hasattr(args, "validation")
    assert config["data"]["n_eval_targets"] == 20
    assert config["data"]["n_train"] == [64]
    assert config["data"]["n_test"] == 500
    assert config["data"]["t_obs"] == 15.0
    assert config["data"]["t_max"] == 30.0
    assert config["data"]["sigma_y"] == 0.05
    assert config["prior"]["bank_size"] == 1024
    assert config["gmvip"]["num_inducing"] == 32
    assert config["training"]["learning_rate"] == 5.0e-3
    assert config["training"]["max_steps"] == 3000
    assert config["training"]["n_mc_train"] == 16
    assert config["training"]["n_mc_eval"] == 1024
    assert config["training"]["regression_coeffs"] == 256
    assert config["likelihood"]["learn_observation_noise"] is True
    assert config["ftip"]["warm_start_from_vip"] is True
    assert config["ftip"]["warm_start_steps"] == 3000
    assert config["ftip"]["warm_start_lr"] == 5.0e-3
    assert config["ftip"]["fine_tune_steps"] == 3000
    assert config["ftip"]["fine_tune_lr"] == 1.0e-4


def test_run_smoke_writes_canonical_artifacts_without_period_metric(tmp_path):
    out_root = tmp_path / "results"

    main(
        [
            "--methods",
            "vip,ftip,gmvip",
            "--seed",
            "0",
            "--target-ids",
            "0",
            "--vip-basis-size",
            "8",
            "--smoke",
            "--output-root",
            str(out_root),
            "--disable-tqdm",
        ]
    )

    for method in ("vip", "ftip", "gmvip"):
        method_dir = out_root / "seed_0" / "S_8" / method
        metrics_path = method_dir / "metrics_per_target_region.csv"
        assert metrics_path.exists()
        metrics_text = metrics_path.read_text(encoding="utf-8")
        assert "far_extrapolation" in metrics_text
        assert "oscillation_period_error" not in metrics_text
        manifest = (method_dir / "manifest.json").read_text(encoding="utf-8")
        assert '"mode": "learned_scalar"' in manifest
        assert '"checkpoint_selection": "none_final_step_only"' in manifest
        assert (method_dir / "checkpoints" / "target_0_ntrain_4.pt").exists()
