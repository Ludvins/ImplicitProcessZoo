import copy

import numpy as np
import torch

from experiments.volterra.datasets import load_lotka_volterra_tasks
from experiments.volterra.generate import generate_bank
from experiments.volterra.priors import LotkaVolterraPrior
from experiments.volterra.run import (
    DEFAULT_LOTKA_VOLTERRA_CONFIG,
    SMOKE_LOTKA_VOLTERRA_CONFIG,
    FreshLotkaVolterraSIPPrior,
    build_model,
    fit_model,
    predictive_function_samples,
)


def test_volterra_paper_defaults_use_standardized_coefficient_budget():
    config = DEFAULT_LOTKA_VOLTERRA_CONFIG

    assert config["training"]["regression_coeffs"] == 20
    assert config["training"]["max_steps"] == 400
    assert config["ftip"]["warm_start_from_vip"] is True
    assert config["ftip"]["training_overrides"]["regression_coeffs"] == 20
    assert config["ftip"]["fine_tune_training"]["max_steps"] == 400
    assert config["ftip"]["fine_tune_training"]["early_stopping_patience"] == 401


def _write_small_lv_dataset(root):
    t = np.arange(0.0, 30.0 + 0.5, 0.5, dtype=np.float64)
    target_y, target_theta = generate_bank(2, t_grid=t, seed=240)
    np.savez_compressed(root / "target_paths.npz", t=t, y=target_y, theta=target_theta)


def test_sip_prior_adapter_fresh_draws_change_between_calls():
    t = np.linspace(0.0, 1.0, 5)
    base = LotkaVolterraPrior(
        t, y_mean=np.zeros((1, 2)), y_std=np.ones((1, 2)), num_samples=3, seed=11
    )
    X = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)

    fresh = FreshLotkaVolterraSIPPrior(base, num_samples=3, seed=5, fresh_prior_samples=True)
    first = fresh(X, 3)
    second = fresh(X, 3)

    assert base._latent_cache == {}

    fixed = FreshLotkaVolterraSIPPrior(base, num_samples=3, seed=5, fresh_prior_samples=False)
    fixed_first = fixed(X, 3)
    fixed_second = fixed(X, 3)

    assert not torch.allclose(first, second)
    assert torch.allclose(fixed_first, fixed_second)


def test_volterra_sip_builds_trains_and_predicts(tmp_path):
    _write_small_lv_dataset(tmp_path)
    config = copy.deepcopy(SMOKE_LOTKA_VOLTERRA_CONFIG)
    config["data"]["root"] = str(tmp_path)

    tasks = load_lotka_volterra_tasks(
        tmp_path,
        seed=0,
        n_eval_targets=1,
        n_train_times=6,
        prior_bank_size=8,
        dtype=torch.float64,
    )
    task = tasks[0]
    model = build_model(
        "sip", task, config, seed=0, device=torch.device("cpu"), dtype=torch.float64
    )
    optimizer = torch.optim.Adam(model.vi_parameters(), lr=1.0e-3)

    loss = model._train_step(optimizer, task.X_train, task.y_train)
    samples = predictive_function_samples(model, "sip", task.X_val, 3, seed=123)

    assert torch.isfinite(loss)
    assert samples.shape == (3, task.X_val.shape[0], 2)
    assert model.generative_function.fresh_prior_samples is True


def test_volterra_empirical_gmvip_defaults_to_joint_outputs(tmp_path):
    _write_small_lv_dataset(tmp_path)
    config = copy.deepcopy(SMOKE_LOTKA_VOLTERRA_CONFIG)
    config["data"]["root"] = str(tmp_path)
    task = load_lotka_volterra_tasks(
        tmp_path,
        seed=0,
        n_eval_targets=1,
        n_train_times=6,
        prior_bank_size=16,
        dtype=torch.float64,
    )[0]

    model = build_model(
        "gmvip_empirical",
        task,
        config,
        seed=0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    samples = predictive_function_samples(model, "gmvip_empirical", task.X_val, 4, seed=123)
    loss, _ = model.elbo_loss(
        task.X_train,
        task.y_train,
        num_samples=4,
        num_data=task.X_train.shape[0],
    )
    loss.backward()

    joint_dim = int(config["gmvip"]["num_inducing"]) * 2
    assert config["gmvip"]["joint_output_covariance"] is True
    assert model.joint_output_covariance is True
    assert model.operator.joint_outputs is True
    assert model.operator.K_ZZ.shape == (joint_dim, joint_dim)
    assert model.coefficients.scale_tril.shape == (joint_dim, joint_dim)
    assert samples.shape == (4, task.X_val.shape[0], 2)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(loss)
    assert model.coefficients.raw_scale_tril.grad is not None


def test_volterra_training_free_prior_and_empirical_gp_baselines(tmp_path):
    _write_small_lv_dataset(tmp_path)
    config = copy.deepcopy(SMOKE_LOTKA_VOLTERRA_CONFIG)
    config["data"]["root"] = str(tmp_path)
    config["empirical_gp"]["bank_size"] = 16
    task = load_lotka_volterra_tasks(
        tmp_path,
        seed=0,
        n_eval_targets=1,
        n_train_times=6,
        prior_bank_size=16,
        dtype=torch.float64,
    )[0]

    for method in ("analog_prior", "empirical_gp"):
        model = build_model(
            method,
            task,
            config,
            seed=0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        info = fit_model(model, method, task, config, seed=0, device=torch.device("cpu"))
        samples = predictive_function_samples(model, method, task.X_val, 8, seed=123)

        assert info["steps"] == 0
        assert samples.shape == (8, task.X_val.shape[0], 2)
        assert torch.isfinite(samples).all()

    assert model.cross_covariance.shape == (2 * task.X_plot.shape[0], 2 * task.X_train.shape[0])
    query_idx = model._query_indices(task.X_val)
    weights = torch.cholesky_solve(
        (model.observed_targets - model.observed_mean).unsqueeze(-1),
        model.observed_cholesky,
    ).squeeze(-1)
    expected_mean = model.mean[query_idx] + model.cross_covariance[query_idx] @ weights
    empirical_mean = model.predict_f_samples(task.X_val, 8192, seed=456).mean(dim=0).reshape(-1)
    assert torch.allclose(empirical_mean, expected_mean, atol=4.0e-2, rtol=3.0e-2)


def test_volterra_gmvip_surrogate_prior_is_standard_normal_and_training_free(tmp_path):
    _write_small_lv_dataset(tmp_path)
    config = copy.deepcopy(SMOKE_LOTKA_VOLTERRA_CONFIG)
    config["data"]["root"] = str(tmp_path)
    task = load_lotka_volterra_tasks(
        tmp_path,
        seed=0,
        n_eval_targets=1,
        n_train_times=6,
        prior_bank_size=16,
        dtype=torch.float64,
    )[0]

    model = build_model(
        "gmvip_surrogate_prior",
        task,
        config,
        seed=0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    info = fit_model(
        model,
        "gmvip_surrogate_prior",
        task,
        config,
        seed=0,
        device=torch.device("cpu"),
    )
    samples = predictive_function_samples(
        model, "gmvip_surrogate_prior", task.X_val, 8, seed=123
    )

    joint_dim = int(config["gmvip"]["num_inducing"]) * 2
    assert info["steps"] == 0
    assert torch.count_nonzero(model.coefficients.mean) == 0
    assert torch.allclose(
        model.coefficients.scale_tril,
        torch.eye(joint_dim, dtype=torch.float64),
    )
    assert samples.shape == (8, task.X_val.shape[0], 2)
    assert torch.isfinite(samples).all()
