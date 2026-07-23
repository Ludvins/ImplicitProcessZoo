import copy
from types import SimpleNamespace

import numpy as np
import torch

from experiments.volterra.benchmark import (
    DEFAULT_LOTKA_VOLTERRA_CONFIG,
    SMOKE_LOTKA_VOLTERRA_CONFIG,
    FreshLotkaVolterraSIPPrior,
    LotkaVolterraPrior,
    _model_noise_std_norm,
    build_model,
    fit_model,
    generate_bank,
    load_lotka_volterra_tasks,
    predictive_function_samples,
)


class _CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.weights_after_steps = []

    def _train_step(self, optimizer, _X, _y):
        optimizer.zero_grad(set_to_none=True)
        target = torch.as_tensor(
            len(self.weights_after_steps) + 1,
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        loss = (self.weight - target).square()
        loss.backward()
        optimizer.step()
        self.weights_after_steps.append(float(self.weight.detach()))
        return loss


def test_volterra_defaults_use_standardized_coefficient_budget():
    config = DEFAULT_LOTKA_VOLTERRA_CONFIG

    assert config["training"]["regression_coeffs"] == 20
    assert config["training"]["max_steps"] == 800
    assert config["training"]["n_mc_eval"] == 1024
    assert config["likelihood"]["learn_observation_noise"] is True
    assert config["ftip"]["warm_start_from_vip"] is True
    assert config["ftip"]["training_overrides"]["regression_coeffs"] == 20
    assert config["ftip"]["training_overrides"]["n_mc_train"] == 8
    assert config["ftip"]["warm_start_training"]["n_mc_train"] == 4
    assert config["ftip"]["warm_start_training"]["max_steps"] == 400
    assert config["ftip"]["fine_tune_training"]["max_steps"] == 400
    assert "early_stopping_patience" not in config["training"]
    assert "eval_interval" not in config["training"]
    assert "early_stopping_patience" not in config["ftip"]["warm_start_training"]
    assert "eval_interval" not in config["ftip"]["warm_start_training"]
    assert "early_stopping_patience" not in config["ftip"]["fine_tune_training"]
    assert "eval_interval" not in config["ftip"]["fine_tune_training"]
    assert config["gmvip"]["prior_bank_size"] == 512
    assert config["gmvip"]["num_inducing"] == 96
    assert config["gmvip"]["joint_output_covariance"] is True
    assert config["gmvip"]["training_overrides"]["max_steps"] == 800
    assert "early_stopping_patience" not in config["gmvip"]["training_overrides"]
    assert "eval_interval" not in config["gmvip"]["training_overrides"]


def test_fit_model_uses_exact_final_scheduled_step():
    model = _CountingModel()
    task = SimpleNamespace(
        X_train=torch.zeros((2, 1), dtype=torch.float64),
        y_train=torch.zeros((2, 1), dtype=torch.float64),
    )
    config = {
        "training": {
            "max_steps": 3,
            "learning_rate": 0.1,
            "batch_size": "full",
            "disable_tqdm": True,
        }
    }

    info = fit_model(model, "counting", task, config, device=torch.device("cpu"))

    assert info["steps"] == 3
    assert info["checkpoint"] == "final_step"
    assert len(model.weights_after_steps) == 3
    assert float(model.weight.detach()) == model.weights_after_steps[-1]


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
    samples = predictive_function_samples(model, "sip", task.X_test, 3, seed=123)

    assert torch.isfinite(loss)
    assert samples.shape == (3, task.X_test.shape[0], 2)
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
    samples = predictive_function_samples(model, "gmvip_empirical", task.X_test, 4, seed=123)
    loss, _ = model.elbo_loss(
        task.X_train,
        task.y_train,
        num_samples=4,
        num_data=task.X_train.shape[0],
    )
    loss.backward()

    joint_dim = int(config["gmvip"]["num_inducing"]) * 2
    expected_bank_size = config["gmvip"].get("prior_bank_size", config["prior"]["bank_size"])
    assert config["gmvip"]["joint_output_covariance"] is True
    assert model.operator.num_bank_samples == expected_bank_size
    assert model.operator.bank_Z.shape[0] == expected_bank_size
    assert model.joint_output_covariance is True
    assert model.operator.joint_outputs is True
    assert model.operator.K_ZZ.shape == (joint_dim, joint_dim)
    assert model.coefficients.scale_tril.shape == (joint_dim, joint_dim)
    assert samples.shape == (4, task.X_test.shape[0], 2)
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
        info = fit_model(model, method, task, config, device=torch.device("cpu"))
        samples = predictive_function_samples(model, method, task.X_test, 8, seed=123)

        assert info["steps"] == 0
        assert info["checkpoint"] == "fixed_predictive"
        assert samples.shape == (8, task.X_test.shape[0], 2)
        assert torch.isfinite(samples).all()

    assert model.cross_covariance.shape == (2 * task.X_plot.shape[0], 2 * task.X_train.shape[0])
    query_idx = model._query_indices(task.X_test)
    weights = torch.cholesky_solve(
        (model.observed_targets - model.observed_mean).unsqueeze(-1),
        model.observed_cholesky,
    ).squeeze(-1)
    expected_mean = model.mean[query_idx] + model.cross_covariance[query_idx] @ weights
    empirical_mean = model.predict_f_samples(task.X_test, 8192, seed=456).mean(dim=0).reshape(-1)
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
        device=torch.device("cpu"),
    )
    samples = predictive_function_samples(model, "gmvip_surrogate_prior", task.X_test, 8, seed=123)

    joint_dim = int(config["gmvip"]["num_inducing"]) * 2
    assert info["steps"] == 0
    assert torch.count_nonzero(model.coefficients.mean) == 0
    assert torch.allclose(
        model.coefficients.scale_tril,
        torch.eye(joint_dim, dtype=torch.float64),
    )
    assert samples.shape == (8, task.X_test.shape[0], 2)
    assert torch.isfinite(samples).all()


def test_ftip_and_vip_use_the_same_prior_basis_seed(tmp_path):
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
    vip = build_model("vip", task, config, seed=17, device=torch.device("cpu"), dtype=torch.float64)
    ftip = build_model(
        "ftip", task, config, seed=17, device=torch.device("cpu"), dtype=torch.float64
    )

    assert vip.generative_function.seed == ftip.generative_function.seed
    assert torch.allclose(
        vip.generative_function.sample_latents(8),
        ftip.generative_function.sample_latents(8),
    )


def test_volterra_learnable_noise_is_per_output_and_initialized_consistently(tmp_path):
    _write_small_lv_dataset(tmp_path)
    config = copy.deepcopy(SMOKE_LOTKA_VOLTERRA_CONFIG)
    config["data"]["root"] = str(tmp_path)
    config["likelihood"]["learn_observation_noise"] = True
    task = load_lotka_volterra_tasks(
        tmp_path,
        seed=0,
        n_eval_targets=1,
        n_train_times=6,
        prior_bank_size=16,
        dtype=torch.float64,
    )[0]

    for method in ("vip", "ftip", "gmvip_empirical"):
        model = build_model(
            method,
            task,
            config,
            seed=0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        learned_noise = _model_noise_std_norm(model, task, config)

        if method == "gmvip_empirical":
            assert model.likelihood.log_noise.requires_grad
        else:
            assert model.log_variance.requires_grad
        assert learned_noise.shape == (2,)
        assert torch.allclose(learned_noise, task.noise_std)
