import copy

import numpy as np
import torch

from experiments.volterra.datasets import load_lotka_volterra_tasks
from experiments.volterra.generate import generate_bank
from experiments.volterra.priors import LotkaVolterraPrior
from experiments.volterra.run import (
    FreshLotkaVolterraSIPPrior,
    SMOKE_LOTKA_VOLTERRA_CONFIG,
    build_model,
    predictive_function_samples,
)


def _write_small_lv_dataset(root):
    t = np.arange(0.0, 30.0 + 0.5, 0.5, dtype=np.float64)
    target_y, target_theta = generate_bank(2, t_grid=t, seed=240)
    np.savez_compressed(root / "target_paths.npz", t=t, y=target_y, theta=target_theta)


def test_sip_prior_adapter_fresh_draws_change_between_calls():
    t = np.linspace(0.0, 1.0, 5)
    base = LotkaVolterraPrior(t, y_mean=np.zeros((1, 2)), y_std=np.ones((1, 2)), num_samples=3, seed=11)
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
    model = build_model("sip", task, config, seed=0, device=torch.device("cpu"), dtype=torch.float64)
    optimizer = torch.optim.Adam(model.vi_parameters(), lr=1.0e-3)

    loss = model._train_step(optimizer, task.X_train, task.y_train)
    samples = predictive_function_samples(model, "sip", task.X_val, 3, seed=123)

    assert torch.isfinite(loss)
    assert samples.shape == (3, task.X_val.shape[0], 2)
    assert model.generative_function.fresh_prior_samples is True
