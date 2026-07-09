import numpy as np
import torch

from experiments.simulator_forecasting.generate import generate_dataset
from experiments.simulator_forecasting.datasets import load_damped_oscillator_tasks


def test_damped_oscillator_task_generation_and_splits(tmp_path):
    generate_dataset(tmp_path, n_targets=2, n_prior=8, n_test=31, t_max=30.0, seed=0)

    tasks = load_damped_oscillator_tasks(
        tmp_path,
        seed=0,
        n_eval_targets=1,
        n_train=6,
        prior_bank_size=8,
        context_points=10,
        dtype=torch.float64,
    )
    task = tasks[0]
    train_t = np.asarray(task.metadata["train_t"])
    t_grid = np.asarray(task.metadata["t_grid"])

    assert np.all(train_t >= 0.0)
    assert np.all(train_t <= 8.0)
    assert t_grid[0] == 0.0
    assert t_grid[-1] == 30.0
    assert task.X_train.shape == (6, 1)
    assert task.y_train.shape == (6, 1)
    assert task.X_plot.shape == (31, 1)
    assert task.y_plot_true.shape == (31, 1)
    assert task.X_context_observed.shape == (10, 1)
    assert task.X_context_full.shape == (10, 1)


def test_damped_oscillator_task_reuses_same_target_for_same_seed(tmp_path):
    generate_dataset(tmp_path, n_targets=1, n_prior=8, n_test=31, t_max=30.0, seed=3)

    first = load_damped_oscillator_tasks(tmp_path, seed=11, n_eval_targets=1, n_train=4, prior_bank_size=8)[0]
    second = load_damped_oscillator_tasks(tmp_path, seed=11, n_eval_targets=1, n_train=4, prior_bank_size=8)[0]

    assert first.metadata["latent"] == second.metadata["latent"]
    assert first.metadata["train_t"] == second.metadata["train_t"]
    assert torch.allclose(first.y_plot_true, second.y_plot_true)


def test_misspecified_dataset_stores_dragged_targets_but_loader_prior_has_zero_drag(tmp_path):
    generate_dataset(tmp_path, n_targets=3, n_prior=8, n_test=31, t_max=30.0, misspecified=True, seed=5)
    tasks = load_damped_oscillator_tasks(tmp_path, seed=0, n_eval_targets=1, n_train=4, prior_bank_size=8)
    task = tasks[0]

    target_drag = float(task.metadata["latent"][7])
    prior_latents = task.prior.sample_latents(4, seed=9)

    assert 0.02 <= target_drag <= 0.08
    assert torch.allclose(prior_latents[:, 7], torch.zeros(4, dtype=prior_latents.dtype))
