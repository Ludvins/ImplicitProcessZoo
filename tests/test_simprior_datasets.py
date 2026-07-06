import numpy as np
import torch

from experiments.volterra.datasets import load_lotka_volterra_tasks
from experiments.volterra.generate import generate_bank


def _write_small_lv_dataset(root):
    t = np.arange(0.0, 30.0 + 0.5, 0.5, dtype=np.float64)
    target_y, target_theta = generate_bank(3, t_grid=t, seed=200)
    np.savez_compressed(root / "target_paths.npz", t=t, y=target_y, theta=target_theta)
    return target_theta


def test_lotka_volterra_task_uses_live_ode_prior_without_prior_bank(tmp_path):
    _write_small_lv_dataset(tmp_path)

    tasks = load_lotka_volterra_tasks(
        tmp_path,
        seed=0,
        n_eval_targets=1,
        n_train_times=6,
        prior_bank_size=6,
        dtype=torch.float64,
    )
    task = tasks[0]
    latents = task.prior.sample_latents(3, seed=123)
    values = task.prior.evaluate(task.X_train[:2], latents)

    assert latents.shape == (3, 6)
    assert values.shape == (3, 2, 2)


def test_lotka_volterra_task_splits_and_normalization(tmp_path):
    _write_small_lv_dataset(tmp_path)

    tasks = load_lotka_volterra_tasks(
        tmp_path,
        seed=0,
        n_eval_targets=1,
        n_train_times=6,
        prior_bank_size=6,
        dtype=torch.float64,
    )
    task = tasks[0]
    train_idx = np.asarray(task.metadata["train_indices"])
    t_grid = np.asarray(task.metadata["t_grid"])
    y_mean = torch.tensor(task.metadata["y_mean"], dtype=torch.float64).reshape(1, 2)
    y_std = torch.tensor(task.metadata["y_std"], dtype=torch.float64).reshape(1, 2)

    assert t_grid[train_idx[0]] == 0.0
    assert t_grid[train_idx[-1]] <= 15.0
    assert task.X_train.shape == (6, 1)
    assert task.y_train.shape == (6, 2)
    assert torch.allclose(task.y_train.mean(dim=0), torch.zeros(2, dtype=torch.float64), atol=1e-12)

    reconstructed_val = task.y_val * y_std + y_mean
    expected_val = torch.tensor(task.metadata["y_val_physical"], dtype=torch.float64)
    reconstructed_test = task.y_test * y_std + y_mean
    expected_test = torch.tensor(task.metadata["y_test_physical"], dtype=torch.float64)

    assert torch.allclose(reconstructed_val, expected_val)
    assert torch.allclose(reconstructed_test, expected_test)
