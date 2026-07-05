from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.simprior.interfaces import SimPriorTask
from experiments.simprior.priors import LotkaVolterraPrior


def normalize_time(t: np.ndarray | torch.Tensor, t_max: float = 30.0) -> np.ndarray:
    return 2.0 * (np.asarray(t, dtype=np.float64) / float(t_max)) - 1.0


def _select_train_indices(t: np.ndarray, n_train_times: int, seed: int) -> np.ndarray:
    train_pool = np.flatnonzero((t >= 0.0) & (t <= 15.0))
    first = int(train_pool[0])
    last = int(train_pool[-1])
    interior = train_pool[(train_pool != first) & (train_pool != last)]
    needed = max(0, int(n_train_times) - 2)
    if needed > interior.size:
        raise ValueError("n_train_times is larger than the available Lotka-Volterra train grid.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(interior, size=needed, replace=False) if needed else np.array([], dtype=int)
    return np.sort(np.concatenate([[first, last], chosen]).astype(int))


def _every_fifth(indices: np.ndarray) -> np.ndarray:
    return indices[::5].astype(int)


def _metadata_json(root: Path) -> dict:
    path = root / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_lotka_volterra_tasks(
    root: str | Path,
    *,
    seed: int = 0,
    n_eval_targets: int = 20,
    n_train_times: int = 40,
    noise_scale: float = 0.03,
    prior_bank_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> list[SimPriorTask]:
    root = Path(root)
    prior_npz = np.load(root / "prior_paths.npz")
    target_npz = np.load(root / "target_paths.npz")
    t = target_npz["t"].astype(np.float64)
    target_y = target_npz["y"].astype(np.float64)
    target_theta = target_npz["theta"].astype(np.float64)
    t_max = float(t[-1])
    target_count = min(int(n_eval_targets), int(target_y.shape[0]))
    base_meta = _metadata_json(root)
    device = torch.device(device or "cpu")

    prior_y = prior_npz["y"]
    prior_theta = prior_npz["theta"] if "theta" in prior_npz.files else None
    if prior_bank_size is not None:
        prior_y = prior_y[: int(prior_bank_size)]
        if prior_theta is not None:
            prior_theta = prior_theta[: int(prior_bank_size)]

    val_idx = _every_fifth(np.flatnonzero((t > 15.0) & (t <= 20.0)))
    test_idx = np.flatnonzero(t > 20.0).astype(int)
    plot_idx = np.arange(t.shape[0], dtype=int)
    X_all = normalize_time(t, t_max=t_max).reshape(-1, 1)

    tasks: list[SimPriorTask] = []
    for target_id in range(target_count):
        clean = target_y[target_id]
        train_idx = _select_train_indices(t, n_train_times, seed + 10_000 * target_id)
        y_train_clean = clean[train_idx]
        train_output_std = np.std(y_train_clean, axis=0, keepdims=True)
        noise_std = np.maximum(float(noise_scale) * train_output_std, 1e-8)
        rng = np.random.default_rng(seed + 20_000 + target_id)
        y_train_noisy = y_train_clean + rng.normal(0.0, noise_std, size=y_train_clean.shape)
        y_mean = y_train_noisy.mean(axis=0, keepdims=True)
        y_std = np.maximum(y_train_noisy.std(axis=0, keepdims=True), 1e-6)
        noise_std_norm = (noise_std / y_std).reshape(2)

        def norm_y(values: np.ndarray) -> np.ndarray:
            return (values - y_mean) / y_std

        prior = LotkaVolterraPrior(
            prior_npz["t"],
            prior_y,
            prior_theta,
            y_mean=y_mean,
            y_std=y_std,
            num_samples=int(prior_bank_size or prior_y.shape[0]),
            seed=seed + 30_000 + target_id,
            device=device,
            dtype=dtype,
        )
        metadata = {
            **base_meta,
            "target_id": int(target_id),
            "theta": target_theta[target_id].tolist(),
            "t_grid": t.tolist(),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "plot_indices": plot_idx.tolist(),
            "y_mean": y_mean.reshape(2).tolist(),
            "y_std": y_std.reshape(2).tolist(),
            "noise_std": noise_std.reshape(2).tolist(),
            "noise_std_norm": noise_std_norm.tolist(),
            "y_train_physical": y_train_noisy.tolist(),
            "y_train_clean_physical": y_train_clean.tolist(),
            "y_val_physical": clean[val_idx].tolist(),
            "y_test_physical": clean[test_idx].tolist(),
            "y_plot_true_physical": clean.tolist(),
        }
        tasks.append(
            SimPriorTask(
                name=f"lotka_volterra_target_{target_id}",
                X_train=torch.as_tensor(X_all[train_idx], dtype=dtype, device=device),
                y_train=torch.as_tensor(norm_y(y_train_noisy), dtype=dtype, device=device),
                X_val=torch.as_tensor(X_all[val_idx], dtype=dtype, device=device),
                y_val=torch.as_tensor(norm_y(clean[val_idx]), dtype=dtype, device=device),
                X_test=torch.as_tensor(X_all[test_idx], dtype=dtype, device=device),
                y_test=torch.as_tensor(norm_y(clean[test_idx]), dtype=dtype, device=device),
                X_plot=torch.as_tensor(X_all[plot_idx], dtype=dtype, device=device),
                y_plot_true=torch.as_tensor(norm_y(clean[plot_idx]), dtype=dtype, device=device),
                noise_std=torch.as_tensor(noise_std_norm, dtype=dtype, device=device),
                prior=prior,
                metadata=metadata,
            )
        )
    return tasks
