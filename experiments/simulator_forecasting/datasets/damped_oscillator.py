from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.simulator_forecasting.interfaces import ForecastingTask
from experiments.simulator_forecasting.priors import DampedOscillatorPrior


def normalize_time(t: np.ndarray | torch.Tensor, t_max: float = 30.0) -> np.ndarray:
    return 2.0 * (np.asarray(t, dtype=np.float64) / float(t_max)) - 1.0


def _metadata_json(root: Path) -> dict:
    path = root / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _interp_scalar(t_grid: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.interp(query.astype(np.float64), t_grid.astype(np.float64), values.astype(np.float64))


def load_damped_oscillator_tasks(
    root: str | Path,
    *,
    seed: int = 0,
    n_eval_targets: int = 20,
    n_train: int = 16,
    t_obs: float = 8.0,
    sigma_y: float | None = None,
    prior_bank_size: int | None = None,
    context_points: int = 128,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> list[ForecastingTask]:
    root = Path(root)
    target_npz = np.load(root / "target_paths.npz")
    t_grid = target_npz["t"].astype(np.float64)
    target_y = target_npz["y"].astype(np.float64)
    target_latents = target_npz["latents"].astype(np.float64)
    if target_y.ndim == 2:
        target_y = target_y[..., None]
    base_meta = _metadata_json(root)
    t_max = float(base_meta.get("t_max", t_grid[-1]))
    t_obs = float(t_obs)
    sigma_y = float(base_meta.get("sigma_y", 0.05) if sigma_y is None else sigma_y)
    target_count = min(int(n_eval_targets), int(target_y.shape[0]))
    reference_bank_size = int(prior_bank_size or base_meta.get("n_prior", 4096))
    device = torch.device(device or "cpu")
    X_plot_np = normalize_time(t_grid, t_max=t_max).reshape(-1, 1)
    context_observed_np = normalize_time(np.linspace(0.0, t_obs, int(context_points)), t_max=t_max).reshape(-1, 1)
    context_full_np = normalize_time(np.linspace(0.0, t_max, int(context_points)), t_max=t_max).reshape(-1, 1)

    tasks: list[ForecastingTask] = []
    for target_id in range(target_count):
        clean = target_y[target_id, :, 0]
        rng = np.random.default_rng(int(seed) + 10_000 * target_id + 97 * int(n_train))
        train_t = np.sort(rng.uniform(0.0, t_obs, size=int(n_train)).astype(np.float64))
        y_train_clean = _interp_scalar(t_grid, clean, train_t).reshape(-1, 1)
        y_train_noisy = y_train_clean + rng.normal(0.0, sigma_y, size=y_train_clean.shape)
        y_mean = y_train_noisy.mean(axis=0, keepdims=True)
        y_std = np.maximum(y_train_noisy.std(axis=0, keepdims=True), 1e-6)
        noise_std_norm = (np.array([[sigma_y]], dtype=np.float64) / y_std).reshape(1)

        def norm_y(values: np.ndarray) -> np.ndarray:
            return (values - y_mean) / y_std

        prior = DampedOscillatorPrior(
            t_grid,
            y_mean=y_mean,
            y_std=y_std,
            num_samples=reference_bank_size,
            reference_bank_size=reference_bank_size,
            seed=int(seed) + 30_000 + target_id,
            forcing_delta=float(base_meta.get("forcing_delta", 0.1)),
            rho=float(base_meta.get("rho", 0.98)),
            sigma_u=float(base_meta.get("sigma_u", 0.05)),
            sample_drag=False,
            device=device,
            dtype=dtype,
        )
        metadata = {
            **base_meta,
            "target_id": int(target_id),
            "n_train": int(n_train),
            "t_obs": t_obs,
            "t_grid": t_grid.tolist(),
            "train_t": train_t.tolist(),
            "test_indices": list(range(t_grid.shape[0])),
            "latent": target_latents[target_id].tolist(),
            "y_mean": y_mean.reshape(1).tolist(),
            "y_std": y_std.reshape(1).tolist(),
            "sigma_y": float(sigma_y),
            "noise_std_norm": noise_std_norm.tolist(),
            "y_train_physical": y_train_noisy.tolist(),
            "y_train_clean_physical": y_train_clean.tolist(),
            "y_test_physical": clean.reshape(-1, 1).tolist(),
            "y_plot_true_physical": clean.reshape(-1, 1).tolist(),
        }
        tasks.append(
            ForecastingTask(
                name=f"damped_oscillator_target_{target_id}_ntrain_{int(n_train)}",
                X_train=torch.as_tensor(normalize_time(train_t, t_max=t_max).reshape(-1, 1), dtype=dtype, device=device),
                y_train=torch.as_tensor(norm_y(y_train_noisy), dtype=dtype, device=device),
                X_test=torch.as_tensor(X_plot_np, dtype=dtype, device=device),
                y_test=torch.as_tensor(norm_y(clean.reshape(-1, 1)), dtype=dtype, device=device),
                X_plot=torch.as_tensor(X_plot_np, dtype=dtype, device=device),
                y_plot_true=torch.as_tensor(norm_y(clean.reshape(-1, 1)), dtype=dtype, device=device),
                X_context_observed=torch.as_tensor(context_observed_np, dtype=dtype, device=device),
                X_context_full=torch.as_tensor(context_full_np, dtype=dtype, device=device),
                noise_std=torch.as_tensor(noise_std_norm, dtype=dtype, device=device),
                prior=prior,
                metadata=metadata,
            )
        )
    return tasks
