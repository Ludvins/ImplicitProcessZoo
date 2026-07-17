"""Recompute Volterra diagnostics on the held-out test partition only."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.common import (
    oscillation_period_error,
    peak_time_error,
    phase_lag_error,
    positivity_violation_rate,
)
from experiments.volterra.datasets import load_lotka_volterra_tasks
from experiments.volterra.metrics import (
    lotka_volterra_residual_score,
    nearest_prior_mse,
)

RECOMPUTED_METRICS = (
    "nearest_prior_mse",
    "nearest_prior_mse_median",
    "ode_residual",
    "prey_peak_time_error",
    "predator_peak_time_error",
    "oscillation_period_error",
    "prey_predator_phase_lag_error",
    "positivity_violation_rate",
)


def _backup(path: Path) -> None:
    backup = path.with_name(f"{path.stem}.pre_test_only{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)


def _load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _task_cache_key(config: dict, seed: int, method: str) -> tuple:
    data = config.get("data", {})
    prior = config.get("prior", {})
    bank_size = prior.get("bank_size")
    if method == "oracle_prior_bank":
        bank_size = config.get("oracle_prior_bank", {}).get("bank_size")
    return (
        str(data.get("root", "data/simprior/lotka_volterra")),
        int(seed),
        int(data.get("n_eval_targets", 20)),
        int(data.get("n_train_times", 80)),
        float(data.get("noise_scale", 0.03)),
        None if bank_size is None else int(bank_size),
    )


def _load_tasks(config: dict, seed: int, method: str):
    key = _task_cache_key(config, seed, method)
    root, run_seed, n_targets, n_train, noise_scale, bank_size = key
    tasks = load_lotka_volterra_tasks(
        root,
        seed=run_seed,
        n_eval_targets=n_targets,
        n_train_times=n_train,
        noise_scale=noise_scale,
        prior_bank_size=bank_size,
        device="cpu",
        dtype=torch.float64,
    )
    return key, tasks


def _float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def recompute_run(
    run_dir: Path,
    *,
    task_cache: dict[tuple, list],
    prior_cache: dict[tuple, torch.Tensor],
) -> dict:
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    rows_path = run_dir / "metrics_per_target.csv"
    if not (config_path.exists() and metrics_path.exists() and rows_path.exists()):
        raise FileNotFoundError(f"Incomplete Volterra result directory: {run_dir}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    method = str(metrics["method"])
    run_seed = int(metrics["seed"])
    cache_key = _task_cache_key(config, run_seed, method)
    if cache_key not in task_cache:
        _, task_cache[cache_key] = _load_tasks(config, run_seed, method)
    tasks = task_cache[cache_key]

    rows, fieldnames = _load_rows(rows_path)
    for metric in RECOMPUTED_METRICS:
        if metric not in fieldnames:
            fieldnames.append(metric)
    if "metric_partition" not in fieldnames:
        fieldnames.append("metric_partition")

    for row in rows:
        target_id = int(row["target_id"])
        task = tasks[target_id]
        prediction_path = run_dir / "predictions" / f"target_{target_id}.npz"
        with np.load(prediction_path) as payload:
            samples = torch.as_tensor(payload["samples"], dtype=torch.float64)
            truth = torch.as_tensor(payload["y_true"], dtype=torch.float64)
            t_grid = torch.as_tensor(payload["t_plot"], dtype=torch.float64)

        test_idx = torch.as_tensor(task.metadata["test_indices"], dtype=torch.long)
        test_samples = samples[:, test_idx]
        test_truth = truth[test_idx]
        test_t = t_grid[test_idx]

        target_seed = int(row["seed"])
        prior_key = (*cache_key, target_id, target_seed)
        if prior_key not in prior_cache:
            prior_ids = task.prior.sample_indices(
                min(512, task.prior.num_paths), seed=target_seed + 701
            )
            prior_cache[prior_key] = task.prior.evaluate_raw(task.X_test, prior_ids).cpu()
        prior_test = prior_cache[prior_key]

        nearest = nearest_prior_mse(test_samples[:128], prior_test, chunk_size=32)
        residual = lotka_volterra_residual_score(test_samples[:64], test_t)
        values = {
            "nearest_prior_mse": _float(nearest["mean"]),
            "nearest_prior_mse_median": _float(nearest["median"]),
            "ode_residual": _float(residual.mean()),
            "prey_peak_time_error": _float(
                peak_time_error(test_samples, test_truth, test_t, channel=0)
            ),
            "predator_peak_time_error": _float(
                peak_time_error(test_samples, test_truth, test_t, channel=1)
            ),
            "oscillation_period_error": _float(
                oscillation_period_error(test_samples, test_truth, test_t, channels=(0, 1))
            ),
            "prey_predator_phase_lag_error": _float(
                phase_lag_error(test_samples, test_truth, test_t)
            ),
            "positivity_violation_rate": _float(positivity_violation_rate(test_samples)),
        }
        row.update({key: repr(value) for key, value in values.items()})
        row["metric_partition"] = "test_(20,30]"

    _backup(rows_path)
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = metrics.setdefault("summary", {})
    for key in RECOMPUTED_METRICS:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "stderr": float(np.nanstd(values) / max(1.0, np.sqrt(values.size))),
        }
    metrics["metric_partition"] = "test_(20,30]"
    metrics["recomputed_from_saved_predictions"] = True
    _backup(metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "method": method,
        "targets": len(rows),
        "metric_partition": metrics["metric_partition"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute Volterra diagnostic metrics on the test interval (20, 30]."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict]:
    args = parse_args(argv)
    task_cache: dict[tuple, list] = {}
    prior_cache: dict[tuple, torch.Tensor] = {}
    results = [
        recompute_run(run_dir, task_cache=task_cache, prior_cache=prior_cache)
        for run_dir in args.run_dirs
    ]
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
