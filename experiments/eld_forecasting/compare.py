from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _backfill_interval_metrics(
    frame: pd.DataFrame,
    method_dir: Path,
    *,
    levels: tuple[float, ...] = (0.8,),
) -> pd.DataFrame:
    """Derive newly requested interval metrics from saved prediction samples."""
    missing_levels = [
        level
        for level in levels
        if f"cov{int(round(100 * level))}" not in frame
        or f"width{int(round(100 * level))}" not in frame
    ]
    required = {"target_id", "region_start_idx", "region_stop_idx"}
    if not missing_levels or not required.issubset(frame.columns):
        return frame

    result = frame.copy()
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, row in result.iterrows():
        target_id = int(row["target_id"])
        if target_id not in cache:
            prediction_path = method_dir / "predictions" / f"target_{target_id}.npz"
            if not prediction_path.is_file():
                continue
            with np.load(prediction_path) as prediction:
                cache[target_id] = (
                    np.asarray(prediction["samples"], dtype=np.float64),
                    np.asarray(prediction["truth"], dtype=np.float64),
                )
        samples, truth = cache[target_id]
        start = int(row["region_start_idx"])
        stop = int(row["region_stop_idx"])
        region_samples = samples[:, start:stop]
        region_truth = truth[start:stop]
        for level in missing_levels:
            suffix = int(round(100 * level))
            alpha = 0.5 * (1.0 - level)
            lower = np.quantile(region_samples, alpha, axis=0)
            upper = np.quantile(region_samples, 1.0 - alpha, axis=0)
            result.loc[index, f"cov{suffix}"] = float(
                np.mean((region_truth >= lower) & (region_truth <= upper))
            )
            result.loc[index, f"width{suffix}"] = float(np.mean(upper - lower))
    return result


def aggregate(results_root: str | Path) -> pd.DataFrame:
    root = Path(results_root)
    paths = list(root.glob("*/*/metrics_per_target_region.csv"))
    if not paths:
        raise FileNotFoundError(f"No ELD metrics_per_target_region.csv files found under {root}.")
    configs = [(path.parent / "config.yaml").read_text(encoding="utf-8") for path in paths]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("ELD result configurations differ; refusing to aggregate them.")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = _backfill_interval_metrics(frame, path.parent)
        if "run_seed" not in frame:
            try:
                frame["run_seed"] = int(path.parent.name.removeprefix("seed_"))
            except ValueError as error:
                raise ValueError(f"Cannot infer run seed from {path}.") from error
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    duplicate_keys = ["method", "run_seed", "target_id", "region"]
    if data.duplicated(duplicate_keys).any():
        raise ValueError("ELD results contain duplicate method/seed/target/region rows.")
    identity_counts = data.groupby(["run_seed", "target_id"])[["client_id", "start_time"]].nunique()
    if bool((identity_counts > 1).any().any()):
        raise ValueError("ELD methods do not share identical target identities.")
    target_sets = {
        method: set(zip(group["run_seed"], group["target_id"], group["region"]))
        for method, group in data.groupby("method")
    }
    reference_targets = next(iter(target_sets.values()))
    if any(targets != reference_targets for targets in target_sets.values()):
        raise ValueError("ELD methods do not contain identical seed/target/region rows.")
    metrics = [
        "rmse",
        "nll",
        "crps",
        "cqm",
        "cov80",
        "cov90",
        "cov95",
        "width80",
        "width90",
        "width95",
        "peak_magnitude_error",
        "peak_timing_error_hours",
        "train_time_sec",
        "eval_time_sec",
    ]
    rows = []
    seed_groups = [("all", data)] + [
        (str(int(run_seed)), group) for run_seed, group in data.groupby("run_seed")
    ]
    for run_seed, seed_group in seed_groups:
        for (method, region), group in seed_group.groupby(["method", "region"]):
            row = {
                "run_seed": run_seed,
                "method": method,
                "region": region,
                "n": len(group),
            }
            for metric in metrics:
                if metric not in group:
                    continue
                values = group[metric].to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                if not finite.size:
                    continue
                row[f"{metric}_median"] = float(np.median(finite))
                row[f"{metric}_q25"] = float(np.quantile(finite, 0.25))
                row[f"{metric}_q75"] = float(np.quantile(finite, 0.75))
                row[f"{metric}_mean"] = float(np.mean(finite))
            rows.append(row)
    result = pd.DataFrame(rows)
    result["_seed_order"] = result["run_seed"].map(
        lambda value: -1 if str(value) == "all" else int(value)
    )
    return (
        result.sort_values(["_seed_order", "region", "method"])
        .drop(columns="_seed_order")
        .reset_index(drop=True)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ELD forecasting metrics.")
    parser.add_argument("--results-root", default="results/eld_forecasting")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    summary = aggregate(args.results_root)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
