from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
        "cov90",
        "cov95",
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
