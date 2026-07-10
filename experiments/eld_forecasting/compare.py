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
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    if "methodology_version" not in data:
        raise ValueError("ELD methodology-v1 and v2 results cannot be mixed; use a v2-only root.")
    data = data[data["methodology_version"] == 2]
    if data.empty:
        raise ValueError("No methodology-version 2 ELD rows were found under the selected root.")
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
    for keys, group in data.groupby(["method", "region", "stress"], dropna=False):
        method, region, stress = keys
        row = {"method": method, "region": region, "stress": stress, "n": len(group)}
        for metric in metrics:
            if metric in group:
                values = group[metric].to_numpy(dtype=np.float64)
                row[f"{metric}_mean"] = float(np.nanmean(values))
                row[f"{metric}_stderr"] = float(
                    np.nanstd(values) / max(1.0, np.sqrt(np.isfinite(values).sum()))
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["region", "stress", "method"]).reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ELD forecasting metrics.")
    parser.add_argument("--results-root", default="results/eld_forecasting_v2")
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
