"""Aggregate VIP/FTIP coefficient-ablation result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

IDENTIFIER_COLUMNS = {
    "experiment",
    "method",
    "metric_partition",
    "seed",
    "target_id",
}


def summarize(results_root: Path) -> dict[str, str]:
    frames = []
    coefficient_dirs = [
        path
        for path in results_root.glob("s*")
        if path.is_dir() and path.name.removeprefix("s").isdigit()
    ]
    for coefficient_dir in sorted(
        coefficient_dirs, key=lambda path: int(path.name.removeprefix("s"))
    ):
        coefficients = int(coefficient_dir.name.removeprefix("s"))
        for path in sorted(
            coefficient_dir.glob("lotka_volterra/*/seed_*/metrics_per_target.csv")
        ):
            frame = pd.read_csv(path)
            frame.insert(0, "regression_coeffs", coefficients)
            frame.insert(1, "run_seed", int(path.parent.name.removeprefix("seed_")))
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No coefficient-ablation metrics found under {results_root}")

    combined = pd.concat(frames, ignore_index=True)
    group_columns = ["method", "regression_coeffs"]
    numeric_columns = [
        column
        for column in combined.select_dtypes(include="number").columns
        if column not in IDENTIFIER_COLUMNS | {"regression_coeffs", "run_seed"}
    ]
    grouped = combined.groupby(group_columns, sort=True)[numeric_columns]
    mean = grouped.mean().add_suffix("_mean").reset_index()
    standard_deviation = grouped.std(ddof=1).add_suffix("_std")
    mean_std = pd.concat(
        [grouped.mean().add_suffix("_mean"), standard_deviation], axis=1
    ).reset_index()
    median = grouped.median().add_suffix("_median")
    q25 = grouped.quantile(0.25).add_suffix("_q25")
    q75 = grouped.quantile(0.75).add_suffix("_q75")
    median_iqr = pd.concat([median, q25, q75], axis=1).reset_index()

    combined_path = results_root / "metrics_per_target_all.csv"
    mean_path = results_root / "summary_mean.csv"
    mean_std_path = results_root / "summary_mean_std.csv"
    median_path = results_root / "summary_median_iqr.csv"
    manifest_path = results_root / "ablation_manifest.json"
    combined.to_csv(combined_path, index=False)
    mean.to_csv(mean_path, index=False)
    mean_std.to_csv(mean_std_path, index=False)
    median_iqr.to_csv(median_path, index=False)

    manifest = {
        "experiment": "lotka_volterra",
        "methods": sorted(combined["method"].unique().tolist()),
        "regression_coeffs": sorted(combined["regression_coeffs"].unique().tolist()),
        "run_seeds": sorted(combined["run_seed"].unique().tolist()),
        "targets_per_configuration": int(
            combined.groupby(group_columns + ["run_seed"]).size().min()
        ),
        "vip_steps": 400,
        "ftip_warm_start_vip_steps": 400,
        "ftip_additional_steps": 400,
        "metric_partition": "test_(20,30]",
        "outputs": {
            "per_target": str(combined_path),
            "mean": str(mean_path),
            "mean_std": str(mean_std_path),
            "median_iqr": str(median_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in {
        "per_target": combined_path,
        "mean": mean_path,
        "mean_std": mean_std_path,
        "median_iqr": median_path,
        "manifest": manifest_path,
    }.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/volterra_coeff_ablation")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = parse_args(argv)
    outputs = summarize(args.results_root)
    print(json.dumps(outputs, indent=2))
    return outputs


if __name__ == "__main__":
    main()
