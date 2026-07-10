from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.common import write_csv_rows

from .plots import plot_metric_by_region


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _coerce(row: dict) -> dict:
    result = dict(row)
    for key in ("seed", "target_id", "n_train", "train_steps"):
        if key in result and result[key] not in ("", None):
            result[key] = int(float(result[key]))
    for key in (
        "rmse",
        "nlpd",
        "crps",
        "cov90",
        "cov95",
        "width90",
        "width95",
        "train_time_sec",
        "eval_time_sec",
    ):
        if key in result and result[key] not in ("", None):
            result[key] = float(result[key])
    return result


def collect_rows(results_root: str | Path) -> list[dict]:
    root = Path(results_root)
    rows: list[dict] = []
    for path in root.glob("*/seed_*/metrics_per_target_region.csv"):
        rows.extend(_coerce(row) for row in _read_rows(path))
    return rows


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["n_train"], row["region"]), []).append(row)
    fields = ["method", "n_train", "region"]
    metrics = ["rmse", "nlpd", "crps", "cov90", "cov95", "width90", "width95"]
    fields.extend(f"{metric}_mean" for metric in metrics)
    fields.extend(f"{metric}_stderr" for metric in metrics)
    output_rows = []
    for (method, n_train, region), group in sorted(groups.items()):
        out = {"method": method, "n_train": n_train, "region": region}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group if metric in row], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.nanmean(values)) if values.size else np.nan
            out[f"{metric}_stderr"] = (
                float(np.nanstd(values) / max(1.0, np.sqrt(values.size))) if values.size else np.nan
            )
        output_rows.append(out)
    write_csv_rows(path, output_rows, fieldnames=fields)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize simulator forecasting method outputs.")
    parser.add_argument("--results-root", default="results/simprior/simulator_forecasting")
    parser.add_argument("--out", default=None)
    parser.add_argument("--plot-metric", default="nlpd")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = parse_args(argv)
    root = Path(args.results_root)
    out = Path(args.out) if args.out is not None else root / "summary.csv"
    rows = collect_rows(root)
    write_summary(out, rows)
    if rows:
        try:
            plot_metric_by_region(
                root / f"{args.plot_metric}_by_region", rows=rows, metric=args.plot_metric
            )
        except ImportError as exc:
            print(f"Skipping plot generation: {exc}")
    return {"summary": str(out)}


if __name__ == "__main__":
    main()
