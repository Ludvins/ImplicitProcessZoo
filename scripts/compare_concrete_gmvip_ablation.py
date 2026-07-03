"""Summarize Concrete GMVIP ablation JSON results."""

import argparse
import csv
import json
from pathlib import Path


METRICS = ("RMSE", "NLL", "CRPS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Concrete GMVIP ablation result JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--dataset", default="concrete")
    parser.add_argument("--output", default="results/concrete_gmvip_ablation.csv")
    parser.add_argument("--include_baselines", action="store_true")
    return parser.parse_args()


def load_rows(results_dir, dataset, include_baselines):
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        name = path.name.lower()
        if dataset.lower() not in name:
            continue
        if "gmvip" not in name and not include_baselines:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            model = record.get("model_type") or record.get("model") or path.stem
            if model != "gmvip" and not include_baselines:
                continue
            test = record.get("test", {})
            hyper = record.get("hyperparameters", {})
            row = {
                "file": str(path),
                "dataset": record.get("dataset", dataset),
                "model": model,
                "seed": record.get("seed", ""),
                "operator": hyper.get("gmvip_operator_type", ""),
                "posterior": hyper.get("gmvip_posterior_type", ""),
                "learn_Z": hyper.get("gmvip_learn_Z", ""),
                "learn_prior": hyper.get("gmvip_learn_prior", ""),
                "mean_mode": hyper.get("gmvip_mean_mode", ""),
                "inducing_scale": hyper.get("gmvip_inducing_scale", ""),
                "num_inducing": hyper.get("gmvip_num_inducing", ""),
                "num_train_samples": hyper.get("gmvip_num_train_samples", ""),
            }
            for metric in METRICS:
                row[metric] = test.get(metric, "")
            rows.append(row)
    return rows


def sort_key(row):
    return (
        row["model"],
        str(row["operator"]),
        str(row["posterior"]),
        str(row["learn_Z"]),
        str(row["learn_prior"]),
        str(row["seed"]),
    )


def main():
    args = parse_args()
    rows = sorted(
        load_rows(args.results_dir, args.dataset, args.include_baselines),
        key=sort_key,
    )
    if not rows:
        raise SystemExit("No matching result JSON files found.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
