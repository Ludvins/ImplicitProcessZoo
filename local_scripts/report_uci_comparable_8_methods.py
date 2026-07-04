"""Aggregate comparable 8-method UCI result JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results" / "uci_comparable_8_methods_30k"
DEFAULT_REPORT = REPO / "outputs" / "reports" / "uci_comparable_8_methods_30k_summary.md"
METRICS = ("RMSE", "NLL", "CRPS", "CQM")


def pretty_dataset(dataset: str) -> str:
    return {"boston": "Boston", "concrete": "Concrete", "energy": "Energy"}.get(
        dataset,
        dataset.title(),
    )


def method_label(result: dict) -> str:
    model = result.get("model")
    hp = result.get("hyperparameters", {})
    if model == "map":
        return "MAP"
    if model == "mfvi":
        return "MFVI"
    if model == "fbnn":
        return "FBNN"
    if model == "tfsvi":
        return "TFSVI"
    if model == "vip":
        prior = "Tunable Prior" if hp.get("vip_learn_prior", True) else "Fixed Prior"
        return f"VIP {prior}"
    if model == "ftip":
        prior = "Tunable Prior" if hp.get("ftip_learn_prior", True) else "Fixed Prior"
        return f"FTIP {prior}"
    if model == "gmvip":
        prior = "Tunable Prior" if hp.get("gmvip_learn_prior", False) else "Fixed Prior"
        return f"GMVIP {prior}"
    if model == "sip":
        prior = "Tunable Prior" if hp.get("sip_learn_prior", True) else "Fixed Prior"
        return f"SIP {prior}"
    return str(model).upper()


def load_results(path: Path) -> list[dict]:
    records: list[dict] = []
    for file in sorted(path.glob("*.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            item["_source_file"] = str(file)
            records.append(item)
    return records


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(pretty_dataset(record["dataset"]), method_label(record))].append(record)

    rows: list[dict] = []
    for (dataset, label), items in grouped.items():
        row = {"dataset": dataset, "method": label, "n": len(items)}
        seeds = sorted(item.get("hyperparameters", {}).get("seed") for item in items)
        row["seeds"] = seeds
        for metric in METRICS:
            values = [
                float(item["test"][metric])
                for item in items
                if metric in item.get("test", {}) and item["test"][metric] is not None
            ]
            if not values:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                continue
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return sorted(rows, key=lambda row: (row["dataset"], row["RMSE_mean"] if row["RMSE_mean"] is not None else float("inf")))


def render_markdown(rows: list[dict]) -> str:
    lines = ["# UCI Comparable 8-Method Summary", ""]
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset in sorted(by_dataset):
        lines.extend([f"## {dataset}", ""])
        lines.append("| Method | n | seeds | RMSE | NLL | CRPS | CQM |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
        for row in by_dataset[dataset]:
            def fmt(metric: str) -> str:
                mean = row.get(f"{metric}_mean")
                std = row.get(f"{metric}_std")
                if mean is None:
                    return ""
                return f"{mean:.4f} +/- {std:.4f}"

            seeds = ",".join(str(seed) for seed in row["seeds"])
            lines.append(
                f"| {row['method']} | {row['n']} | {seeds} | "
                f"{fmt('RMSE')} | {fmt('NLL')} | {fmt('CRPS')} | {fmt('CQM')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    records = load_results(args.results_dir)
    rows = summarize(records)
    report = render_markdown(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
