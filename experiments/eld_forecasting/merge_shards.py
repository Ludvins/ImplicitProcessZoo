from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from experiments.common import write_csv_rows, write_json
from experiments.eld_forecasting.run import _summarize


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def merge_method_shards(
    shard_roots: list[str | Path],
    output_root: str | Path,
    *,
    method: str,
    seed: int = 0,
) -> Path:
    if not shard_roots:
        raise ValueError("At least one shard root is required.")
    sources = [Path(root) / method / f"seed_{seed}" for root in shard_roots]
    missing = [str(source) for source in sources if not source.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing method shard directories: {missing}")

    configs = [(source / "config.yaml").read_text(encoding="utf-8") for source in sources]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Shard configs differ; refusing to combine incomparable runs.")

    rows: list[dict[str, str]] = []
    runtimes: list[dict] = []
    target_sources: dict[int, Path] = {}
    for source in sources:
        shard_rows = _read_csv(source / "metrics_per_target_region.csv")
        rows.extend(shard_rows)
        for runtime in json.loads((source / "runtime.json").read_text(encoding="utf-8")):
            target_id = int(runtime["target_id"])
            if target_id in target_sources:
                raise ValueError(f"Target {target_id} occurs in more than one shard.")
            prediction = source / "predictions" / f"target_{target_id}.npz"
            if not prediction.is_file():
                raise FileNotFoundError(f"Missing prediction artifact: {prediction}")
            target_sources[target_id] = prediction
            runtimes.append(runtime)

    row_targets = {int(row["target_id"]) for row in rows}
    if row_targets != set(target_sources):
        raise ValueError("Metric-row targets do not match runtime/prediction targets.")

    destination = Path(output_root) / method / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.yaml").write_text(configs[0], encoding="utf-8")
    for _target_id, prediction in sorted(target_sources.items()):
        _atomic_copy(prediction, destination / "predictions" / prediction.name)

    rows.sort(key=lambda row: (int(row["target_id"]), str(row["region"])))
    runtimes.sort(key=lambda row: int(row["target_id"]))
    write_csv_rows(destination / "metrics_per_target_region.csv", rows)
    write_json(destination / "runtime.json", runtimes)
    write_json(
        destination / "metrics.json",
        {
            "method": method,
            "run_seed": int(seed),
            "targets": sorted(target_sources),
            "summary": _summarize(rows),
            "merged_shards": [str(Path(root)) for root in shard_roots],
        },
    )
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge disjoint ELD method shards.")
    parser.add_argument("--shard-root", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    output = merge_method_shards(
        args.shard_root,
        args.output_root,
        method=args.method,
        seed=args.seed,
    )
    print(output)
    return output


if __name__ == "__main__":
    main()
