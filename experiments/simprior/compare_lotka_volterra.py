from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.simprior.plots import DEFAULT_METHOD_LABELS, plot_lv_shared_axis_method_comparison


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_target_ids(value: str | None) -> list[int]:
    return [int(item) for item in _parse_csv_list(value)]


def _read_metric_rows(results_root: Path, method: str, seed: int) -> dict[int, dict[str, float | str]]:
    path = results_root / method / f"seed_{seed}" / "metrics_per_target.csv"
    rows: dict[int, dict[str, float | str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"experiment", "method"}:
                    parsed[key] = value
                elif value == "":
                    parsed[key] = np.nan
                else:
                    parsed[key] = float(value)
            rows[int(parsed["target_id"])] = parsed
    return rows


def _metric_score(row: dict[str, float | str], metric: str) -> float:
    value = float(row[metric])
    if metric.startswith("cov"):
        nominal = float(metric.removeprefix("cov")) / 100.0
        return abs(value - nominal)
    return value


def rank_gmvip_win_loss_targets(
    metrics_by_method: dict[str, dict[int, dict[str, float | str]]],
    *,
    gmvip_method: str,
    selection_methods: list[str],
    metric: str,
    n: int,
) -> dict[str, list[dict[str, object]]]:
    common_targets = sorted(set.intersection(*(set(metrics_by_method[method]) for method in selection_methods)))
    wins: list[dict[str, object]] = []
    losses: list[dict[str, object]] = []
    for target_id in common_targets:
        scores = {
            method: _metric_score(metrics_by_method[method][target_id], metric)
            for method in selection_methods
        }
        ranked = sorted(selection_methods, key=lambda method: scores[method])
        best = ranked[0]
        gmvip_score = scores[gmvip_method]
        if best == gmvip_method:
            runner_up_score = scores[ranked[1]] if len(ranked) > 1 else gmvip_score
            margin = runner_up_score - gmvip_score
            wins.append(
                {
                    "target_id": target_id,
                    "best_method": best,
                    "margin": float(margin),
                    "scores": {method: float(scores[method]) for method in selection_methods},
                }
            )
        else:
            margin = gmvip_score - scores[best]
            losses.append(
                {
                    "target_id": target_id,
                    "best_method": best,
                    "margin": float(margin),
                    "scores": {method: float(scores[method]) for method in selection_methods},
                }
            )
    wins.sort(key=lambda item: float(item["margin"]), reverse=True)
    losses.sort(key=lambda item: float(item["margin"]), reverse=True)
    return {"wins": wins[:n], "losses": losses[:n]}


def _load_prediction(results_root: Path, method: str, seed: int, target_id: int) -> dict[str, np.ndarray]:
    path = results_root / method / f"seed_{seed}" / "predictions" / f"target_{target_id}.npz"
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _plot_target(
    *,
    results_root: Path,
    out_dir: Path,
    plot_methods: list[str],
    seed: int,
    target_id: int,
    title: str,
    stem: str,
) -> Path:
    predictions = {
        method: _load_prediction(results_root, method, seed, target_id)
        for method in plot_methods
    }
    path_base = out_dir / stem
    plot_lv_shared_axis_method_comparison(
        path_base,
        predictions_by_method=predictions,
        methods=plot_methods,
        title=title,
    )
    return path_base.with_suffix(".png")


def build_comparison_plots(args: argparse.Namespace) -> dict[str, object]:
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else results_root / "shared_axes"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_methods = _parse_csv_list(args.plot_methods)
    selection_methods = _parse_csv_list(args.selection_methods)
    target_ids = _parse_target_ids(args.target_ids)
    metrics_by_method = {
        method: _read_metric_rows(results_root, method, int(args.seed))
        for method in sorted(set(selection_methods + [args.gmvip_method]))
    }

    written: list[str] = []
    for target_id in target_ids:
        label = f"target {target_id}"
        path = _plot_target(
            results_root=results_root,
            out_dir=out_dir,
            plot_methods=plot_methods,
            seed=int(args.seed),
            target_id=target_id,
            title=f"Lotka-Volterra shared-axis comparison, {label}",
            stem=f"target_{target_id}_shared_axes",
        )
        written.append(str(path))

    ranking = rank_gmvip_win_loss_targets(
        metrics_by_method,
        gmvip_method=args.gmvip_method,
        selection_methods=selection_methods,
        metric=args.metric,
        n=int(args.n_win_loss),
    )
    rank_path = out_dir / f"{args.gmvip_method}_{args.metric}_win_loss_seed_{args.seed}.json"
    rank_path.write_text(json.dumps(ranking, indent=2), encoding="utf-8")

    gmvip_label = DEFAULT_METHOD_LABELS.get(args.gmvip_method, args.gmvip_method)
    for kind in ("wins", "losses"):
        kind_singular = "win" if kind == "wins" else "loss"
        for entry in ranking[kind]:
            target_id = int(entry["target_id"])
            best_method = str(entry["best_method"])
            best_label = DEFAULT_METHOD_LABELS.get(best_method, best_method)
            title = (
                f"{gmvip_label} {kind_singular} on {args.metric}, target {target_id}"
                if kind == "wins"
                else f"{gmvip_label} loss on {args.metric}, target {target_id}; best: {best_label}"
            )
            path = _plot_target(
                results_root=results_root,
                out_dir=out_dir,
                plot_methods=plot_methods,
                seed=int(args.seed),
                target_id=target_id,
                title=title,
                stem=f"{args.gmvip_method}_{args.metric}_{kind_singular}_target_{target_id}",
            )
            written.append(str(path))

    summary = {"ranking": ranking, "ranking_path": str(rank_path), "figures": written}
    summary_path = out_dir / f"shared_axis_summary_seed_{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared-axis Lotka-Volterra comparison plots.")
    parser.add_argument("--results-root", default="results/simprior/lotka_volterra")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-methods", default="vip,gmvip_empirical,ftip,oracle_prior_bank")
    parser.add_argument("--selection-methods", default="vip,gmvip_empirical,ftip")
    parser.add_argument("--gmvip-method", default="gmvip_empirical")
    parser.add_argument("--metric", default="rmse")
    parser.add_argument("--n-win-loss", type=int, default=3)
    parser.add_argument("--target-ids", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    summary = build_comparison_plots(args)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
