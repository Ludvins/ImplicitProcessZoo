from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHOD_LABELS = {
    "analog_prior": "Prior predictive",
    "gmvip_surrogate_prior": "GMVIP surrogate prior",
    "empirical_gp": "Empirical GP",
    "map": "MAP",
    "mfvi": "MFVI",
    "vip": "VIP",
    "ftip": "FTIP",
    "sip": "SIP",
    "gmvip_empirical": "GMVIP",
    "gmvip_rbf": "GMVIP-RBF",
    "oracle_prior_bank": "Oracle",
}
METRICS = (
    ("rmse", "RMSE $\\downarrow$"),
    ("nll", "NLL $\\downarrow$"),
    ("crps", "CRPS $\\downarrow$"),
    ("cov90", "Cov$_{90}$"),
    ("ode_residual", "ODE residual $\\downarrow$"),
)
FIXED_NLL_DEFINITION = "equal_weight_gaussian_mixture_with_fixed_observation_variance"
LEARNED_NLL_DEFINITION = "equal_weight_gaussian_mixture_with_learned_observation_variance"
LEARNABLE_NOISE_METHODS = {
    "map",
    "mfvi",
    "vip",
    "ftip",
    "sip",
    "gmvip_empirical",
    "gmvip_rbf",
}
SPECIES_COLORS = ("#4C78A8", "#F58518")
SPECIES_MEAN_COLORS = ("#1F5A93", "#D95F02")


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _parse_csv(raw: str, cast=str) -> tuple:
    values = tuple(cast(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values:
        raise ValueError("A comma-separated argument was empty.")
    return values


def _parse_target_ids(raw: str) -> tuple[int, ...]:
    ids: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start, stop = token.split(":", 1)
            ids.extend(range(int(start), int(stop)))
        elif "-" in token:
            start, stop = token.split("-", 1)
            ids.extend(range(int(start), int(stop) + 1))
        else:
            ids.append(int(token))
    if not ids:
        raise ValueError("--target-ids must select at least one target.")
    return tuple(dict.fromkeys(ids))


def _method_dir(root: Path, seed: int, basis_size: int, method: str) -> Path:
    return root / f"seed_{seed}" / f"S_{basis_size}" / method


def _expected_nll_definition(method: str, noise_mode: str) -> str:
    if noise_mode == "learned" and method in LEARNABLE_NOISE_METHODS:
        return LEARNED_NLL_DEFINITION
    return FIXED_NLL_DEFINITION


def _load_manifest(
    root: Path,
    seed: int,
    basis_size: int,
    method: str,
    *,
    noise_mode: str = "learned",
) -> dict:
    method_dir = _method_dir(root, seed, basis_size, method)
    path = method_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "experiment": "lotka_volterra",
        "method": method,
        "seed": int(seed),
        "vip_basis_size": int(basis_size),
        "nll": _expected_nll_definition(method, noise_mode),
        "checkpoint_selection": "none_final_step_only",
        "data_usage": {
            "training": "t<=15",
            "unused_gap": "15<t<=20",
            "test": "20<t<=30",
        },
        "status": "complete",
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Incompatible manifest {path}: {mismatches}")
    if not manifest.get("protocol_hash"):
        raise RuntimeError(f"Manifest has no protocol hash: {path}")
    return manifest


def _validate_manifests(
    root: Path,
    seed: int,
    selections: list[tuple[int, str]],
    *,
    noise_mode: str = "learned",
) -> dict[tuple[int, str], dict]:
    manifests = {
        (basis_size, method): _load_manifest(
            root,
            seed,
            basis_size,
            method,
            noise_mode=noise_mode,
        )
        for basis_size, method in selections
    }
    dataset_hashes = {manifest["dataset"]["sha256"] for manifest in manifests.values()}
    if len(dataset_hashes) != 1:
        raise RuntimeError("Selected results were produced from different target datasets.")
    return manifests


def _load_prediction(
    root: Path,
    *,
    seed: int,
    basis_size: int,
    method: str,
    target_id: int,
) -> dict[str, np.ndarray]:
    path = _method_dir(root, seed, basis_size, method) / "predictions" / f"target_{target_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction artifact: {path}")
    with np.load(path) as payload:
        prediction = {key: payload[key] for key in payload.files}
    if int(prediction["evaluation_samples"]) != 1024:
        raise RuntimeError(f"{path} does not contain exactly 1,024 evaluation samples.")
    if prediction["samples"].shape[0] != 1024:
        raise RuntimeError(f"{path} sample array has the wrong leading dimension.")
    return prediction


def _read_metrics(root: Path, seed: int, basis_size: int, method: str) -> list[dict]:
    path = _method_dir(root, seed, basis_size, method) / "metrics_per_target.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        converted.append(
            {
                key: (
                    value if key in {"experiment", "method", "metric_partition"} else float(value)
                )
                for key, value in row.items()
                if value not in {"", None}
            }
        )
    return converted


def _mean_se(rows: list[dict], metric: str) -> tuple[float, float]:
    values = np.asarray(
        [float(row[metric]) for row in rows if metric in row and np.isfinite(row[metric])],
        dtype=np.float64,
    )
    if values.size == 0:
        return float("nan"), float("nan")
    stderr = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return float(values.mean()), stderr


def _select_metric_targets(
    rows: list[dict],
    target_ids: tuple[int, ...] | None,
) -> list[dict]:
    if target_ids is None:
        return rows
    selected = [row for row in rows if int(row["target_id"]) in target_ids]
    found = {int(row["target_id"]) for row in selected}
    missing = sorted(set(target_ids) - found)
    if missing:
        raise RuntimeError(f"Metrics are missing requested target IDs: {missing}")
    return selected


def _finite_minmax(values: list[np.ndarray]) -> tuple[float, float]:
    finite = [np.ravel(value[np.isfinite(value)]) for value in values]
    finite = [value for value in finite if value.size]
    if not finite:
        return -1.0, 1.0
    stacked = np.concatenate(finite)
    lo, hi = float(stacked.min()), float(stacked.max())
    pad = 0.055 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def plot_trajectory_grid(
    path_base: Path,
    *,
    predictions: dict[str, dict[str, np.ndarray]],
    methods: tuple[str, ...],
) -> dict[str, str]:
    plt = _plt()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    limits = []
    for dim in range(2):
        values = []
        for prediction in predictions.values():
            lower, upper = np.quantile(prediction["samples"], [0.05, 0.95], axis=0)
            values.extend(
                (
                    prediction["y_true"][:, dim],
                    prediction["y_train"][:, dim],
                    prediction["mean"][:, dim],
                    lower[:, dim],
                    upper[:, dim],
                )
            )
        limits.append(_finite_minmax(values))

    fig, axes = plt.subplots(
        2,
        len(methods),
        figsize=(max(3.0 * len(methods), 9.0), 4.6),
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes).reshape(2, -1)
    for column, method in enumerate(methods):
        prediction = predictions[method]
        t = prediction["t_plot"]
        samples = prediction["samples"]
        lower, upper = np.quantile(samples, [0.05, 0.95], axis=0)
        for dim in range(2):
            ax = axes[dim, column]
            ax.axvspan(0.0, 15.0, color="#EAF2F8", alpha=0.62, linewidth=0)
            ax.axvspan(15.0, 20.0, color="#F7E8B5", alpha=0.48, linewidth=0)
            ax.axvspan(20.0, 30.0, color="#F2D7D5", alpha=0.42, linewidth=0)
            ax.axvline(15.0, color="#666666", linestyle="--", linewidth=0.7)
            ax.axvline(20.0, color="#666666", linestyle=":", linewidth=0.8)
            for sample in samples[:20, :, dim]:
                ax.plot(t, sample, color=SPECIES_COLORS[dim], alpha=0.18, linewidth=0.75)
            ax.fill_between(
                t,
                lower[:, dim],
                upper[:, dim],
                color=SPECIES_COLORS[dim],
                alpha=0.20,
                linewidth=0,
            )
            ax.plot(
                t,
                prediction["mean"][:, dim],
                color=SPECIES_MEAN_COLORS[dim],
                linewidth=1.9,
            )
            ax.plot(t, prediction["y_true"][:, dim], color="black", linewidth=1.15)
            ax.scatter(
                prediction["y_train_x"],
                prediction["y_train"][:, dim],
                color="black",
                s=15,
                zorder=5,
            )
            ax.set_ylim(*limits[dim])
            ax.grid(alpha=0.2)
            if column == 0:
                ax.set_ylabel("prey" if dim == 0 else "predator")
            if dim == 0:
                ax.set_title(METHOD_LABELS.get(method, method))
            else:
                ax.set_xlabel("time")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#EAF2F8", alpha=0.62, label="training"),
        plt.Rectangle((0, 0), 1, 1, color="#F7E8B5", alpha=0.48, label="unused gap"),
        plt.Rectangle((0, 0), 1, 1, color="#F2D7D5", alpha=0.42, label="test"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    png = path_base.with_suffix(".png")
    pdf = path_base.with_suffix(".pdf")
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _rank_wrappers(values: dict[str, float], metric: str) -> dict[str, str]:
    if metric == "cov90":
        ordered = sorted(values, key=lambda key: (abs(values[key] - 0.9), key))
    else:
        ordered = sorted(values, key=lambda key: (values[key], key))
    wrappers = {method: "" for method in values}
    if ordered:
        wrappers[ordered[0]] = "best"
    if len(ordered) > 1:
        wrappers[ordered[1]] = "second"
    return wrappers


def _format_cell(mean: float, stderr: float, wrapper: str) -> str:
    value = f"{mean:.2f} \\pm {stderr:.2f}"
    return f"$\\{wrapper}{{{value}}}$" if wrapper else f"${value}$"


def write_main_table(
    path: Path,
    *,
    root: Path,
    seed: int,
    basis_size: int,
    methods: tuple[str, ...],
    aggregate_target_ids: tuple[int, ...] | None = None,
) -> str:
    summaries = {
        method: {
            metric: _mean_se(
                _select_metric_targets(
                    _read_metrics(root, seed, basis_size, method),
                    aggregate_target_ids,
                ),
                metric,
            )
            for metric, _ in METRICS
        }
        for method in methods
    }
    wrappers = {
        metric: _rank_wrappers({method: summaries[method][metric][0] for method in methods}, metric)
        for metric, _ in METRICS
    }
    lines = [
        "\\begin{tabular}{l" + "c" * len(METRICS) + "}",
        "\\toprule",
        "Method & " + " & ".join(label for _, label in METRICS) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        cells = [
            _format_cell(*summaries[method][metric], wrappers[metric][method])
            for metric, _ in METRICS
        ]
        lines.append(f"{METHOD_LABELS.get(method, method)} & " + " & ".join(cells) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def write_basis_table(
    path: Path,
    *,
    root: Path,
    seed: int,
    basis_sizes: tuple[int, ...],
    aggregate_target_ids: tuple[int, ...] | None = None,
) -> str:
    rows = []
    for method in ("vip", "ftip"):
        for basis_size in basis_sizes:
            metrics = _select_metric_targets(
                _read_metrics(root, seed, basis_size, method),
                aggregate_target_ids,
            )
            rows.append(
                (
                    method,
                    basis_size,
                    {metric: _mean_se(metrics, metric) for metric, _ in METRICS},
                )
            )
    gmvip_metrics = _select_metric_targets(
        _read_metrics(root, seed, basis_sizes[0], "gmvip_empirical"),
        aggregate_target_ids,
    )
    rows.append(
        (
            "gmvip_empirical",
            None,
            {metric: _mean_se(gmvip_metrics, metric) for metric, _ in METRICS},
        )
    )
    wrappers = {
        metric: _rank_wrappers(
            {f"{method}:{basis_size}": summary[metric][0] for method, basis_size, summary in rows},
            metric,
        )
        for metric, _ in METRICS
    }
    lines = [
        "\\begin{tabular}{ll" + "c" * len(METRICS) + "}",
        "\\toprule",
        "Method & Representation & " + " & ".join(label for _, label in METRICS) + " \\\\",
        "\\midrule",
    ]
    for method, basis_size, summary in rows:
        row_key = f"{method}:{basis_size}"
        representation = f"$S={basis_size}$" if basis_size is not None else "$M=96$"
        cells = [_format_cell(*summary[metric], wrappers[metric][row_key]) for metric, _ in METRICS]
        lines.append(
            f"{METHOD_LABELS.get(method, method)} & {representation} & "
            + " & ".join(cells)
            + " \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def write_noise_comparison(
    tex_path: Path,
    json_path: Path,
    *,
    fixed_root: Path,
    learned_root: Path,
    seed: int,
    basis_size: int,
    fixed_methods: tuple[str, ...],
    learned_methods: tuple[str, ...],
    aggregate_target_ids: tuple[int, ...] | None = None,
) -> dict[str, str]:
    row_specs = [
        (f"{method}:fixed", f"{METHOD_LABELS.get(method, method)} (fixed)", fixed_root, method)
        for method in fixed_methods
    ]
    row_specs.extend(
        (
            f"{method}:learned",
            f"{METHOD_LABELS.get(method, method)} (learned)",
            learned_root,
            method,
        )
        for method in learned_methods
    )
    report_rows = {}
    for key, label, root, method in row_specs:
        rows = _select_metric_targets(
            _read_metrics(root, seed, basis_size, method),
            aggregate_target_ids,
        )
        summary = {
            metric: dict(zip(("mean", "stderr"), _mean_se(rows, metric), strict=True))
            for metric, _ in METRICS
        }
        noise_summary = {}
        for metric in (
            "observation_noise_std_prey",
            "observation_noise_std_predator",
        ):
            if all(metric in row for row in rows):
                noise_summary[metric] = dict(
                    zip(("mean", "stderr"), _mean_se(rows, metric), strict=True)
                )
        report_rows[key] = {
            "label": label,
            "method": method,
            "noise_mode": "learned" if root == learned_root else "fixed",
            "metrics": summary,
            "observation_noise": noise_summary,
        }

    wrappers = {
        metric: _rank_wrappers(
            {key: report_rows[key]["metrics"][metric]["mean"] for key, *_ in row_specs},
            metric,
        )
        for metric, _ in METRICS
    }
    lines = [
        "\\begin{tabular}{l" + "c" * len(METRICS) + "}",
        "\\toprule",
        "Method & " + " & ".join(label for _, label in METRICS) + " \\\\",
        "\\midrule",
    ]
    for key, label, *_ in row_specs:
        cells = [
            _format_cell(
                report_rows[key]["metrics"][metric]["mean"],
                report_rows[key]["metrics"][metric]["stderr"],
                wrappers[metric][key],
            )
            for metric, _ in METRICS
        ]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "basis_size": basis_size,
                "aggregate_target_ids": aggregate_target_ids,
                "fixed_results_root": str(fixed_root),
                "learned_results_root": str(learned_root),
                "rows": report_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"table": str(tex_path), "summary": str(json_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render canonical Volterra artifacts.")
    parser.add_argument("--results-root", default="results/volterra")
    parser.add_argument(
        "--noise-mode",
        choices=("learned", "fixed"),
        default="learned",
        help="Expected trained-model likelihood-noise protocol (default: learned).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--vip-basis-size", type=int, default=20)
    parser.add_argument("--vip-basis-sizes", default="20,64,128,256")
    parser.add_argument("--target-ids", default="9")
    parser.add_argument(
        "--aggregate-target-ids",
        default=None,
        help="Optional target IDs/ranges used to aggregate tables; figures use --target-ids.",
    )
    parser.add_argument(
        "--fixed-noise-results-root",
        default=None,
        help="Optional fixed-noise sensitivity root used for a comparison table.",
    )
    parser.add_argument("--noise-comparison-methods", default="vip,ftip,gmvip_empirical")
    parser.add_argument("--out-dir", default="outputs/volterra")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    root = Path(args.results_root)
    methods = _parse_csv(args.methods)
    basis_size = int(args.vip_basis_size)
    basis_sizes = _parse_csv(args.vip_basis_sizes, int)
    target_ids = _parse_target_ids(args.target_ids)
    aggregate_target_ids = (
        _parse_target_ids(args.aggregate_target_ids)
        if args.aggregate_target_ids is not None
        else None
    )
    selections = [(basis_size, method) for method in methods]
    selections.extend((size, method) for size in basis_sizes for method in ("vip", "ftip"))
    selections.append((basis_sizes[0], "gmvip_empirical"))
    standard_manifests = _validate_manifests(
        root,
        int(args.seed),
        list(dict.fromkeys(selections)),
        noise_mode=args.noise_mode,
    )

    out_dir = Path(args.out_dir)
    figures = []
    for target_id in target_ids:
        predictions = {
            method: _load_prediction(
                root,
                seed=int(args.seed),
                basis_size=basis_size,
                method=method,
                target_id=target_id,
            )
            for method in methods
        }
        figures.append(
            {
                "target_id": target_id,
                **plot_trajectory_grid(
                    out_dir / f"volterra_target_{target_id}",
                    predictions=predictions,
                    methods=methods,
                ),
            }
        )
    result = {
        "figures": figures,
        "main_table": write_main_table(
            out_dir / "volterra_main_table.tex",
            root=root,
            seed=int(args.seed),
            basis_size=basis_size,
            methods=methods,
            aggregate_target_ids=aggregate_target_ids,
        ),
        "basis_table": write_basis_table(
            out_dir / "volterra_basis_table.tex",
            root=root,
            seed=int(args.seed),
            basis_sizes=basis_sizes,
            aggregate_target_ids=aggregate_target_ids,
        ),
        "aggregate_target_ids": aggregate_target_ids,
    }
    if args.fixed_noise_results_root is not None:
        fixed_root = Path(args.fixed_noise_results_root)
        comparison_methods = _parse_csv(args.noise_comparison_methods)
        fixed_manifests = _validate_manifests(
            fixed_root,
            int(args.seed),
            [(basis_size, method) for method in methods],
            noise_mode="fixed",
        )
        dataset_hashes = {
            manifest["dataset"]["sha256"]
            for manifest in (*standard_manifests.values(), *fixed_manifests.values())
        }
        if len(dataset_hashes) != 1:
            raise RuntimeError("Fixed- and learned-noise results use different target datasets.")
        result["noise_comparison"] = write_noise_comparison(
            out_dir / "volterra_noise_comparison_table.tex",
            out_dir / "volterra_noise_comparison.json",
            fixed_root=fixed_root,
            learned_root=root,
            seed=int(args.seed),
            basis_size=basis_size,
            fixed_methods=methods,
            learned_methods=comparison_methods,
            aggregate_target_ids=aggregate_target_ids,
        )
    report_manifest = out_dir / "volterra_report.json"
    result["report_manifest"] = str(report_manifest)
    report_manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
