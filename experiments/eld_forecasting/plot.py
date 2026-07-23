from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


DEFAULT_COLORS = {
    "analog": "#777777",
    "seasonal_naive": "#9c755f",
    "empirical_gaussian": "#59a14f",
    "gmvip_empirical_exact": "#e15759",
    "vip": "#4e79a7",
    "vip_512": "#59a14f",
    "ftip": "#f28e2b",
    "gmvip_empirical": "#b07aa1",
}

DEFAULT_LABELS = {
    "analog": "Analog prior",
    "seasonal_naive": "Seasonal naive",
    "empirical_gaussian": "Empirical Gaussian",
    "gmvip_empirical_exact": "GMVIP exact",
    "vip": "VIP",
    "vip_512": "VIP-512",
    "ftip": "FTIP",
    "gmvip_empirical": "GMVIP",
}
TABLE_METRICS = (
    ("rmse", "RMSE $\\downarrow$", None),
    ("nll", "NLL $\\downarrow$", None),
    ("crps", "CRPS $\\downarrow$", None),
    ("cqm", "CQM $\\downarrow$", None),
    ("cov80", "Cov. 80\\%", 0.80),
    ("cov90", "Cov. 90\\%", 0.90),
)

PAPER_OBSERVED_WINDOW_COLOR = "#EAF2F8"
PAPER_POSTERIOR_COLOR = "#4C78A8"
PAPER_MEAN_COLOR = "#1F5A93"
PAPER_COLUMN_WIDTH = 3.0
PAPER_ROW_HEIGHT = 4.3 / 2.0
PAPER_MIN_WIDTH = 9.0
PAPER_TITLE_FONTSIZE = 12
PAPER_TICK_LABELSIZE = 8.5
PAPER_POSTERIOR_SAMPLE_LINES = 20
PDF_METADATA = {
    "Creator": "ImplicitProcessZoo",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
}


def _method_dir(results_root: Path, method: str, *, seed: int, basis_size: int) -> Path:
    return results_root / f"seed_{seed}" / f"S_{basis_size}" / method


def _method_dirs(
    results_root: Path,
    methods: list[str] | None,
    *,
    seed: int = 0,
    basis_size: int = 20,
) -> list[Path]:
    if methods:
        return [
            _method_dir(results_root, method, seed=seed, basis_size=basis_size)
            for method in methods
        ]
    root = results_root / f"seed_{seed}" / f"S_{basis_size}"
    return sorted(path for path in root.iterdir() if path.is_dir())


def _available_target_ids(method_dirs: list[Path]) -> list[int]:
    ids = set()
    for method_dir in method_dirs:
        pred_dir = method_dir / "predictions"
        for path in pred_dir.glob("target_*.npz"):
            try:
                ids.add(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
    return sorted(ids)


def _load_prediction(method_dir: Path, target_id: int):
    path = method_dir / "predictions" / f"target_{target_id}.npz"
    if not path.exists():
        return None
    return np.load(path)


def _forecast_start(data, context_t: np.ndarray) -> float:
    if "forecast_start_hour" in data:
        return float(data["forecast_start_hour"])
    if context_t.size > 1:
        return float(context_t[-1] + np.median(np.diff(context_t)))
    return float(context_t[-1]) if context_t.size else 0.0


def _samples_2d(data) -> np.ndarray:
    samples = np.asarray(data["samples"], dtype=float)
    if samples.ndim == 3:
        samples = samples[..., 0]
    return samples


def _reference_series(data):
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    truth = np.asarray(data["truth"], dtype=float).reshape(-1)
    if "context_t" in data:
        context_t = np.asarray(data["context_t"], dtype=float).reshape(-1)
        context_y = np.asarray(data["context_y"], dtype=float).reshape(-1)
    else:
        context_t = np.asarray(data["train_t"], dtype=float).reshape(-1)
        context_y = np.asarray(data["train_y"], dtype=float).reshape(-1)
    return t, truth, context_t, context_y


def _validate_loaded_predictions(loaded, *, target_id: int, seed: int) -> None:
    reference_method, reference = loaded[0]
    reference_t, reference_truth, reference_context_t, reference_context_y = _reference_series(
        reference
    )
    reference_identity = tuple(
        str(reference[key]) if key in reference else None for key in ("client_id", "start_time")
    )
    if int(reference["evaluation_samples"]) != 1024 or _samples_2d(reference).shape[0] != 1024:
        raise ValueError(
            f"Prediction artifact for seed {seed}, target {target_id} does not contain 1,024 samples."
        )
    for method, data in loaded[1:]:
        t, truth, context_t, context_y = _reference_series(data)
        identity = tuple(
            str(data[key]) if key in data else None for key in ("client_id", "start_time")
        )
        if int(data["evaluation_samples"]) != 1024 or _samples_2d(data).shape[0] != 1024:
            raise ValueError(
                f"{method} prediction for seed {seed}, target {target_id} has the wrong sample count."
            )
        if identity != reference_identity or not all(
            np.array_equal(left, right)
            for left, right in (
                (t, reference_t),
                (truth, reference_truth),
                (context_t, reference_context_t),
                (context_y, reference_context_y),
            )
        ):
            raise ValueError(
                f"Prediction artifact mismatch for seed {seed}, target {target_id}: "
                f"{reference_method} and {method} do not describe the same trajectory."
            )


def _save_figure(fig, path_base: Path, formats: list[str], *, dpi: int = 180) -> list[Path]:
    written = []
    path_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        suffix = fmt.strip().lower().lstrip(".")
        if not suffix:
            continue
        out_path = path_base.with_suffix(f".{suffix}")
        if suffix == "png":
            fig.savefig(out_path, dpi=int(dpi))
        elif suffix == "pdf":
            fig.savefig(out_path, metadata=PDF_METADATA)
        else:
            fig.savefig(out_path)
        written.append(out_path)
    return written


def _load_target_predictions(
    results_root: Path,
    target_id: int,
    methods: list[str] | None = None,
    *,
    seed: int = 0,
    basis_size: int = 20,
) -> list[tuple[str, np.lib.npyio.NpzFile]]:
    method_dirs = _method_dirs(results_root, methods, seed=seed, basis_size=basis_size)
    loaded = []
    for method_dir in method_dirs:
        data = _load_prediction(method_dir, target_id)
        if data is not None:
            method = method_dir.name
            loaded.append((method, data))
    if not loaded:
        raise FileNotFoundError(
            f"No prediction files found for target {target_id} seed {seed} under {results_root}."
        )
    _validate_loaded_predictions(loaded, target_id=target_id, seed=seed)
    return loaded


def plot_target(
    results_root: Path,
    target_id: int,
    output_dir: Path,
    methods: list[str] | None = None,
    *,
    seed: int = 0,
    basis_size: int = 20,
    formats: list[str] | None = None,
) -> list[Path]:
    plt = _pyplot()
    formats = formats or ["png"]
    loaded = _load_target_predictions(
        results_root, target_id, methods, seed=seed, basis_size=basis_size
    )
    reference = loaded[0][1]
    t, truth, context_t, context_y = _reference_series(reference)
    forecast_start = _forecast_start(reference, context_t)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, truth, color="black", linewidth=2.0, label="truth", zorder=5)
    ax.scatter(context_t, context_y, color="black", s=18, label="observed context", zorder=6)
    ax.axvline(forecast_start, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    for method, data in loaded:
        color = DEFAULT_COLORS.get(method)
        samples = _samples_2d(data)
        mean = samples.mean(axis=0)
        lower = np.quantile(samples, 0.05, axis=0)
        upper = np.quantile(samples, 0.95, axis=0)
        ax.plot(t, mean, linewidth=1.8, color=color, label=method)
        ax.fill_between(t, lower, upper, color=color, alpha=0.16, linewidth=0)

    ax.set_title(f"ELD held-out trajectory target {target_id}")
    ax.set_xlabel("hours since window start")
    ax.set_ylabel("load")
    ax.set_xlim(float(t.min()), float(t.max()))
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()

    outputs = _save_figure(fig, output_dir / f"seed_{seed}_target_{target_id}", formats)
    plt.close(fig)
    return outputs


def plot_target_method_grid(
    results_root: Path,
    target_id: int,
    output_dir: Path,
    methods: list[str],
    *,
    seed: int = 0,
    basis_size: int = 20,
    formats: list[str] | None = None,
) -> list[Path]:
    plt = _pyplot()
    formats = formats or ["png"]
    loaded = _load_target_predictions(
        results_root, target_id, methods, seed=seed, basis_size=basis_size
    )
    labels = {method: DEFAULT_LABELS.get(method, method) for method, _ in loaded}
    width = max(PAPER_COLUMN_WIDTH * len(loaded), PAPER_MIN_WIDTH)

    fig, axes = plt.subplots(
        1,
        len(loaded),
        figsize=(width, PAPER_ROW_HEIGHT),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax, (method, data) in zip(axes, loaded):
        t, truth, train_t, train_y = _reference_series(data)
        samples = _samples_2d(data)
        mean = samples.mean(axis=0)
        q05 = np.quantile(samples, 0.05, axis=0)
        q95 = np.quantile(samples, 0.95, axis=0)
        forecast_start = _forecast_start(data, train_t)

        ax.axvspan(
            0.0,
            forecast_start,
            color=PAPER_OBSERVED_WINDOW_COLOR,
            alpha=0.7,
            linewidth=0,
            zorder=-2,
        )
        ax.axvline(
            forecast_start,
            color="black",
            linestyle="--",
            linewidth=0.85,
            alpha=0.6,
            zorder=2,
        )
        for sample in samples[:PAPER_POSTERIOR_SAMPLE_LINES]:
            ax.plot(t, sample, color=PAPER_POSTERIOR_COLOR, alpha=0.20, linewidth=0.8, zorder=1)
        ax.fill_between(t, q05, q95, color=PAPER_POSTERIOR_COLOR, alpha=0.22, linewidth=0, zorder=2)
        ax.plot(t, mean, color=PAPER_MEAN_COLOR, linewidth=1.9, zorder=4)
        ax.plot(t, truth, color="black", linewidth=1.2, zorder=5)
        ax.scatter(train_t, train_y, color="black", s=18, zorder=6)
        ax.set_title(labels[method], fontsize=PAPER_TITLE_FONTSIZE)
        ax.tick_params(axis="both", labelsize=PAPER_TICK_LABELSIZE)
        ax.grid(alpha=0.22)
        ax.set_xlim(0.0, float(t[-1]))

    fig.tight_layout()
    method_tag = "_".join(method for method, _ in loaded)
    outputs = _save_figure(
        fig, output_dir / f"seed_{seed}_target_{target_id}_{method_tag}_grid", formats
    )
    plt.close(fig)
    return outputs


def _read_metric_rows(
    root: Path,
    *,
    methods: list[str],
    seeds: list[int],
    basis_size: int,
) -> list[dict]:
    rows = []
    dataset_hashes = set()
    for seed in seeds:
        for method in methods:
            method_dir = _method_dir(root, method, seed=seed, basis_size=basis_size)
            manifest_path = method_dir / "manifest.json"
            metric_path = method_dir / "metrics_per_target_region.csv"
            if not manifest_path.is_file() or not metric_path.is_file():
                raise FileNotFoundError(f"Missing canonical electricity artifacts in {method_dir}.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_noise = (
                "learned_scalar"
                if method in {"vip", "vip_512", "ftip", "gmvip_empirical"}
                else "fixed_scalar"
            )
            expected = {
                "schema_version": 2,
                "experiment": "electricity_forecasting",
                "method": method,
                "seed": seed,
                "vip_basis_size": basis_size,
                "evaluation_samples": 1024,
                "checkpoint_selection": "none_final_step_only",
                "status": "complete",
            }
            mismatches = {
                key: (manifest.get(key), value)
                for key, value in expected.items()
                if manifest.get(key) != value
            }
            if manifest.get("observation_noise", {}).get("mode") != expected_noise:
                mismatches["observation_noise.mode"] = (
                    manifest.get("observation_noise", {}).get("mode"),
                    expected_noise,
                )
            if manifest.get("data_usage", {}).get("validation") != "none":
                mismatches["data_usage.validation"] = (
                    manifest.get("data_usage", {}).get("validation"),
                    "none",
                )
            if method == "gmvip_empirical" and int(
                manifest.get("config", {}).get("gmvip", {}).get("num_inducing", -1)
            ) != 96:
                mismatches["gmvip.num_inducing"] = (
                    manifest.get("config", {}).get("gmvip", {}).get("num_inducing"),
                    96,
                )
            if mismatches:
                raise RuntimeError(f"Incompatible electricity manifest {manifest_path}: {mismatches}")
            dataset_hashes.add(manifest["dataset"]["sha256"])
            with metric_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("region") != "test_forecast":
                        continue
                    converted = dict(row)
                    for metric, *_ in TABLE_METRICS:
                        converted[metric] = float(row[metric])
                    converted["run_seed"] = int(row["run_seed"])
                    converted["target_id"] = int(row["target_id"])
                    rows.append(converted)
    if len(dataset_hashes) != 1:
        raise RuntimeError("Electricity runs were produced from different processed datasets.")
    expected_count = len(methods) * len(seeds) * 25
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} electricity rows, found {len(rows)}.")
    return rows


def _rank_wrappers(values: dict[str, float], nominal: float | None) -> dict[str, str]:
    ordered = sorted(
        values,
        key=(lambda key: abs(values[key] - nominal))
        if nominal is not None
        else (lambda key: values[key]),
    )
    wrappers = {key: "" for key in values}
    if ordered:
        wrappers[ordered[0]] = "best"
    if len(ordered) > 1:
        wrappers[ordered[1]] = "second"
    return wrappers


def write_main_table(
    tex_path: Path,
    json_path: Path,
    *,
    root: Path,
    methods: list[str],
    seeds: list[int],
    basis_size: int,
) -> dict:
    rows = _read_metric_rows(
        root,
        methods=methods,
        seeds=seeds,
        basis_size=basis_size,
    )
    summaries = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summaries[method] = {}
        for metric, _label, _nominal in TABLE_METRICS:
            values = np.asarray([row[metric] for row in method_rows], dtype=np.float64)
            summaries[method][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "count": int(values.size),
            }
    wrappers = {
        metric: _rank_wrappers(
            {method: summaries[method][metric]["mean"] for method in methods},
            nominal,
        )
        for metric, _label, nominal in TABLE_METRICS
    }
    lines = [
        "\\begin{tabular}{l" + "c" * len(TABLE_METRICS) + "}",
        "\\toprule",
        "Method & " + " & ".join(label for _metric, label, _nominal in TABLE_METRICS) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        cells = []
        for metric, _label, nominal in TABLE_METRICS:
            mean = summaries[method][metric]["mean"]
            std = summaries[method][metric]["std"]
            if nominal is not None:
                mean *= 100.0
                std *= 100.0
            cell = f"${mean:.2f} \\pm {std:.2f}$"
            wrapper = wrappers[metric][method]
            if wrapper:
                cell = f"\\{wrapper}{{{cell}}}"
            cells.append(cell)
        lines.append(f"{DEFAULT_LABELS.get(method, method)} & " + " & ".join(cells) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "seeds": seeds,
        "targets_per_seed": 25,
        "basis_size": basis_size,
        "aggregation": "mean_plus_sample_standard_deviation_ddof_1",
        "rows": summaries,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"table": str(tex_path), "summary": str(json_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render canonical electricity artifacts.")
    parser.add_argument("--results-root", default="results/electricity")
    parser.add_argument(
        "--methods",
        default="analog,vip,ftip,empirical_gaussian,gmvip_empirical",
    )
    parser.add_argument("--figure-methods", default="vip,ftip,gmvip_empirical")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--vip-basis-size", type=int, default=20)
    parser.add_argument("--figure-seed", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=18)
    parser.add_argument("--out-dir", default="outputs/electricity")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    figure_methods = [
        item.strip() for item in args.figure_methods.split(",") if item.strip()
    ]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    table = write_main_table(
        out_dir / "electricity_main_table.tex",
        out_dir / "electricity_summary.json",
        root=results_root,
        methods=methods,
        seeds=seeds,
        basis_size=int(args.vip_basis_size),
    )
    figure = plot_target_method_grid(
        results_root,
        int(args.target_id),
        out_dir,
        figure_methods,
        seed=int(args.figure_seed),
        basis_size=int(args.vip_basis_size),
        formats=["png", "pdf"],
    )
    result = {"table": table, "figure": [str(path) for path in figure]}
    report_path = out_dir / "electricity_report.json"
    result["report"] = str(report_path)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
