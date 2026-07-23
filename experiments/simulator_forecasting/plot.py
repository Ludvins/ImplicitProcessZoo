from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_RESULTS_ROOT = Path("results/oscillator")
DEFAULT_OUTPUT_ROOT = Path("outputs/oscillator")
DEFAULT_METHODS = ("vip", "ftip", "gmvip")
DEFAULT_LABELS = {"vip": "VIP", "ftip": "FTIP", "gmvip": "GMVIP"}
DEFAULT_COLORS = {"vip": "#4E79A7", "ftip": "#F28E2B", "gmvip": "#B07AA1"}
TABLE_METRICS = (
    ("rmse", "RMSE $\\downarrow$", None),
    ("nlpd", "NLL $\\downarrow$", None),
    ("crps", "CRPS $\\downarrow$", None),
    ("cov90", "Cov$_{90}$", 0.90),
)
PDF_METADATA = {
    "Creator": "ImplicitProcessZoo",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _method_dir(root: Path, method: str, *, seed: int, basis_size: int) -> Path:
    return root / f"seed_{seed}" / f"S_{basis_size}" / method


def _read_manifest(method_dir: Path) -> dict[str, object]:
    path = method_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_manifests(
    results_root: Path,
    methods: tuple[str, ...],
    *,
    seed: int,
    basis_size: int,
    expected_targets: int,
) -> dict[str, dict[str, object]]:
    manifests = {
        method: _read_manifest(
            _method_dir(results_root, method, seed=seed, basis_size=basis_size)
        )
        for method in methods
    }
    dataset_hashes = set()
    for method, manifest in manifests.items():
        if manifest.get("schema_version") != 2:
            raise ValueError(f"{method}: unsupported manifest schema.")
        if manifest.get("experiment") != "damped_oscillator":
            raise ValueError(f"{method}: wrong experiment in manifest.")
        if manifest.get("method") != method:
            raise ValueError(f"{method}: method/manifest mismatch.")
        if manifest.get("status") != "complete":
            raise ValueError(f"{method}: result run is not complete.")
        if int(manifest.get("seed", -1)) != seed:
            raise ValueError(f"{method}: seed mismatch.")
        if int(manifest.get("vip_basis_size", -1)) != basis_size:
            raise ValueError(f"{method}: VIP/FTIP basis-size mismatch.")
        if int(manifest.get("evaluation_samples", -1)) != 1024:
            raise ValueError(f"{method}: canonical results require exactly 1,024 samples.")
        if manifest.get("checkpoint_selection") != "none_final_step_only":
            raise ValueError(f"{method}: validation/checkpoint selection is not allowed.")
        noise = manifest.get("observation_noise", {})
        if not isinstance(noise, dict) or noise.get("mode") != "learned_scalar":
            raise ValueError(f"{method}: observation noise must be learned.")
        usage = manifest.get("data_usage", {})
        if not isinstance(usage, dict) or usage.get("training") != "t<=15":
            raise ValueError(f"{method}: incorrect training interval.")
        if usage.get("validation") != "none":
            raise ValueError(f"{method}: validation data must not be used.")
        completed = manifest.get("completed_targets", [])
        if len(completed) != expected_targets:
            raise ValueError(
                f"{method}: expected {expected_targets} completed targets, got {len(completed)}."
            )
        config = manifest.get("config", {})
        if method == "gmvip":
            gmvip = config.get("gmvip", {}) if isinstance(config, dict) else {}
            if int(gmvip.get("num_inducing", -1)) != 32:
                raise ValueError("gmvip: canonical oscillator protocol requires M=32.")
        dataset = manifest.get("dataset", {})
        if isinstance(dataset, dict):
            dataset_hashes.add(dataset.get("sha256"))
    if len(dataset_hashes) != 1:
        raise ValueError("Methods were evaluated on different oscillator datasets.")
    return manifests


def _load_prediction(
    results_root: Path,
    method: str,
    *,
    seed: int,
    basis_size: int,
    target_id: int,
    n_train: int,
) -> dict[str, np.ndarray]:
    path = (
        _method_dir(results_root, method, seed=seed, basis_size=basis_size)
        / "predictions"
        / f"target_{target_id}_ntrain_{n_train}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction artifact: {path}")
    with np.load(path) as payload:
        prediction = {key: payload[key] for key in payload.files}
    samples = np.asarray(prediction["samples"])
    if int(prediction["evaluation_samples"]) != 1024 or samples.shape[0] != 1024:
        raise ValueError(f"{path} does not contain exactly 1,024 posterior samples.")
    if "observation_noise_std" not in prediction:
        raise ValueError(f"{path} does not record learned observation noise.")
    return prediction


def _vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values if values.ndim == 1 else values[:, 0]


def _validate_predictions(predictions: dict[str, dict[str, np.ndarray]]) -> None:
    reference_method = next(iter(predictions))
    reference = predictions[reference_method]
    for method, prediction in predictions.items():
        for key in ("t", "truth", "train_t", "train_y"):
            if not np.array_equal(np.asarray(prediction[key]), np.asarray(reference[key])):
                raise ValueError(
                    f"{reference_method} and {method} do not describe the same target."
                )


def _save_figure(fig, path_base: Path, *, dpi: int) -> list[str]:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    png = path_base.with_suffix(".png")
    pdf = path_base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf, metadata=PDF_METADATA)
    return [str(png), str(pdf)]


def plot_target(
    results_root: Path,
    output_root: Path,
    methods: tuple[str, ...],
    *,
    seed: int,
    basis_size: int,
    target_id: int,
    n_train: int,
    dpi: int,
) -> list[str]:
    predictions = {
        method: _load_prediction(
            results_root,
            method,
            seed=seed,
            basis_size=basis_size,
            target_id=target_id,
            n_train=n_train,
        )
        for method in methods
    }
    _validate_predictions(predictions)
    plt = _pyplot()
    fig, axes = plt.subplots(
        1, len(methods), figsize=(max(3.0 * len(methods), 9.0), 2.35), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        prediction = predictions[method]
        t = np.asarray(prediction["t"], dtype=float)
        truth = _vector(prediction["truth"])
        train_t = np.asarray(prediction["train_t"], dtype=float)
        train_y = _vector(prediction["train_y"])
        samples = np.asarray(prediction["samples"], dtype=float)
        if samples.ndim == 3:
            samples = samples[..., 0]
        mean = samples.mean(axis=0)
        q05, q95 = np.quantile(samples, (0.05, 0.95), axis=0)

        ax.axvspan(0.0, 15.0, color="#EAF2F8", alpha=0.75, linewidth=0)
        ax.axvspan(15.0, 20.0, color="#FFF4E6", alpha=0.55, linewidth=0)
        ax.axvspan(20.0, 30.0, color="#FCEBEC", alpha=0.45, linewidth=0)
        ax.axvline(15.0, color="black", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.axvline(20.0, color="black", linestyle=":", linewidth=0.8, alpha=0.55)
        for sample in samples[:20]:
            ax.plot(t, sample, color=DEFAULT_COLORS[method], alpha=0.13, linewidth=0.7)
        ax.fill_between(t, q05, q95, color=DEFAULT_COLORS[method], alpha=0.20, linewidth=0)
        ax.plot(t, mean, color=DEFAULT_COLORS[method], linewidth=1.8)
        ax.plot(t, truth, color="black", linewidth=1.15)
        ax.scatter(train_t, train_y, color="black", s=15, zorder=5)
        ax.set_title(DEFAULT_LABELS.get(method, method))
        ax.set_xlim(float(t.min()), float(t.max()))
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("$y(t)$")
    for ax in axes:
        ax.set_xlabel("$t$")
    fig.tight_layout()
    written = _save_figure(
        fig, output_root / f"oscillator_target_{target_id}", dpi=dpi
    )
    plt.close(fig)
    return written


def _read_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_metric(mean: float, std: float, *, coverage: bool) -> str:
    if coverage:
        return f"${100.0 * mean:.1f} \\pm {100.0 * std:.1f}$"
    return f"${mean:.2f} \\pm {std:.2f}$"


def write_main_table(
    results_root: Path,
    output_root: Path,
    methods: tuple[str, ...],
    *,
    seed: int,
    basis_size: int,
    expected_targets: int,
) -> tuple[Path, Path]:
    values: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        rows = _read_metric_rows(
            _method_dir(results_root, method, seed=seed, basis_size=basis_size)
            / "metrics_per_target_region.csv"
        )
        rows = [row for row in rows if row["region"] == "far_extrapolation"]
        target_ids = {int(row["target_id"]) for row in rows}
        if len(rows) != expected_targets or len(target_ids) != expected_targets:
            raise ValueError(
                f"{method}: expected one far-extrapolation row for each of "
                f"{expected_targets} targets."
            )
        values[method] = {
            metric: np.asarray([float(row[metric]) for row in rows], dtype=float)
            for metric, _, _ in TABLE_METRICS
        }

    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "aggregation": "mean_plus_sample_standard_deviation",
        "ddof": 1,
        "region": "far_extrapolation",
        "targets": expected_targets,
        "methods": {},
    }
    lines = [
        "\\begin{tabular}{l" + "c" * len(TABLE_METRICS) + "}",
        "\\toprule",
        "Method & " + " & ".join(label for _, label, _ in TABLE_METRICS) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        cells = []
        method_summary: dict[str, object] = {}
        for metric, _, nominal in TABLE_METRICS:
            metric_values = values[method][metric]
            mean = float(metric_values.mean())
            std = float(metric_values.std(ddof=1))
            cells.append(_format_metric(mean, std, coverage=nominal is not None))
            method_summary[metric] = {"mean": mean, "std": std}
        summary["methods"][method] = method_summary
        lines.append(f"{DEFAULT_LABELS.get(method, method)} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])

    table_path = output_root / "oscillator_main_table.tex"
    summary_path = output_root / "oscillator_summary.json"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return table_path, summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and report the canonical damped-oscillator benchmark."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vip-basis-size", type=int, default=256)
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--expected-targets", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    methods = _split_csv(args.methods)
    if not methods:
        raise ValueError("At least one method is required.")
    _validate_manifests(
        args.results_root,
        methods,
        seed=args.seed,
        basis_size=args.vip_basis_size,
        expected_targets=args.expected_targets,
    )
    figures = plot_target(
        args.results_root,
        args.output_root,
        methods,
        seed=args.seed,
        basis_size=args.vip_basis_size,
        target_id=args.target_id,
        n_train=args.n_train,
        dpi=args.dpi,
    )
    table, summary = write_main_table(
        args.results_root,
        args.output_root,
        methods,
        seed=args.seed,
        basis_size=args.vip_basis_size,
        expected_targets=args.expected_targets,
    )
    report = {
        "results_root": str(args.results_root),
        "methods": list(methods),
        "seed": args.seed,
        "vip_basis_size": args.vip_basis_size,
        "evaluation_samples": 1024,
        "observation_noise": "learned_scalar",
        "period_metric": "removed",
        "figure": figures,
        "table": str(table),
        "summary": str(summary),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "oscillator_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
