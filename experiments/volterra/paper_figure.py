from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD_LABELS = {
    "vip": "VIP",
    "gmvip_empirical": "GMVIP",
}

METHOD_COLORS = {
    "vip": "#4C78A8",
    "gmvip_empirical": "#2A9D8F",
}

VALIDATION_COLOR = "#E7C95B"
PREDICTION_START = 20.0


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return plt, Line2D, Patch


def _path_base(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() in {".png", ".pdf"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_prediction(results_root: Path, method: str, seed: int, target_id: int) -> dict[str, np.ndarray]:
    path = results_root / method / f"seed_{seed}" / "predictions" / f"target_{target_id}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file for {method}, seed {seed}, target {target_id}: {path}"
        )
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _finite_minmax(values: list[np.ndarray], pad_fraction: float = 0.055) -> tuple[float, float]:
    finite = [np.ravel(value[np.isfinite(value)]) for value in values]
    finite = [value for value in finite if value.size]
    if not finite:
        return -1.0, 1.0
    stacked = np.concatenate(finite)
    lo = float(stacked.min())
    hi = float(stacked.max())
    pad = pad_fraction * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _shared_limits(predictions: dict[str, dict[str, np.ndarray]]) -> tuple[list[tuple[float, float]], tuple[float, float], tuple[float, float]]:
    trajectory_limits: list[tuple[float, float]] = []
    for dim in range(2):
        values: list[np.ndarray] = []
        for pred in predictions.values():
            lower, upper = np.quantile(pred["samples"], [0.05, 0.95], axis=0)
            values.extend(
                [
                    pred["y_true"][:, dim],
                    pred["y_train"][:, dim],
                    pred["mean"][:, dim],
                    lower[:, dim],
                    upper[:, dim],
                ]
            )
        trajectory_limits.append(_finite_minmax(values))

    phase_x: list[np.ndarray] = []
    phase_y: list[np.ndarray] = []
    for pred in predictions.values():
        samples = pred["samples"]
        phase_x.extend([pred["y_true"][:, 0], pred["y_train"][:, 0], pred["mean"][:, 0], samples[:, :, 0]])
        phase_y.extend([pred["y_true"][:, 1], pred["y_train"][:, 1], pred["mean"][:, 1], samples[:, :, 1]])
    return trajectory_limits, _finite_minmax(phase_x), _finite_minmax(phase_y)


def plot_paper_lotka_volterra_figure(
    path_base: str | Path,
    *,
    predictions_by_method: dict[str, dict[str, np.ndarray]],
    methods: tuple[str, str] = ("vip", "gmvip_empirical"),
    n_phase_samples: int = 18,
) -> dict[str, str]:
    plt, Line2D, Patch = _plt()
    path_base = _path_base(path_base)
    method_list = list(methods)
    trajectory_limits, phase_xlim, phase_ylim = _shared_limits(
        {method: predictions_by_method[method] for method in method_list}
    )

    fig = plt.figure(figsize=(7.35, 3.95))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.073,
        right=0.995,
        bottom=0.255,
        top=0.895,
        width_ratios=(1.0, 1.0, 0.92),
        wspace=0.34,
        hspace=0.16,
    )
    axes = np.array(
        [[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(len(method_list))]
    )

    titles = ("Prey Trajectory", "Predator Trajectory", "Phase Portrait")
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9.4, pad=5.0)

    for row, method in enumerate(method_list):
        pred = predictions_by_method[method]
        t = pred["t_plot"]
        y_true = pred["y_true"]
        y_train_x = pred["y_train_x"]
        y_train = pred["y_train"]
        samples = pred["samples"]
        mean = pred["mean"]
        lower, upper = np.quantile(samples, [0.05, 0.95], axis=0)
        color = METHOD_COLORS.get(method, "#4C78A8")
        label = METHOD_LABELS.get(method, method)

        for dim in range(2):
            ax = axes[row, dim]
            ax.axvspan(15.0, PREDICTION_START, color=VALIDATION_COLOR, alpha=0.28, linewidth=0, zorder=0)
            ax.fill_between(t, lower[:, dim], upper[:, dim], color=color, alpha=0.17, linewidth=0, zorder=1)
            ax.plot(t, mean[:, dim], color=color, linewidth=1.55, zorder=3)
            ax.plot(t, y_true[:, dim], color="black", linewidth=1.15, zorder=4)
            ax.scatter(y_train_x, y_train[:, dim], s=7, color="black", alpha=0.72, zorder=5)
            ax.axvline(
                PREDICTION_START,
                color="#5B5B5B",
                linestyle="--",
                linewidth=0.75,
                alpha=0.78,
                zorder=2,
            )
            ax.set_ylim(*trajectory_limits[dim])
            ax.grid(alpha=0.22, linewidth=0.55)
            ax.tick_params(axis="both", labelsize=7.4, width=0.7, length=3)
            if row == len(method_list) - 1:
                ax.set_xlabel("Time", fontsize=8.2)
            else:
                ax.tick_params(labelbottom=False)
            if dim == 0:
                ax.set_ylabel(label, fontsize=8.2, rotation=90)

        phase_ax = axes[row, 2]
        for sample in samples[:n_phase_samples]:
            phase_ax.plot(sample[:, 0], sample[:, 1], color=color, alpha=0.13, linewidth=0.72, zorder=1)
        phase_ax.plot(mean[:, 0], mean[:, 1], color=color, linewidth=1.35, zorder=3)
        phase_ax.plot(y_true[:, 0], y_true[:, 1], color="black", linewidth=1.15, zorder=4)
        phase_ax.scatter(y_train[:, 0], y_train[:, 1], s=7, color="black", alpha=0.72, zorder=5)
        phase_ax.set_xlim(*phase_xlim)
        phase_ax.set_ylim(*phase_ylim)
        phase_ax.grid(alpha=0.22, linewidth=0.55)
        phase_ax.tick_params(axis="both", labelsize=7.4, width=0.7, length=3)
        if row == len(method_list) - 1:
            phase_ax.set_xlabel("Prey", fontsize=8.2)
        else:
            phase_ax.tick_params(labelbottom=False)
        phase_ax.set_ylabel("Predator", fontsize=8.2)

    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS["vip"], lw=1.7, label="VIP Posterior Mean"),
        Line2D([0], [0], color=METHOD_COLORS["gmvip_empirical"], lw=1.7, label="GMVIP Posterior Mean"),
        Patch(facecolor="#6F87A6", alpha=0.18, edgecolor="none", label="90% Interval"),
        Line2D([0], [0], color="#6F87A6", lw=1.0, alpha=0.35, label="Phase Posterior Samples"),
        Line2D([0], [0], color="black", lw=1.2, label="Truth"),
        Line2D([0], [0], marker="o", color="black", lw=0, markersize=3.6, alpha=0.72, label="Observations"),
        Patch(facecolor=VALIDATION_COLOR, alpha=0.28, edgecolor="none", label="Validation Window"),
        Line2D([0], [0], color="#5B5B5B", linestyle="--", lw=0.85, label="Predictions Start"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
        fontsize=7.1,
        handlelength=1.9,
        columnspacing=1.15,
        handletextpad=0.45,
    )

    png_path = path_base.with_suffix(".png")
    pdf_path = path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def build_figure(args: argparse.Namespace) -> dict[str, object]:
    results_root = Path(args.results_root)
    methods = (args.vip_method, args.gmvip_method)
    predictions = {
        method: _load_prediction(results_root, method, int(args.seed), int(args.target_id))
        for method in methods
    }
    written = plot_paper_lotka_volterra_figure(
        args.out,
        predictions_by_method=predictions,
        methods=methods,
        n_phase_samples=int(args.n_phase_samples),
    )
    return {
        "target_id": int(args.target_id),
        "seed": int(args.seed),
        "results_root": str(results_root),
        "methods": list(methods),
        "figures": written,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paper-ready VIP/GMVIP Lotka-Volterra combined figure."
    )
    parser.add_argument("--results-root", default="results/simprior_paper_ready_defaults/lotka_volterra")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--vip-method", default="vip")
    parser.add_argument("--gmvip-method", default="gmvip_empirical")
    parser.add_argument("--n-phase-samples", type=int, default=18)
    parser.add_argument(
        "--out",
        default="results/simprior_paper_ready_defaults/lotka_volterra_vip_gmvip_target0_combined_paper",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    summary = build_figure(args)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
