"""Publication-style plots for the synthetic prior-fidelity experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

METHOD_COLORS = {
    "true_null": "#4d4d4d",
    "vip": "#1f77b4",
    "gmvip": "#bc4b51",
}
METHOD_LABELS = {
    "true_null": "True prior split",
    "vip": "VIP surrogate",
    "gmvip": "GMVIP surrogate",
}


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional experiment dependency
        raise RuntimeError(
            "Plotting requires matplotlib. Install the project's experiments dependencies."
        ) from exc
    return plt


def _save_figure(fig, path_base: Path, *, dpi: int = 220) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_prior_samples(
    output_dir: str | Path,
    payload: dict[str, np.ndarray],
    *,
    max_curves: int = 30,
) -> None:
    """Plot true, VIP, and GMVIP samples for their designated default settings."""
    plt = _pyplot()
    x = np.asarray(payload["x_grid"])
    panels = [
        ("true_reference", "Original BNN prior", "#4d4d4d"),
        ("vip_default", "VIP surrogate", METHOD_COLORS["vip"]),
        ("gmvip_default", "GMVIP surrogate", METHOD_COLORS["gmvip"]),
    ]
    all_values = np.concatenate([np.asarray(payload[key]).reshape(-1) for key, _, _ in panels])
    lower, upper = np.quantile(all_values, [0.005, 0.995])
    padding = 0.05 * max(upper - lower, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), sharex=True, sharey=True)
    for axis, (key, title, color) in zip(axes, panels):
        samples = np.asarray(payload[key])
        count = min(int(max_curves), samples.shape[0])
        indices = np.linspace(0, samples.shape[0] - 1, count, dtype=int)
        axis.plot(x, samples[indices].T, color=color, alpha=0.16, linewidth=0.75)
        axis.plot(x, samples.mean(axis=0), color=color, linewidth=1.8)
        q05, q95 = np.quantile(samples, [0.05, 0.95], axis=0)
        axis.fill_between(x, q05, q95, color=color, alpha=0.16, linewidth=0)
        axis.set_title(title)
        axis.set_xlabel(r"$x$")
        axis.grid(alpha=0.2, linewidth=0.6)
    axes[0].set_ylabel(r"$f(x)$")
    axes[0].set_ylim(lower - padding, upper + padding)
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "prior_samples")
    plt.close(fig)


def plot_pointwise_w1(
    output_dir: str | Path,
    profile_rows: list[dict],
) -> None:
    """Plot mean pointwise marginal W1 with seed-level standard errors."""
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    for method in ("true_null", "vip", "gmvip"):
        rows = [row for row in profile_rows if row["method"] == method]
        if not rows:
            continue
        x = np.asarray([row["x"] for row in rows], dtype=np.float64)
        mean = np.asarray([row["mean"] for row in rows], dtype=np.float64)
        stderr = np.asarray([row["stderr"] for row in rows], dtype=np.float64)
        color = METHOD_COLORS[method]
        axis.plot(x, mean, color=color, linewidth=1.8, label=METHOD_LABELS[method])
        axis.fill_between(x, mean - stderr, mean + stderr, color=color, alpha=0.16)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"Pointwise $W_1$")
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "pointwise_w1_default")
    plt.close(fig)


def _summary_value(row: dict, metric: str, suffix: str) -> float:
    return float(row[f"{metric}_{suffix}"])


def _null_band(axis, summary_rows: list[dict], metric: str) -> None:
    null_rows = [row for row in summary_rows if row["method"] == "true_null"]
    if not null_rows:
        return
    mean = _summary_value(null_rows[0], metric, "mean")
    stderr = _summary_value(null_rows[0], metric, "stderr")
    axis.axhline(
        mean,
        color=METHOD_COLORS["true_null"],
        linestyle="--",
        linewidth=1.2,
        label=METHOD_LABELS["true_null"],
    )
    axis.axhspan(
        max(0.0, mean - stderr),
        mean + stderr,
        color=METHOD_COLORS["true_null"],
        alpha=0.10,
        linewidth=0,
    )


def _plot_matched_metrics(
    output_dir: Path,
    summary_rows: list[dict],
    *,
    metrics: tuple[tuple[str, str], ...],
    filename: str,
) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 3.8), squeeze=False)
    for axis, (metric, ylabel) in zip(axes[0], metrics):
        _null_band(axis, summary_rows, metric)
        for method in ("vip", "gmvip"):
            rows = [
                row
                for row in summary_rows
                if row["method"] == method and bool(row["in_matched_sweep"])
            ]
            rows.sort(key=lambda row: int(row["coefficient_dim"]))
            if not rows:
                continue
            x = np.asarray([int(row["coefficient_dim"]) for row in rows])
            mean = np.asarray([_summary_value(row, metric, "mean") for row in rows])
            stderr = np.asarray([_summary_value(row, metric, "stderr") for row in rows])
            axis.errorbar(
                x,
                mean,
                yerr=stderr,
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=1.6,
                capsize=2.5,
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({int(value) for value in x}) if "x" in locals() else [])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_xlabel("Coefficient dimension")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2, linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(fig, output_dir / filename)
    plt.close(fig)


def plot_matched_sweeps(output_dir: str | Path, summary_rows: list[dict]) -> None:
    """Plot core, moment, and robustness metrics against coefficient dimension."""
    output_dir = Path(output_dir)
    _plot_matched_metrics(
        output_dir,
        summary_rows,
        metrics=(
            ("joint_sw2", r"Joint sliced $W_2$"),
            ("marginal_w1_mean", r"Mean marginal $W_1$"),
        ),
        filename="distance_vs_dimension",
    )
    _plot_matched_metrics(
        output_dir,
        summary_rows,
        metrics=(
            ("mean_rmse", "Standardized mean RMSE"),
            ("covariance_rel_fro", "Relative covariance error"),
        ),
        filename="moment_errors_vs_dimension",
    )
    _plot_matched_metrics(
        output_dir,
        summary_rows,
        metrics=(
            ("energy_distance", "Energy distance"),
            ("rbf_mmd2", r"RBF MMD$^2$"),
        ),
        filename="robustness_metrics_vs_dimension",
    )


def plot_gmvip_bank_sensitivity(
    output_dir: str | Path,
    summary_rows: list[dict],
) -> None:
    """Plot GMVIP distance sensitivity to its empirical operator bank size."""
    plt = _pyplot()
    rows = [row for row in summary_rows if row["method"] == "gmvip" and bool(row["in_bank_sweep"])]
    rows.sort(key=lambda row: int(row["operator_bank_size"]))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for axis, (metric, ylabel) in zip(
        axes,
        (
            ("joint_sw2", r"Joint sliced $W_2$"),
            ("marginal_w1_mean", r"Mean marginal $W_1$"),
        ),
    ):
        _null_band(axis, summary_rows, metric)
        x = np.asarray([int(row["operator_bank_size"]) for row in rows])
        mean = np.asarray([_summary_value(row, metric, "mean") for row in rows])
        stderr = np.asarray([_summary_value(row, metric, "stderr") for row in rows])
        axis.errorbar(
            x,
            mean,
            yerr=stderr,
            color=METHOD_COLORS["gmvip"],
            marker="o",
            linewidth=1.6,
            capsize=2.5,
            label=METHOD_LABELS["gmvip"],
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(x)
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_xlabel("GMVIP operator-bank size")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(fig, Path(output_dir) / "gmvip_bank_sensitivity")
    plt.close(fig)
