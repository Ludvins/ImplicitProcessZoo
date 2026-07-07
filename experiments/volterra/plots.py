from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path_base: str | Path, **savefig_kwargs) -> None:
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=180, **savefig_kwargs)
    fig.savefig(path_base.with_suffix(".pdf"), **savefig_kwargs)


DEFAULT_METHOD_LABELS = {
    "map": "MAP",
    "mfvi": "MFVI",
    "vip": "VIP",
    "ftip": "FTIP",
    "sip": "SIP",
    "gmvip_empirical": "GM-VIP empirical",
    "gmvip_rbf": "GM-VIP RBF",
    "oracle_prior_bank": "Oracle prior bank",
}


DEFAULT_METHOD_COLORS = {
    "map": "#9D755D",
    "mfvi": "#B279A2",
    "vip": "#4C78A8",
    "ftip": "#F58518",
    "sip": "#E45756",
    "gmvip_empirical": "#2A9D8F",
    "gmvip_rbf": "#54A24B",
    "oracle_prior_bank": "#6F4E7C",
}


def _finite_minmax(values: list[np.ndarray], pad_fraction: float = 0.06) -> tuple[float, float]:
    finite = [np.ravel(value[np.isfinite(value)]) for value in values]
    finite = [value for value in finite if value.size]
    if not finite:
        return -1.0, 1.0
    all_values = np.concatenate(finite)
    lo = float(all_values.min())
    hi = float(all_values.max())
    pad = float(pad_fraction) * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def plot_lv_posterior_trajectory(
    path_base: str | Path,
    *,
    t: np.ndarray,
    y_true: np.ndarray,
    train_t: np.ndarray,
    train_y: np.ndarray,
    samples: np.ndarray,
    method: str,
    n_samples: int = 10,
) -> None:
    plt = _plt()
    mean = samples.mean(axis=0)
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    species = ("prey", "predator")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for dim, ax in enumerate(axes):
        ax.axvspan(0.0, 15.0, color="#EAF2F8", alpha=0.7, label="train" if dim == 0 else None)
        ax.axvspan(15.0, 20.0, color="#F7F4E8", alpha=0.7, label="val" if dim == 0 else None)
        ax.axvspan(20.0, float(t[-1]), color="#F2ECEC", alpha=0.7, label="test" if dim == 0 else None)
        for sample in samples[:n_samples, :, dim]:
            ax.plot(t, sample, color="#4C78A8", alpha=0.22, linewidth=1)
        ax.fill_between(t, lower[:, dim], upper[:, dim], color="#4C78A8", alpha=0.22)
        ax.plot(t, mean[:, dim], color="#1F5A93", linewidth=2, label="posterior mean" if dim == 0 else None)
        ax.plot(t, y_true[:, dim], color="black", linewidth=1.5, label="truth" if dim == 0 else None)
        ax.scatter(train_t, train_y[:, dim], color="black", s=18, zorder=4, label="observed" if dim == 0 else None)
        ax.set_ylabel(species[dim])
        ax.grid(alpha=0.2)
    axes[0].set_title(method)
    axes[-1].set_xlabel("time")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    _save(fig, path_base)
    plt.close(fig)


def plot_lv_phase_portrait(
    path_base: str | Path,
    *,
    y_true: np.ndarray,
    train_y: np.ndarray,
    samples: np.ndarray,
    method: str,
    n_samples: int = 10,
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    for sample in samples[:n_samples]:
        ax.plot(sample[:, 0], sample[:, 1], color="#4C78A8", alpha=0.28, linewidth=1)
    ax.plot(y_true[:, 0], y_true[:, 1], color="black", linewidth=1.8, label="truth")
    ax.scatter(train_y[:, 0], train_y[:, 1], color="#D55E00", s=20, zorder=4, label="observed")
    ax.set_xlabel("prey")
    ax.set_ylabel("predator")
    ax.set_title(method)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path_base)
    plt.close(fig)


def plot_lv_prior_vs_posterior(
    path_base: str | Path,
    *,
    t: np.ndarray,
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    n_samples: int = 20,
) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    species = ("prey", "predator")
    for dim in range(2):
        for sample in prior_samples[:n_samples, :, dim]:
            axes[dim, 0].plot(t, sample, color="#72B7B2", alpha=0.25, linewidth=1)
        for sample in posterior_samples[:n_samples, :, dim]:
            axes[dim, 1].plot(t, sample, color="#4C78A8", alpha=0.25, linewidth=1)
        axes[dim, 0].set_ylabel(species[dim])
        axes[dim, 0].grid(alpha=0.2)
        axes[dim, 1].grid(alpha=0.2)
    axes[0, 0].set_title("prior samples")
    axes[0, 1].set_title("posterior samples")
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    fig.tight_layout()
    _save(fig, path_base)
    plt.close(fig)


def plot_lv_shared_axis_method_comparison(
    path_base: str | Path,
    *,
    predictions_by_method: dict[str, dict[str, np.ndarray]],
    methods: list[str] | tuple[str, ...] | None = None,
    method_labels: dict[str, str] | None = None,
    method_colors: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    """Plot Lotka-Volterra method predictions with comparable axes.

    The prey column shares one y-axis across methods, and the predator column
    shares another. This avoids visually favoring methods through independent
    axis scaling.
    """
    plt = _plt()
    methods = list(methods or predictions_by_method)
    labels = {**DEFAULT_METHOD_LABELS, **(method_labels or {})}
    colors = {**DEFAULT_METHOD_COLORS, **(method_colors or {})}

    ylims = []
    for dim in range(2):
        values: list[np.ndarray] = []
        for method in methods:
            pred = predictions_by_method[method]
            samples = pred["samples"]
            lower, upper = np.quantile(samples, [0.05, 0.95], axis=0)
            values.extend(
                [
                    pred["y_true"][:, dim],
                    pred["y_train"][:, dim],
                    pred["mean"][:, dim],
                    lower[:, dim],
                    upper[:, dim],
                ]
            )
        ylims.append(_finite_minmax(values))

    fig, axes = plt.subplots(
        len(methods),
        2,
        figsize=(12.8, 2.55 * len(methods)),
        sharex=True,
        sharey="col",
        constrained_layout=True,
    )
    if len(methods) == 1:
        axes = np.expand_dims(axes, axis=0)
    species = ("prey", "predator")
    for row, method in enumerate(methods):
        pred = predictions_by_method[method]
        t = pred["t_plot"]
        y_true = pred["y_true"]
        y_train_x = pred["y_train_x"]
        y_train = pred["y_train"]
        samples = pred["samples"]
        mean = pred["mean"]
        lower, upper = np.quantile(samples, [0.05, 0.95], axis=0)
        color = colors.get(method, "#4C78A8")
        for dim, name in enumerate(species):
            ax = axes[row, dim]
            ax.fill_between(t, lower[:, dim], upper[:, dim], color=color, alpha=0.18, linewidth=0)
            ax.plot(t, mean[:, dim], color=color, linewidth=1.8, label="posterior mean")
            ax.plot(t, y_true[:, dim], color="black", linewidth=1.2, label="truth")
            ax.scatter(y_train_x, y_train[:, dim], s=11, color="black", alpha=0.62, zorder=4, label="train obs")
            ax.axvspan(15.0, 20.0, color="#C7CEDB", alpha=0.16, linewidth=0)
            ax.axvline(20.0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.55)
            ax.set_ylim(*ylims[dim])
            ax.grid(alpha=0.22)
            if row == 0:
                ax.set_title(name)
            if dim == 0:
                ax.set_ylabel(labels.get(method, method))
            if row == len(methods) - 1:
                ax.set_xlabel("time")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:3], legend_labels[:3], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    if title is not None:
        fig.suptitle(title, y=1.055, fontsize=12)
    _save(fig, path_base, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curve(path_base: str | Path, curves: dict[str, dict[float, float]]) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1, linestyle="--")
    for method, coverage in curves.items():
        levels = np.array(sorted(coverage), dtype=np.float64)
        values = np.array([coverage[level] for level in levels], dtype=np.float64)
        ax.plot(levels, values, marker="o", label=method)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path_base)
    plt.close(fig)


def plot_metric_bars(path_base: str | Path, metrics_by_method: dict[str, dict[str, float]], metric_names: tuple[str, ...]) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.2 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]
    methods = list(metrics_by_method)
    x = np.arange(len(methods))
    for ax, metric in zip(axes, metric_names):
        values = [metrics_by_method[method].get(metric, np.nan) for method in methods]
        ax.bar(x, values, color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, path_base)
    plt.close(fig)
