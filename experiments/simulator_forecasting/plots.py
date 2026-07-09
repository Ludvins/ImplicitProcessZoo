from __future__ import annotations

from pathlib import Path

import numpy as np


METHOD_LABELS = {
    "gmvip": "GMVIP",
    "gmvip_cov": "GM-VIP Cov",
    "gmvip_rbf": "GM-VIP RBF",
    "vip": "VIP",
    "ftip": "FTIP",
    "sip": "SIP",
    "map": "MAP",
    "deep_ensemble": "Deep Ensemble",
    "mfvi": "MFVI",
    "fbnn_observed": "fBNN observed",
    "fbnn_full": "fBNN full",
    "tfsvi_observed": "TFSVI observed",
    "tfsvi_full": "TFSVI full",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path_base: str | Path) -> None:
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))


def plot_prior_samples(
    path_base: str | Path,
    *,
    t: np.ndarray,
    samples: np.ndarray,
    n_samples: int = 20,
    t_obs: float = 8.0,
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for sample in samples[:n_samples, :, 0]:
        ax.plot(t, sample, color="#2A9D8F", alpha=0.28, linewidth=1)
    ax.axvline(float(t_obs), color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("time")
    ax.set_ylabel("x(t)")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    _save(fig, path_base)
    plt.close(fig)


def plot_posterior_forecast(
    path_base: str | Path,
    *,
    t: np.ndarray,
    y_true: np.ndarray,
    train_t: np.ndarray,
    train_y: np.ndarray,
    samples: np.ndarray,
    method: str,
    n_samples: int = 20,
    t_obs: float = 8.0,
) -> None:
    plt = _plt()
    mean = samples.mean(axis=0)
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.axvspan(0.0, float(t_obs), color="#EAF2F8", alpha=0.7)
    ax.axvline(float(t_obs), color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    for sample in samples[:n_samples, :, 0]:
        ax.plot(t, sample, color="#4C78A8", alpha=0.20, linewidth=0.9)
    ax.fill_between(t, lower[:, 0], upper[:, 0], color="#4C78A8", alpha=0.22)
    ax.plot(t, mean[:, 0], color="#1F5A93", linewidth=2.0, label="posterior mean")
    ax.plot(t, y_true[:, 0], color="black", linewidth=1.3, label="truth")
    ax.scatter(train_t, train_y[:, 0], color="black", s=18, zorder=4, label="observed")
    ax.set_title(METHOD_LABELS.get(method, method))
    ax.set_xlabel("time")
    ax.set_ylabel("x(t)")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path_base)
    plt.close(fig)


def plot_metric_by_region(path_base: str | Path, *, rows: list[dict], metric: str) -> None:
    plt = _plt()
    preferred = ["interpolation", "near_extrapolation", "medium_extrapolation", "far_extrapolation"]
    present = list(dict.fromkeys(str(row["region"]) for row in rows))
    regions = [region for region in preferred if region in present] + [region for region in present if region not in preferred]
    methods = list(dict.fromkeys(row["method"] for row in rows))
    x = np.arange(len(regions))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for idx, method in enumerate(methods):
        values = []
        for region in regions:
            vals = [float(row[metric]) for row in rows if row["method"] == method and row["region"] == region]
            values.append(float(np.mean(vals)) if vals else np.nan)
        ax.bar(x + (idx - 0.5 * (len(methods) - 1)) * width, values, width=width, label=METHOD_LABELS.get(method, method))
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=18, ha="right")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, path_base)
    plt.close(fig)
