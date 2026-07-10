from __future__ import annotations

import argparse
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
    "vip": "#4e79a7",
    "vip_512": "#59a14f",
    "ftip": "#f28e2b",
    "gmvip_empirical": "#b07aa1",
}

DEFAULT_LABELS = {
    "analog": "Analog",
    "seasonal_naive": "Seasonal naive",
    "vip": "VIP",
    "vip_512": "VIP-512",
    "ftip": "FTIP",
    "gmvip_empirical": "GMVIP",
}

PAPER_OBSERVED_WINDOW_COLOR = "#EAF2F8"
PAPER_POSTERIOR_COLOR = "#4C78A8"
PAPER_MEAN_COLOR = "#1F5A93"
PAPER_COLUMN_WIDTH = 3.0
PAPER_ROW_HEIGHT = 4.3 / 2.0
PAPER_MIN_WIDTH = 9.0
PAPER_TITLE_FONTSIZE = 12
PAPER_TICK_LABELSIZE = 8.5
PAPER_POSTERIOR_SAMPLE_LINES = 20


def _method_dirs(results_root: Path, methods: list[str] | None, *, seed: int = 0) -> list[Path]:
    if methods:
        return [results_root / method / f"seed_{seed}" for method in methods]
    return sorted(path for path in results_root.glob(f"*/seed_{seed}") if path.is_dir())


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
    data = np.load(path)
    if "methodology_version" not in data or int(data["methodology_version"]) != 2:
        raise ValueError(f"Refusing non-v2 ELD prediction artifact: {path}")
    return data


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
) -> list[tuple[str, np.lib.npyio.NpzFile]]:
    method_dirs = _method_dirs(results_root, methods, seed=seed)
    loaded = []
    for method_dir in method_dirs:
        data = _load_prediction(method_dir, target_id)
        if data is not None:
            method = method_dir.parent.name
            loaded.append((method, data))
    if not loaded:
        raise FileNotFoundError(
            f"No prediction files found for target {target_id} seed {seed} under {results_root}."
        )
    return loaded


def plot_target(
    results_root: Path,
    target_id: int,
    output_dir: Path,
    methods: list[str] | None = None,
    *,
    seed: int = 0,
    formats: list[str] | None = None,
) -> list[Path]:
    plt = _pyplot()
    formats = formats or ["png"]
    loaded = _load_target_predictions(results_root, target_id, methods, seed=seed)
    reference = loaded[0][1]
    t, truth, context_t, context_y = _reference_series(reference)
    t_obs = float(context_t[-1]) if context_t.size else 0.0

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, truth, color="black", linewidth=2.0, label="truth", zorder=5)
    ax.scatter(context_t, context_y, color="black", s=18, label="observed context", zorder=6)
    ax.axvline(t_obs, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

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
    formats: list[str] | None = None,
) -> list[Path]:
    plt = _pyplot()
    formats = formats or ["png"]
    loaded = _load_target_predictions(results_root, target_id, methods, seed=seed)
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
        t_obs = float(train_t[-1]) if train_t.size else 0.0

        ax.axvspan(0.0, t_obs, color=PAPER_OBSERVED_WINDOW_COLOR, alpha=0.7, linewidth=0, zorder=-2)
        ax.axvline(t_obs, color="black", linestyle="--", linewidth=0.85, alpha=0.6, zorder=2)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ELD prediction illustrations.")
    parser.add_argument("--results-root", default="results/eld_forecasting_v2")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--target-ids", default=None, help="Comma-separated target ids. Defaults to all available."
    )
    parser.add_argument("--methods", default=None, help="Comma-separated methods to include.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=("overlay", "method_grid"), default="overlay")
    parser.add_argument(
        "--formats", default="png", help="Comma-separated output formats, e.g. png,pdf."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "figures"
    methods = (
        [item.strip() for item in args.methods.split(",") if item.strip()] if args.methods else None
    )
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    method_dirs = _method_dirs(results_root, methods, seed=int(args.seed))
    if args.target_ids:
        target_ids = [int(item.strip()) for item in args.target_ids.split(",") if item.strip()]
    else:
        target_ids = _available_target_ids(method_dirs)
    outputs = []
    for target_id in target_ids:
        if args.layout == "method_grid":
            grid_methods = methods or ["vip", "ftip", "gmvip_empirical"]
            outputs.extend(
                plot_target_method_grid(
                    results_root,
                    target_id,
                    output_dir,
                    grid_methods,
                    seed=int(args.seed),
                    formats=formats,
                )
            )
        else:
            outputs.extend(
                plot_target(
                    results_root,
                    target_id,
                    output_dir,
                    methods=methods,
                    seed=int(args.seed),
                    formats=formats,
                )
            )
    for path in outputs:
        print(path)
    return outputs


if __name__ == "__main__":
    main()
