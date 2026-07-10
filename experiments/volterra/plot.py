from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_RESULTS_ROOT = Path("results/simprior_paper_ready_defaults/lotka_volterra")
DEFAULT_METHOD_ROOTS = {
    "vip": DEFAULT_RESULTS_ROOT,
    "ftip": Path("results/simprior_search_ordering/ftip_steps625_mc8_coeff128/lotka_volterra"),
    "gmvip_empirical": Path(
        "results/simprior_search_ordering/gmvip_bank512_z96_beta1_steps800/lotka_volterra"
    ),
}
DEFAULT_OUTPUT_DIR = Path("results/simprior_search_ordering")

METHOD_LABELS = {
    "map": "MAP",
    "mfvi": "MFVI",
    "vip": "VIP",
    "ftip": "FTIP",
    "sip": "SIP",
    "gmvip_empirical": "GMVIP",
    "gmvip_rbf": "GMVIP-RBF",
    "oracle_prior_bank": "Oracle",
}
METHOD_SLUGS = {
    "gmvip_empirical": "gmvip",
    "oracle_prior_bank": "oracle",
}

SPECIES_COLORS = ("#4C78A8", "#F58518")
SPECIES_MEAN_COLORS = ("#1F5A93", "#D95F02")
OBSERVED_WINDOW_COLOR = "#EAF2F8"
OBSERVED_CUTOFF = 15.0
N_POSTERIOR_SAMPLE_LINES = 20


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _path_base(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() in {".png", ".pdf"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _method_slug(method: str) -> str:
    return METHOD_SLUGS.get(method, method)


def _parse_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(method.strip() for method in str(raw).split(",") if method.strip())
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    return methods


def _parse_method_roots(values: list[str] | None) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected --method-root METHOD=PATH, got {value!r}.")
        method, root = value.split("=", 1)
        method = method.strip()
        if not method:
            raise ValueError(f"Missing method name in --method-root {value!r}.")
        roots[method] = Path(root.strip())
    return roots


def _resolved_method_roots(
    methods: tuple[str, ...],
    *,
    results_root: Path,
    method_root_overrides: dict[str, Path],
) -> dict[str, Path]:
    use_default_roots = results_root == DEFAULT_RESULTS_ROOT
    roots = {
        method: DEFAULT_METHOD_ROOTS.get(method, results_root)
        if use_default_roots
        else results_root
        for method in methods
    }
    roots.update(method_root_overrides)
    return roots


def _load_prediction(
    results_root: Path, method: str, seed: int, target_id: int
) -> dict[str, np.ndarray]:
    path = results_root / method / f"seed_{seed}" / "predictions" / f"target_{target_id}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file for {method}, seed {seed}, target {target_id}: {path}"
        )
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _available_target_ids(results_root: Path, method: str, seed: int) -> list[int]:
    pred_dir = results_root / method / f"seed_{seed}" / "predictions"
    ids: list[int] = []
    for path in pred_dir.glob("target_*.npz"):
        match = re.fullmatch(r"target_(\d+)", path.stem)
        if match:
            ids.append(int(match.group(1)))
    if not ids:
        raise FileNotFoundError(f"No target prediction files found in {pred_dir}")
    return sorted(ids)


def _parse_target_ids(
    raw: str, *, first_root: Path, first_method: str, seed: int
) -> tuple[int, ...]:
    raw = str(raw).strip()
    if raw.lower() == "all":
        return tuple(_available_target_ids(first_root, first_method, seed))
    ids: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, stop = chunk.split("-", 1)
            start_id = int(start)
            stop_id = int(stop)
            step = 1 if stop_id >= start_id else -1
            ids.extend(range(start_id, stop_id + step, step))
        else:
            ids.append(int(chunk))
    if not ids:
        raise ValueError("--target-ids must contain at least one target id.")
    return tuple(dict.fromkeys(ids))


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


def _shared_species_limits(
    predictions: dict[str, dict[str, np.ndarray]],
) -> tuple[tuple[float, float], ...]:
    limits: list[tuple[float, float]] = []
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
        limits.append(_finite_minmax(values))
    return tuple(limits)


def plot_trajectory_grid(
    path_base: str | Path,
    *,
    predictions_by_method: dict[str, dict[str, np.ndarray]],
    methods: tuple[str, ...],
) -> dict[str, str]:
    plt = _plt()
    path_base = _path_base(path_base)
    y_limits = _shared_species_limits({method: predictions_by_method[method] for method in methods})

    fig, axes = plt.subplots(
        2,
        len(methods),
        figsize=(max(3.0 * len(methods), 9.0), 4.3),
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes).reshape(2, -1)

    for col, method in enumerate(methods):
        pred = predictions_by_method[method]
        t = pred["t_plot"]
        y_true = pred["y_true"]
        y_train_x = pred["y_train_x"]
        y_train = pred["y_train"]
        samples = pred["samples"]
        mean = pred["mean"]
        lower, upper = np.quantile(samples, [0.05, 0.95], axis=0)

        for dim, species_color in enumerate(SPECIES_COLORS):
            ax = axes[dim, col]
            ax.axvspan(
                0.0, OBSERVED_CUTOFF, color=OBSERVED_WINDOW_COLOR, alpha=0.7, linewidth=0, zorder=-2
            )
            ax.axvline(
                OBSERVED_CUTOFF,
                color="black",
                linestyle="--",
                linewidth=0.85,
                alpha=0.6,
                zorder=2,
            )
            for sample in samples[:N_POSTERIOR_SAMPLE_LINES, :, dim]:
                ax.plot(t, sample, color=species_color, alpha=0.20, linewidth=0.8, zorder=1)
            ax.fill_between(
                t,
                lower[:, dim],
                upper[:, dim],
                color=species_color,
                alpha=0.22,
                linewidth=0,
                zorder=2,
            )
            ax.plot(t, mean[:, dim], color=SPECIES_MEAN_COLORS[dim], linewidth=1.9, zorder=4)
            ax.plot(t, y_true[:, dim], color="black", linewidth=1.2, zorder=5)
            ax.scatter(y_train_x, y_train[:, dim], color="black", s=18, zorder=6)
            ax.set_ylim(*y_limits[dim])
            ax.grid(alpha=0.22)
            ax.tick_params(axis="both", labelsize=8.5)
            if col == 0:
                ax.set_ylabel("prey" if dim == 0 else "predator", fontsize=10)
            if dim == 1:
                ax.set_xlabel("time", fontsize=10)
            if dim == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=12)

    png_path = path_base.with_suffix(".png")
    pdf_path = path_base.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _output_base(
    args: argparse.Namespace, methods: tuple[str, ...], target_id: int, multiple_targets: bool
) -> Path:
    if args.out:
        base = _path_base(args.out)
        return base.parent / f"{base.name}_target{target_id}" if multiple_targets else base
    method_slug = "_".join(_method_slug(method) for method in methods)
    return Path(args.out_dir) / f"lotka_volterra_{method_slug}_target{target_id}"


def build_figures(args: argparse.Namespace) -> dict[str, object]:
    methods = _parse_methods(args.methods)
    results_root = Path(args.results_root)
    method_roots = _resolved_method_roots(
        methods,
        results_root=results_root,
        method_root_overrides=_parse_method_roots(args.method_root),
    )
    target_ids = _parse_target_ids(
        str(args.target_ids),
        first_root=method_roots[methods[0]],
        first_method=methods[0],
        seed=int(args.seed),
    )

    figure_rows = []
    for target_id in target_ids:
        predictions = {
            method: _load_prediction(method_roots[method], method, int(args.seed), int(target_id))
            for method in methods
        }
        written = plot_trajectory_grid(
            _output_base(args, methods, target_id, multiple_targets=len(target_ids) > 1),
            predictions_by_method=predictions,
            methods=methods,
        )
        figure_rows.append({"target_id": int(target_id), **written})

    return {
        "target_ids": [int(target_id) for target_id in target_ids],
        "seed": int(args.seed),
        "results_root": str(results_root),
        "method_roots": {method: str(method_roots[method]) for method in methods},
        "methods": list(methods),
        "figures": figure_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Lotka-Volterra posterior trajectories for saved experiment results."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--method-root",
        action="append",
        default=[],
        help="Override one method root as METHOD=PATH.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default="vip,ftip,gmvip_empirical")
    parser.add_argument(
        "--target-ids", default="9", help="Comma-separated ids, ranges like 0-4, or all."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--out",
        default=None,
        help="Single output path base. For multiple targets, target ids are appended.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    summary = build_figures(args)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
