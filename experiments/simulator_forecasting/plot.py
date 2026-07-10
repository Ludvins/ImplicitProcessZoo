from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_RESULTS_ROOT = "results/simulator_forecasting_tobs15_20targets/simulator_forecasting"
DEFAULT_METHODS = ("vip", "ftip", "gmvip")
DEFAULT_TARGET_ID = 0
DEFAULT_N_TRAIN = 64
DEFAULT_T_OBS = 15.0

METHOD_LABELS = {
    "vip": "VIP",
    "ftip": "FTIP",
    "gmvip": "GMVIP",
    "gmvip_cov": "GMVIP",
    "gmvip_rbf": "GMVIP RBF",
    "map": "MAP",
    "deep_ensemble": "Deep Ensemble",
    "mfvi": "MFVI",
    "fbnn_observed": "fBNN observed",
    "fbnn_full": "fBNN full",
    "tfsvi_observed": "TFSVI observed",
    "tfsvi_full": "TFSVI full",
}

METHOD_ALIASES = {
    "gmvip_cov": "gmvip",
}

METHOD_DIR_ALIASES = {
    "gmvip": ("gmvip", "gmvip_cov"),
}

OBSERVED_WINDOW_COLOR = "#EAF2F8"
POSTERIOR_COLOR = "#4C78A8"
MEAN_COLOR = "#1F5A93"
VOLTERRA_COLUMN_WIDTH = 3.0
VOLTERRA_ROW_HEIGHT = 4.3 / 2.0
VOLTERRA_MIN_WIDTH = 9.0
TITLE_FONTSIZE = 12
TICK_LABELSIZE = 8.5
N_POSTERIOR_SAMPLE_LINES = 20


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _split_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _canonical_method(method: str) -> str:
    method = str(method).strip()
    return METHOD_ALIASES.get(method, method)


def _parse_methods(value: str | None) -> tuple[str, ...]:
    methods = tuple(_canonical_method(method) for method in _split_csv(value))
    return tuple(dict.fromkeys(methods))


def _parse_target_ids(args: argparse.Namespace) -> list[int]:
    if args.target_ids is None:
        return [int(args.target_id)]
    ids: list[int] = []
    for item in _split_csv(args.target_ids):
        if "-" in item:
            start_raw, stop_raw = item.split("-", 1)
            start, stop = int(start_raw), int(stop_raw)
            step = 1 if stop >= start else -1
            ids.extend(range(start, stop + step, step))
        else:
            ids.append(int(item))
    return list(dict.fromkeys(ids))


def _parse_labels(methods: tuple[str, ...], labels: str | None) -> dict[str, str]:
    if labels is None:
        return {method: METHOD_LABELS.get(method, method) for method in methods}
    parts = _split_csv(labels)
    if len(parts) != len(methods):
        raise ValueError(f"--labels must have {len(methods)} comma-separated values.")
    return dict(zip(methods, parts))


def _parse_method_roots(method_roots: list[str] | None) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for item in method_roots or []:
        if "=" not in item:
            raise ValueError(f"--method-root must have form method=path, got {item!r}.")
        method, root = item.split("=", 1)
        method = _canonical_method(method.strip())
        if not method:
            raise ValueError(f"--method-root must name a method, got {item!r}.")
        roots[method] = Path(root.strip())
    return roots


def _method_slug(method: str) -> str:
    label = METHOD_LABELS.get(method, method)
    label = label.replace("GMVIP RBF", "gmvip_rbf").replace("GMVIP", "gmvip")
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _path_base(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() in {".png", ".pdf", ".svg"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _default_out_dir(results_root: Path) -> Path:
    return results_root / "figures"


def _default_stem(target_id: int, n_train: int, methods: tuple[str, ...]) -> str:
    method_part = "_".join(_method_slug(method) for method in methods)
    return f"forecast_target{int(target_id)}_ntrain{int(n_train)}_{method_part}"


def _format_stem(stem: str, *, target_id: int, n_train: int, methods: tuple[str, ...]) -> str:
    return stem.format(
        target_id=int(target_id),
        n_train=int(n_train),
        methods="_".join(_method_slug(method) for method in methods),
    )


def _output_base(
    args: argparse.Namespace, *, target_id: int, methods: tuple[str, ...], n_targets: int
) -> Path:
    if args.out is not None:
        out = str(args.out)
        if n_targets > 1 and "{target_id}" not in out:
            out = f"{out}_target{{target_id}}"
        return _path_base(
            _format_stem(out, target_id=target_id, n_train=args.n_train, methods=methods)
        )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else _default_out_dir(Path(args.results_root))
    )
    stem = args.stem or _default_stem(target_id, int(args.n_train), methods)
    return _path_base(
        out_dir / _format_stem(stem, target_id=target_id, n_train=args.n_train, methods=methods)
    )


def _load_prediction(
    results_root: Path, method: str, seed: int, target_id: int, n_train: int
) -> dict[str, np.ndarray]:
    candidates = METHOD_DIR_ALIASES.get(method, (method,))
    paths = [
        results_root
        / method_dir
        / f"seed_{seed}"
        / "predictions"
        / f"target_{target_id}_ntrain_{n_train}.npz"
        for method_dir in candidates
    ]
    path = next((candidate for candidate in paths if candidate.exists()), paths[0])
    if not path.exists():
        tried = ", ".join(str(candidate) for candidate in paths)
        raise FileNotFoundError(f"Missing prediction file for {method}; tried {tried}")
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values
    return values[:, 0]


def _quantile(prediction: dict[str, np.ndarray], key: str, q: float) -> np.ndarray:
    if key in prediction:
        return _vector(prediction[key])
    return _vector(np.quantile(prediction["samples"], q, axis=0))


def _sample_curve(sample: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample)
    if sample.ndim == 1:
        return sample
    return sample[:, 0]


def plot_forecast_grid(
    path_base: str | Path,
    *,
    predictions_by_method: dict[str, dict[str, np.ndarray]],
    methods: tuple[str, ...] = DEFAULT_METHODS,
    labels: dict[str, str] | None = None,
    t_obs: float = DEFAULT_T_OBS,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 180,
) -> dict[str, str]:
    plt = _plt()
    path_base = _path_base(path_base)
    labels = labels or {method: METHOD_LABELS.get(method, method) for method in methods}
    width = max(VOLTERRA_COLUMN_WIDTH * len(methods), VOLTERRA_MIN_WIDTH)

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(width, VOLTERRA_ROW_HEIGHT),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax, method in zip(axes, methods):
        pred = predictions_by_method[method]
        t = np.asarray(pred["t"])
        truth = _vector(pred["truth"])
        train_t = np.asarray(pred["train_t"])
        train_y = _vector(pred["train_y"])
        mean = _vector(pred["mean"])
        q05 = _quantile(pred, "q05", 0.05)
        q95 = _quantile(pred, "q95", 0.95)

        ax.axvspan(
            0.0, float(t_obs), color=OBSERVED_WINDOW_COLOR, alpha=0.7, linewidth=0, zorder=-2
        )
        ax.axvline(float(t_obs), color="black", linestyle="--", linewidth=0.85, alpha=0.6, zorder=2)
        if "samples" in pred:
            for sample in np.asarray(pred["samples"])[:N_POSTERIOR_SAMPLE_LINES]:
                ax.plot(
                    t,
                    _sample_curve(sample),
                    color=POSTERIOR_COLOR,
                    alpha=0.20,
                    linewidth=0.8,
                    zorder=1,
                )
        ax.fill_between(t, q05, q95, color=POSTERIOR_COLOR, alpha=0.22, linewidth=0, zorder=2)
        ax.plot(t, mean, color=MEAN_COLOR, linewidth=1.9, zorder=4)
        ax.plot(t, truth, color="black", linewidth=1.2, zorder=5)
        ax.scatter(train_t, train_y, color="black", s=18, zorder=6)
        ax.set_title(labels.get(method, method), fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
        ax.grid(alpha=0.22)
        ax.set_xlim(0.0, float(t[-1]))

    fig.tight_layout()

    written: dict[str, str] = {}
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt:
            continue
        path = path_base.with_suffix(f".{fmt}")
        if fmt == "png":
            fig.savefig(path, dpi=int(dpi))
        else:
            fig.savefig(path)
        written[fmt] = str(path)
    plt.close(fig)
    return written


def build_figures(args: argparse.Namespace) -> dict[str, object]:
    methods = _parse_methods(args.methods)
    if not methods:
        raise ValueError("At least one method/model is required.")
    target_ids = _parse_target_ids(args)
    labels = _parse_labels(methods, args.labels)
    formats = _split_csv(args.formats) or ("png", "pdf")
    default_root = Path(args.results_root)
    root_overrides = _parse_method_roots(args.method_root)

    figures = []
    for target_id in target_ids:
        predictions = {
            method: _load_prediction(
                results_root=root_overrides.get(method, default_root),
                method=method,
                seed=int(args.seed),
                target_id=int(target_id),
                n_train=int(args.n_train),
            )
            for method in methods
        }
        out = _output_base(
            args, target_id=int(target_id), methods=methods, n_targets=len(target_ids)
        )
        written = plot_forecast_grid(
            out,
            predictions_by_method=predictions,
            methods=methods,
            labels=labels,
            t_obs=float(args.t_obs),
            formats=formats,
            dpi=int(args.dpi),
        )
        figures.append({"target_id": int(target_id), "figures": written})

    return {
        "seed": int(args.seed),
        "n_train": int(args.n_train),
        "t_obs": float(args.t_obs),
        "results_root": str(default_root),
        "method_roots": {method: str(root) for method, root in root_overrides.items()},
        "methods": list(methods),
        "target_ids": [int(target_id) for target_id in target_ids],
        "figures": figures,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot simulator forecasting posterior trajectories."
    )
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--method-root",
        action="append",
        default=[],
        help="Per-method result root override: method=path.",
    )
    parser.add_argument("--methods", "--models", dest="methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--labels", default=None, help="Optional comma-separated display labels matching --methods."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=DEFAULT_TARGET_ID)
    parser.add_argument(
        "--target-ids", default=None, help="Comma-separated ids or inclusive ranges, e.g. 0,3-5."
    )
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--t-obs", type=float, default=DEFAULT_T_OBS)
    parser.add_argument(
        "--out",
        default=None,
        help="Output path base. Use {target_id}, {n_train}, or {methods} placeholders.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--stem",
        default=None,
        help="Filename stem for --out-dir. Supports {target_id}, {n_train}, {methods}.",
    )
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    summary = build_figures(args)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
