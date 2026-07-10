from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import yaml

from experiments.common import write_csv_rows
from experiments.eld_forecasting import run as base_run
from experiments.eld_forecasting.datasets import (
    _select_targets,
    build_window_specs,
    load_processed,
    make_task_from_spec,
    processed_exists,
)
from experiments.eld_forecasting.metrics import metrics_by_region

CANDIDATE_PRIOR_SELECTIONS = (
    "same_client_prefix_nn",
    "same_client_calendar_prefix_nn",
    "other_client_prefix_nn",
    "other_client_calendar_prefix_nn",
    "other_client_calendar",
    "calendar_prefix_nn",
    "prefix_nn",
    "calendar",
)


def _hours_to_points(hours: float) -> int:
    points = int(round(float(hours) / 0.25))
    if abs(points * 0.25 - float(hours)) > 1.0e-8:
        raise ValueError("ELD windows must be multiples of 15 minutes.")
    return points


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_years(value: str) -> set[int]:
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def _device_dtype(args: argparse.Namespace):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return device, dtype


def _base_config(
    args: argparse.Namespace, *, prefix_points: int, window_points: int, prior_selection: str
) -> dict:
    config = copy.deepcopy(base_run.PILOT_CONFIG)
    config["method"] = args.methods
    config["data"].update(
        {
            "root": args.root,
            "split": "test",
            "n_targets": args.n_targets,
            "window_length": window_points,
            "prefix_length": prefix_points,
            "prior_years": sorted(_parse_years(args.prior_years)),
            "target_years": sorted(_parse_years(args.target_years)),
            "noise_std_norm": args.noise_std_norm,
        }
    )
    config["prior"].update({"bank_size": args.prior_bank_size, "selection": prior_selection})
    config["training"].update(
        {
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "n_mc_train": args.n_mc_train,
            "n_mc_eval": args.n_mc_eval,
            "regression_coeffs": args.regression_coeffs,
            "disable_tqdm": args.disable_tqdm,
        }
    )
    config["gmvip"].update(
        {
            "num_inducing": args.num_inducing if args.num_inducing is not None else prefix_points,
            "beta": args.gmvip_beta,
        }
    )
    return config


def _final_regions(train_hours: float, val_hours: float, window_hours: float) -> dict:
    context_end = float(train_hours) + float(val_hours)
    regions = {
        "observed_context": (0.0, context_end, True),
        "test_forecast": (context_end, float(window_hours), False),
    }
    if context_end < 24.0:
        regions["same_day_test"] = (context_end, min(24.0, float(window_hours)), False)
    if window_hours > 24.0:
        regions["next_day_test"] = (max(24.0, context_end), float(window_hours), False)
    return regions


def _make_task(
    data,
    specs,
    target,
    *,
    target_id: int,
    prefix_points: int,
    window_points: int,
    prior_selection: str,
    args: argparse.Namespace,
    seed: int,
    device,
    dtype,
):
    prior_years = _parse_years(args.prior_years)
    target_years = _parse_years(args.target_years)
    task = make_task_from_spec(
        data,
        specs,
        target,
        target_id=target_id,
        prior_years=prior_years,
        target_years=target_years,
        split_name="test",
        bank_size=args.prior_bank_size,
        window_length=window_points,
        prefix_length=prefix_points,
        noise_std_norm=args.noise_std_norm,
        seed=seed,
        device=device,
        dtype=dtype,
        prefix_eps=1.0e-3,
        prior_selection=prior_selection,
    )
    return task


def _score_validation(
    model, method: str, task, config: dict, *, seed: int, train_hours: float, val_hours: float
) -> dict:
    eval_samples = int(base_run._training_config(config).n_mc_eval)
    samples_norm = base_run.predictive_function_samples(
        model, method, task.X_plot, eval_samples, seed + 701
    )
    samples = base_run._unnormalize(task, samples_norm)
    y_true = base_run._unnormalize(task, task.y_plot_true)
    t_grid = torch.as_tensor(task.metadata["t_grid"], dtype=samples.dtype, device=samples.device)
    noise_std = torch.as_tensor(
        [float(task.metadata["sigma_y"])], dtype=samples.dtype, device=samples.device
    )
    rows = metrics_by_region(
        samples,
        y_true,
        t_grid,
        noise_std,
        levels=tuple(config.get("metrics", {}).get("levels", [0.9, 0.95])),
        regions={"validation": (float(train_hours), float(train_hours) + float(val_hours), False)},
    )
    return rows["validation"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    write_csv_rows(path, rows)


def _select_prior_rules(
    targets,
    data,
    specs,
    *,
    args: argparse.Namespace,
    device,
    dtype,
    train_points: int,
    window_points: int,
) -> tuple[dict[int, str], list[dict]]:
    candidates = _parse_csv(args.candidate_prior_selections)
    unknown = sorted(set(candidates) - set(CANDIDATE_PRIOR_SELECTIONS))
    if unknown:
        raise ValueError(f"Unknown candidate prior selections: {unknown}")
    selection_config = _base_config(
        args,
        prefix_points=train_points,
        window_points=window_points,
        prior_selection=candidates[0],
    )
    selection_config["method"] = args.selection_method
    selection_config["training"]["max_steps"] = args.selection_max_steps
    selection_config["gmvip"]["num_inducing"] = train_points
    selection_config["metrics"]["regions"] = {
        "validation": (
            float(args.train_hours),
            float(args.train_hours) + float(args.val_hours),
            False,
        )
    }

    chosen: dict[int, str] = {}
    rows: list[dict] = []
    for target_id, target in enumerate(targets):
        best_rule = None
        best_score = math.inf
        for cand_idx, prior_selection in enumerate(candidates):
            target_seed = int(args.seed) + 1000 * target_id + 37 * cand_idx
            config = copy.deepcopy(selection_config)
            config["prior"]["selection"] = prior_selection
            task = _make_task(
                data,
                specs,
                target,
                target_id=target_id,
                prefix_points=train_points,
                window_points=window_points,
                prior_selection=prior_selection,
                args=args,
                seed=target_seed,
                device=device,
                dtype=dtype,
            )
            model = base_run.build_model(
                args.selection_method, task, config, seed=target_seed, device=device, dtype=dtype
            )
            train_info = base_run.fit_model(model, task, config, device=device)
            metrics = _score_validation(
                model,
                args.selection_method,
                task,
                config,
                seed=target_seed,
                train_hours=args.train_hours,
                val_hours=args.val_hours,
            )
            score = float(metrics[args.selection_metric])
            if score < best_score:
                best_score = score
                best_rule = prior_selection
            rows.append(
                {
                    "target_id": target_id,
                    "client_id": task.metadata["client_id"],
                    "start_time": task.metadata["start_time"],
                    "candidate_prior_selection": prior_selection,
                    "selection_method": args.selection_method,
                    "selection_metric": args.selection_metric,
                    "selection_score": score,
                    "train_hours": float(args.train_hours),
                    "val_hours": float(args.val_hours),
                    "train_time_sec": train_info["train_time_sec"],
                    "train_steps": train_info["steps"],
                    **{f"validation_{key}": value for key, value in metrics.items()},
                }
            )
        if best_rule is None:
            raise RuntimeError("No prior-selection candidate was evaluated.")
        chosen[target_id] = best_rule
        for row in rows:
            if row["target_id"] == target_id:
                row["selected_prior_selection"] = best_rule
                row["selected"] = row["candidate_prior_selection"] == best_rule
    return chosen, rows


def _run_final_methods(
    targets,
    data,
    specs,
    selected: dict[int, str],
    *,
    args: argparse.Namespace,
    device,
    dtype,
    context_points: int,
    window_points: int,
    output_dir: Path,
) -> dict:
    methods = base_run._requested_methods(args.methods)
    final_config = _base_config(
        args,
        prefix_points=context_points,
        window_points=window_points,
        prior_selection="validation_selected",
    )
    final_config["metrics"]["regions"] = _final_regions(
        args.train_hours, args.val_hours, args.window_hours
    )
    final_config["validation_bank_selection"] = {
        "train_hours": float(args.train_hours),
        "val_hours": float(args.val_hours),
        "test_hours": float(args.window_hours - args.train_hours - args.val_hours),
        "selection_method": args.selection_method,
        "selection_metric": args.selection_metric,
        "candidate_prior_selections": _parse_csv(args.candidate_prior_selections),
    }

    summaries = {}
    for method in methods:
        method_dir = output_dir / method / f"seed_{args.seed}"
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "config.yaml").write_text(
            yaml.safe_dump(final_config, sort_keys=False), encoding="utf-8"
        )
        rows = []
        runtimes = []
        for target_id, target in enumerate(targets):
            target_seed = int(args.seed) + 1000 * target_id
            prior_selection = selected[target_id]
            task = _make_task(
                data,
                specs,
                target,
                target_id=target_id,
                prefix_points=context_points,
                window_points=window_points,
                prior_selection=prior_selection,
                args=args,
                seed=target_seed,
                device=device,
                dtype=dtype,
            )
            task.metadata.update(
                {
                    "selection_protocol": "train_context_validation_refit_test",
                    "selection_train_hours": float(args.train_hours),
                    "selection_val_hours": float(args.val_hours),
                    "selection_context_hours": float(args.train_hours + args.val_hours),
                    "selection_test_hours": float(
                        args.window_hours - args.train_hours - args.val_hours
                    ),
                    "selection_method": args.selection_method,
                    "selection_metric": args.selection_metric,
                    "selected_prior_selection": prior_selection,
                }
            )
            model = base_run.build_model(
                method, task, final_config, seed=target_seed, device=device, dtype=dtype
            )
            train_info = base_run.fit_model(model, task, final_config, device=device)
            target_rows = base_run.evaluate_target(
                model, method, task, final_config, seed=target_seed, out_dir=method_dir
            )
            for row in target_rows:
                row["train_time_sec"] = float(train_info["train_time_sec"])
                row["train_steps"] = int(train_info["steps"])
                row["loss_start"] = train_info["loss_start"]
                row["loss_end"] = train_info["loss_end"]
                row["selection_method"] = args.selection_method
                row["selection_metric"] = args.selection_metric
                row["selected_prior_selection"] = prior_selection
                row["selection_train_hours"] = float(args.train_hours)
                row["selection_val_hours"] = float(args.val_hours)
                row["selection_context_hours"] = float(args.train_hours + args.val_hours)
                row["selection_test_hours"] = float(
                    args.window_hours - args.train_hours - args.val_hours
                )
            rows.extend(target_rows)
            runtimes.append(
                {"target_id": target_id, "selected_prior_selection": prior_selection, **train_info}
            )
        summary = {
            "method": method,
            "seed": int(args.seed),
            "targets": list(range(len(targets))),
            "summary": base_run._summarize(rows),
        }
        (method_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (method_dir / "runtime.json").write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
        _write_csv(method_dir / "metrics_per_target_region.csv", rows)
        summaries[method] = summary
    return summaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ELD forecasting with per-target validation-bank selection."
    )
    parser.add_argument("--root", default="data/electricity_load_diagrams")
    parser.add_argument(
        "--output-dir", default="results/eld_forecasting_valbank_context15_val5_test28"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-targets", type=int, default=3)
    parser.add_argument("--methods", default="analog,vip,gmvip_empirical")
    parser.add_argument(
        "--selection-method",
        choices=["analog", "vip", "vip_512", "ftip", "gmvip_empirical"],
        default="analog",
    )
    parser.add_argument(
        "--selection-metric", choices=["rmse", "nll", "crps", "cqm"], default="crps"
    )
    parser.add_argument(
        "--candidate-prior-selections",
        default="same_client_prefix_nn,same_client_calendar_prefix_nn,calendar_prefix_nn,prefix_nn,calendar",
    )
    parser.add_argument("--prior-years", default="2011,2012,2013")
    parser.add_argument("--target-years", default="2014")
    parser.add_argument("--window-hours", type=float, default=48.0)
    parser.add_argument("--train-hours", type=float, default=15.0)
    parser.add_argument("--val-hours", type=float, default=5.0)
    parser.add_argument("--prior-bank-size", type=int, default=512)
    parser.add_argument("--num-inducing", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--selection-max-steps", type=int, default=0)
    parser.add_argument("--n-mc-train", type=int, default=8)
    parser.add_argument("--n-mc-eval", type=int, default=256)
    parser.add_argument("--regression-coeffs", type=int, default=128)
    parser.add_argument("--noise-std-norm", type=float, default=0.05)
    parser.add_argument("--gmvip-beta", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float32")
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.train_hours <= 0 or args.val_hours <= 0:
        raise ValueError("--train-hours and --val-hours must be positive.")
    if args.train_hours + args.val_hours >= args.window_hours:
        raise ValueError("train + validation hours must leave a non-empty test horizon.")
    if not processed_exists(args.root):
        raise FileNotFoundError(f"Processed ELD data not found under {args.root!r}.")

    device, dtype = _device_dtype(args)
    train_points = _hours_to_points(args.train_hours)
    context_points = _hours_to_points(args.train_hours + args.val_hours)
    window_points = _hours_to_points(args.window_hours)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.root)
    specs = build_window_specs(
        data,
        window_length=window_points,
        prefix_length=context_points,
        min_nonzero_fraction=0.9,
        min_prefix_std=1.0e-3,
    )
    targets = _select_targets(
        specs, years=_parse_years(args.target_years), n_targets=args.n_targets, seed=args.seed
    )

    start = time.time()
    selected, selection_rows = _select_prior_rules(
        targets,
        data,
        specs,
        args=args,
        device=device,
        dtype=dtype,
        train_points=train_points,
        window_points=window_points,
    )
    _write_csv(output_dir / "selection_decisions.csv", selection_rows)
    summaries = _run_final_methods(
        targets,
        data,
        specs,
        selected,
        args=args,
        device=device,
        dtype=dtype,
        context_points=context_points,
        window_points=window_points,
        output_dir=output_dir,
    )
    manifest = {
        "elapsed_sec": time.time() - start,
        "train_points": train_points,
        "validation_points": context_points - train_points,
        "context_points": context_points,
        "window_points": window_points,
        "selected_prior_rules": selected,
        "summaries": summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
