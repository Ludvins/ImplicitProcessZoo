"""Small reporting helpers shared by benchmark entrypoints."""

import csv
import json
import os
import sys


DEFAULT_METRICS = (
    "RMSE",
    "NLL",
    "CRPS",
    "CQM",
    "ADE",
    "FDE",
    "Error",
    "AUROC",
)

MODEL_DISPLAY_NAMES = {
    "fbnn": "FBNN",
    "ftip": "FTIP",
    "gmvip": "GMVIP",
    "map": "MAP",
    "mfvi": "MFVI",
    "tfsvi": "TFSVI",
    "vip": "VIP",
}

DATASET_DISPLAY_NAMES = {
    "boston": "Boston",
    "concrete": "Concrete",
    "energy": "Energy",
    "kin8nm": "Kin8nm",
    "naval": "Naval",
    "power": "Power",
    "protein": "Protein",
    "wine": "Wine",
    "year": "Year",
    "Year": "Year",
    "yacht": "Yacht",
}

DISCREPANCY_DISPLAY_NAMES = {
    "energy": "Energy",
    "mmd": "MMD",
    "prior_whitened_gaussian_kl": "Prior-Whitened Gaussian KL",
    "prior_whitened_sliced_kl": "Prior-Whitened Sliced KL",
    "sample_sliced_kl": "Sample Sliced KL",
    "sample_sliced_knn_kl": "Sample Sliced kNN KL",
    "sample_sliced_gaussian_kl": "Sample Sliced Gaussian KL",
    "sample_sliced_quantile_transport_kl": "Sliced Quantile-Transport KL",
    "sample_sliced_rank_kl": "Sample Sliced Rank KL",
    "spectral_projected_kl": "Spectral Projected KL",
    "spectral_sliced_kl": "Spectral Sliced KL",
    "sliced_wasserstein": "Sliced Wasserstein",
    "stein": "Stein",
    "sinkhorn": "Sinkhorn",
}


def pretty_model_name(model_type):
    """Human-readable model label for reports and W&B run names."""
    model_type = str(model_type)
    return MODEL_DISPLAY_NAMES.get(model_type, model_type.replace("_", " ").upper())


def canonical_model_type(model_type):
    """Canonical method id."""
    return str(model_type or "").lower()


def pretty_dataset_name(dataset_name):
    """Human-readable dataset label for reports and W&B run names."""
    dataset_name = str(dataset_name)
    if dataset_name in DATASET_DISPLAY_NAMES:
        return DATASET_DISPLAY_NAMES[dataset_name]
    return dataset_name.replace("_", " ").replace("-", " ").title()


def pretty_discrepancy_name(discrepancy):
    """Human-readable discrepancy label for W&B run names."""
    discrepancy = str(discrepancy)
    if discrepancy in DISCREPANCY_DISPLAY_NAMES:
        return DISCREPANCY_DISPLAY_NAMES[discrepancy]
    return discrepancy.replace("_", " ").replace("-", " ").title()


def wandb_run_name(prefix, *, dataset=None, model=None, seed=None, suffix=None):
    """Build stable, readable W&B run names."""
    parts = [str(prefix)]
    if dataset is not None:
        parts.append(pretty_dataset_name(dataset))
    if model is not None:
        parts.append(pretty_model_name(model))
    if suffix:
        parts.append(str(suffix))
    if seed is not None:
        parts.append(f"seed {seed}")
    return " | ".join(parts)


def add_wandb_args(parser):
    """Add optional Weights & Biases tracking flags to a benchmark parser."""
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases experiment tracking.",
    )
    parser.add_argument("--wandb_project", default="gmvip")
    parser.add_argument("--wandb_entity", default="ludvins")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument(
        "--wandb_mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="W&B mode. Leave unset for W&B's default behavior.",
    )
    parser.add_argument(
        "--wandb_dir",
        default=None,
        help="Directory for W&B run files. Defaults to the benchmark output_dir.",
    )
    parser.add_argument("--wandb_run_id", default=None)
    parser.add_argument("--wandb_resume", default=None)
    parser.add_argument(
        "--wandb_log_every",
        type=int,
        default=100,
        help="Training-step interval for logging scalar loss/LR to W&B.",
    )


def init_wandb_run(args, *, name=None, group=None, tags=None, config=None):
    """Initialize a W&B run when --wandb is enabled.

    The import is intentionally lazy so W&B remains an optional dependency for
    users who only want local JSON/CSV outputs.
    """
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases tracking was requested with --wandb, but the "
            "`wandb` package is not installed. Install it with "
            "`pip install wandb` or run without --wandb."
        ) from exc

    run_dir = getattr(args, "wandb_dir", None) or getattr(args, "output_dir", None)
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)

    run_tags = list(getattr(args, "wandb_tags", []) or [])
    if tags:
        run_tags.extend(tags)

    run_config = {}
    if hasattr(args, "__dict__"):
        run_config.update(_json_safe_dict(vars(args)))
    if config:
        run_config.update(_json_safe_dict(config))

    return wandb.init(
        project=getattr(args, "wandb_project", "gmvip"),
        entity=getattr(args, "wandb_entity", None),
        group=getattr(args, "wandb_group", None) or group,
        name=getattr(args, "wandb_name", None) or name,
        tags=run_tags or None,
        config=run_config,
        mode=getattr(args, "wandb_mode", None),
        dir=run_dir,
        id=getattr(args, "wandb_run_id", None),
        resume=getattr(args, "wandb_resume", None),
    )


def wandb_log(data, step=None):
    """Log to the active W&B run if one exists."""
    wandb = sys.modules.get("wandb")
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    wandb.log(_json_safe_dict(data), step=step)


def wandb_log_train_step(args, step, loss, optimizer=None, model=None, model_type=None):
    if not getattr(args, "wandb", False):
        return
    log_every = max(1, int(getattr(args, "wandb_log_every", 100)))
    if step % log_every != 0:
        return
    payload = {"train/loss": _to_float(loss)}
    if optimizer is not None and optimizer.param_groups:
        payload["train/lr"] = float(optimizer.param_groups[0]["lr"])
    if model is not None:
        payload.update(training_decomposition(model, model_type=model_type))
    wandb_log(payload, step=step)


def training_decomposition(model, model_type=None):
    """Return latest data-fit and regularizer pieces from a trained model.

    The model classes expose method-specific buffers after each ``_train_step``.
    This helper translates them into stable W&B metric names while keeping
    method-specific terms visible for debugging.
    """
    payload = {}
    data_fit = _last_scalar(model, "data_terms")
    if data_fit is None:
        data_fit = _last_scalar(model, "bb_alphas")
    if data_fit is not None:
        payload["train/data_fit"] = data_fit

    kl = _last_scalar(model, "KLs")
    prior_reg = _last_scalar(model, "prior_regularizers")
    beta = _last_scalar(model, "betas")
    function_term = _last_scalar(model, "function_terms")

    normalized_type = canonical_model_type(model_type)
    if normalized_type == "ap_fsvi" or beta is not None or function_term is not None:
        if function_term is None:
            function_term = kl
        if function_term is not None:
            if normalized_type == "fcfsvi":
                payload["train/fcfsvi_context_kl"] = function_term
                payload["train/fcfsvi/context_kl"] = function_term
            elif normalized_type == "gmvip":
                payload["train/gmvip_kl"] = function_term
                payload["train/gmvip/latent_kl_unweighted"] = function_term
            else:
                payload["train/ap_fsvi_discrepancy"] = function_term
                payload["train/apfsvi/discrepancy_unweighted"] = function_term
            payload["train/function_regularizer_unweighted"] = function_term
        if beta is not None:
            if normalized_type == "fcfsvi":
                payload["train/fcfsvi_beta"] = beta
                payload["train/fcfsvi/beta"] = beta
            elif normalized_type == "gmvip":
                payload["train/gmvip_beta"] = beta
                payload["train/gmvip/beta"] = beta
            else:
                payload["train/ap_fsvi_beta"] = beta
                payload["train/apfsvi/beta"] = beta
        if function_term is not None:
            weighted = (beta if beta is not None else 1.0) * function_term
            payload["train/regularizer"] = weighted
            if normalized_type == "fcfsvi":
                payload["train/fcfsvi_weighted_context_kl"] = weighted
                payload["train/fcfsvi/context_kl_weighted"] = weighted
            elif normalized_type == "gmvip":
                payload["train/gmvip_weighted_kl"] = weighted
                payload["train/gmvip/latent_kl_weighted"] = weighted
            else:
                payload["train/ap_fsvi_weighted_discrepancy"] = weighted
                payload["train/apfsvi/discrepancy_weighted"] = weighted

        if normalized_type == "gmvip":
            metrics = getattr(model, "last_train_metrics", {}) or {}
            for metric_name in ("flow_logdet_mean", "flow_kl_std"):
                value = metrics.get(metric_name)
                if value is not None:
                    payload[f"train/gmvip/{metric_name}"] = _to_float(value)

    else:
        if kl is not None:
            payload["train/kl"] = kl
            payload["train/elbo_kl"] = kl
            if normalized_type == "vip":
                payload["train/vip/kl"] = kl
            elif normalized_type == "fbnn":
                payload["train/fbnn/functional_kl"] = kl
            elif normalized_type == "tfsvi":
                payload["train/tfsvi/functional_kl"] = kl
            elif normalized_type == "mfvi":
                payload["train/mfvi/kl"] = kl
            elif normalized_type == "map":
                payload["train/map/l2"] = kl
        if prior_reg is not None:
            payload["train/prior_regularizer"] = prior_reg
            payload["train/evidence_regularizer"] = prior_reg
            if normalized_type == "vip":
                payload["train/vip/prior_regularizer"] = prior_reg
            elif normalized_type == "ftip":
                payload["train/ftip/prior_regularizer"] = prior_reg
        pieces = [value for value in (kl, prior_reg) if value is not None]
        if pieces:
            total_regularizer = sum(pieces)
            payload["train/regularizer"] = total_regularizer
            payload["train/objective_regularizer"] = total_regularizer
            payload["train/regularizer_total"] = total_regularizer

    if normalized_type == "fcfsvi":
        kl_raw = _last_scalar(model, "nf_kl_raws")
        prior_flow_nll = _last_scalar(model, "prior_flow_nlls")
        posterior_flow_nll = _last_scalar(model, "posterior_flow_nlls")
        prior_flow_train_nll = _last_scalar(model, "prior_flow_train_nlls")
        prior_flow_val_nll = _last_scalar(model, "prior_flow_val_nlls")
        prior_flow_converged = _last_scalar(model, "prior_flow_converged_flags")
        posterior_flow_nll_before = _last_scalar(model, "posterior_flow_nlls_before")
        posterior_flow_nll_after = _last_scalar(model, "posterior_flow_nlls_after")
        posterior_flow_train_nll = _last_scalar(model, "posterior_flow_train_nlls")
        posterior_flow_val_nll = _last_scalar(model, "posterior_flow_val_nlls")
        prior_flow_relative_improvement = _last_scalar(
            model, "prior_flow_relative_improvements"
        )
        prior_flow_update_count = _last_scalar(model, "prior_flow_update_counts")
        posterior_flow_relative_improvement = _last_scalar(
            model, "posterior_flow_relative_improvements"
        )
        posterior_flow_update_count = _last_scalar(model, "posterior_flow_update_counts")
        posterior_flow_converged = _last_scalar(
            model, "posterior_flow_converged_flags"
        )
        posterior_flow_fit_samples = _last_scalar(
            model, "posterior_flow_fit_sample_counts"
        )
        posterior_flow_val_samples = _last_scalar(
            model, "posterior_flow_val_sample_counts"
        )
        context_opt_kl_before = _last_scalar(
            model, "context_optimization_kls_before"
        )
        context_opt_kl_after = _last_scalar(
            model, "context_optimization_kls_after"
        )
        context_opt_update_count = _last_scalar(
            model, "context_optimization_update_counts"
        )
        context_input_norm = _last_scalar(model, "context_input_norms")
        if kl_raw is not None:
            payload["train/fcfsvi_kl_raw"] = kl_raw
        if prior_flow_nll is not None:
            payload["train/fcfsvi_prior_flow_nll"] = prior_flow_nll
        if prior_flow_train_nll is not None:
            payload["train/fcfsvi_prior_flow_train_nll"] = prior_flow_train_nll
        if prior_flow_val_nll is not None:
            payload["train/fcfsvi_prior_flow_val_nll"] = prior_flow_val_nll
        if prior_flow_relative_improvement is not None:
            payload["train/fcfsvi_prior_flow_relative_improvement"] = (
                prior_flow_relative_improvement
            )
        if prior_flow_update_count is not None:
            payload["train/fcfsvi_prior_flow_updates"] = prior_flow_update_count
        if prior_flow_converged is not None:
            payload["train/fcfsvi_prior_flow_converged"] = prior_flow_converged
        if posterior_flow_nll is not None:
            payload["train/fcfsvi_posterior_flow_nll"] = posterior_flow_nll
        if posterior_flow_nll_before is not None:
            payload["train/fcfsvi_posterior_flow_nll_before"] = posterior_flow_nll_before
        if posterior_flow_nll_after is not None:
            payload["train/fcfsvi_posterior_flow_nll_after"] = posterior_flow_nll_after
        if posterior_flow_nll_before is not None and posterior_flow_nll_after is not None:
            payload["train/fcfsvi_posterior_flow_nll_drop"] = (
                posterior_flow_nll_before - posterior_flow_nll_after
            )
        if posterior_flow_train_nll is not None:
            payload["train/fcfsvi_posterior_flow_train_nll"] = posterior_flow_train_nll
        if posterior_flow_val_nll is not None:
            payload["train/fcfsvi_posterior_flow_val_nll"] = posterior_flow_val_nll
        if posterior_flow_relative_improvement is not None:
            payload["train/fcfsvi_posterior_flow_relative_improvement"] = (
                posterior_flow_relative_improvement
            )
        if posterior_flow_update_count is not None:
            payload["train/fcfsvi_posterior_flow_updates"] = posterior_flow_update_count
        if posterior_flow_converged is not None:
            payload["train/fcfsvi_posterior_flow_converged"] = posterior_flow_converged
        if posterior_flow_fit_samples is not None:
            payload["train/fcfsvi_posterior_flow_fit_samples"] = posterior_flow_fit_samples
        if posterior_flow_val_samples is not None:
            payload["train/fcfsvi_posterior_flow_val_samples"] = posterior_flow_val_samples
        if context_opt_kl_before is not None:
            payload["train/fcfsvi_context_opt_kl_before"] = context_opt_kl_before
        if context_opt_kl_after is not None:
            payload["train/fcfsvi_context_opt_kl_after"] = context_opt_kl_after
        if context_opt_update_count is not None:
            payload["train/fcfsvi_context_opt_updates"] = context_opt_update_count
        if context_input_norm is not None:
            payload["train/fcfsvi_context_input_norm"] = context_input_norm

    base_kl = _last_scalar(model, "base_KLs")
    flow_ldj = _last_scalar(model, "flow_ldj")
    if base_kl is not None:
        payload["train/ftip_base_kl"] = base_kl
    if flow_ldj is not None:
        payload["train/ftip_flow_ldj"] = flow_ldj

    if "train/data_fit" in payload and "train/regularizer" in payload:
        payload["train/reconstructed_loss"] = (
            payload["train/data_fit"] + payload["train/regularizer"]
        )
    return {key: value for key, value in payload.items() if value is not None}


def wandb_log_eval(step, train_metrics=None, validation_metrics=None):
    payload = {}
    payload.update(_prefix_metrics("train_eval", train_metrics or {}))
    payload.update(_prefix_metrics("validation", validation_metrics or {}))
    wandb_log(payload, step=step)


def wandb_log_result(result):
    """Log final train/test metrics and update the W&B summary."""
    wandb = sys.modules.get("wandb")
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    payload = {
        "train_time_s": result.get("train_time_s"),
    }
    payload.update(_prefix_metrics("final/train", result.get("train", {})))
    payload.update(_prefix_metrics("final/test", result.get("test", {})))
    payload.update(_prefix_metrics("final/prior", result.get("prior", {})))
    if "ood" in result:
        payload.update(_prefix_metrics("final/ood", result.get("ood", {})))
    payload = {k: v for k, v in payload.items() if v is not None}
    wandb.log(_json_safe_dict(payload))
    wandb.run.summary.update(_json_safe_dict(payload))


def finish_wandb_run(run=None):
    wandb = sys.modules.get("wandb")
    if wandb is None:
        return
    active = run or getattr(wandb, "run", None)
    if active is not None:
        active.finish()


def flatten_results(results):
    flat = []
    for item in results:
        if isinstance(item, list):
            flat.extend(flatten_results(item))
        elif item is not None:
            flat.append(item)
    return flat


def comparison_rows(results, split="test", metrics=DEFAULT_METRICS):
    rows = []
    for result in flatten_results(results):
        values = result.get(split, {})
        row = {
            "dataset": result.get("dataset", ""),
            "model": result.get("model", ""),
            "train_time_s": result.get("train_time_s", ""),
        }
        for metric in metrics:
            if metric in values:
                row[metric] = values[metric]
        rows.append(row)
    return rows


def _prefix_metrics(prefix, metrics):
    prefixed = {}
    for key, value in (metrics or {}).items():
        value = _to_float(value)
        if value is not None:
            prefixed[f"{prefix}/{key}"] = value
    return prefixed


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (int, float, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _last_scalar(obj, attr):
    values = getattr(obj, attr, None)
    if values is None or len(values) == 0:
        return None
    return _to_float(values[-1])


def _json_safe_dict(values):
    return {key: _json_safe(value) for key, value in values.items()}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def print_comparison(results, split="test", primary="RMSE"):
    rows = comparison_rows(results, split=split)
    if not rows:
        return

    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for dataset, dataset_rows in by_dataset.items():
        metrics = [m for m in DEFAULT_METRICS if any(m in r for r in dataset_rows)]
        sort_metric = primary if any(primary in r for r in dataset_rows) else metrics[0]
        dataset_rows = sorted(
            dataset_rows,
            key=lambda r: float(r.get(sort_metric, float("inf"))),
        )
        columns = ["model"] + metrics + ["train_time_s"]
        widths = {
            col: max(len(col), *(len(_format_cell(row.get(col, ""))) for row in dataset_rows))
            for col in columns
        }
        print(f"\n{split.title()} comparison: {dataset} (sorted by {sort_metric})")
        print("  " + "  ".join(col.ljust(widths[col]) for col in columns))
        print("  " + "  ".join("-" * widths[col] for col in columns))
        for row in dataset_rows:
            print("  " + "  ".join(
                _format_cell(row.get(col, "")).ljust(widths[col]) for col in columns
            ))


def save_comparison(results, output_dir, name, split="test"):
    rows = comparison_rows(results, split=split)
    if not rows:
        return None, None
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{name}_{split}_comparison.json")
    csv_path = os.path.join(output_dir, f"{name}_{split}_comparison.csv")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    columns = ["dataset", "model"] + [
        metric for metric in DEFAULT_METRICS if any(metric in row for row in rows)
    ] + ["train_time_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _format_cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

