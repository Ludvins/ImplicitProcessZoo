"""Train selected models on the Variational-LLA synthetic dataset and plot predictions.

Examples
--------
python -m scripts.synthetic_plot --models mfvi fbnn vip tfsvi ftip gmvip
python -m scripts.synthetic_plot --models all --iterations 2000 --device cuda

All model/training options from ``scripts.uci_benchmark`` are accepted. This
entrypoint fixes the dataset to ``variational_lla`` and adds plotting-specific
options.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    plt = None
    LinearSegmentedColormap = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    Image = None
    ImageDraw = None
    ImageFont = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_utils import pretty_model_name
from scripts.uci_benchmark import (
    REGRESSION_MODELS,
    _ckpt_path,
    _fbnn_pred_components,
    _gmvip_pred_components,
    _is_fcfsvi_model,
    _tfsvi_pred_components,
    build_model,
    parse_args as parse_uci_args,
    train_with_metrics,
)
from src.utils.dataset import Test_Dataset, Training_Dataset, get_dataset

import scripts.uci_benchmark as uci_benchmark


DEFAULT_MODELS = ["map", "mfvi", "vip", "fbnn", "sip", "tfsvi", "ftip", "gmvip"]
SYNTHETIC_WEIGHT_LOG_SIGMA_INIT = {
    "mfvi": -5.0,
    "fbnn": -3.0,
}
SYNTHETIC_BB_ALPHA = {
    "mfvi": 0.0,
}
SYNTHETIC_SIP_DEFAULTS = {
    "sip_inducing_method": "train_quantiles",
    "sip_jitter": 1e-4,
    "sip_learn_prior": False,
}
SYNTHETIC_SIP_FLAGS = {
    "sip_inducing_method": ("--sip_inducing_method",),
    "sip_jitter": ("--sip_jitter",),
    "sip_learn_prior": ("--sip_learn_prior", "--no-sip_learn_prior"),
}
SYNTHETIC_VIP_DEFAULTS = {
    "vip_learn_prior": False,
}
SYNTHETIC_VIP_FLAGS = {
    "vip_learn_prior": ("--vip_learn_prior", "--no-vip_learn_prior"),
}
SYNTHETIC_TFSVI_DEFAULTS = {
    "tfsvi_S_ctx": 3,
    "tfsvi_K_ctx": 16,
    "tfsvi_num_train_samples": 5,
    "iterations": 5000,
}
SYNTHETIC_TFSVI_FLAGS = {
    "tfsvi_S_ctx": ("--tfsvi_S_ctx",),
    "tfsvi_K_ctx": ("--tfsvi_K_ctx",),
    "tfsvi_num_train_samples": ("--tfsvi_num_train_samples",),
    "iterations": ("--iterations", "--epochs", "--default_iterations", "--tfsvi_steps"),
}
SYNTHETIC_DATASET_NAME = "variational_lla"
SYNTHETIC_DATASET_LABEL = "Variational-LLA"
STEP_OVERRIDE_MODELS = list(REGRESSION_MODELS)
SAMPLE_BAND_MODELS = {
    "sip",
    "ap_fsvi",
    "fbnn",
    "fcfsvi",
    "ftip",
    "gmvip",
    "mfvi",
    "tfsvi",
    "vip",
}
PREDICTIVE_INTERVAL_MODELS = {"vip"}
MODEL_COLORS = {
    "mfvi": "#2ca02c",
    "fbnn": "#ff7f0e",
    "vip": "#1f77b4",
    "tfsvi": "#9467bd",
    "ftip": "#d62728",
    "ap_fsvi": "#17becf",
    "fcfsvi": "#8c564b",
    "gmvip": "#bcbd22",
    "sip": "#e377c2",
    "map": "#4d4d4d",
}
DATA_PANEL_SCATTER_STYLE = {
    "s": 4,
    "c": "#4d4d4d",
    "alpha": 0.55,
    "linewidths": 0,
    "edgecolors": "none",
    "zorder": 20,
}
DATA_OVERLAY_SCATTER_STYLE = {
    "s": 8,
    "c": "#202020",
    "alpha": 0.62,
    "linewidths": 0,
    "edgecolors": "none",
    "zorder": 20,
}

MATHERON_GMVIP_DEFAULTS = {
    "batch_size": 400,
    "gmvip_inducing_method": "train_quantiles",
    "gmvip_learn_Z": False,
    "gmvip_learn_kernel": False,
    "gmvip_learn_prior": False,
    "gmvip_num_eval_samples": 256,
    "gmvip_num_inducing": 256,
    "gmvip_num_operator_bank_samples": 1024,
    "gmvip_num_train_samples": 128,
    "gmvip_operator_type": "empirical",
    "iterations": 30000,
    "lr": 1e-3,
    "save_checkpoint": False,
}

MATHERON_GMVIP_FLAGS = {
    "batch_size": ("--batch_size",),
    "gmvip_inducing_method": ("--gmvip_inducing_method",),
    "gmvip_learn_Z": ("--gmvip_learn_Z", "--no-gmvip_learn_Z"),
    "gmvip_learn_kernel": ("--gmvip_learn_kernel", "--no-gmvip_learn_kernel"),
    "gmvip_learn_prior": ("--gmvip_learn_prior", "--no-gmvip_learn_prior"),
    "gmvip_num_eval_samples": ("--gmvip_num_eval_samples",),
    "gmvip_num_inducing": ("--gmvip_num_inducing",),
    "gmvip_num_operator_bank_samples": ("--gmvip_num_operator_bank_samples",),
    "gmvip_num_train_samples": ("--gmvip_num_train_samples",),
    "gmvip_operator_type": ("--gmvip_operator_type",),
    "iterations": ("--iterations", "--epochs", "--default_iterations", "--gmvip_steps"),
    "lr": ("--lr",),
    "save_checkpoint": ("--save_checkpoint", "--no_save_checkpoint"),
}


@dataclass
class PredictiveGrid:
    means: np.ndarray
    stds: np.ndarray
    mixture_mean: np.ndarray
    mixture_std: np.ndarray
    metrics: dict


def _flag_names(argv):
    flags = set()
    for token in argv:
        if isinstance(token, str) and token.startswith("--"):
            flags.add(token.split("=", 1)[0])
    return flags


def _flag_supplied(synthetic_args, *flags):
    supplied = getattr(synthetic_args, "_supplied_flags", set())
    return any(flag in supplied for flag in flags)


def parse_synthetic_args(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Variational-LLA predictive-distribution plotter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "All remaining model/training flags are forwarded to "
            f"scripts.uci_benchmark. The dataset is always {SYNTHETIC_DATASET_NAME}."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=REGRESSION_MODELS + ["all"],
        help="Models to train and plot.",
    )
    parser.add_argument(
        "--default_iterations",
        type=int,
        default=10000,
        help="Training iterations used when neither --iterations nor --epochs is given.",
    )
    for model_type in STEP_OVERRIDE_MODELS:
        parser.add_argument(
            f"--{model_type}_steps",
            type=int,
            default=None,
            help=f"Override training iterations for only {model_type}.",
        )
    parser.add_argument(
        "--mfvi_weight_log_sigma_init",
        type=float,
        default=None,
        help="Override --weight_log_sigma_init for only MFVI.",
    )
    parser.add_argument(
        "--fbnn_weight_log_sigma_init",
        type=float,
        default=None,
        help="Override --weight_log_sigma_init for only FBNN.",
    )
    parser.add_argument(
        "--vip_sample_functions",
        action="store_true",
        default=False,
        help=(
            "Plot VIP by sampling latent regression coefficients from q(a) "
            "instead of using its analytic predictive interval."
        ),
    )
    parser.add_argument("--grid_points", type=int, default=300)
    parser.add_argument("--plot_samples", type=int, default=60)
    parser.add_argument("--density_bins", type=int, default=220)
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help=(
            "Shared x-axis limits and prediction-grid domain for all panels. "
            "If omitted, use training x min/max padded by --xpad."
        ),
    )
    parser.add_argument(
        "--xpad",
        type=float,
        default=1.0,
        help="Padding added on each side of the training x-domain when --xlim is omitted.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=(-1.0, 1.8),
        metavar=("YMIN", "YMAX"),
        help=(
            "Fixed shared y-axis limits for all panels. The default matches "
            "the GMVIP-style Variational-LLA panel scale."
        ),
    )
    parser.add_argument(
        "--auto_ylim",
        action="store_true",
        default=False,
        help="Compute shared y-axis limits from the plotted predictions instead of --ylim.",
    )
    parser.add_argument(
        "--plot_style",
        choices=["auto", "density", "sample_bands"],
        default="auto",
        help="How model panels render posterior predictions.",
    )
    parser.add_argument("--max_cols", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--figure_name", default="variational_lla_predictive_distributions.png")
    parser.add_argument("--pdf_name", default="variational_lla_predictive_distributions.pdf")
    parser.add_argument("--results_name", default="variational_lla_predictive_distributions.json")
    parser.add_argument(
        "--include_data_panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include a data panel before the model panels. The data panel is "
            "skipped when --overlay_data is enabled."
        ),
    )
    parser.add_argument(
        "--overlay_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay observed Variational-LLA data points on each model panel.",
    )
    parser.add_argument(
        "--plot_predictive_mean",
        action="store_true",
        default=False,
        help="Draw the predictive mixture mean as a black line on model panels.",
    )
    parser.add_argument(
        "--normalize_inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize inputs before training. Use --no-normalize_inputs to train on raw x.",
    )
    parser.add_argument(
        "--gmvip_matheron_preset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the tuned Variational-LLA Matheron/GMVIP plotting defaults "
            "for GMVIP unless overridden: empirical operator, raw inputs for "
            "GMVIP-only runs, fixed train-quantile Z, M=256, S_train=128, "
            "S_eval=256, 30k iterations."
        ),
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Keep training later models if one model fails.",
    )

    synthetic_args, forwarded = parser.parse_known_args(raw_argv)
    synthetic_args._supplied_flags = _flag_names(raw_argv)
    forbidden = {"--dataset", "--model"}
    used_forbidden = [flag for flag in forwarded if flag.split("=", 1)[0] in forbidden]
    if used_forbidden:
        parser.error(
            f"synthetic_plot fixes --dataset {SYNTHETIC_DATASET_NAME} "
            "and uses --models; remove "
            + ", ".join(used_forbidden)
        )

    uci_args = parse_uci_args(
        ["--model", "map", "--dataset", SYNTHETIC_DATASET_NAME, *forwarded]
    )
    if not uci_args._iters_user_supplied:
        uci_args.iterations = synthetic_args.default_iterations
        uci_args.epochs = None
    if uci_args.vip_epochs is None and uci_args.vip_iterations is None:
        uci_args.vip_iterations = uci_args.iterations
        uci_args.vip_epochs = uci_args.epochs
    models = resolve_models(synthetic_args.models)
    if (
        synthetic_args.gmvip_matheron_preset
        and models == ["gmvip"]
        and not _flag_supplied(synthetic_args, "--normalize_inputs", "--no-normalize_inputs")
    ):
        synthetic_args.normalize_inputs = False
    return synthetic_args, uci_args


def resolve_models(models):
    if "all" in models:
        return list(DEFAULT_MODELS)
    seen = set()
    ordered = []
    for model in models:
        if model not in seen:
            ordered.append(model)
            seen.add(model)
    return ordered


def missing_model_reason(model_type):
    if model_type == "ap_fsvi" and getattr(uci_benchmark, "APFSVI", None) is None:
        return "src.ap_fsvi is not available in this workspace"
    if _is_fcfsvi_model(model_type) and getattr(uci_benchmark, "FCFSVI", None) is None:
        return "src.fcfsvi is not available in this workspace"
    return None


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def args_for_model(base_args, model_type):
    args = copy.copy(base_args)
    args.model = model_type
    if model_type == "mfvi" and not args._bb_alpha_user_supplied:
        args.bb_alpha = 0.5
    if (_is_fcfsvi_model(model_type) or model_type == "gmvip") and not args._bb_alpha_user_supplied:
        args.bb_alpha = 0.0
    return args


def args_for_run(base_args, synthetic_args, model_type):
    args = args_for_model(base_args, model_type)
    args.vip_sample_functions = bool(synthetic_args.vip_sample_functions)
    if model_type == "gmvip" and synthetic_args.gmvip_matheron_preset:
        for attr, value in MATHERON_GMVIP_DEFAULTS.items():
            if not _flag_supplied(synthetic_args, *MATHERON_GMVIP_FLAGS[attr]):
                setattr(args, attr, value)
        if args.iterations is not None:
            args.epochs = None
    if model_type == "map" and not _flag_supplied(synthetic_args, "--map_l2"):
        args.map_l2 = 0.0
    if model_type == "sip":
        for attr, value in SYNTHETIC_SIP_DEFAULTS.items():
            if not _flag_supplied(synthetic_args, *SYNTHETIC_SIP_FLAGS[attr]):
                setattr(args, attr, value)
    if model_type == "vip":
        for attr, value in SYNTHETIC_VIP_DEFAULTS.items():
            if not _flag_supplied(synthetic_args, *SYNTHETIC_VIP_FLAGS[attr]):
                setattr(args, attr, value)
    if model_type == "tfsvi":
        for attr, value in SYNTHETIC_TFSVI_DEFAULTS.items():
            if not _flag_supplied(synthetic_args, *SYNTHETIC_TFSVI_FLAGS[attr]):
                setattr(args, attr, value)
        if args.iterations is not None:
            args.epochs = None
    if (
        model_type in SYNTHETIC_BB_ALPHA
        and not _flag_supplied(synthetic_args, "--bb_alpha")
    ):
        args.bb_alpha = SYNTHETIC_BB_ALPHA[model_type]
    if (
        model_type in SYNTHETIC_WEIGHT_LOG_SIGMA_INIT
        and not _flag_supplied(
            synthetic_args,
            "--weight_log_sigma_init",
            f"--{model_type}_weight_log_sigma_init",
        )
    ):
        args.weight_log_sigma_init = SYNTHETIC_WEIGHT_LOG_SIGMA_INIT[model_type]

    steps = getattr(synthetic_args, f"{model_type}_steps", None)
    if steps is not None:
        args.iterations = int(steps)
        args.epochs = None

    weight_override = getattr(
        synthetic_args,
        f"{model_type}_weight_log_sigma_init",
        None,
    )
    if weight_override is not None:
        args.weight_log_sigma_init = float(weight_override)

    vip_steps = getattr(synthetic_args, "vip_steps", None)
    if model_type == "ftip" and vip_steps is not None:
        args.vip_iterations = int(vip_steps)
        args.vip_epochs = None
    return args


def make_train_loader(train_dataset, args):
    use_cuda = bool(args.device and "cuda" in str(args.device))
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 2)
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=0,
        generator=generator,
    )


def _training_args_without_metrics(args, train_loader, epochs=None, iterations=None):
    train_args = copy.copy(args)
    if epochs is None and iterations is None:
        epochs = args.epochs
        iterations = args.iterations
    if iterations is not None:
        train_args.eval_every = max(int(iterations) + 1, int(args.eval_every))
    else:
        total_steps = max(1, int(epochs or 1)) * max(1, len(train_loader))
        train_args.eval_every = max(total_steps + 1, int(args.eval_every))
    return train_args


def train_for_plot(
    dataset_name,
    model_type,
    model,
    args,
    train_loader,
    train_eval_dataset,
    lr=None,
    epochs=None,
    iterations=None,
    desc="Training",
):
    """Train a model for plotting without computing held-out/test metrics."""
    if lr is None:
        lr = args.lr
    train_args = _training_args_without_metrics(
        args,
        train_loader,
        epochs=epochs,
        iterations=iterations,
    )
    t0 = time.time()
    losses, _, diagnostics = train_with_metrics(
        model,
        train_loader,
        train_eval_dataset,
        train_eval_dataset,
        train_args,
        lr=lr,
        epochs=epochs,
        iterations=iterations,
        model_type=model_type,
        desc=desc,
    )
    train_time = time.time() - t0

    actual_epochs = epochs if epochs is not None else args.epochs
    actual_iterations = iterations if iterations is not None else args.iterations
    result = {
        "dataset": dataset_name,
        "model": model_type,
        "train_time_s": round(train_time, 2),
        "training": {
            "epochs": actual_epochs,
            "iterations": actual_iterations,
            "lr": lr,
            "seed": args.seed,
        },
        "losses": losses,
        "diagnostics": diagnostics,
    }

    print(f"  Time: {train_time:.1f}s")
    if args.save_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt = _ckpt_path(args, dataset_name, model_type)
        torch.save(model.state_dict(), ckpt)
        print(f"  Checkpoint: {ckpt}")

    return result, model


def train_one_model(
    dataset_name,
    model_type,
    base_args,
    synthetic_args,
    train_dataset,
    train_eval_dataset,
):
    args = args_for_run(base_args, synthetic_args, model_type)
    seed_everything(args.seed)
    loader = make_train_loader(train_dataset, args)

    if model_type == "ftip" and args.auto_warm_start:
        print("\nFTIP warm start: training VIP source model.")
        vip_args = args_for_model(args, "vip")
        vip_model = build_model(vip_args, train_dataset, model_type="vip")
        _, vip_model = train_for_plot(
            dataset_name,
            "vip",
            vip_model,
            vip_args,
            loader,
            train_eval_dataset,
            lr=args.vip_lr,
            epochs=args.vip_epochs,
            iterations=args.vip_iterations,
            desc="VIP warm start",
        )
        model = build_model(args, train_dataset, model_type="ftip")
        model.warm_start_from_vip(vip_model, learnable_affine=args.learnable_affine)
        del vip_model
        result, model = train_for_plot(
            dataset_name,
            "ftip",
            model,
            args,
            loader,
            train_eval_dataset,
            lr=args.ftip_lr,
            desc="FTIP fine-tuning",
        )
        result["warm_start"] = {
            "enabled": True,
            "vip_iterations": args.vip_iterations,
            "vip_epochs": args.vip_epochs,
            "vip_lr": args.vip_lr,
            "learnable_affine": args.learnable_affine,
        }
        return result, model

    model = build_model(args, train_dataset, model_type=model_type)
    result, model = train_for_plot(
        dataset_name,
        model_type,
        model,
        args,
        loader,
        train_eval_dataset,
        desc=f"{pretty_model_name(model_type)} training",
    )
    return result, model


def normalized_grid(x_orig, train_dataset, dtype, device):
    x_norm = (x_orig - train_dataset.inputs_mean) / train_dataset.inputs_std
    return torch.as_tensor(x_norm, dtype=dtype, device=device)


def make_plot_split(dataset, base_args, synthetic_args):
    if synthetic_args.normalize_inputs:
        return dataset.get_split(base_args.test_size, base_args.seed)

    if not hasattr(dataset, "inputs") or not hasattr(dataset, "targets"):
        raise ValueError(
            "--no-normalize_inputs requires a dataset exposing raw inputs and targets."
        )

    train_dataset = Training_Dataset(
        np.asarray(dataset.inputs),
        np.asarray(dataset.targets),
        normalize_inputs=False,
        normalize_targets=getattr(dataset, "type", None) == "regression",
    )
    train_eval_dataset = Test_Dataset(
        np.asarray(dataset.inputs),
        np.asarray(dataset.targets),
        train_dataset.inputs_mean,
        train_dataset.inputs_std,
    )
    test_dataset = Test_Dataset(
        np.asarray(dataset.inputs),
        np.asarray(dataset.targets),
        train_dataset.inputs_mean,
        train_dataset.inputs_std,
    )
    return train_dataset, train_eval_dataset, test_dataset


def _component_eval_samples(args, model_type):
    if model_type == "ftip":
        return int(args.eval_samples)
    if model_type == "fbnn":
        return int(args.fbnn_num_eval_samples)
    if model_type == "tfsvi":
        return int(args.tfsvi_num_eval_samples)
    if model_type == "mfvi":
        return int(args.mfvi_num_eval_samples)
    if model_type == "gmvip":
        return int(args.gmvip_num_eval_samples)
    if _is_fcfsvi_model(model_type):
        return int(args.fcfsvi_num_eval_samples)
    if model_type == "sip":
        return int(args.sip_num_eval_samples)
    if model_type == "ap_fsvi":
        return int(args.ap_fsvi_num_eval_samples)
    return int(args.regression_coeffs)


def _as_component_arrays(mean, std):
    if mean.ndim == 2:
        mean = mean.unsqueeze(0)
    if mean.ndim != 3:
        raise ValueError(f"Expected mean with shape [S,N,D], got {tuple(mean.shape)}")

    if std is None:
        std = torch.zeros_like(mean)
    elif std.ndim == 0:
        std = std.view(1, 1, 1).expand_as(mean)
    elif std.ndim == 1:
        if std.shape[0] == mean.shape[1]:
            std = std.view(1, mean.shape[1], 1).expand_as(mean)
        elif std.shape[0] == mean.shape[0]:
            std = std.view(mean.shape[0], 1, 1).expand_as(mean)
        else:
            raise ValueError(f"Cannot align std shape {tuple(std.shape)} with mean {tuple(mean.shape)}")
    elif std.ndim == 2:
        if std.shape == mean.shape[:2]:
            std = std.unsqueeze(-1)
        elif std.shape[0] == mean.shape[0] and std.shape[1] == mean.shape[2]:
            std = std.unsqueeze(1)
        elif std.shape == mean.shape[1:]:
            std = std.unsqueeze(0)
        else:
            raise ValueError(f"Cannot align std shape {tuple(std.shape)} with mean {tuple(mean.shape)}")
    elif std.ndim != 3:
        raise ValueError(f"Expected std with <=3 dims, got {tuple(std.shape)}")

    std = std.expand_as(mean)
    means = mean[..., 0].detach().cpu().numpy()
    stds = std[..., 0].clamp_min(1e-8).detach().cpu().numpy()
    return means, stds


def predictive_components(model, model_type, args, x_grid):
    model.eval()
    S = _component_eval_samples(args, model_type)
    with torch.no_grad():
        if model_type == "ftip":
            coeffs = model.sample_flow_coefficients(S)
            mean, std = model.forward_with_coefficients(x_grid, coeffs)
        elif model_type == "fbnn":
            mean, std = _fbnn_pred_components(model, x_grid, S)
        elif model_type == "tfsvi":
            mean, std = _tfsvi_pred_components(model, x_grid, S)
        elif model_type == "mfvi":
            mean, std = _tfsvi_pred_components(model, x_grid, S)
        elif model_type == "gmvip":
            mean, std = _gmvip_pred_components(model, x_grid, S)
        elif model_type == "sip":
            mean = model.predict_f_samples(x_grid, S)
            mean = mean * model.y_std + model.y_mean
            std = torch.exp(0.5 * model.log_variance).view(1, 1, 1) * model.y_std
            std = std.expand_as(mean)
        elif model_type == "vip" and getattr(args, "vip_sample_functions", False):
            S = int(getattr(args, "eval_samples", S))
            if model.dtype != x_grid.dtype:
                x_grid = x_grid.to(model.dtype)
            f = model.generative_function(x_grid)
            m = f.mean(dim=0, keepdim=True)
            phi = (f - m) / model._sqrt_coeffs_m1
            q_sqrt = torch.zeros_like(model._q_sqrt_buf)
            q_sqrt[model._tril_row, model._tril_col] = model.q_sqrt_tri
            eps = torch.randn(
                S,
                model.num_coeffs,
                model.output_dim,
                generator=model.generator,
                dtype=model.dtype,
                device=model.device,
            )
            coeffs = torch.einsum("aid,ijd->ajd", eps, q_sqrt) + model.q_mu.unsqueeze(0)
            mean = torch.einsum("snd,asd->and", phi, coeffs) + m.squeeze(0)
            mean = mean * model.y_std + model.y_mean
            std = torch.exp(0.5 * model.log_variance).view(1, 1, 1) * model.y_std
            std = std.expand_as(mean)
        elif model_type in {"ap_fsvi"} or _is_fcfsvi_model(model_type):
            old_samples = getattr(model, "num_samples", None)
            if old_samples is not None:
                model.num_samples = S
            try:
                mean, std = model(x_grid)
            finally:
                if old_samples is not None:
                    model.num_samples = old_samples
        else:
            mean, std = model(x_grid)

    means, stds = _as_component_arrays(mean, std)
    mixture_mean = means.mean(axis=0)
    mixture_var = (stds ** 2 + means ** 2).mean(axis=0) - mixture_mean ** 2
    mixture_std = np.sqrt(np.maximum(mixture_var, 1e-12))
    return means, stds, mixture_mean, mixture_std


def make_predictive_grid(model, model_type, args, x_grid, result):
    means, stds, mixture_mean, mixture_std = predictive_components(
        model, model_type, args, x_grid
    )
    return PredictiveGrid(
        means=means,
        stds=stds,
        mixture_mean=mixture_mean,
        mixture_std=mixture_std,
        metrics={},
    )


def choose_style(model_type, pred, requested):
    if requested != "auto":
        return requested
    if model_type in SAMPLE_BAND_MODELS and pred.means.shape[0] > 1:
        return "sample_bands"
    if model_type in PREDICTIVE_INTERVAL_MODELS:
        return "predictive_interval"
    return "density"


def plot_density(ax, x, pred, color, y_lim, bins):
    density = mixture_density_grid(pred, y_lim, bins)
    cmap = LinearSegmentedColormap.from_list(
        f"pred_{color}",
        [(1.0, 1.0, 1.0, 0.0), color],
    )
    ax.imshow(
        density,
        extent=[float(x.min()), float(x.max()), y_lim[0], y_lim[1]],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bilinear",
        alpha=0.95,
        zorder=0,
    )


def mixture_density_grid(pred, y_lim, bins):
    y = np.linspace(y_lim[0], y_lim[1], bins, dtype=np.float64)
    means = pred.means.astype(np.float64)
    stds = np.maximum(pred.stds.astype(np.float64), 1e-8)
    density = np.empty((bins, means.shape[1]), dtype=np.float64)
    normalizer = math.sqrt(2.0 * math.pi)
    for start in range(0, bins, 64):
        stop = min(start + 64, bins)
        yy = y[start:stop, None, None]
        z = (yy - means[None, :, :]) / stds[None, :, :]
        block = np.exp(-0.5 * z * z) / (stds[None, :, :] * normalizer)
        density[start:stop] = block.mean(axis=1)
    scale = np.nanpercentile(density, 99.0)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanmax(density) or 1.0)
    return np.clip(density / scale, 0.0, 1.0)


def plot_sample_bands(ax, x, pred, color, max_samples):
    S = pred.means.shape[0]
    if S > max_samples:
        idx = np.linspace(0, S - 1, max_samples, dtype=int)
    else:
        idx = np.arange(S)
    for sample_idx in idx:
        mean = pred.means[sample_idx]
        std = pred.stds[sample_idx]
        ax.fill_between(
            x,
            mean - 2.0 * std,
            mean + 2.0 * std,
            color=color,
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x, mean, color=color, alpha=0.42, linewidth=0.8, zorder=2)


def plot_predictive_interval(ax, x, pred, color):
    lower = pred.mixture_mean - 2.0 * pred.mixture_std
    upper = pred.mixture_mean + 2.0 * pred.mixture_std
    ax.fill_between(x, lower, upper, color=color, alpha=0.28, linewidth=0, zorder=1)
    ax.plot(x, pred.mixture_mean, color=color, alpha=0.95, linewidth=1.4, zorder=2)


def _plot_limits(dataset, x_orig, true_y, predictions, synthetic_args):
    if not synthetic_args.auto_ylim and synthetic_args.ylim is not None:
        lo, hi = (float(v) for v in synthetic_args.ylim)
        if not hi > lo:
            raise ValueError(f"--ylim requires YMAX > YMIN, got {synthetic_args.ylim}")
        return lo, hi

    values = [np.asarray(dataset.targets).reshape(-1)]
    if true_y is not None:
        values.append(np.asarray(true_y).reshape(-1))
    for pred in predictions.values():
        values.extend(
            [
                pred.mixture_mean.reshape(-1),
                (pred.means - 2.0 * pred.stds).reshape(-1),
                (pred.means + 2.0 * pred.stds).reshape(-1),
            ]
        )
    vals = np.concatenate([v[np.isfinite(v)] for v in values if v.size])
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(vals, [0.5, 99.5])
    data_lo = float(np.nanmin(dataset.targets))
    data_hi = float(np.nanmax(dataset.targets))
    if true_y is not None:
        data_lo = min(data_lo, float(np.nanmin(true_y)))
        data_hi = max(data_hi, float(np.nanmax(true_y)))
    lo = min(float(lo), data_lo)
    hi = max(float(hi), data_hi)
    pad = 0.08 * max(hi - lo, 1e-8)
    return lo - pad, hi + pad


def _hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _load_font(size):
    if ImageFont is None:
        return None
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _polyline(draw, points, fill, width=1):
    if len(points) >= 2:
        draw.line(points, fill=fill, width=width, joint="curve")


def figure_panels(synthetic_args, model_order):
    panels = []
    if synthetic_args.include_data_panel and not synthetic_args.overlay_data:
        panels.append("data")
    panels.extend(model_order)
    return panels


def _draw_pillow_data_points(draw, dataset, scale_x, scale_y, left, top, alpha=150, radius=2):
    xs = np.asarray(dataset.inputs).reshape(-1)
    ys = np.asarray(dataset.targets).reshape(-1)
    for x_val, y_val in zip(xs, ys):
        px = scale_x(x_val, left)
        py = scale_y(y_val, top)
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=(30, 30, 30, alpha),
        )


def _pillow_figure(
    path,
    dataset,
    x_orig,
    true_y,
    predictions,
    model_order,
    synthetic_args,
):
    if Image is None or ImageDraw is None:
        raise RuntimeError(
            "Creating figures requires matplotlib or Pillow; neither is available."
        )

    panels = figure_panels(synthetic_args, model_order)
    cols = min(max(1, synthetic_args.max_cols), len(panels))
    rows = int(math.ceil(len(panels) / cols))
    panel_w = 360
    panel_h = 300
    margin_l = 52
    margin_r = 16
    margin_t = 34
    margin_b = 42
    plot_w = panel_w - margin_l - margin_r
    plot_h = panel_h - margin_t - margin_b
    width = cols * panel_w
    height = rows * panel_h

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(18)
    label_font = _load_font(13)
    small_font = _load_font(12)

    y_lim = _plot_limits(dataset, x_orig, true_y, predictions, synthetic_args)
    x_flat = x_orig.reshape(-1)
    x_min, x_max = float(x_flat.min()), float(x_flat.max())
    y_min, y_max = y_lim

    def scale_x(value, left):
        return left + int(round((float(value) - x_min) / max(x_max - x_min, 1e-12) * plot_w))

    def scale_y(value, top):
        return top + int(round((y_max - float(value)) / max(y_max - y_min, 1e-12) * plot_h))

    for panel_idx, panel in enumerate(panels):
        row = panel_idx // cols
        col = panel_idx % cols
        base_x = col * panel_w
        base_y = row * panel_h
        left = base_x + margin_l
        top = base_y + margin_t
        right = left + plot_w
        bottom = top + plot_h

        if panel == "data":
            _draw_pillow_data_points(
                draw,
                dataset,
                scale_x,
                scale_y,
                left,
                top,
                alpha=150,
            )
            if true_y is not None:
                true_points = [
                    (scale_x(x_val, left), scale_y(y_val, top))
                    for x_val, y_val in zip(x_flat, true_y)
                ]
                _polyline(draw, true_points, fill=(0, 0, 0, 255), width=2)
            draw.text((left, base_y + 8), "Data", fill=(0, 0, 0, 255), font=title_font)
        else:
            pred = predictions[panel]
            color = MODEL_COLORS.get(panel, "#1f77b4")
            rgb = _hex_to_rgb(color)
            style = choose_style(panel, pred, synthetic_args.plot_style)
            if style == "density":
                density = mixture_density_grid(pred, y_lim, synthetic_args.density_bins)
                alpha = (np.flipud(density) * 220).astype(np.uint8)
                rgba = np.zeros((density.shape[0], density.shape[1], 4), dtype=np.uint8)
                rgba[..., 0] = rgb[0]
                rgba[..., 1] = rgb[1]
                rgba[..., 2] = rgb[2]
                rgba[..., 3] = alpha
                density_img = Image.fromarray(rgba, mode="RGBA")
                resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
                density_img = density_img.resize((plot_w, plot_h), resample=resample)
                image.alpha_composite(density_img, dest=(left, top))
            elif style == "predictive_interval":
                overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay, "RGBA")
                lower = pred.mixture_mean - 2.0 * pred.mixture_std
                upper = pred.mixture_mean + 2.0 * pred.mixture_std
                upper_pts = [
                    (scale_x(x_val, left), scale_y(y_val, top))
                    for x_val, y_val in zip(x_flat, upper)
                ]
                lower_pts = [
                    (scale_x(x_val, left), scale_y(y_val, top))
                    for x_val, y_val in zip(x_flat[::-1], lower[::-1])
                ]
                overlay_draw.polygon(
                    upper_pts + lower_pts,
                    fill=(rgb[0], rgb[1], rgb[2], 76),
                )
                mean_pts = [
                    (scale_x(x_val, left), scale_y(y_val, top))
                    for x_val, y_val in zip(x_flat, pred.mixture_mean)
                ]
                _polyline(
                    overlay_draw,
                    mean_pts,
                    fill=(rgb[0], rgb[1], rgb[2], 225),
                    width=2,
                )
                image.alpha_composite(overlay)
            else:
                S = pred.means.shape[0]
                if S > synthetic_args.plot_samples:
                    sample_idx = np.linspace(0, S - 1, synthetic_args.plot_samples, dtype=int)
                else:
                    sample_idx = np.arange(S)
                for idx in sample_idx:
                    sample_layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
                    sample_draw = ImageDraw.Draw(sample_layer, "RGBA")
                    mean = pred.means[idx]
                    std = pred.stds[idx]
                    upper = mean + 2.0 * std
                    lower = mean - 2.0 * std
                    upper_pts = [
                        (scale_x(x_val, left), scale_y(y_val, top))
                        for x_val, y_val in zip(x_flat, upper)
                    ]
                    lower_pts = [
                        (scale_x(x_val, left), scale_y(y_val, top))
                        for x_val, y_val in zip(x_flat[::-1], lower[::-1])
                    ]
                    sample_draw.polygon(
                        upper_pts + lower_pts,
                        fill=(rgb[0], rgb[1], rgb[2], 30),
                    )
                    line_pts = [
                        (scale_x(x_val, left), scale_y(y_val, top))
                        for x_val, y_val in zip(x_flat, mean)
                    ]
                    _polyline(
                        sample_draw,
                        line_pts,
                        fill=(rgb[0], rgb[1], rgb[2], 115),
                        width=1,
                    )
                    image.alpha_composite(sample_layer)
            if synthetic_args.overlay_data:
                _draw_pillow_data_points(
                    draw,
                    dataset,
                    scale_x,
                    scale_y,
                    left,
                    top,
                    alpha=165,
                    radius=2,
                )
            if synthetic_args.plot_predictive_mean:
                mean_points = [
                    (scale_x(x_val, left), scale_y(y_val, top))
                    for x_val, y_val in zip(x_flat, pred.mixture_mean)
                ]
                _polyline(draw, mean_points, fill=(0, 0, 0, 255), width=2)
            draw.text(
                (left, base_y + 8),
                pretty_model_name(panel),
                fill=(0, 0, 0, 255),
                font=title_font,
            )

        draw.line((left, bottom, right, bottom), fill=(35, 35, 35, 255), width=1)
        draw.line((left, top, left, bottom), fill=(35, 35, 35, 255), width=1)
        draw.text((left + plot_w // 2 - 4, bottom + 16), "x", fill=(0, 0, 0, 255), font=label_font)
        if col == 0:
            draw.text((base_x + 16, top + plot_h // 2 - 8), "y", fill=(0, 0, 0, 255), font=label_font)
        for tick in np.linspace(x_min, x_max, num=4):
            tx = scale_x(tick, left)
            draw.line((tx, bottom, tx, bottom + 4), fill=(35, 35, 35, 255), width=1)
            draw.text(
                (tx - 10, bottom + 6),
                f"{tick:g}",
                fill=(0, 0, 0, 255),
                font=small_font,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
    image.save(path)
    return path


def save_figure(
    path,
    dataset,
    x_orig,
    true_y,
    predictions,
    model_order,
    synthetic_args,
):
    if plt is None:
        return _pillow_figure(
            path,
            dataset,
            x_orig,
            true_y,
            predictions,
            model_order,
            synthetic_args,
        )
    if LinearSegmentedColormap is None:
        raise RuntimeError("matplotlib.colors is required to create density plots.")

    panels = figure_panels(synthetic_args, model_order)
    cols = min(max(1, synthetic_args.max_cols), len(panels))
    rows = int(math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.45 * cols, 2.1 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    y_lim = _plot_limits(dataset, x_orig, true_y, predictions, synthetic_args)
    x_flat = x_orig.reshape(-1)
    all_axes = axes.reshape(-1)

    for panel_idx, panel in enumerate(panels):
        ax = all_axes[panel_idx]
        if panel == "data":
            ax.scatter(
                np.asarray(dataset.inputs).reshape(-1),
                np.asarray(dataset.targets).reshape(-1),
                **DATA_PANEL_SCATTER_STYLE,
            )
            if true_y is not None:
                ax.plot(x_flat, true_y, color="black", linewidth=1.2)
            ax.set_title("Data", fontsize=11)
        else:
            pred = predictions[panel]
            color = MODEL_COLORS.get(panel, "#1f77b4")
            style = choose_style(panel, pred, synthetic_args.plot_style)
            if style == "density":
                plot_density(ax, x_flat, pred, color, y_lim, synthetic_args.density_bins)
            elif style == "predictive_interval":
                plot_predictive_interval(ax, x_flat, pred, color)
            else:
                plot_sample_bands(ax, x_flat, pred, color, synthetic_args.plot_samples)
            if synthetic_args.overlay_data:
                ax.scatter(
                    np.asarray(dataset.inputs).reshape(-1),
                    np.asarray(dataset.targets).reshape(-1),
                    **DATA_OVERLAY_SCATTER_STYLE,
                )
            if synthetic_args.plot_predictive_mean:
                ax.plot(x_flat, pred.mixture_mean, color="black", linewidth=1.2, zorder=10)
            ax.set_title(pretty_model_name(panel), fontsize=11)

        ax.set_xlim(float(x_flat.min()), float(x_flat.max()))
        ax.set_ylim(*y_lim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8, length=2)

    for ax in all_axes[len(panels) :]:
        ax.axis("off")
    for row in range(rows):
        axes[row, 0].set_ylabel("y", fontsize=9)
    for col in range(cols):
        axes[-1, col].set_xlabel("x", fontsize=9)

    fig.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=synthetic_args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    synthetic_args, base_args = parse_synthetic_args(argv)
    models = resolve_models(synthetic_args.models)
    output_dir = Path(base_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = SYNTHETIC_DATASET_NAME
    dataset = get_dataset(dataset_name)
    train_dataset, train_eval_dataset, _ = make_plot_split(
        dataset,
        base_args,
        synthetic_args,
    )
    dtype = torch.float64 if base_args.dtype == "float64" else torch.float32
    device = torch.device(base_args.device)
    if synthetic_args.xlim is not None:
        x_min, x_max = (float(v) for v in synthetic_args.xlim)
        if not x_max > x_min:
            raise ValueError(f"--xlim requires XMAX > XMIN, got {synthetic_args.xlim}")
    else:
        x_train = np.asarray(dataset.inputs, dtype=np.float64).reshape(-1)
        finite_x = x_train[np.isfinite(x_train)]
        if finite_x.size == 0:
            raise ValueError("Cannot infer plotting x-domain from empty/non-finite training inputs.")
        xpad = float(synthetic_args.xpad)
        if xpad < 0:
            raise ValueError(f"--xpad must be non-negative, got {synthetic_args.xpad}")
        x_min = float(np.min(finite_x)) - xpad
        x_max = float(np.max(finite_x)) + xpad
    x_orig = np.linspace(x_min, x_max, synthetic_args.grid_points)[:, None]
    true_function = getattr(dataset, "true_function", None)
    true_y = (
        np.asarray(true_function(x_orig.reshape(-1))).reshape(-1)
        if callable(true_function)
        else None
    )
    x_grid = normalized_grid(x_orig, train_dataset, dtype, device)

    results = []
    predictions = {}
    plotted_models = []
    failures = {}
    requested_all = "all" in synthetic_args.models

    for model_type in models:
        print(
            f"\n{'=' * 60}\n"
            f"{SYNTHETIC_DATASET_LABEL} model: {pretty_model_name(model_type)}\n"
            f"{'=' * 60}"
        )
        missing = missing_model_reason(model_type)
        if missing is not None:
            failures[model_type] = missing
            if requested_all or synthetic_args.continue_on_error:
                print(f"  [skip] {model_type}: {missing}")
                continue
            raise ImportError(missing)
        model_args = args_for_run(base_args, synthetic_args, model_type)
        try:
            result, model = train_one_model(
                dataset_name,
                model_type,
                base_args,
                synthetic_args,
                train_dataset,
                train_eval_dataset,
            )
            results.append(result)
            predictions[model_type] = make_predictive_grid(
                model,
                model_type,
                model_args,
                x_grid,
                result,
            )
            plotted_models.append(model_type)
        except Exception as exc:
            failures[model_type] = repr(exc)
            if not synthetic_args.continue_on_error:
                raise
            print(f"  [warn] {model_type} failed: {exc!r}")

    figure_path = output_dir / synthetic_args.figure_name
    if plotted_models:
        save_figure(
            figure_path,
            dataset,
            x_orig,
            true_y,
            predictions,
            plotted_models,
            synthetic_args,
        )
        print(f"\nFigure saved to {figure_path}")
        pdf_path = output_dir / synthetic_args.pdf_name if synthetic_args.pdf_name else None
        if pdf_path is not None:
            save_figure(
                pdf_path,
                dataset,
                x_orig,
                true_y,
                predictions,
                plotted_models,
                synthetic_args,
            )
            print(f"PDF saved to {pdf_path}")
    else:
        print("\nNo models completed; figure was not created.")

    summary = {
        "dataset": dataset_name,
        "normalize_inputs": synthetic_args.normalize_inputs,
        "gmvip_matheron_preset": synthetic_args.gmvip_matheron_preset,
        "xlim": [float(x_orig.min()), float(x_orig.max())],
        "ylim": None if synthetic_args.auto_ylim else list(map(float, synthetic_args.ylim)),
        "models": plotted_models,
        "figure": str(figure_path) if plotted_models else None,
        "pdf": str(pdf_path) if plotted_models and pdf_path is not None else None,
        "results": results,
        "failures": failures,
    }
    results_path = output_dir / synthetic_args.results_name
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {results_path}")
    return summary


if __name__ == "__main__":
    main()
