"""Synthetic regression benchmark.

Runs VIP, FTIP, MFVI, FBNN, and TFSVI on the bimodal,
heterocedastic, and skewed synthetic datasets. Each run writes a checkpoint
and a JSON result file under ``results/synthetic`` by default.

Example:
    python -m scripts.synthetic_benchmark --models ftip --datasets bimodal
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset import get_dataset
from src.utils.metrics import MetricsRegression
from src.utils.utils import infinite_loader
from src.priors.generative_functions import BayesianNN, BayesLinear, GP, ExactGP
from src.flows import CouplingFlow, SplineCouplingFlow, SplineCoupling1x1Flow
from src.vip import VIP
from src.ftip import FTIP
from src.mfvi import MFVI
from src.fbnn import FBNN
from src.tfsvi import TFSVI


DATASETS = ["bimodal", "heterocedastic", "skewed"]
MODELS = ["vip", "ftip", "mfvi", "fbnn", "tfsvi"]

# Per-dataset, per-model settings used for the synthetic benchmark.
DATASET_CONFIGS = {
    "bimodal": {
        "shared": {
            "hidden_dims": [10, 10], "activation": "tanh",
            "layer_model": "BayesLinear", "dropout": 0.0,
            "regression_coeffs": 20, "batch_size": 200,
            "cosine_annealing": True, "dtype": "float64",
            "prior_regularizer_scaler": 1.0,
        },
        "vip":  {"lr": 1e-4, "iterations": 200_000, "bb_alpha": 1.0},
        "ftip": {"lr": 1e-3, "iterations": 20_000, "bb_alpha": 1.0,
                 "flow_depth": 2, "num_samples": 20, "eval_samples": 100},
        "mfvi": {"lr": 1e-3, "iterations": 200_000, "bb_alpha": 0.5,
                 "num_samples": 20, "eval_samples": 200},
        "fbnn": {"lr": 1e-3, "iterations": 200_000, "bb_alpha": 1.0,
                 "num_samples": 20, "eval_samples": 100,
                 "num_measurement": 20, "num_context": 20,
                 "context_std": 2.0, "lambda_kl": 1.0,
                 "prior": "gp", "freeze_prior": True},
        "tfsvi": {"lr": 1e-3, "iterations": 200_000, "bb_alpha": 1.0,
                  "num_samples": 20, "eval_samples": 100,
                  "S_ctx": 5, "K_ctx": 20, "sigma_prior": 1.0},
    },
    "heterocedastic": {
        "shared": {
            "hidden_dims": [10, 10], "activation": "tanh",
            "layer_model": "BayesLinear", "dropout": 0.0,
            "regression_coeffs": 20, "batch_size": 200,
            "cosine_annealing": True, "dtype": "float64",
            "prior_regularizer_scaler": 0.1,
        },
        "vip":  {"lr": 1e-4, "iterations": 50_000, "bb_alpha": 1.0},
        "ftip": {"lr": 1e-3, "iterations": 20_000, "bb_alpha": 1.0,
                 "flow_depth": 2, "num_samples": 20, "eval_samples": 100},
        "mfvi": {"lr": 1e-3, "iterations": 50_000, "bb_alpha": 0.5,
                 "num_samples": 20, "eval_samples": 200},
        "fbnn": {"lr": 1e-3, "iterations": 50_000, "bb_alpha": 1.0,
                 "num_samples": 20, "eval_samples": 100,
                 "num_measurement": 20, "num_context": 20,
                 "context_std": 2.0, "lambda_kl": 1.0,
                 "prior": "gp", "freeze_prior": True},
        "tfsvi": {"lr": 1e-3, "iterations": 50_000, "bb_alpha": 1.0,
                  "num_samples": 20, "eval_samples": 100,
                  "S_ctx": 5, "K_ctx": 20, "sigma_prior": 1.0},
    },
    "skewed": {
        "shared": {
            "hidden_dims": [10, 10], "activation": "tanh",
            "layer_model": "BayesLinear", "dropout": 0.0,
            "regression_coeffs": 20, "batch_size": 200,
            "cosine_annealing": True, "dtype": "float64",
            "prior_regularizer_scaler": 1.0,
        },
        "vip":  {"lr": 1e-4, "iterations": 200_000, "bb_alpha": 1.0},
        "ftip": {"lr": 1e-3, "iterations": 20_000, "bb_alpha": 1.0,
                 "flow_depth": 2, "num_samples": 20, "eval_samples": 100},
        "mfvi": {"lr": 1e-3, "iterations": 200_000, "bb_alpha": 0.5,
                 "num_samples": 20, "eval_samples": 200},
        "fbnn": {"lr": 1e-3, "iterations": 200_000, "bb_alpha": 1.0,
                 "num_samples": 20, "eval_samples": 100,
                 "num_measurement": 20, "num_context": 20,
                 "context_std": 2.0, "lambda_kl": 1.0,
                 "prior": "gp", "freeze_prior": True},
        "tfsvi": {"lr": 1e-3, "iterations": 200_000, "bb_alpha": 1.0,
                  "num_samples": 20, "eval_samples": 100,
                  "S_ctx": 5, "K_ctx": 20, "sigma_prior": 1.0},
    },
}

ACTIVATION_FNS = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train VIP/FTIP/MFVI/FBNN/TFSVI on synthetic datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=["all"],
                   choices=MODELS + ["all"],
                   help="Which models to train.")
    p.add_argument("--datasets", nargs="+", default=["all"],
                   choices=DATASETS + ["all"],
                   help="Which synthetic datasets to run on.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="results/synthetic")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-train even when an output JSON already exists.")
    p.add_argument("--iterations_override", type=int, default=None,
                   help="Replace every model's iteration count (for smoke tests).")
    p.add_argument("--prior", choices=["bnn", "gp", "gp_rff"], default=None,
                   help="Override the prior generative function: BNN (default), exact GP (gp), or RFF GP (gp_rff).")
    p.add_argument("--flow_type", type=str, default="spline_1x1",
                   choices=["affine", "spline", "spline_1x1"],
                   help="FTIP flow class. 'affine' = original CouplingFlow, "
                        "'spline' = SplineCouplingFlow (RQ coupling), "
                        "'spline_1x1' = SplineCoupling1x1Flow (spline + Glow "
                        "1x1 LU mixing, default).")
    p.add_argument("--flow_num_bins", type=int, default=8,
                   help="Bins per RQ-spline coupling layer (ignored if flow_type=affine).")
    p.add_argument("--flow_domain", type=float, default=3.0,
                   help="Spline domain half-width B (ignored if flow_type=affine).")
    p.add_argument("--eval_every", type=int, default=2000,
                   help="Iterations between mid-training validation evals.")
    p.add_argument("--use_tqdm", action="store_true", default=True)
    args = p.parse_args()

    if "all" in args.models:
        args.models = MODELS
    if "all" in args.datasets:
        args.datasets = DATASETS
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_bnn(train_ds, shared, num_samples, seed, device, dtype, fix_noise,
               hidden_dims=None):
    """Build the prior generative function.

    ``hidden_dims`` defaults to ``shared["hidden_dims"]``; per-model configs
    can override this when needed.
    """
    if hidden_dims is None:
        hidden_dims = shared["hidden_dims"]
    prior = shared.get("prior", "bnn")
    if prior == "gp":
        return ExactGP(
            num_samples=num_samples,
            input_dim=train_ds.input_dim,
            output_dim=train_ds.output_dim,
            kernel_amp=shared.get("gp_amp", 1.0),
            kernel_length=shared.get("gp_length", 1.0),
            fix_random_noise=fix_noise,
            device=device, seed=seed, dtype=dtype,
        )
    if prior == "gp_rff":
        return GP(
            num_samples=num_samples,
            input_dim=train_ds.input_dim,
            output_dim=train_ds.output_dim,
            inner_layer_dim=shared.get("gp_features", 100),
            kernel_amp=shared.get("gp_amp", 1.0),
            kernel_length=shared.get("gp_length", 1.0),
            fix_random_noise=fix_noise,
            device=device, seed=seed, dtype=dtype,
        )
    return BayesianNN(
        input_dim=train_ds.input_dim,
        num_samples=num_samples,
        structure=hidden_dims,
        activation=ACTIVATION_FNS[shared["activation"]],
        output_dim=train_ds.output_dim,
        layer_model=BayesLinear,
        dropout=shared["dropout"],
        fix_random_noise=fix_noise,
        device=device, seed=seed, dtype=dtype,
    )


def build_vip(shared, cfg, train_ds, seed, device, dtype):
    gen_fn = _build_bnn(train_ds, shared, shared["regression_coeffs"],
                        seed, device, dtype, fix_noise=True)
    return VIP(
        generative_function=gen_fn,
        num_regression_coeffs=shared["regression_coeffs"],
        output_dim=train_ds.output_dim,
        likelihood="regression",
        num_data=len(train_ds),
        bb_alpha=cfg["bb_alpha"],
        use_prior_regularizer=True,
        regularizer_mode="evidence",
        prior_regularizer_scaler=shared["prior_regularizer_scaler"],
        y_mean=train_ds.targets_mean, y_std=train_ds.targets_std,
        dtype=dtype, device=device, seed=seed,
    )


def build_ftip(shared, cfg, train_ds, seed, device, dtype):
    # FTIP can override the BNN width via cfg["hidden_dims"]; other models
    # continue to inherit from shared["hidden_dims"] so their existing
    # checkpoints remain loadable.
    hidden_dims = cfg.get("hidden_dims", shared["hidden_dims"])
    gen_fn = _build_bnn(train_ds, shared, shared["regression_coeffs"],
                        seed, device, dtype, fix_noise=True,
                        hidden_dims=hidden_dims)
    flow_type = shared.get("flow_type", "spline_1x1")
    flow_kwargs = dict(depth=cfg["flow_depth"],
                       input_dim=shared["regression_coeffs"],
                       device=device, dtype=dtype, seed=seed)
    if flow_type == "affine":
        flow = CouplingFlow(**flow_kwargs)
    elif flow_type == "spline":
        flow = SplineCouplingFlow(**flow_kwargs,
                                  num_bins=shared.get("flow_num_bins", 8),
                                  B=shared.get("flow_domain", 3.0))
    elif flow_type == "spline_1x1":
        flow = SplineCoupling1x1Flow(**flow_kwargs,
                                     num_bins=shared.get("flow_num_bins", 8),
                                     B=shared.get("flow_domain", 3.0))
    else:
        raise ValueError(f"Unknown flow_type: {flow_type!r}")
    return FTIP(
        generative_function=gen_fn,
        num_regression_coeffs=shared["regression_coeffs"],
        output_dim=train_ds.output_dim,
        likelihood="regression",
        num_data=len(train_ds),
        bb_alpha=cfg["bb_alpha"],
        use_prior_regularizer=True,
        regularizer_mode="evidence",
        prior_regularizer_scaler=shared["prior_regularizer_scaler"],
        y_mean=train_ds.targets_mean, y_std=train_ds.targets_std,
        dtype=dtype, device=device, seed=seed,
        flow=flow, num_samples=cfg["num_samples"],
    )


def build_mfvi(shared, cfg, train_ds, seed, device, dtype):
    gen_fn = _build_bnn(train_ds, shared, cfg["num_samples"],
                        seed, device, dtype, fix_noise=False)
    return MFVI(
        generative_function=gen_fn,
        output_dim=train_ds.output_dim,
        likelihood="regression",
        num_data=len(train_ds),
        num_samples=cfg["num_samples"],
        bb_alpha=cfg["bb_alpha"],
        y_mean=train_ds.targets_mean, y_std=train_ds.targets_std,
        device=device, dtype=dtype,
    )


def _set_bnn_num_samples(bnn, S):
    """Mutate a BayesianNN's ``num_samples`` and regenerate cached noise.

    Used to rebrand the shared posterior BNN for TFSVI (which needs S=1
    so the BNN forward returns ``[1, N, D]``, and TFSVI then squeezes).
    """
    old = bnn.num_samples
    bnn.num_samples = S
    if not hasattr(bnn, "layers"):
        return
    for layer in bnn.layers:
        if hasattr(layer, "num_samples"):
            layer.num_samples = S
        if (hasattr(layer, "fix_random_noise") and layer.fix_random_noise
                and S != old):
            layer.noise = layer.get_noise(first_call=True)


def build_fbnn(shared, cfg, train_ds, seed, device, dtype):
    # Posterior BNN: same architecture/layer-model as VIP/FTIP. FBNN mutates
    # its ``num_samples`` (and regenerates cached noise) every step, so the
    # construction-time value just sets the initial state.
    gen_fn = _build_bnn(train_ds, shared, cfg["num_samples"],
                        seed, device, dtype, fix_noise=True)
    if cfg["prior"] == "gp":
        prior = GP(
            input_dim=train_ds.input_dim,
            output_dim=train_ds.output_dim,
            inner_layer_dim=shared.get("gp_features", 100),
            seed=seed, device=device, dtype=dtype,
        )
    else:
        # BNN prior: same architecture, different seed (frozen below).
        prior = _build_bnn(train_ds, shared, cfg["num_samples"],
                           seed + 1, device, dtype, fix_noise=True)
    return FBNN(
        generative_function=gen_fn,
        prior_function=prior,
        output_dim=train_ds.output_dim,
        likelihood="regression",
        num_data=len(train_ds),
        num_samples=cfg["num_samples"],
        num_measurement=cfg["num_measurement"],
        num_context=cfg["num_context"],
        context_std=cfg["context_std"],
        bb_alpha=cfg["bb_alpha"],
        lambda_kl=cfg["lambda_kl"],
        freeze_prior=cfg["freeze_prior"],
        y_mean=train_ds.targets_mean, y_std=train_ds.targets_std,
        device=device, dtype=dtype,
    )


def build_tfsvi(shared, cfg, train_ds, seed, device, dtype):
    # Uses the same backbone BNN as VIP/FTIP/FBNN as the architecture
    # template, so TFSVI's q(theta) mirrors the BNN's parameter structure:
    # with SimplerBayesLinear that's 4 scalars per layer (TFSVI doubles to
    # 8), with BayesLinear it's 2P (TFSVI doubles to 4P). ``num_samples``
    # is set to 1 (and cached noise regenerated) so the BNN forward is
    # deterministic in theta and the linearised KL is well-defined.
    gen_fn = _build_bnn(train_ds, shared, shared["regression_coeffs"],
                        seed, device, dtype, fix_noise=True)
    _set_bnn_num_samples(gen_fn, 1)
    return TFSVI(
        input_dim=train_ds.input_dim,
        output_dim=train_ds.output_dim,
        structure=shared["hidden_dims"],
        activation=ACTIVATION_FNS[shared["activation"]],
        likelihood="regression",
        num_data=len(train_ds),
        sigma_prior=cfg["sigma_prior"],
        num_samples=cfg["num_samples"],
        bb_alpha=cfg["bb_alpha"],
        S_ctx=cfg["S_ctx"],
        K_ctx=cfg["K_ctx"],
        y_mean=train_ds.targets_mean, y_std=train_ds.targets_std,
        generative_function=gen_fn,
        device=device, dtype=dtype,
    )


BUILDERS = {"vip": build_vip, "ftip": build_ftip,
            "mfvi": build_mfvi, "fbnn": build_fbnn, "tfsvi": build_tfsvi}


# ---------------------------------------------------------------------------
# Prediction helpers — return (mean_pred, std_pred) of shape (S, N, D) so the
# shared MetricsRegression can consume any model's output.
# ---------------------------------------------------------------------------

def predict_vip(model, x):
    mean, std = model(x)                       # (1, N, D), (1, N, D)
    return mean, std


def predict_ftip(model, x, S):
    old = model.num_samples
    model.num_samples = S
    samples, noise_std = model(x)              # (S, N, D), (S, 1)
    model.num_samples = old
    std = noise_std.unsqueeze(-1).expand_as(samples)
    return samples, std


def predict_mfvi(model, x, S):
    F = model.predict_f_samples(x, S)          # (S, N, D), normalized
    mean = F * model.y_std + model.y_mean
    std_scalar = torch.sqrt(torch.exp(model.log_variance)) * model.y_std
    std = std_scalar.expand_as(mean)
    return mean, std


def predict_fbnn(model, x, S):
    # FBNN's posterior BNN caches S weight-noise samples; switch to the
    # eval-time S and restore afterwards so training-time state is preserved.
    old_S = model.num_samples
    model._set_num_samples(S)
    model.num_samples = S
    try:
        F = model.predict_f_samples(x, S)      # (S, N, D), normalized
    finally:
        model._set_num_samples(old_S)
        model.num_samples = old_S
    mean = F * model.y_std + model.y_mean
    std_scalar = torch.sqrt(torch.exp(model.log_variance)) * model.y_std
    std = std_scalar.expand_as(mean)
    return mean, std


def predict_tfsvi(model, x, S):
    # TFSVI uses vmap over q(theta) samples internally; just call
    # predict_f_samples to draw S parameter samples per call.
    F = model.predict_f_samples(x, S)          # (S, N, D), normalized
    mean = F * model.y_std + model.y_mean
    std_scalar = torch.sqrt(torch.exp(model.log_variance)) * model.y_std
    std = std_scalar.expand_as(mean)
    return mean, std


def model_predict(model_type, model, x, shared, cfg):
    """Dispatch to the right predict function; returns (mean, std), both (S,N,D)."""
    if model_type == "vip":
        return predict_vip(model, x)
    if model_type == "ftip":
        return predict_ftip(model, x, cfg["eval_samples"])
    if model_type == "mfvi":
        return predict_mfvi(model, x, cfg["eval_samples"])
    if model_type == "fbnn":
        return predict_fbnn(model, x, cfg["eval_samples"])
    if model_type == "tfsvi":
        return predict_tfsvi(model, x, cfg["eval_samples"])
    raise ValueError(model_type)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _metrics_batch_size(model_type, cfg):
    """Pick a batch size that keeps CRPS memory in check (256M pair entries)."""
    if model_type == "vip":
        S = 1
    elif model_type in ("ftip", "mfvi", "fbnn", "tfsvi"):
        S = cfg["eval_samples"]
    else:
        raise ValueError(model_type)
    S_crps = min(S, 100)
    return max(16, min(512, 256_000_000 // max(1, S_crps * S_crps)))


def evaluate(model_type, model, dataset, shared, cfg, device, dtype, light=False):
    x = torch.tensor(dataset.inputs, dtype=dtype, device=device)
    y = torch.tensor(dataset.targets, dtype=dtype, device=device)
    batch_size = _metrics_batch_size(model_type, cfg)
    metrics = MetricsRegression(num_data=len(dataset), device=device)
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            mean, std = model_predict(model_type, model, xb, shared, cfg)
            metrics.update(yb, loss=torch.tensor(0.0),
                           mean_pred=mean, std_pred=std, light=light)
    d = metrics.get_dict()
    model.train()
    if light:
        return {"RMSE": d["RMSE"], "NLL": d["NLL"]}
    return d


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _train_iters(model, train_loader, train_test_ds, test_ds,
                 lr, iterations, shared, cfg, model_type, args,
                 device, dtype, desc="Training"):
    """Shared iteration loop for VIP / FTIP / MFVI."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if shared["cosine_annealing"]:
        T_max = max(1, math.ceil(iterations / len(train_loader)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=lr / 100
        )

    losses = []
    metrics_history = {"iterations": [], "train": [], "validation": []}
    model.train()
    data_stream = infinite_loader(train_loader)
    iters_per_epoch = len(train_loader)
    loop = (tqdm(range(iterations), unit=" iter", desc=desc)
            if args.use_tqdm else range(iterations))

    for i in loop:
        inputs, target = next(data_stream)
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        loss = model._train_step(optimizer, inputs, target)
        losses.append(loss.detach().item())

        if scheduler is not None and (i + 1) % iters_per_epoch == 0:
            scheduler.step()

        if (i + 1) % args.eval_every == 0 and train_test_ds is not None:
            metrics_history["iterations"].append(i + 1)
            metrics_history["train"].append(evaluate(
                model_type, model, train_test_ds, shared, cfg,
                device, dtype, light=True,
            ))
            metrics_history["validation"].append(evaluate(
                model_type, model, test_ds, shared, cfg,
                device, dtype, light=True,
            ))
        if args.use_tqdm:
            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return losses, metrics_history


def _collect_diagnostics(model_type, model):
    d = {}
    for attr in ("KLs", "bb_alphas", "prior_regularizers"):
        if hasattr(model, attr):
            d[attr] = [float(v) for v in getattr(model, attr)]
    if model_type == "ftip":
        if hasattr(model, "base_KLs"):
            d["base_KLs"] = [float(v) for v in model.base_KLs]
        if hasattr(model, "flow_ldj"):
            d["flow_ldj"] = [float(v) for v in model.flow_ldj]
    return d


# ---------------------------------------------------------------------------
# Per-run orchestration
# ---------------------------------------------------------------------------

def _hyperparameter_snapshot(model_type, shared, cfg, args):
    hp = dict(shared)
    hp.update(cfg)
    hp["seed"] = args.seed
    hp["device"] = args.device
    return hp


def _out_paths(args, dataset_name, model_type):
    # Non-default FTIP flows append a tag so the three flow types don't collide.
    if model_type == "ftip":
        ft = getattr(args, "flow_type", "affine")
        flow_tag = "" if ft == "affine" else f"_{ft.replace('_', '')}"
    else:
        flow_tag = ""
    base = os.path.join(
        args.output_dir,
        f"{model_type}_{dataset_name}{flow_tag}_seed{args.seed}",
    )
    return base + ".json", base + ".pt"


def run_single(dataset_name, model_type, args):
    json_path, ckpt_path = _out_paths(args, dataset_name, model_type)
    if os.path.exists(json_path) and not args.overwrite:
        print(f"  Skipping {model_type}/{dataset_name}: {json_path} exists.")
        with open(json_path) as f:
            return json.load(f)

    full_cfg = DATASET_CONFIGS[dataset_name]
    shared = dict(full_cfg["shared"])
    cfg = dict(full_cfg[model_type])
    if args.iterations_override is not None:
        cfg["iterations"] = args.iterations_override
    if args.prior is not None:
        shared["prior"] = args.prior
    # Propagate FTIP flow knobs from CLI into shared so build_ftip can read them.
    shared["flow_type"] = args.flow_type
    shared["flow_num_bins"] = args.flow_num_bins
    shared["flow_domain"] = args.flow_domain

    device = torch.device(args.device)
    dtype = torch.float64 if shared["dtype"] == "float64" else torch.float32

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  Model: {model_type.upper()}  |  "
          f"iters={cfg['iterations']}  lr={cfg['lr']}")
    print(f"{'='*60}")

    dataset = get_dataset(dataset_name)
    train_ds, train_test_ds, test_ds = dataset.get_split(args.test_size, args.seed)
    torch.manual_seed(args.seed)

    use_cuda = "cuda" in args.device
    train_loader = DataLoader(
        train_ds, batch_size=shared["batch_size"], shuffle=True,
        pin_memory=use_cuda, num_workers=0,
    )

    model = BUILDERS[model_type](shared, cfg, train_ds, args.seed, device, dtype)

    # FBNN's measurement reservoir + TFSVI's context-sampling pool are
    # normally populated inside their .fit(); we drive training through
    # _train_iters / _train_step instead, so populate them upfront here.
    if model_type == "fbnn":
        model._fill_reservoir(train_loader)
    elif model_type == "tfsvi":
        all_X = [inputs for inputs, _ in train_loader]
        model._train_inputs = torch.cat(all_X, dim=0)

    t0 = time.time()
    diagnostics = {}
    metrics_history = {"iterations": [], "train": [], "validation": []}
    losses, metrics_history = _train_iters(
            model, train_loader, train_test_ds, test_ds,
            lr=cfg["lr"], iterations=cfg["iterations"],
            shared=shared, cfg=cfg, model_type=model_type, args=args,
            device=device, dtype=dtype,
            desc=f"{model_type.upper()} on {dataset_name}",
        )
    diagnostics = _collect_diagnostics(model_type, model)

    train_time = time.time() - t0

    train_m = evaluate(model_type, model, train_test_ds, shared, cfg,
                       device, dtype, light=False)
    test_m = evaluate(model_type, model, test_ds, shared, cfg,
                      device, dtype, light=False)

    result = {
        "dataset": dataset_name,
        "model": model_type,
        "hyperparameters": _hyperparameter_snapshot(model_type, shared, cfg, args),
        "train_time_s": round(train_time, 2),
        "train": train_m,
        "test": test_m,
        "losses": losses,
        "metrics_history": metrics_history,
        "diagnostics": diagnostics,
    }

    for split, m in [("Train", train_m), ("Test", test_m)]:
        print(f"  {model_type.upper()} {split}: RMSE={m['RMSE']:.4f}  "
              f"NLL={m['NLL']:.4f}  CRPS={m['CRPS']:.4f}  CQM={m['CQM']:.4f}")
    print(f"  Time: {train_time:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {ckpt_path}")
    print(f"  Saved {json_path}")
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_results = []
    for ds in args.datasets:
        for mt in args.models:
            all_results.append(run_single(ds, mt, args))


if __name__ == "__main__":
    main()
