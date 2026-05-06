"""Binary classification benchmark.

Runs HIGGS, SUSY, and Rectangles with VIP, FTIP, MFVI, FBNN, or TFSVI.
The evaluation path converts each model's output to probability samples
before computing binary metrics.

Example:
    python -m scripts.binary_classification_benchmark --model ftip --dataset Rectangles
"""

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset import get_dataset
from src.utils.likelihood import inv_probit
from src.utils.metrics import MetricsBinary
from src.utils.utils import infinite_loader
from src.priors.generative_functions import (
    BayesianNN, BayesLinear, SimplerBayesLinear, GP,
)
from src.flows import CouplingFlow, SplineCouplingFlow, SplineCoupling1x1Flow
from src.vip import VIP
from src.ftip import FTIP
from src.mfvi import MFVI
from src.fbnn import FBNN
from src.tfsvi import TFSVI

BINARY_DATASETS = ["HIGGS", "SUSY", "Rectangles"]

ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}

LAYER_MODELS = {
    "BayesLinear": BayesLinear,
    "SimplerBayesLinear": SimplerBayesLinear,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Binary classification benchmark (HIGGS / SUSY / Rectangles)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Experiment ---
    p.add_argument("--model", type=str, required=True,
                    choices=["vip", "ftip", "mfvi", "fbnn", "tfsvi"],
                    help="Model to train.")
    p.add_argument("--dataset", type=str, required=True,
                    help="Dataset name (HIGGS / SUSY / Rectangles) or 'all'.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "float64"], help="Tensor dtype.")
    p.add_argument("--device", type=str, default=None,
                    help="Torch device (default: cuda if available, else cpu).")
    p.add_argument("--output_dir", type=str, default="results",
                    help="Directory to save result JSON files.")

    # --- Generative function (BNN MLP — these are tabular datasets) ---
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[100, 100],
                    help="Hidden layer widths for the BayesianNN backbone.")
    p.add_argument("--activation", type=str, default="relu",
                    choices=list(ACTIVATIONS.keys()),
                    help="Activation function for the BayesianNN.")
    p.add_argument("--layer_model", type=str, default="BayesLinear",
                    choices=list(LAYER_MODELS.keys()),
                    help="Bayesian layer type.")
    p.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate in the generative model.")
    p.add_argument("--weight_log_sigma_init", type=float, default=None,
                    help="Initial value of weight_log_sigma. Default: -1.0 "
                         "for MFVI/FBNN (avoid posterior collapse), 0.0 for "
                         "VIP/FTIP/TFSVI.")

    # --- Shared VIP / FTIP / MFVI ---
    p.add_argument("--regression_coeffs", type=int, default=20,
                    help="Number of regression coefficients (S).")
    p.add_argument("--bb_alpha", type=float, default=0.5,
                    help="BB-alpha parameter (0 = ELBO, 1 = BB-alpha energy).")
    p.add_argument("--use_prior_regularizer", action="store_true", default=False,
                    help="Enable prior regularizer.")
    p.add_argument("--regularizer_mode", type=str, default="evidence",
                    choices=["evidence", "KL"],
                    help="Prior regularizer mode.")
    p.add_argument("--prior_regularizer_scaler", type=float, default=1.0,
                    help="Prior regularizer scaling factor.")

    # --- FTIP-specific ---
    p.add_argument("--flow_type", type=str, default="spline_1x1",
                    choices=["affine", "spline", "spline_1x1"],
                    help="FTIP flow class.")
    p.add_argument("--flow_num_bins", type=int, default=8,
                    help="Bins per RQ-spline coupling layer (ignored if flow_type=affine).")
    p.add_argument("--flow_domain", type=float, default=3.0,
                    help="Spline domain half-width B (ignored if flow_type=affine).")
    p.add_argument("--flow_depth", type=int, default=4,
                    help="Number of coupling layers in the normalizing flow.")
    p.add_argument("--num_samples", type=int, default=200,
                    help="Number of MC posterior samples (FTIP training).")
    p.add_argument("--eval_samples", type=int, default=1000,
                    help="Number of MC samples used at evaluation time.")
    p.add_argument("--warm_start_from", type=str, default=None,
                    help="Path to a VIP checkpoint (.pt) for warm-starting FTIP.")
    p.add_argument("--learnable_affine", action="store_true", default=True,
                    help="Make the affine warm-start layer trainable.")
    p.add_argument("--no_learnable_affine", action="store_true",
                    help="Fix the affine warm-start layer (not trainable).")

    # --- Auto warm-start (VIP -> FTIP) ---
    p.add_argument("--auto_warm_start", action="store_true", default=True,
                    help="Train VIP first, then warm-start FTIP (FTIP only).")
    p.add_argument("--no_auto_warm_start", action="store_true",
                    help="Disable auto warm-start; train FTIP from scratch.")
    p.add_argument("--vip_epochs", type=int, default=None,
                    help="Epochs for VIP pre-training phase.")
    p.add_argument("--vip_iterations", type=int, default=None,
                    help="Iterations for VIP pre-training phase.")
    p.add_argument("--vip_lr", type=float, default=1e-3,
                    help="Learning rate for VIP pre-training phase.")
    p.add_argument("--ftip_lr", type=float, default=1e-4,
                    help="Learning rate for FTIP fine-tuning phase.")

    # --- FBNN-specific ---
    p.add_argument("--fbnn_prior", type=str, default="gp",
                    choices=["gp", "bnn"],
                    help="fBNN prior family: 'gp' (RFF GP) or 'bnn'.")
    p.add_argument("--fbnn_freeze_prior", action="store_true", default=False,
                    help="Freeze the prior's parameters.")
    p.add_argument("--fbnn_gp_inner_dim", type=int, default=10,
                    help="Inner-layer dim of the RFF GP prior.")
    p.add_argument("--fbnn_num_measurement", type=int, default=20,
                    help="# training-point measurements for the functional KL.")
    p.add_argument("--fbnn_num_context", type=int, default=20,
                    help="# OOD context points sampled from N(0, context_std^2).")
    p.add_argument("--fbnn_context_std", type=float, default=2.0,
                    help="Std of the Gaussian from which context points are sampled.")
    p.add_argument("--fbnn_lambda_kl", type=float, default=1.0,
                    help="Weight on the functional KL term in the FBNN ELBO.")
    p.add_argument("--fbnn_num_eval_samples", type=int, default=200,
                    help="MC posterior samples used at FBNN evaluation time.")

    # --- TFSVI-specific ---
    p.add_argument("--tfsvi_sigma_prior", type=float, default=1.0,
                    help="Prior std for the parameter Gaussian.")
    p.add_argument("--tfsvi_S_ctx", type=int, default=5,
                    help="# context sets in the max-KL estimator.")
    p.add_argument("--tfsvi_K_ctx", type=int, default=20,
                    help="# points per context set.")
    p.add_argument("--tfsvi_num_train_samples", type=int, default=20,
                    help="MC parameter samples per TFSVI training step.")
    p.add_argument("--tfsvi_num_eval_samples", type=int, default=200,
                    help="MC parameter samples used at TFSVI evaluation time.")

    # --- MFVI-specific ---
    p.add_argument("--mfvi_num_eval_samples", type=int, default=200,
                    help="MC weight samples used at MFVI evaluation time.")

    # --- Checkpointing ---
    p.add_argument("--save_checkpoint", action="store_true", default=True,
                    help="Save model checkpoint after training.")
    p.add_argument("--no_save_checkpoint", action="store_true",
                    help="Disable saving model checkpoint.")

    # --- Training ---
    p.add_argument("--batch_size", type=int, default=1024,
                    help="Training batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--iterations", type=int, default=None,
                    help="Number of training iterations (mutually exclusive with --epochs).")
    p.add_argument("--epochs", type=int, default=None,
                    help="Number of training epochs (mutually exclusive with --iterations).")
    p.add_argument("--eval_every", type=int, default=2000,
                    help="Compute light metrics on train/test every N iterations.")
    p.add_argument("--eval_subset", type=int, default=50000,
                    help="Cap on examples used in light eval / final eval (per split). "
                         "HIGGS/SUSY are too large for full eval per step.")
    p.add_argument("--cosine_annealing", action="store_true", default=True,
                    help="Use cosine annealing LR schedule.")
    p.add_argument("--no_cosine_annealing", action="store_true",
                    help="Disable cosine annealing.")
    args = p.parse_args()

    # Resolve mutually-exclusive flags
    if args.no_cosine_annealing:
        args.cosine_annealing = False
    if args.no_save_checkpoint:
        args.save_checkpoint = False
    if args.no_auto_warm_start:
        args.auto_warm_start = False
    if args.no_learnable_affine:
        args.learnable_affine = False

    if args.warm_start_from:
        args.auto_warm_start = False

    # HIGGS/SUSY have millions of rows — default to an iteration budget.
    if args.iterations is None and args.epochs is None:
        args.iterations = 50000

    if args.vip_epochs is None and args.vip_iterations is None:
        args.vip_epochs = args.epochs
        args.vip_iterations = args.iterations

    # Default weight init: -1.0 (sigma~0.37) keeps BNN-basis logits bounded.
    # On binary tabular data with deep MLPs, sigma=1.0 lets logits explode to
    # |F|>2000, saturating inv_probit and crushing VIP NLL/ECE. -1.0 matches
    # the MFVI/FBNN classification default and works across all 5 models here.
    if args.weight_log_sigma_init is None:
        args.weight_log_sigma_init = -1.0

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def _ensure_layers_alias(gen_fn):
    """MFVI/FBNN iterate ``gen_fn.layers``; alias when needed."""
    if hasattr(gen_fn, 'layers'):
        return
    pieces = []
    if hasattr(gen_fn, 'head'):
        pieces.extend(list(gen_fn.head))
    if pieces:
        object.__setattr__(gen_fn, 'layers', pieces)


def _set_bnn_num_samples(bnn, S):
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


def _set_bnn_fix_random_noise(bnn, fix):
    if hasattr(bnn, "fix_random_noise"):
        bnn.fix_random_noise = fix
    if not hasattr(bnn, "layers"):
        return
    for layer in bnn.layers:
        if hasattr(layer, "fix_random_noise"):
            layer.fix_random_noise = fix


def build_model(args, train_dataset, model_type=None):
    """Build a binary-classification model.

    output_dim is fixed to 1 (Bernoulli/probit likelihood). num_classes=2 is
    passed for API compatibility but unused on the binary path.
    """
    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    gen_fn = BayesianNN(
        input_dim=train_dataset.input_dim,
        num_samples=args.regression_coeffs,
        structure=args.hidden_dims,
        activation=ACTIVATIONS[args.activation],
        output_dim=1,
        layer_model=LAYER_MODELS[args.layer_model],
        dropout=args.dropout,
        weight_log_sigma_init=args.weight_log_sigma_init,
        device=device,
        seed=args.seed,
        dtype=dtype,
    )

    common = dict(
        generative_function=gen_fn,
        num_regression_coeffs=args.regression_coeffs,
        output_dim=1,
        likelihood="binary",
        num_data=len(train_dataset),
        num_classes=2,
        bb_alpha=args.bb_alpha,
        use_prior_regularizer=args.use_prior_regularizer,
        regularizer_mode=args.regularizer_mode,
        prior_regularizer_scaler=args.prior_regularizer_scaler,
        y_mean=0.0,
        y_std=1.0,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    if model_type == "vip":
        return VIP(**common)

    if model_type == "ftip":
        flow = _build_flow(
            args,
            input_dim=args.regression_coeffs * 1,
            device=device, dtype=dtype,
        )
        return FTIP(**common, flow=flow, num_samples=args.num_samples)

    if model_type == "mfvi":
        _ensure_layers_alias(gen_fn)
        _set_bnn_fix_random_noise(gen_fn, False)
        return MFVI(
            generative_function=gen_fn,
            output_dim=1,
            likelihood="binary",
            num_data=len(train_dataset),
            num_samples=args.regression_coeffs,
            num_classes=2,
            bb_alpha=args.bb_alpha,
            y_mean=0.0,
            y_std=1.0,
            device=device,
            dtype=dtype,
        )

    if model_type == "tfsvi":
        # Function-space VI (Rudner et al., 2022). Reuses the shared
        # ``gen_fn`` as the architecture template so TFSVI's q(theta)
        # mirrors the BNN's parameter structure: with SimplerBayesLinear
        # that's 4 scalars per layer (TFSVI doubles to 8), with BayesLinear
        # it's 2P (TFSVI doubles to 4P). This keeps the structure-of-q
        # consistent with the MFVI/VIP/FTIP/FBNN baselines. ``num_samples``
        # is set to 1 (and cached noise regenerated) so the BNN forward
        # returns ``[1, N, D]`` and is deterministic in theta — required
        # for TFSVI's linearised KL.
        _ensure_layers_alias(gen_fn)
        _set_bnn_num_samples(gen_fn, 1)
        return TFSVI(
            input_dim=train_dataset.input_dim,
            output_dim=1,
            structure=args.hidden_dims,
            activation=ACTIVATIONS[args.activation],
            likelihood="binary",
            num_data=len(train_dataset),
            sigma_prior=args.tfsvi_sigma_prior,
            num_samples=args.tfsvi_num_train_samples,
            num_classes=2,
            bb_alpha=args.bb_alpha,
            S_ctx=args.tfsvi_S_ctx,
            K_ctx=args.tfsvi_K_ctx,
            y_mean=0.0,
            y_std=1.0,
            generative_function=gen_fn,
            device=device,
            dtype=dtype,
        )

    if model_type == "fbnn":
        _ensure_layers_alias(gen_fn)
        if args.fbnn_prior == "gp":
            prior = GP(
                input_dim=train_dataset.input_dim,
                output_dim=1,
                inner_layer_dim=args.fbnn_gp_inner_dim,
                seed=args.seed,
                device=device,
                dtype=dtype,
            )
        else:
            prior = BayesianNN(
                input_dim=train_dataset.input_dim,
                num_samples=args.regression_coeffs,
                structure=args.hidden_dims,
                activation=ACTIVATIONS[args.activation],
                output_dim=1,
                layer_model=LAYER_MODELS[args.layer_model],
                dropout=args.dropout,
                fix_random_noise=True,
                device=device,
                seed=args.seed + 1,
                dtype=dtype,
            )
        return FBNN(
            generative_function=gen_fn,
            prior_function=prior,
            output_dim=1,
            likelihood="binary",
            num_data=len(train_dataset),
            num_samples=args.regression_coeffs,
            num_measurement=args.fbnn_num_measurement,
            num_context=args.fbnn_num_context,
            context_std=args.fbnn_context_std,
            bb_alpha=args.bb_alpha,
            lambda_kl=args.fbnn_lambda_kl,
            num_classes=2,
            y_mean=0.0,
            y_std=1.0,
            freeze_prior=args.fbnn_freeze_prior,
            device=device,
            dtype=dtype,
        )

    raise ValueError(f"Unknown model_type: {model_type!r}")


def _build_flow(args, input_dim, device, dtype):
    common_kw = dict(depth=args.flow_depth, input_dim=input_dim,
                     device=device, dtype=dtype, seed=args.seed)
    if args.flow_type == "affine":
        return CouplingFlow(**common_kw)
    if args.flow_type == "spline":
        return SplineCouplingFlow(
            **common_kw, num_bins=args.flow_num_bins, B=args.flow_domain,
        )
    if args.flow_type == "spline_1x1":
        return SplineCoupling1x1Flow(
            **common_kw, num_bins=args.flow_num_bins, B=args.flow_domain,
        )
    raise ValueError(f"Unknown flow_type: {args.flow_type!r}")


def _eval_samples_for(args, model_type):
    if model_type == "fbnn":
        return args.fbnn_num_eval_samples
    if model_type == "tfsvi":
        return args.tfsvi_num_eval_samples
    if model_type == "mfvi":
        return args.mfvi_num_eval_samples
    return args.eval_samples


def _to_probabilities(model_type, samples):
    """Normalize each model's binary output to probability samples (S, N, 1).

    VIP returns ``(mean_prob, var)`` — the mean is already a probit-marginalized
    probability, shape (1, N, 1). FTIP and TFSVI return raw f-samples (logits)
    that need ``inv_probit``. MFVI and FBNN already apply ``inv_probit`` inside
    ``predict_y_samples``.
    """
    if model_type in ("ftip", "tfsvi"):
        return inv_probit(samples)
    return samples


def evaluate(model, dataset, args, model_type=None, batch_size=4096,
             cap=None, light=False):
    """Compute binary metrics on a dataset.

    cap : int or None
        If set, truncate to the first ``cap`` examples (used for HIGGS/SUSY
        where 5–10M-row eval per call is wasteful). The split is contiguous,
        so the cap acts on a deterministic prefix.
    """
    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    x = torch.tensor(dataset.inputs, dtype=dtype, device=device)
    y = torch.tensor(dataset.targets, dtype=dtype, device=device)
    if cap is not None and x.shape[0] > cap:
        x = x[:cap]
        y = y[:cap]

    metrics = MetricsBinary(num_data=len(dataset), device=device)
    was_training = model.training
    model.eval()

    old_mc = getattr(model, 'num_mc_samples', None)
    if model_type == "vip":
        model.num_mc_samples = args.eval_samples
    eval_S = _eval_samples_for(args, model_type)
    fbnn_old_S = None
    if model_type == "fbnn":
        fbnn_old_S = model.num_samples
        model._set_num_samples(eval_S)
        model.num_samples = eval_S

    with torch.no_grad():
        if model_type == "ftip":
            a = model.sample_flow_coefficients(args.eval_samples)
        for i in range(0, x.shape[0], batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            if model_type == "ftip":
                samples, _ = model.forward_with_coefficients(xb, a)
            elif model_type == "vip":
                samples, _ = model(xb)
            else:  # mfvi / fbnn / tfsvi
                samples = model.predict_y_samples(xb, eval_S)
            probs = _to_probabilities(model_type, samples)
            metrics.update(yb, loss=torch.tensor(0.0), mean_pred=probs)

    if old_mc is not None:
        model.num_mc_samples = old_mc
    if fbnn_old_S is not None:
        model._set_num_samples(fbnn_old_S)
        model.num_samples = fbnn_old_S
    if was_training:
        model.train()

    d = metrics.get_dict()
    if light:
        return {"Error": d["Error"], "NLL": d["NLL"]}
    return d


def train_with_metrics(model, train_loader, train_test_dataset, validation_dataset,
                       args, lr=None, epochs=None, iterations=None,
                       model_type=None, desc="Training"):
    if lr is None:
        lr = args.lr
    if model_type is None:
        model_type = args.model
    if epochs is None and iterations is None:
        epochs = args.epochs
        iterations = args.iterations

    device = torch.device(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if args.cosine_annealing:
        if iterations is not None:
            T_max = max(1, math.ceil(iterations / len(train_loader)))
        else:
            T_max = max(1, epochs)
        eta_min = lr / 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min,
        )

    losses = []
    metrics_history = {"iterations": [], "train": [], "validation": []}

    if model_type == "tfsvi":
        all_X = [inp for inp, _ in train_loader]
        model._train_inputs = torch.cat(all_X, dim=0).to(device)

    model.train()

    if iterations is not None:
        total = iterations
        data_stream = infinite_loader(train_loader)
        iters_per_epoch = len(train_loader)
        loop = tqdm(range(total), unit=" iter", desc=desc)
        for i in loop:
            inputs, target = next(data_stream)
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = model._train_step(optimizer, inputs, target)
            losses.append(loss.item())

            if scheduler is not None and (i + 1) % iters_per_epoch == 0:
                scheduler.step()

            if (i + 1) % args.eval_every == 0:
                metrics_history["iterations"].append(i + 1)
                metrics_history["train"].append(evaluate(
                    model, train_test_dataset, args, model_type=model_type,
                    cap=args.eval_subset, light=True,
                ))
                metrics_history["validation"].append(evaluate(
                    model, validation_dataset, args, model_type=model_type,
                    cap=args.eval_subset, light=True,
                ))
            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    else:
        loop = tqdm(range(epochs), unit=" epoch", desc=desc)
        it = 0
        for _ in loop:
            for inputs, target in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                loss = model._train_step(optimizer, inputs, target)
                losses.append(loss.item())
                it += 1
                if it % args.eval_every == 0:
                    metrics_history["iterations"].append(it)
                    metrics_history["train"].append(evaluate(
                        model, train_test_dataset, args, model_type=model_type,
                        cap=args.eval_subset, light=True,
                    ))
                    metrics_history["validation"].append(evaluate(
                        model, validation_dataset, args, model_type=model_type,
                        cap=args.eval_subset, light=True,
                    ))
            if scheduler is not None:
                scheduler.step()
            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    if model_type in ("vip", "ftip"):
        diagnostics = {
            "KLs": [float(v) for v in model.KLs],
            "bb_alphas": [float(v) for v in model.bb_alphas],
            "prior_regularizers": [float(v) for v in model.prior_regularizers],
        }
        if model_type == "ftip":
            diagnostics["base_KLs"] = [float(v) for v in model.base_KLs]
            diagnostics["flow_ldj"] = [float(v) for v in model.flow_ldj]
    else:
        diagnostics = {
            "KLs": [float(v) for v in getattr(model, "KLs", [])],
            "bb_alphas": [float(v) for v in getattr(model, "bb_alphas", [])],
        }

    return losses, metrics_history, diagnostics


def _ckpt_path(args, dataset_name, model_type):
    alpha_tag = f"_alpha{args.bb_alpha}"
    layer_tag = "_simpler" if args.layer_model == "SimplerBayesLinear" else "_bayes"
    return os.path.join(
        args.output_dir,
        f"{model_type}_{dataset_name}{alpha_tag}{layer_tag}_seed{args.seed}.pt",
    )


def _build_result(dataset_name, model_type, model, args, train_loader,
                  train_test_dataset, test_dataset, lr=None,
                  epochs=None, iterations=None, desc="Training"):
    if lr is None:
        lr = args.lr

    t0 = time.time()
    losses, metrics_history, diagnostics = train_with_metrics(
        model, train_loader, train_test_dataset, test_dataset, args,
        lr=lr, epochs=epochs, iterations=iterations,
        model_type=model_type, desc=desc,
    )
    train_time = time.time() - t0

    train_metrics = evaluate(
        model, train_test_dataset, args, model_type=model_type,
        cap=args.eval_subset,
    )
    test_metrics = evaluate(
        model, test_dataset, args, model_type=model_type,
        cap=args.eval_subset,
    )

    actual_epochs = epochs if epochs is not None else args.epochs
    actual_iterations = iterations if iterations is not None else args.iterations

    hyperparameters = {
        "lr": lr,
        "batch_size": args.batch_size,
        "iterations": actual_iterations,
        "epochs": actual_epochs,
        "cosine_annealing": args.cosine_annealing,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "layer_model": args.layer_model,
        "dropout": args.dropout,
        "weight_log_sigma_init": args.weight_log_sigma_init,
        "regression_coeffs": args.regression_coeffs,
        "bb_alpha": args.bb_alpha,
        "use_prior_regularizer": args.use_prior_regularizer,
        "regularizer_mode": args.regularizer_mode,
        "prior_regularizer_scaler": args.prior_regularizer_scaler,
        "eval_subset": args.eval_subset,
    }
    if model_type == "ftip":
        hyperparameters.update({
            "flow_type": args.flow_type,
            "flow_depth": args.flow_depth,
            "flow_num_bins": args.flow_num_bins,
            "flow_domain": args.flow_domain,
            "num_samples": args.num_samples,
            "eval_samples": args.eval_samples,
        })
    elif model_type == "fbnn":
        hyperparameters.update({
            "fbnn_prior": args.fbnn_prior,
            "fbnn_freeze_prior": args.fbnn_freeze_prior,
            "fbnn_gp_inner_dim": args.fbnn_gp_inner_dim,
            "fbnn_num_measurement": args.fbnn_num_measurement,
            "fbnn_num_context": args.fbnn_num_context,
            "fbnn_context_std": args.fbnn_context_std,
            "fbnn_lambda_kl": args.fbnn_lambda_kl,
            "fbnn_num_eval_samples": args.fbnn_num_eval_samples,
        })
    elif model_type == "tfsvi":
        hyperparameters.update({
            "tfsvi_sigma_prior": args.tfsvi_sigma_prior,
            "tfsvi_S_ctx": args.tfsvi_S_ctx,
            "tfsvi_K_ctx": args.tfsvi_K_ctx,
            "tfsvi_num_train_samples": args.tfsvi_num_train_samples,
            "tfsvi_num_eval_samples": args.tfsvi_num_eval_samples,
        })
    elif model_type == "mfvi":
        hyperparameters["mfvi_num_eval_samples"] = args.mfvi_num_eval_samples

    result = {
        "dataset": dataset_name,
        "model": model_type,
        "task": "binary",
        "hyperparameters": hyperparameters,
        "train_time_s": round(train_time, 2),
        "train": train_metrics,
        "test": test_metrics,
        "losses": losses,
        "metrics_history": metrics_history,
        "diagnostics": diagnostics,
    }

    for split, m in [("Train", train_metrics), ("Test", test_metrics)]:
        print(
            f"  {model_type.upper()} {split}: "
            f"Error={m['Error']:.4f}  NLL={m['NLL']:.4f}  "
            f"AUC={m['AUC']:.4f}  ECE={m['ECE']:.4f}"
        )
    print(f"  Time: {train_time:.1f}s")

    if args.save_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt = _ckpt_path(args, dataset_name, model_type)
        torch.save(model.state_dict(), ckpt)
        print(f"  Checkpoint: {ckpt}")

    return result, model


def run_single(dataset_name, args):
    use_warm_start = args.model == "ftip" and args.auto_warm_start

    header = "FTIP (warm-start from VIP)" if use_warm_start else args.model.upper()
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  Model: {header}  |  Task: binary")
    print(f"{'='*60}")

    dataset = get_dataset(dataset_name)
    train_dataset, train_test_dataset, test_dataset = dataset.get_split(0.1, args.seed)

    use_cuda = args.device and "cuda" in args.device
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_cuda, num_workers=0,
    )

    results = []

    if use_warm_start:
        vip_ep, vip_it = args.vip_epochs, args.vip_iterations
        ftip_ep, ftip_it = args.epochs, args.iterations
        if vip_it is not None and ftip_it is not None:
            phase_v = f"{vip_it} iters"
            phase_f = f"{ftip_it} iters"
        else:
            phase_v = f"{vip_ep} epochs"
            phase_f = f"{ftip_ep} epochs"

        print(f"\n  Phase 1: Training VIP for warm-start ({phase_v}, lr={args.vip_lr})")
        vip_model = build_model(args, train_dataset, model_type="vip")
        vip_ws_result, vip_model = _build_result(
            dataset_name, "vip", vip_model, args, train_loader,
            train_test_dataset, test_dataset, lr=args.vip_lr,
            epochs=vip_ep, iterations=vip_it,
            desc="VIP pre-training",
        )

        vip_ws_state = {n: p.data.clone() for n, p in vip_model.named_parameters()}

        print(f"\n  VIP baseline: continuing for {phase_f}")
        vip_baseline_result, _ = _build_result(
            dataset_name, "vip", vip_model, args, train_loader,
            train_test_dataset, test_dataset, lr=args.vip_lr,
            epochs=ftip_ep, iterations=ftip_it,
            desc="VIP baseline (continued)",
        )
        vip_baseline_result["train_time_s"] = round(
            vip_ws_result["train_time_s"] + vip_baseline_result["train_time_s"], 2
        )
        results.append(vip_baseline_result)

        for n, p in vip_model.named_parameters():
            p.data.copy_(vip_ws_state[n])
        del vip_ws_state

        print(f"\n  Phase 2: Fine-tuning FTIP ({phase_f}, lr={args.ftip_lr})")
        ftip_model = build_model(args, train_dataset, model_type="ftip")
        ftip_model.warm_start_from_vip(vip_model, learnable_affine=args.learnable_affine)
        del vip_model

        ftip_result, _ = _build_result(
            dataset_name, "ftip", ftip_model, args, train_loader,
            train_test_dataset, test_dataset, lr=args.ftip_lr,
            epochs=ftip_ep, iterations=ftip_it,
            desc="FTIP fine-tuning",
        )
        ftip_result["warm_start"] = {
            "enabled": True,
            "vip_epochs": vip_ep,
            "vip_iterations": vip_it,
            "vip_lr": args.vip_lr,
            "vip_checkpoint": _ckpt_path(args, dataset_name, "vip"),
            "learnable_affine": args.learnable_affine,
        }
        ftip_result["total_time_s"] = round(
            vip_ws_result["train_time_s"] + ftip_result["train_time_s"], 2
        )
        results.append(ftip_result)

        vip_err = vip_baseline_result["test"]["Error"]
        ftip_err = ftip_result["test"]["Error"]
        vip_nll = vip_baseline_result["test"]["NLL"]
        ftip_nll = ftip_result["test"]["NLL"]
        print(
            f"\n  Delta (FTIP - VIP baseline): "
            f"Error={ftip_err - vip_err:+.4f}  NLL={ftip_nll - vip_nll:+.4f}"
        )
        print(f"  FTIP total time: {ftip_result['total_time_s']:.1f}s")
    else:
        model = build_model(args, train_dataset)
        if args.warm_start_from and args.model == "ftip":
            device = torch.device(args.device)
            vip_model = build_model(args, train_dataset, model_type="vip")
            vip_model.load_state_dict(
                torch.load(args.warm_start_from, map_location=device, weights_only=True)
            )
            model.warm_start_from_vip(vip_model, learnable_affine=args.learnable_affine)
            del vip_model
            print(
                f"  Warm-started from {args.warm_start_from} "
                f"(affine learnable={args.learnable_affine})"
            )
        result, _ = _build_result(
            dataset_name, args.model, model, args, train_loader,
            train_test_dataset, test_dataset,
        )
        if args.warm_start_from:
            result["warm_start"] = {
                "enabled": True,
                "warm_start_from": args.warm_start_from,
                "learnable_affine": args.learnable_affine,
            }
        results.append(result)

    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.dataset == "all":
        datasets = BINARY_DATASETS
    else:
        datasets = [args.dataset]

    all_results = []
    for ds in datasets:
        alpha_tag = f"_alpha{args.bb_alpha}"
        layer_tag = "_simpler" if args.layer_model == "SimplerBayesLinear" else "_bayes"
        out_path = os.path.join(
            args.output_dir,
            f"binary_{args.model}_{ds}{alpha_tag}{layer_tag}_seed{args.seed}.json",
        )
        if os.path.exists(out_path):
            print(f"\n  Skipping {ds} seed={args.seed}: {out_path} already exists")
            with open(out_path) as f:
                all_results.extend(json.load(f))
            continue

        ds_results = run_single(ds, args)
        all_results.extend(ds_results)

        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(ds_results, f, indent=2)
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
