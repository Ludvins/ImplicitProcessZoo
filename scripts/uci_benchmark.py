"""UCI regression benchmark.

Runs the UCI regression datasets with VIP, FTIP, AP-FSVI, MFVI, FBNN,
TFSVI, or MAP.
Each run writes a JSON result file and, by default, a checkpoint.

Example:
    python -m scripts.uci_benchmark --model ftip --dataset boston
"""

import argparse
import copy
import json
import os
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset import get_dataset
from src.utils.metrics import MetricsRegression
from src.utils.utils import infinite_loader
from src.priors.generative_functions import BayesianNN, BayesLinear, GP
from src.flows import CouplingFlow, SplineCouplingFlow, SplineCoupling1x1Flow
from src.vip import VIP
from src.ftip import FTIP
from src.ap_fsvi import APFSVI
from src.fbnn import FBNN
from src.tfsvi import TFSVI
from src.mfvi import MFVI
from src.map_baseline import DeterministicMAP
from scripts.benchmark_utils import (
    add_wandb_args,
    finish_wandb_run,
    init_wandb_run,
    pretty_discrepancy_name,
    wandb_log_eval,
    wandb_log_result,
    wandb_log_train_step,
    wandb_run_name,
)

UCI_REGRESSION_DATASETS = [
    "boston", "energy", "concrete", "naval", "power",
    "protein", "kin8nm", "yatch", "winered",
]

# Default per-dataset training-iteration budgets, mirroring the FTIP
# cold-start BayesLinear / alpha=1.0 sweep recorded under
# results/uci/ftip_*_alpha1.0_bayes_*. Used only when the user does
# not pass --iterations or --epochs explicitly.
DEFAULT_UCI_ITERS = {
    "boston":   30_000,
    "concrete": 30_000,
    "energy":   30_000,
    "protein":  30_000,
    "kin8nm":   60_000,
    "naval":    60_000,
    "power":    60_000,
    "winered":  60_000,
    "yatch":    60_000,
}

ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}

LAYER_MODELS = {
    "BayesLinear": BayesLinear,
}

REGRESSION_MODELS = ["vip", "ftip", "fbnn", "tfsvi", "mfvi", "ap_fsvi", "map"]


def generate_ood_points(test_dataset, n_ood=None, seed=42):
    """Generate OOD points via data augmentation in normalized input space.

    Combines two strategies:
      - Perturbed test points with large Gaussian noise (std=3.0)
      - Uniform far-field samples from [-5, 5]^d
    """
    rng = np.random.RandomState(seed)
    inputs = test_dataset.inputs  # already normalized
    n_test, input_dim = inputs.shape

    if n_ood is None:
        n_ood = n_test

    n_perturbed = n_ood // 2
    n_uniform = n_ood - n_perturbed

    # Strategy 1: perturb test points with large noise
    idx = rng.choice(n_test, size=n_perturbed, replace=True)
    perturbed = inputs[idx] + rng.randn(n_perturbed, input_dim) * 3.0

    # Strategy 2: uniform far-field
    uniform = rng.uniform(-5.0, 5.0, size=(n_uniform, input_dim))

    return np.concatenate([perturbed, uniform], axis=0).astype(inputs.dtype)


def compute_predictive_entropy(mean_pred, std_pred, n_mc=1000, batch_size=512):
    """MC estimate of differential entropy for a Gaussian mixture.

    The predictive is p(y|x) = (1/S) sum_s N(y; mu_s, sigma_s^2).
    H[y|x] = -E_{y~p}[log p(y|x)] estimated via MC sampling.

    Processes points in batches along the N dimension to avoid OOM.

    Parameters
    ----------
    mean_pred : (S, N, D)
    std_pred  : (S, N, D)
    n_mc      : number of MC samples
    batch_size : int, points per batch along N

    Returns
    -------
    entropy : (N,) per-point differential entropy
    """
    std_pred = std_pred.clamp(min=1e-8)
    S, N, D = mean_pred.shape
    device, dtype = mean_pred.device, mean_pred.dtype

    log_S = torch.log(torch.tensor(S, dtype=dtype, device=device))
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=dtype, device=device))

    all_entropy = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        mu_b = mean_pred[:, start:end, :]   # (S, B, D)
        std_b = std_pred[:, start:end, :]   # (S, B, D)
        B = end - start

        # Draw MC samples from the mixture
        comp_idx = torch.randint(S, (n_mc, B), device=device)
        n_idx = torch.arange(B, device=device).unsqueeze(0).expand(n_mc, B)
        mu_chosen = mu_b[comp_idx, n_idx, :]
        std_chosen = std_b[comp_idx, n_idx, :]
        y_samples = mu_chosen + std_chosen * torch.randn_like(mu_chosen)  # (n_mc, B, D)

        # Evaluate log p(y|x) under the full mixture
        y_exp = y_samples.unsqueeze(1)      # (n_mc, 1, B, D)
        mu_exp = mu_b.unsqueeze(0)          # (1, S, B, D)
        std_exp = std_b.unsqueeze(0)        # (1, S, B, D)
        var_exp = std_exp ** 2

        log_comp = -0.5 * (log2pi + var_exp.log() + (y_exp - mu_exp) ** 2 / var_exp)
        log_comp = log_comp.sum(-1)         # (n_mc, S, B)

        log_mix = torch.logsumexp(log_comp, dim=1) - log_S  # (n_mc, B)
        all_entropy.append(-log_mix.mean(dim=0))             # (B,)

    return torch.cat(all_entropy, dim=0)


def _batched_entropy(model, x, model_type, eval_samples, a=None, batch_size=2048):
    """Compute per-point predictive entropy without materializing full (S, N, D)."""
    all_entropy = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size]
        if model_type == "ftip" and a is not None:
            mean, std = model.forward_with_coefficients(xb, a)
            std = std.unsqueeze(-1).expand_as(mean)
        elif model_type == "vip":
            mean, std = model(xb)
        elif model_type == "fbnn":
            mean, std = _fbnn_pred_components(model, xb)
        elif model_type == "tfsvi":
            mean, std = _tfsvi_pred_components(model, xb, eval_samples)
        elif model_type == "mfvi":
            mean, std = _tfsvi_pred_components(model, xb, eval_samples)
        else:
            if hasattr(model, "num_samples"):
                old_ns = model.num_samples
                model.num_samples = eval_samples
                mean, std = model(xb)
                model.num_samples = old_ns
            else:
                mean, std = model(xb)
            if std.dim() == mean.dim() - 1:
                std = std.unsqueeze(-1).expand_as(mean)
        all_entropy.append(compute_predictive_entropy(mean, std))
    return torch.cat(all_entropy, dim=0)


def evaluate_ood(model, test_dataset, args, model_type=None, seed=42):
    """Evaluate OOD detection via predictive entropy AUROC."""
    from sklearn.metrics import roc_auc_score

    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Generate OOD points
    ood_inputs = generate_ood_points(test_dataset, seed=seed)

    x_id = torch.tensor(test_dataset.inputs, dtype=dtype, device=device)
    x_ood = torch.tensor(ood_inputs, dtype=dtype, device=device)

    model.eval()
    with torch.no_grad():
        # For FTIP: sample flow coefficients once (data-independent)
        a = model.sample_flow_coefficients(args.eval_samples) if model_type == "ftip" else None
        entropy_id = _batched_entropy(model, x_id, model_type, args.eval_samples, a)
        entropy_ood = _batched_entropy(model, x_ood, model_type, args.eval_samples, a)

    # AUROC: label 0 = in-distribution, 1 = OOD, score = entropy
    labels = np.concatenate([np.zeros(len(entropy_id)), np.ones(len(entropy_ood))])
    scores = np.concatenate([entropy_id.cpu().numpy(), entropy_ood.cpu().numpy()])
    auroc = roc_auc_score(labels, scores)

    return {
        "AUROC": float(auroc),
        "entropy_id_mean": float(entropy_id.mean().cpu()),
        "entropy_id_std": float(entropy_id.std().cpu()),
        "entropy_ood_mean": float(entropy_ood.mean().cpu()),
        "entropy_ood_std": float(entropy_ood.std().cpu()),
        "n_id": int(len(entropy_id)),
        "n_ood": int(len(entropy_ood)),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="UCI regression benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Experiment ---
    p.add_argument("--model", type=str, required=True,
                    choices=REGRESSION_MODELS + ["all"],
                    help="Model to train.")
    p.add_argument("--dataset", type=str, required=True,
                    help="Dataset name or 'all' for all UCI regression datasets.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--test_size", type=float, default=0.1,
                    help="Fraction of data used for testing.")
    p.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "float64"], help="Tensor dtype.")
    p.add_argument("--device", type=str, default=None,
                    help="Torch device (default: cuda if available, else cpu).")
    p.add_argument("--test_ood", action="store_true", default=False,
                    help="Evaluate OOD detection using predictive entropy as score.")
    p.add_argument("--output_dir", type=str, default="results",
                    help="Directory to save result JSON files.")

    # --- BayesianNN (generative function / prior) ---
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[10, 10],
                    help="Hidden layer widths for the BayesianNN prior.")
    p.add_argument("--activation", type=str, default="tanh",
                    choices=list(ACTIVATIONS.keys()),
                    help="Activation function for the BayesianNN.")
    p.add_argument("--layer_model", type=str, default="BayesLinear",
                    choices=list(LAYER_MODELS.keys()),
                    help="Bayesian layer type. Benchmarks use full BayesLinear.")
    p.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate in the BayesianNN.")

    # --- Model (shared VIP / FTIP) ---
    p.add_argument("--regression_coeffs", type=int, default=20,
                    help="Number of regression coefficients (S).")
    p.add_argument("--bb_alpha", type=float, default=None,
                    help="BB-alpha parameter (0 = ELBO, 1 = BB-alpha energy). "
                         "If unset: 1.0 globally, but 0.5 for --model mfvi.")
    p.add_argument("--use_prior_regularizer", action="store_true", default=False,
                    help="Enable the method's optional prior regularizer.")
    p.add_argument("--no_prior_regularizer", action="store_true",
                    help="Disable prior regularizer.")
    p.add_argument("--regularizer_mode", type=str, default="evidence",
                    choices=["evidence", "KL"],
                    help="Prior regularizer mode.")
    p.add_argument("--prior_regularizer_scaler", type=float, default=1.0,
                    help="Prior regularizer scaling factor.")

    # --- FTIP-specific ---
    p.add_argument("--flow_type", type=str, default="spline_1x1",
                    choices=["affine", "spline", "spline_1x1"],
                    help="FTIP flow class. 'affine' = original CouplingFlow "
                         "(affine coupling), 'spline' = SplineCouplingFlow "
                         "(RQ spline coupling), 'spline_1x1' = "
                         "SplineCoupling1x1Flow (spline coupling + Glow 1x1 LU "
                         "mixing, default).")
    p.add_argument("--flow_num_bins", type=int, default=8,
                    help="Bins per RQ-spline coupling layer (ignored if "
                         "flow_type=affine).")
    p.add_argument("--flow_domain", type=float, default=3.0,
                    help="Spline domain half-width B (ignored if "
                         "flow_type=affine).")
    p.add_argument("--flow_depth", type=int, default=2,
                    help="Number of coupling layers in the normalizing flow (FTIP only).")
    p.add_argument("--num_samples", type=int, default=200,
                    help="Number of MC posterior samples (FTIP only).")
    p.add_argument("--eval_samples", type=int, default=1000,
                    help="Number of MC samples used at evaluation time (FTIP only).")
    p.add_argument("--warm_start_from", type=str, default=None,
                    help="Path to a VIP checkpoint (.pt) for warm-starting FTIP.")
    p.add_argument("--learnable_affine", action="store_true", default=True,
                    help="Make the affine warm-start layer trainable.")
    p.add_argument("--no_learnable_affine", action="store_true",
                    help="Fix the affine warm-start layer (not trainable).")


    # --- FBNN-specific ---
    p.add_argument("--fbnn_prior", type=str, default="gp",
                    choices=["gp", "bnn"],
                    help="fBNN prior family: 'gp' (RFF GP, paper default) "
                         "or 'bnn' (Bayesian NN with SSGE prior score).")
    p.add_argument("--fbnn_freeze_prior", action="store_true", default=False,
                    help="Freeze the prior's parameters. By default the GP "
                         "kernel hyperparameters are LEARNED jointly, "
                         "matching Sun et al. 2019.")
    p.add_argument("--fbnn_gp_inner_dim", type=int, default=10,
                    help="Inner-layer dim of the RFF GP prior (number of "
                         "random features).")
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

    # --- TFSVI-specific (Rudner et al., 2022) ---
    p.add_argument("--tfsvi_sigma_prior", type=float, default=1.0,
                    help="Prior std for the parameter Gaussian "
                         "p(theta) = N(0, sigma_prior^2 I).")
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
                    help="MC weight samples used at MFVI evaluation time. "
                         "Training uses --regression_coeffs as the per-step "
                         "MC count (matches the other methods in this script).")

    # --- AP-FSVI-specific ---
    p.add_argument("--ap_fsvi_prior", type=str, choices=["gp", "bnn"], default="bnn",
                    help="AP-FSVI function prior: exact RBF GP or frozen BNN prior.")
    p.add_argument("--ap_fsvi_weight_log_sigma_init", type=float, default=None,
                    help="Initial posterior weight log sigma for AP-FSVI.")
    p.add_argument("--ap_fsvi_num_samples", type=int, default=32,
                    help="Posterior function samples per AP-FSVI step.")
    p.add_argument("--ap_fsvi_num_prior_samples", type=int, default=64,
                    help="Prior samples per AP-FSVI regularizer step.")
    p.add_argument("--ap_fsvi_num_eval_samples", type=int, default=200,
                    help="Posterior samples used at AP-FSVI evaluation time.")
    p.add_argument("--ap_fsvi_num_measurement", type=int, default=64,
                    help="Measurement points for the AP-FSVI function regularizer.")
    p.add_argument("--ap_fsvi_adaptive_measure_points", action="store_true", default=False,
                    help="Adapt AP-FSVI measurement points.")
    p.add_argument("--ap_fsvi_adaptive_measure_mode", type=str, default="gradient",
                    choices=["gradient", "candidate", "candidate_then_one_step"],
                    help="Adaptive measurement point strategy.")
    p.add_argument("--ap_fsvi_adaptive_measure_every", type=int, default=1,
                    help="Adapt AP-FSVI measurement points every N optimizer steps.")
    p.add_argument("--ap_fsvi_adaptive_candidate_pool_multiplier", type=int, default=4,
                    help="Candidate pool size multiplier for adaptive AP-FSVI.")
    p.add_argument("--ap_fsvi_adaptive_candidate_pool_size", type=int, default=None,
                    help="Fixed adaptive candidate pool size.")
    p.add_argument("--ap_fsvi_adaptive_num_samples", type=int, default=None,
                    help="Posterior samples used for adaptive candidate scoring.")
    p.add_argument("--ap_fsvi_adaptive_num_prior_samples", type=int, default=None,
                    help="Prior samples used for adaptive candidate scoring.")
    p.add_argument("--ap_fsvi_adaptive_num_projections", type=int, default=None,
                    help="Projection count used for adaptive candidate scoring.")
    p.add_argument("--ap_fsvi_adaptive_measure_steps", type=int, default=3,
                    help="Gradient ascent steps for adaptive measurement points.")
    p.add_argument("--ap_fsvi_adaptive_measure_lr", type=float, default=0.05,
                    help="Gradient ascent learning rate for adaptive measurement points.")
    p.add_argument("--ap_fsvi_adaptive_measure_domain_limit", type=float, default=None,
                    help="Optional clamp radius for adaptive measurement points.")
    p.add_argument("--ap_fsvi_beta", type=float, default=1.0,
                    help="Weight on AP-FSVI function-space regularizer.")
    p.add_argument("--ap_fsvi_beta_start", type=float, default=0.0,
                    help="Initial beta for AP-FSVI warmup.")
    p.add_argument("--ap_fsvi_beta_warmup_steps", type=int, default=5000,
                    help="Linear AP-FSVI beta warmup steps.")
    p.add_argument("--ap_fsvi_data_pretrain_steps", type=int, default=1000,
                    help="Steps with beta forced to zero.")
    p.add_argument("--ap_fsvi_data_loss", choices=["expected_nll", "predictive_nll"],
                    default="expected_nll",
                    help="AP-FSVI data term estimator.")
    p.add_argument("--ap_fsvi_discrepancy", type=str, default="sample_sliced_kl",
                    choices=[
                        "mmd",
                        "energy",
                        "sliced_wasserstein",
                        "stein",
                        "sinkhorn",
                        "prior_whitened_gaussian_kl",
                        "prior_whitened_sliced_kl",
                        "spectral_sliced_kl",
                        "spectral_projected_kl",
                        "sample_sliced_kl",
                        "sample_sliced_knn_kl",
                        "sample_sliced_gaussian_kl",
                        "sample_sliced_quantile_transport_kl",
                    ],
                    help="AP-FSVI function-space discrepancy.")
    p.add_argument("--ap_fsvi_discrepancy_projections", type=int, default=128,
                    help="Projection count for sliced AP-FSVI discrepancies.")
    p.add_argument("--ap_fsvi_sample_projection_mode", type=str, default="random",
                    choices=["random", "fixed_random", "prior_pca", "discrepancy_pca", "fixed_orthogonal"],
                    help="Projection rule for AP-FSVI sample_sliced_kl.")
    p.add_argument("--ap_fsvi_sample_knn_k", type=int, default=3,
                    help="Neighbor count for AP-FSVI sample_sliced_knn_kl.")
    p.add_argument("--ap_fsvi_quantile_transport_k", type=int, default=3,
                    help="Local spacing window for AP-FSVI sliced quantile-transport KL.")
    p.add_argument("--ap_fsvi_sinkhorn_epsilon", type=float, default=1.0,
                    help="AP-FSVI Sinkhorn entropy regularization.")
    p.add_argument("--ap_fsvi_sinkhorn_iterations", type=int, default=50,
                    help="AP-FSVI Sinkhorn iterations.")
    p.add_argument("--ap_fsvi_log_variance_init", type=float, default=-5.0,
                    help="Initial log observation variance for AP-FSVI.")
    p.add_argument("--ap_fsvi_measurement_weights", type=float, nargs=3,
                    default=[0.2, 0.2, 0.6],
                    metavar=("DATA", "NEAR", "DOMAIN"),
                    help="Data/near-data/domain AP-FSVI measurement weights.")
    p.add_argument("--ap_fsvi_near_data_noise", type=float, default=0.1,
                    help="Std of near-data AP-FSVI measurement perturbations.")
    p.add_argument("--ap_fsvi_domain_std", type=float, default=2.5,
                    help="Std for AP-FSVI Gaussian domain measurement points.")
    p.add_argument("--ap_fsvi_max_grad_norm", type=float, default=None,
                    help="Optional AP-FSVI gradient clipping norm.")

    # --- MAP-specific ---
    p.add_argument("--map_l2", type=float, default=1e-4,
                    help="L2 weight penalty for deterministic MAP baseline.")
    p.add_argument("--map_log_variance_init", type=float, default=-5.0,
                    help="Initial log observation variance for MAP baseline.")

    # --- Auto warm-start (VIP -> FTIP pipeline) ---
    p.add_argument("--auto_warm_start", action="store_true", default=True,
                    help="Automatically train VIP first, then warm-start FTIP (FTIP only).")
    p.add_argument("--no_auto_warm_start", action="store_true",
                    help="Disable auto warm-start; train FTIP from scratch.")
    p.add_argument("--vip_epochs", type=int, default=None,
                    help="Epochs for VIP pre-training phase (auto warm-start only).")
    p.add_argument("--vip_iterations", type=int, default=None,
                    help="Iterations for VIP pre-training phase (auto warm-start only).")
    p.add_argument("--vip_lr", type=float, default=1e-3,
                    help="Learning rate for VIP pre-training phase.")
    p.add_argument("--ftip_lr", type=float, default=1e-4,
                    help="Learning rate for FTIP fine-tuning phase (auto warm-start only). "
                         "1e-4 lets the flow escape the VIP warm-start init; smaller values "
                         "(e.g. 1e-5) leave the spline coupling layers frozen at the affine "
                         "VIP posterior.")

    # --- Checkpointing ---
    p.add_argument("--save_checkpoint", action="store_true", default=True,
                    help="Save model checkpoint after training.")
    p.add_argument("--no_save_checkpoint", action="store_true",
                    help="Disable saving model checkpoint.")

    # --- Training ---
    p.add_argument("--batch_size", type=int, default=100,
                    help="Training batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--iterations", type=int, default=None,
                    help="Number of training iterations (mutually exclusive with --epochs).")
    p.add_argument("--epochs", type=int, default=None,
                    help="Number of training epochs (mutually exclusive with --iterations).")
    p.add_argument("--eval_every", type=int, default=1000,
                    help="Compute light metrics on train/test every N iterations.")
    p.add_argument("--cosine_annealing", action="store_true", default=True,
                    help="Use cosine annealing LR schedule.")
    p.add_argument("--no_cosine_annealing", action="store_true",
                    help="Disable cosine annealing.")
    p.add_argument("--compile", action="store_true", default=False,
                    help="Use torch.compile for faster training (requires Triton).")
    add_wandb_args(p)
    args = p.parse_args()

    # Resolve mutually-exclusive flags
    if args.no_prior_regularizer:
        args.use_prior_regularizer = False
    if args.no_cosine_annealing:
        args.cosine_annealing = False
    if args.no_save_checkpoint:
        args.save_checkpoint = False
    if args.no_auto_warm_start:
        args.auto_warm_start = False
    if args.no_learnable_affine:
        args.learnable_affine = False

    # Disable auto_warm_start if an explicit checkpoint path is given
    if args.warm_start_from:
        args.auto_warm_start = False

    # Track whether the user explicitly chose a training budget; if not,
    # main() picks per-dataset iters from DEFAULT_UCI_ITERS so the script
    # mirrors the FTIP cold-start sweep by default (30k for the small
    # group, 60k for the rest).
    args._iters_user_supplied = (
        args.iterations is not None or args.epochs is not None
    )
    # Detect whether the user passed --bb_alpha so MFVI can use its
    # benchmark default without overriding an explicit choice.
    args._bb_alpha_user_supplied = args.bb_alpha is not None
    if args.bb_alpha is None:
        args.bb_alpha = 1.0

    # Default VIP pre-training length: same as main training length
    if args.vip_epochs is None and args.vip_iterations is None:
        args.vip_epochs = args.epochs
        args.vip_iterations = args.iterations

    if args.ap_fsvi_weight_log_sigma_init is None:
        args.ap_fsvi_weight_log_sigma_init = 0.0
    if args.ap_fsvi_num_samples is None:
        args.ap_fsvi_num_samples = args.regression_coeffs

    # Default device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def _set_bnn_num_samples(bnn, S):
    """Mutate a BayesianNN's ``num_samples`` and regenerate cached noise.

    Mirrors :meth:`FBNN._set_num_samples` but works on any BayesianNN —
    used to rebrand the shared ``gen_fn`` for TFSVI (which needs S=1 so
    the BNN forward returns ``[1, N, D]``).
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


def _set_bnn_fix_random_noise(bnn, fix):
    """Mutate ``fix_random_noise`` on a BayesianNN (and all its layers).

    Used to repurpose the shared ``gen_fn`` for MFVI, which needs
    ``fix_random_noise=False`` so each forward draws fresh weight noise
    (the variational posterior is over (weight_mu, weight_log_sigma) and
    the gradient signal must average over fresh ε ~ N(0, I) per step).
    """
    if hasattr(bnn, "fix_random_noise"):
        bnn.fix_random_noise = fix
    if not hasattr(bnn, "layers"):
        return
    for layer in bnn.layers:
        if hasattr(layer, "fix_random_noise"):
            layer.fix_random_noise = fix


def build_model(args, train_dataset, model_type=None):
    """Build a UCI regression model.

    Parameters
    ----------
    model_type : str or None
        Override args.model (useful for building VIP during auto warm-start).
    """
    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    def _arg(name, default=None):
        return getattr(args, name, default)

    if model_type == "map":
        return DeterministicMAP(
            input_dim=train_dataset.input_dim,
            output_dim=train_dataset.output_dim,
            structure=args.hidden_dims,
            activation=ACTIVATIONS[args.activation],
            num_data=len(train_dataset),
            l2=_arg("map_l2", 1e-4),
            y_mean=train_dataset.targets_mean,
            y_std=train_dataset.targets_std,
            log_variance_init=_arg("map_log_variance_init", -5.0),
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    if model_type == "ap_fsvi":
        ap_samples = _arg("ap_fsvi_num_samples", 32)
        prior_samples = _arg("ap_fsvi_num_prior_samples", 64)
        gen_fn = BayesianNN(
            input_dim=train_dataset.input_dim,
            num_samples=ap_samples,
            structure=args.hidden_dims,
            activation=ACTIVATIONS[args.activation],
            output_dim=train_dataset.output_dim,
            layer_model=BayesLinear,
            dropout=args.dropout,
            fix_random_noise=False,
            weight_log_sigma_init=_arg("ap_fsvi_weight_log_sigma_init", 0.0),
            device=device,
            seed=args.seed,
            dtype=dtype,
        )
        prior_fn = None
        if _arg("ap_fsvi_prior", "bnn") == "bnn":
            prior_fn = BayesianNN(
                input_dim=train_dataset.input_dim,
                num_samples=prior_samples,
                structure=args.hidden_dims,
                activation=ACTIVATIONS[args.activation],
                output_dim=train_dataset.output_dim,
                layer_model=BayesLinear,
                dropout=args.dropout,
                fix_random_noise=False,
                zero_mean_prior=True,
                weight_log_sigma_init=0.0,
                device=device,
                seed=args.seed + 1,
                dtype=dtype,
            )
        return APFSVI(
            generative_function=gen_fn,
            prior_function=prior_fn,
            input_dim=train_dataset.input_dim,
            output_dim=train_dataset.output_dim,
            likelihood="regression",
            num_data=len(train_dataset),
            num_samples=ap_samples,
            num_prior_samples=prior_samples,
            num_measurement=_arg("ap_fsvi_num_measurement", 64),
            adaptive_measure_points=_arg("ap_fsvi_adaptive_measure_points", False),
            adaptive_measure_mode=_arg("ap_fsvi_adaptive_measure_mode", "gradient"),
            adaptive_measure_every=_arg("ap_fsvi_adaptive_measure_every", 1),
            adaptive_candidate_pool_multiplier=_arg(
                "ap_fsvi_adaptive_candidate_pool_multiplier", 4
            ),
            adaptive_candidate_pool_size=_arg("ap_fsvi_adaptive_candidate_pool_size", None),
            adaptive_num_samples=_arg("ap_fsvi_adaptive_num_samples", None),
            adaptive_num_prior_samples=_arg("ap_fsvi_adaptive_num_prior_samples", None),
            adaptive_num_projections=_arg("ap_fsvi_adaptive_num_projections", None),
            adaptive_measure_steps=_arg("ap_fsvi_adaptive_measure_steps", 3),
            adaptive_measure_lr=_arg("ap_fsvi_adaptive_measure_lr", 0.05),
            adaptive_measure_domain_limit=_arg(
                "ap_fsvi_adaptive_measure_domain_limit", None
            ),
            beta=_arg("ap_fsvi_beta", 1.0),
            beta_start=_arg("ap_fsvi_beta_start", 0.0),
            beta_warmup_steps=_arg("ap_fsvi_beta_warmup_steps", 5000),
            data_pretrain_steps=_arg("ap_fsvi_data_pretrain_steps", 1000),
            data_loss=_arg("ap_fsvi_data_loss", "expected_nll"),
            measurement_weights=_arg("ap_fsvi_measurement_weights", [0.2, 0.2, 0.6]),
            near_data_noise=_arg("ap_fsvi_near_data_noise", 0.1),
            domain_std=_arg("ap_fsvi_domain_std", 2.5),
            function_discrepancy=_arg("ap_fsvi_discrepancy", "sample_sliced_kl"),
            discrepancy_num_projections=_arg("ap_fsvi_discrepancy_projections", 128),
            sample_projection_mode=_arg("ap_fsvi_sample_projection_mode", "random"),
            sample_knn_k=_arg("ap_fsvi_sample_knn_k", 3),
            quantile_transport_k=_arg("ap_fsvi_quantile_transport_k", 3),
            sinkhorn_epsilon=_arg("ap_fsvi_sinkhorn_epsilon", 1.0),
            sinkhorn_iterations=_arg("ap_fsvi_sinkhorn_iterations", 50),
            log_variance_init=_arg("ap_fsvi_log_variance_init", -5.0),
            max_grad_norm=_arg("ap_fsvi_max_grad_norm", None),
            y_mean=train_dataset.targets_mean,
            y_std=train_dataset.targets_std,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    gen_fn = BayesianNN(
        input_dim=train_dataset.input_dim,
        num_samples=args.regression_coeffs,
        structure=args.hidden_dims,
        activation=ACTIVATIONS[args.activation],
        output_dim=train_dataset.output_dim,
        layer_model=BayesLinear,
        dropout=args.dropout,
        device=device,
        seed=args.seed,
        dtype=dtype,
    )

    common = dict(
        generative_function=gen_fn,
            num_regression_coeffs=_arg("regression_coeffs", 20),
        output_dim=train_dataset.output_dim,
        likelihood="regression",
        num_data=len(train_dataset),
        bb_alpha=args.bb_alpha,
        use_prior_regularizer=args.use_prior_regularizer,
        regularizer_mode=args.regularizer_mode,
        prior_regularizer_scaler=args.prior_regularizer_scaler,
        y_mean=train_dataset.targets_mean,
        y_std=train_dataset.targets_std,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    if model_type == "mfvi":
        # Mean-field VI in weight space.  Reuses the shared ``gen_fn`` —
        # same architecture / layer_model as VIP/FTIP/FBNN/TFSVI — but
        # mutated to ``fix_random_noise=False`` so each forward draws
        # fresh ε ~ N(0, I) (required for an unbiased ELBO gradient over
        # q(theta) = N(weight_mu, exp(2 weight_log_sigma) I)).
        _set_bnn_fix_random_noise(gen_fn, False)
        return MFVI(
            generative_function=gen_fn,
            output_dim=train_dataset.output_dim,
            likelihood="regression",
            num_data=len(train_dataset),
            num_samples=args.regression_coeffs,
            bb_alpha=args.bb_alpha,
            y_mean=train_dataset.targets_mean,
            y_std=train_dataset.targets_std,
            device=device,
            dtype=dtype,
        )

    if model_type == "tfsvi":
        # Function-space VI (Rudner et al., 2022). Reuses the shared
        # ``gen_fn`` as the architecture template so TFSVI's q(theta)
        # mirrors the BNN's full BayesLinear parameter structure. This keeps the structure-of-q
        # consistent with the MFVI/VIP/FTIP/FBNN baselines. ``num_samples``
        # is set to 1 (and cached noise regenerated) so the BNN forward
        # returns ``[1, N, D]`` and is deterministic in theta — required
        # for TFSVI's linearised KL.
        _set_bnn_num_samples(gen_fn, 1)
        return TFSVI(
            input_dim=train_dataset.input_dim,
            output_dim=train_dataset.output_dim,
            structure=args.hidden_dims,
            activation=ACTIVATIONS[args.activation],
            likelihood="regression",
            num_data=len(train_dataset),
            sigma_prior=args.tfsvi_sigma_prior,
            num_samples=args.tfsvi_num_train_samples,
            bb_alpha=args.bb_alpha,
            S_ctx=args.tfsvi_S_ctx,
            K_ctx=args.tfsvi_K_ctx,
            y_mean=train_dataset.targets_mean,
            y_std=train_dataset.targets_std,
            generative_function=gen_fn,
            device=device,
            dtype=dtype,
        )

    if model_type == "fbnn":
        # Reuses the shared ``gen_fn`` as the variational posterior BNN
        # (same architecture as VIP/FTIP). FBNN.``_set_num_samples`` will
        # mutate ``gen_fn.num_samples`` and regenerate cached noise on
        # every train/eval pass anyway, so the construction-time value
        # is irrelevant.
        if args.fbnn_prior == "gp":
            prior = GP(
                input_dim=train_dataset.input_dim,
                output_dim=train_dataset.output_dim,
                inner_layer_dim=args.fbnn_gp_inner_dim,
                seed=args.seed,
                device=device,
                dtype=dtype,
            )
        else:
            # BNN prior: same architecture, different seed (frozen anyway).
            prior = BayesianNN(
                input_dim=train_dataset.input_dim,
                num_samples=args.regression_coeffs,
                structure=args.hidden_dims,
                activation=ACTIVATIONS[args.activation],
                output_dim=train_dataset.output_dim,
                layer_model=BayesLinear,
                dropout=args.dropout,
                fix_random_noise=True,
                device=device,
                seed=args.seed + 1,
                dtype=dtype,
            )
        return FBNN(
            generative_function=gen_fn,
            prior_function=prior,
            output_dim=train_dataset.output_dim,
            likelihood="regression",
            num_data=len(train_dataset),
            num_samples=args.regression_coeffs,
            num_measurement=args.fbnn_num_measurement,
            num_context=args.fbnn_num_context,
            context_std=args.fbnn_context_std,
            bb_alpha=args.bb_alpha,
            lambda_kl=args.fbnn_lambda_kl,
            y_mean=train_dataset.targets_mean,
            y_std=train_dataset.targets_std,
            freeze_prior=args.fbnn_freeze_prior,
            device=device,
            dtype=dtype,
        )

    if model_type == "vip":
        model = VIP(**common)
    else:
        flow = _build_flow(
            args,
            input_dim=args.regression_coeffs,
            device=device, dtype=dtype,
        )
        model = FTIP(**common, flow=flow, num_samples=args.num_samples)

    return model


def _build_flow(args, input_dim, device, dtype):
    """Construct an FTIP flow based on ``args.flow_type``."""
    common = dict(depth=args.flow_depth, input_dim=input_dim,
                  device=device, dtype=dtype, seed=args.seed)
    if args.flow_type == "affine":
        return CouplingFlow(**common)
    if args.flow_type == "spline":
        return SplineCouplingFlow(**common, num_bins=args.flow_num_bins,
                                  B=args.flow_domain)
    if args.flow_type == "spline_1x1":
        return SplineCoupling1x1Flow(**common, num_bins=args.flow_num_bins,
                                     B=args.flow_domain)
    raise ValueError(f"Unknown flow_type: {args.flow_type!r}")



def _fbnn_pred_components(model, xb, S=None):
    """Return per-sample (mean, std) for FBNN on the denormalized scale.

    Each posterior function sample defines a Gaussian
        N(f_s(x) * y_std + y_mean,  exp(log_var) * y_std^2)
    on the original target scale. MetricsRegression mixes these as
    Gaussian-mixture components for RMSE/NLL/CRPS/CQM.
    """
    if S is None:
        S = model.num_samples
    F = model.predict_f_samples(xb, S=S)                   # (S, N, D)
    mean = F * model.y_std + model.y_mean
    sigma = torch.sqrt(torch.exp(model.log_variance)) * model.y_std
    std = sigma.expand_as(mean)
    return mean, std


def _tfsvi_pred_components(model, xb, S):
    """Same Gaussian-mixture form as FBNN: each parameter sample defines
    N(f_s(x)*y_std + y_mean, exp(log_var)*y_std^2) on the original scale.
    """
    F = model.predict_f_samples(xb, S=S)                   # (S, N, D)
    mean = F * model.y_std + model.y_mean
    sigma = torch.sqrt(torch.exp(model.log_variance)) * model.y_std
    std = sigma.expand_as(mean)
    return mean, std


def evaluate(model, dataset, args, model_type=None, batch_size=None):
    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    x = torch.tensor(dataset.inputs, dtype=dtype, device=device)
    y = torch.tensor(dataset.targets, dtype=dtype, device=device)

    # CRPS subsamples to max 100 samples, so S'=min(S,100) for memory calc
    if batch_size is None:
        if model_type == "ftip":
            S = args.eval_samples
        elif model_type == "fbnn":
            S = args.fbnn_num_eval_samples
        elif model_type == "tfsvi":
            S = args.tfsvi_num_eval_samples
        elif model_type == "mfvi":
            S = args.mfvi_num_eval_samples
        else:
            S = args.regression_coeffs
        S_crps = min(S, 100)
        batch_size = max(16, min(512, 256_000_000 // max(1, S_crps * S_crps)))

    metrics = MetricsRegression(num_data=len(dataset), device=device)
    model.eval()
    # Switch FBNN to eval-time S (and back at the end)
    fbnn_old_S = None
    if model_type == "fbnn":
        fbnn_old_S = model.num_samples
        model._set_num_samples(args.fbnn_num_eval_samples)
        model.num_samples = args.fbnn_num_eval_samples
    with torch.no_grad():
        # For FTIP: sample flow coefficients once (data-independent)
        a = model.sample_flow_coefficients(args.eval_samples) if model_type == "ftip" else None
        for i in range(0, x.shape[0], batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            if model_type == "ftip":
                mean, std = model.forward_with_coefficients(xb, a)
                std = std.unsqueeze(-1).expand_as(mean)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False)
            elif model_type == "fbnn":
                mean, std = _fbnn_pred_components(model, xb)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False)
            elif model_type == "tfsvi":
                mean, std = _tfsvi_pred_components(model, xb, args.tfsvi_num_eval_samples)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False)
            elif model_type == "mfvi":
                mean, std = _tfsvi_pred_components(model, xb, args.mfvi_num_eval_samples)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False)
            else:
                mean, std = model(xb)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False)
    if fbnn_old_S is not None:
        model._set_num_samples(fbnn_old_S)
        model.num_samples = fbnn_old_S
    return metrics.get_dict()


def evaluate_prior(model, dataset, args, num_prior_samples=200, batch_size=2048):
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    x = torch.tensor(dataset.inputs, dtype=dtype, device=device)

    model.eval()
    all_var = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            prior_samples = model.forward_prior(xb, num_prior_samples)
            all_var.append(prior_samples.var(dim=0))
    per_input_var = torch.cat(all_var, dim=0)

    return {
        "var_mean": float(per_input_var.mean().cpu().numpy()),
        "var_var": float(per_input_var.var().cpu().numpy()),
    }


def evaluate_light(model, dataset, args, model_type=None, batch_size=2048):
    """Compute light metrics (RMSE, NLL) without CRPS/CQM for speed."""
    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    x = torch.tensor(dataset.inputs, dtype=dtype, device=device)
    y = torch.tensor(dataset.targets, dtype=dtype, device=device)

    metrics = MetricsRegression(num_data=len(dataset), device=device)
    model.eval()
    with torch.no_grad():
        # For FTIP: sample flow coefficients once (data-independent)
        a = model.sample_flow_coefficients(args.eval_samples) if model_type == "ftip" else None
        for i in range(0, x.shape[0], batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            if model_type == "ftip":
                mean, std = model.forward_with_coefficients(xb, a)
                std = std.unsqueeze(-1).expand_as(mean)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
            elif model_type == "fbnn":
                mean, std = _fbnn_pred_components(model, xb)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
            elif model_type == "tfsvi":
                # Light-eval: cap S for speed but still cover the predictive.
                S_light = min(64, args.tfsvi_num_eval_samples)
                mean, std = _tfsvi_pred_components(model, xb, S_light)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
            elif model_type == "mfvi":
                S_light = min(64, args.mfvi_num_eval_samples)
                mean, std = _tfsvi_pred_components(model, xb, S_light)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
            else:
                mean, std = model(xb)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
    d = metrics.get_dict()
    model.train()
    return {"RMSE": d["RMSE"], "NLL": d["NLL"]}


def train_with_metrics(model, train_loader, train_test_dataset, validation_dataset, args,
                       lr=None, epochs=None, iterations=None, model_type=None, desc="Training"):
    """Custom training loop that periodically evaluates light metrics.

    Parameters
    ----------
    epochs, iterations : int or None
        Override args.epochs / args.iterations for this call.
    """
    if lr is None:
        lr = args.lr
    if model_type is None:
        model_type = args.model
    if epochs is None and iterations is None:
        epochs = args.epochs
        iterations = args.iterations

    device = torch.device(args.device)

    wd = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    if args.cosine_annealing:
        if iterations is not None:
            T_max = max(1, math.ceil(iterations / len(train_loader)))
        else:
            T_max = max(1, epochs)
        eta_min = lr / 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    losses = []
    metrics_history = {"iterations": [], "train": [], "validation": []}

    # TFSVI samples its KL context set from `model._train_inputs`; populate
    # it here since we bypass `model.fit()` and call `_train_step` directly.
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
            wandb_log_train_step(
                args, i + 1, loss, optimizer=optimizer, model=model,
                model_type=model_type,
            )

            if scheduler is not None and (i + 1) % iters_per_epoch == 0:
                scheduler.step()

            if (i + 1) % args.eval_every == 0:
                metrics_history["iterations"].append(i + 1)
                train_eval = evaluate_light(
                    model, train_test_dataset, args, model_type=model_type
                )
                validation_eval = evaluate_light(
                    model, validation_dataset, args, model_type=model_type
                )
                metrics_history["train"].append(train_eval)
                metrics_history["validation"].append(validation_eval)
                wandb_log_eval(i + 1, train_eval, validation_eval)

            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    else:
        total_epochs = epochs
        loop = tqdm(range(total_epochs), unit=" epoch", desc=desc)
        it = 0
        for _ in loop:
            for inputs, target in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                loss = model._train_step(optimizer, inputs, target)
                losses.append(loss.item())
                it += 1
                wandb_log_train_step(
                    args, it, loss, optimizer=optimizer, model=model,
                    model_type=model_type,
                )

                if it % args.eval_every == 0:
                    metrics_history["iterations"].append(it)
                    train_eval = evaluate_light(
                        model, train_test_dataset, args, model_type=model_type
                    )
                    validation_eval = evaluate_light(
                        model, validation_dataset, args, model_type=model_type
                    )
                    metrics_history["train"].append(train_eval)
                    metrics_history["validation"].append(validation_eval)
                    wandb_log_eval(it, train_eval, validation_eval)

            if scheduler is not None:
                scheduler.step()
            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    # Extract training diagnostics from model buffers (model-specific).
    diagnostics = {}
    for attr in (
        "KLs", "bb_alphas", "prior_regularizers", "data_terms",
        "function_terms", "betas", "l2_terms",
    ):
        if hasattr(model, attr):
            diagnostics[attr] = [float(v) for v in getattr(model, attr)]
    if model_type == "ftip":
        diagnostics["base_KLs"] = [float(v) for v in model.base_KLs]
        diagnostics["flow_ldj"] = [float(v) for v in model.flow_ldj]

    return losses, metrics_history, diagnostics


def _flow_tag(args, model_type):
    """Suffix that distinguishes FTIP result files by flow_type.

    Empty for non-FTIP models and for the default affine flow (preserves
    backward compat with existing filenames).
    """
    if model_type != "ftip":
        return ""
    ft = getattr(args, "flow_type", "affine")
    if ft == "affine":
        return ""
    if ft == "spline":
        return "_spline"
    if ft == "spline_1x1":
        return "_spline1x1"
    return f"_{ft}"


def _layer_tag(args, model_type=None):
    """Filename tag indicating the Bayesian-layer parameterization."""
    if model_type == "map":
        return "_det"
    return "_bayes"


def _variant_tag(args, model_type):
    """Filename tag for variants that share the same top-level model name."""
    if model_type != "ap_fsvi":
        return ""
    discrepancy = getattr(args, "ap_fsvi_discrepancy", "mmd")
    tag = f"_{discrepancy}"
    if discrepancy in (
        "sample_sliced_kl",
        "sample_sliced_knn_kl",
        "sample_sliced_gaussian_kl",
        "prior_whitened_sliced_kl",
        "spectral_projected_kl",
        "spectral_sliced_kl",
    ):
        mode = getattr(args, "ap_fsvi_sample_projection_mode", "random")
        tag = f"{tag}_{mode}"
    return tag


def _ckpt_path(args, dataset_name, model_type):
    """Build a checkpoint path."""
    alpha_tag = f"_alpha{args.bb_alpha}"
    layer_tag = _layer_tag(args, model_type)
    flow_tag = _flow_tag(args, model_type)
    variant_tag = _variant_tag(args, model_type)
    return os.path.join(
        args.output_dir,
        f"{model_type}_{dataset_name}{alpha_tag}{layer_tag}{flow_tag}{variant_tag}_seed{args.seed}.pt",
    )


def _build_result(dataset_name, model_type, model, args, train_loader,
                  train_test_dataset, test_dataset, lr=None,
                  epochs=None, iterations=None, desc="Training"):
    """Train a model, evaluate it, and return the result dict."""
    if lr is None:
        lr = args.lr

    if args.compile:
        try:
            if hasattr(model, "nelbo"):
                model.nelbo = torch.compile(model.nelbo)
        except Exception:
            print("  [warn] torch.compile unavailable, running without it")

    run = init_wandb_run(
        args,
        name=wandb_run_name(
            "UCI",
            dataset=dataset_name,
            model=model_type,
            suffix=_uci_wandb_suffix(args, model_type),
            seed=args.seed,
        ),
        group=_uci_wandb_group(dataset_name, model_type, args),
        tags=[
            "uci",
            dataset_name,
            model_type,
            args.layer_model,
        ],
        config={
            "dataset_name": dataset_name,
            "model_type": model_type,
        },
    )

    t0 = time.time()
    losses, metrics_history, diagnostics = train_with_metrics(
        model, train_loader, train_test_dataset, test_dataset, args,
        lr=lr, epochs=epochs, iterations=iterations,
        model_type=model_type, desc=desc,
    )
    train_time = time.time() - t0

    train_metrics = evaluate(model, train_test_dataset, args, model_type=model_type)
    test_metrics = evaluate(model, test_dataset, args, model_type=model_type)
    # FBNN / TFSVI / MFVI do not expose an implicit-process prior in (mean, var) form.
    prior_stats = (
        evaluate_prior(model, test_dataset, args)
        if model_type not in ("fbnn", "tfsvi", "mfvi", "map")
        else {"var_mean": float("nan"), "var_var": float("nan")}
    )

    actual_epochs = epochs if epochs is not None else args.epochs
    actual_iterations = iterations if iterations is not None else args.iterations

    hyperparameters = {
        "lr": lr,
        "batch_size": args.batch_size,
        "iterations": actual_iterations,
        "epochs": actual_epochs,
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "layer_model": args.layer_model,
        "dropout": args.dropout,
        "regression_coeffs": args.regression_coeffs,
        "bb_alpha": args.bb_alpha,
        "use_prior_regularizer": args.use_prior_regularizer,
        "regularizer_mode": args.regularizer_mode,
        "prior_regularizer_scaler": args.prior_regularizer_scaler,
        "cosine_annealing": args.cosine_annealing,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
    }
    if model_type == "ftip":
        hyperparameters["flow_depth"] = args.flow_depth
        hyperparameters["flow_type"] = args.flow_type
        hyperparameters["flow_num_bins"] = args.flow_num_bins
        hyperparameters["flow_domain"] = args.flow_domain
        hyperparameters["num_samples"] = args.num_samples
        hyperparameters["eval_samples"] = args.eval_samples
    if model_type == "mfvi":
        hyperparameters["mfvi_num_eval_samples"] = args.mfvi_num_eval_samples
    if model_type == "tfsvi":
        hyperparameters["tfsvi_sigma_prior"] = args.tfsvi_sigma_prior
        hyperparameters["tfsvi_S_ctx"] = args.tfsvi_S_ctx
        hyperparameters["tfsvi_K_ctx"] = args.tfsvi_K_ctx
        hyperparameters["tfsvi_num_train_samples"] = args.tfsvi_num_train_samples
        hyperparameters["tfsvi_num_eval_samples"] = args.tfsvi_num_eval_samples
    if model_type == "fbnn":
        hyperparameters["fbnn_prior"] = args.fbnn_prior
        hyperparameters["fbnn_freeze_prior"] = args.fbnn_freeze_prior
        hyperparameters["fbnn_gp_inner_dim"] = args.fbnn_gp_inner_dim
        hyperparameters["fbnn_num_measurement"] = args.fbnn_num_measurement
        hyperparameters["fbnn_num_context"] = args.fbnn_num_context
        hyperparameters["fbnn_context_std"] = args.fbnn_context_std
        hyperparameters["fbnn_lambda_kl"] = args.fbnn_lambda_kl
        hyperparameters["fbnn_num_eval_samples"] = args.fbnn_num_eval_samples


    if model_type == "ap_fsvi":
        hyperparameters.update({
            "ap_fsvi_layer_model": "BayesLinear",
            "ap_fsvi_prior": args.ap_fsvi_prior,
            "ap_fsvi_weight_log_sigma_init": args.ap_fsvi_weight_log_sigma_init,
            "ap_fsvi_num_samples": args.ap_fsvi_num_samples,
            "ap_fsvi_num_prior_samples": (
                args.ap_fsvi_num_prior_samples or args.ap_fsvi_num_samples
            ),
            "ap_fsvi_num_eval_samples": args.ap_fsvi_num_eval_samples,
            "ap_fsvi_num_measurement": args.ap_fsvi_num_measurement,
            "ap_fsvi_adaptive_measure_points": args.ap_fsvi_adaptive_measure_points,
            "ap_fsvi_adaptive_measure_mode": args.ap_fsvi_adaptive_measure_mode,
            "ap_fsvi_adaptive_measure_every": args.ap_fsvi_adaptive_measure_every,
            "ap_fsvi_beta": args.ap_fsvi_beta,
            "ap_fsvi_beta_start": args.ap_fsvi_beta_start,
            "ap_fsvi_beta_warmup_steps": args.ap_fsvi_beta_warmup_steps,
            "ap_fsvi_data_pretrain_steps": args.ap_fsvi_data_pretrain_steps,
            "ap_fsvi_data_loss": args.ap_fsvi_data_loss,
            "ap_fsvi_discrepancy": args.ap_fsvi_discrepancy,
            "ap_fsvi_discrepancy_projections": args.ap_fsvi_discrepancy_projections,
            "ap_fsvi_sample_projection_mode": args.ap_fsvi_sample_projection_mode,
            "ap_fsvi_sample_knn_k": args.ap_fsvi_sample_knn_k,
            "ap_fsvi_quantile_transport_k": args.ap_fsvi_quantile_transport_k,
            "ap_fsvi_sinkhorn_epsilon": args.ap_fsvi_sinkhorn_epsilon,
            "ap_fsvi_sinkhorn_iterations": args.ap_fsvi_sinkhorn_iterations,
            "ap_fsvi_log_variance_init": args.ap_fsvi_log_variance_init,
            "ap_fsvi_measurement_weights": args.ap_fsvi_measurement_weights,
            "ap_fsvi_near_data_noise": args.ap_fsvi_near_data_noise,
            "ap_fsvi_domain_std": args.ap_fsvi_domain_std,
            "ap_fsvi_max_grad_norm": args.ap_fsvi_max_grad_norm,
        })
    if model_type == "map":
        hyperparameters["map_l2"] = args.map_l2
        hyperparameters["map_log_variance_init"] = args.map_log_variance_init

    result = {
        "dataset": dataset_name,
        "model": model_type,
        "hyperparameters": hyperparameters,
        "train_time_s": round(train_time, 2),
        "train": train_metrics,
        "test": test_metrics,
        "prior": prior_stats,
        "losses": losses,
        "metrics_history": metrics_history,
        "diagnostics": diagnostics,
    }

    for split, m in [("Train", train_metrics), ("Test", test_metrics)]:
        print(f"  {model_type.upper()} {split}: RMSE={m['RMSE']:.4f}  NLL={m['NLL']:.4f}"
              f"  CRPS={m['CRPS']:.4f}  CQM={m['CQM']:.4f}")
    print(f"  Prior: mean(Var[f])={prior_stats['var_mean']:.6f}  "
          f"var(Var[f])={prior_stats['var_var']:.6f}")
    print(f"  Time: {train_time:.1f}s")

    if args.test_ood:
        ood_metrics = evaluate_ood(model, test_dataset, args, model_type=model_type, seed=args.seed)
        result["test"].update(ood_metrics)
        result["ood"] = ood_metrics
        print(f"  OOD: AUROC={ood_metrics['AUROC']:.4f}  "
              f"H(in)={ood_metrics['entropy_id_mean']:.4f}+/-{ood_metrics['entropy_id_std']:.4f}  "
              f"H(ood)={ood_metrics['entropy_ood_mean']:.4f}+/-{ood_metrics['entropy_ood_std']:.4f}")

    wandb_log_result(result)

    # Save checkpoint
    if args.save_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt = _ckpt_path(args, dataset_name, model_type)
        torch.save(model.state_dict(), ckpt)
        print(f"  Checkpoint: {ckpt}")

    return result, model


def _wandb_run_metadata(args, dataset_name):
    suffix = None
    group_parts = ["uci_120k_cluster", dataset_name, args.model]
    if args.model == "ap_fsvi":
        suffix = pretty_discrepancy_name(args.ap_fsvi_discrepancy)
        group_parts.append(args.ap_fsvi_discrepancy)
        if args.ap_fsvi_discrepancy in (
            "sample_sliced_kl",
            "sample_sliced_knn_kl",
            "sample_sliced_gaussian_kl",
            "prior_whitened_sliced_kl",
            "spectral_projected_kl",
            "spectral_sliced_kl",
        ):
            suffix = f"{suffix}/{args.ap_fsvi_sample_projection_mode}"
            group_parts.append(args.ap_fsvi_sample_projection_mode)

    name = wandb_run_name(
        "UCI 120k",
        dataset=dataset_name,
        model=args.model,
        suffix=suffix,
        seed=args.seed,
    )
    tags = ["uci", "120k", dataset_name, args.model]
    if args.model == "ap_fsvi":
        tags.extend([args.ap_fsvi_discrepancy, args.ap_fsvi_sample_projection_mode])
    group = "_".join(str(part) for part in group_parts)
    return name, group, tags


def run_single(dataset_name, args):
    """Run benchmark on a single dataset. Returns a list of result dicts."""
    wandb_name, wandb_group, wandb_tags = _wandb_run_metadata(args, dataset_name)
    wandb_run = init_wandb_run(
        args, name=wandb_name, group=wandb_group, tags=wandb_tags
    )

    try:
        return _run_single(dataset_name, args)
    finally:
        finish_wandb_run(wandb_run)


def _run_single(dataset_name, args):
    """Run benchmark on a single dataset. Returns a list of result dicts."""
    use_warm_start = args.model == "ftip" and args.auto_warm_start

    header = f"FTIP (warm-start from VIP)" if use_warm_start else args.model.upper()
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  Model: {header}")
    print(f"{'='*60}")

    dataset = get_dataset(dataset_name)
    train_dataset, train_test_dataset, test_dataset = dataset.get_split(args.test_size, args.seed)

    use_cuda = args.device and "cuda" in args.device
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_cuda, num_workers=0,
    )

    results = []

    if use_warm_start:
        # Compute fair-comparison VIP budget (same total as VIP pre-train + FTIP fine-tune)
        vip_ep = args.vip_epochs
        vip_it = args.vip_iterations
        ftip_ep = args.epochs
        ftip_it = args.iterations

        if vip_it is not None and ftip_it is not None:
            total_it = vip_it + ftip_it
            total_ep = None
            budget_str = f"{total_it} iters (={vip_it}+{ftip_it})"
            vip_phase_str = f"{vip_it} iters"
            ftip_phase_str = f"{ftip_it} iters"
        elif vip_ep is not None and ftip_ep is not None:
            total_ep = vip_ep + ftip_ep
            total_it = None
            budget_str = f"{total_ep} epochs (={vip_ep}+{ftip_ep})"
            vip_phase_str = f"{vip_ep} epochs"
            ftip_phase_str = f"{ftip_ep} epochs"
        else:
            # Mixed: convert to iterations for fair comparison
            iters_per_epoch = len(train_loader)
            vip_total = vip_it if vip_it is not None else vip_ep * iters_per_epoch
            ftip_total = ftip_it if ftip_it is not None else ftip_ep * iters_per_epoch
            total_it = vip_total + ftip_total
            total_ep = None
            budget_str = f"{total_it} iters (={vip_total}+{ftip_total})"
            vip_phase_str = f"{vip_it} iters" if vip_it else f"{vip_ep} epochs"
            ftip_phase_str = f"{ftip_it} iters" if ftip_it else f"{ftip_ep} epochs"

        # ── Train VIP once: first half for warm-start, full budget for baseline ──
        # Phase 1: train VIP for vip_epochs/iters (warm-start source)
        print(f"\n  Phase 1: Training VIP for warm-start ({vip_phase_str}, lr={args.vip_lr})")

        vip_model = build_model(args, train_dataset, model_type="vip")
        vip_ws_result, vip_model = _build_result(
            dataset_name, "vip", vip_model, args, train_loader,
            train_test_dataset, test_dataset, lr=args.vip_lr,
            epochs=vip_ep, iterations=vip_it,
            desc="VIP pre-training",
        )

        # Save warm-start checkpoint for FTIP (before continuing)
        vip_ws_state = {n: p.data.clone() for n, p in vip_model.named_parameters()}

        # Continue training VIP for the remaining budget (= ftip budget) to get fair baseline
        print(f"\n  VIP baseline: continuing for {ftip_phase_str} (total={budget_str})")
        vip_baseline_result, _ = _build_result(
            dataset_name, "vip", vip_model, args, train_loader,
            train_test_dataset, test_dataset, lr=args.vip_lr,
            epochs=ftip_ep, iterations=ftip_it,
            desc="VIP baseline (continued)",
        )
        # Update baseline result to reflect total training
        vip_baseline_result["train_time_s"] = round(
            vip_ws_result["train_time_s"] + vip_baseline_result["train_time_s"], 2
        )
        results.append(vip_baseline_result)

        # Restore warm-start checkpoint for FTIP
        for n, p in vip_model.named_parameters():
            p.data.copy_(vip_ws_state[n])
        del vip_ws_state

        # ── Phase 2: Warm-start & fine-tune FTIP ──
        print(f"\n  Phase 2: Fine-tuning FTIP ({ftip_phase_str}, lr={args.ftip_lr}, cosine)")

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

        # Print comparison (FTIP vs fair-budget VIP baseline)
        vip_rmse = vip_baseline_result["test"]["RMSE"]
        ftip_rmse = ftip_result["test"]["RMSE"]
        vip_nll = vip_baseline_result["test"]["NLL"]
        ftip_nll = ftip_result["test"]["NLL"]
        print(f"\n  Delta (FTIP - VIP baseline): RMSE={ftip_rmse - vip_rmse:+.4f}  NLL={ftip_nll - vip_nll:+.4f}")
        print(f"  FTIP total time: {ftip_result['total_time_s']:.1f}s")

    else:
        # ── Single-model training (VIP or cold-start FTIP) ──
        model = build_model(args, train_dataset)

        # Manual warm-start from external checkpoint
        if args.warm_start_from and args.model == "ftip":
            device = torch.device(args.device)
            vip_model = build_model(args, train_dataset, model_type="vip")
            vip_model.load_state_dict(
                torch.load(args.warm_start_from, map_location=device, weights_only=True)
            )
            model.warm_start_from_vip(vip_model, learnable_affine=args.learnable_affine)
            del vip_model
            print(f"  Warm-started from {args.warm_start_from} "
                  f"(affine learnable={args.learnable_affine})")

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
        datasets = UCI_REGRESSION_DATASETS
    else:
        datasets = [args.dataset]

    models = REGRESSION_MODELS if args.model == "all" else [args.model]
    all_results = []
    for ds in datasets:
        for model_type in models:
            run_args = copy.copy(args)
            run_args.model = model_type

            # Per-dataset default iters when the user didn't supply a budget.
            # vip_iterations defaults to the same value (used for FTIP warm).
            if not run_args._iters_user_supplied:
                run_args.iterations = DEFAULT_UCI_ITERS.get(ds, 30_000)
                # MFVI gets a larger default budget than the flow-based models.
                if run_args.model == "mfvi":
                    run_args.iterations *= 10
                run_args.epochs = None
                if run_args.vip_epochs is None and run_args.vip_iterations is None:
                    run_args.vip_iterations = run_args.iterations
                    run_args.vip_epochs = None
            # MFVI also defaults to alpha=0.5 (vs 1.0 for the rest), unless
            # the user explicitly passed --bb_alpha.
            if run_args.model == "mfvi" and not run_args._bb_alpha_user_supplied:
                run_args.bb_alpha = 0.5

            # Check if results already exist
            alpha_tag = f"_alpha{run_args.bb_alpha}"
            layer_tag = _layer_tag(run_args, run_args.model)
            flow_tag = _flow_tag(run_args, run_args.model)
            variant_tag = _variant_tag(run_args, run_args.model)
            out_path = os.path.join(
                run_args.output_dir,
                f"{run_args.model}_{ds}{alpha_tag}{layer_tag}{flow_tag}{variant_tag}_seed{run_args.seed}.json",
            )
            if os.path.exists(out_path):
                print(f"\n  Skipping {run_args.model}/{ds} seed={run_args.seed}: {out_path} already exists")
                with open(out_path) as f:
                    loaded = json.load(f)
                all_results.extend(loaded if isinstance(loaded, list) else [loaded])
                continue

            ds_results = run_single(ds, run_args)
            all_results.extend(ds_results)

            # Save per-dataset results immediately (so partial runs are preserved)
            os.makedirs(run_args.output_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(ds_results, f, indent=2)
            print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
