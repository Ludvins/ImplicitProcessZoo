"""UCI regression benchmark.

Runs the UCI regression datasets with VIP, FTIP, GMVIP, SIP,
MFVI, FBNN, TFSVI, or MAP.
Each run writes a JSON result file and, by default, a checkpoint.

Example:
    python -m experiments.uci.benchmark --model ftip --dataset boston
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.benchmark_utils import (
    add_wandb_args,
    finish_wandb_run,
    init_wandb_run,
    wandb_log_eval,
    wandb_log_result,
    wandb_log_train_step,
)
from implicit_process_zoo.fbnn import FBNN
from implicit_process_zoo.flows import CouplingFlow, SplineCoupling1x1Flow, SplineCouplingFlow
from implicit_process_zoo.ftip import FTIP
from implicit_process_zoo.gmvip import GeneralizedMatheronVIP, initialize_inducing_points
from implicit_process_zoo.map_baseline import DeterministicMAP
from implicit_process_zoo.mfvi import MFVI
from implicit_process_zoo.priors.generative_functions import GP, BayesianNN, BayesLinear
from implicit_process_zoo.sip import SIP
from implicit_process_zoo.tfsvi import TFSVI
from implicit_process_zoo.utils import (
    build_training_checkpoint,
    load_training_checkpoint,
    load_warm_start_state,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from implicit_process_zoo.utils.dataset import get_dataset
from implicit_process_zoo.utils.metrics import MetricsRegression
from implicit_process_zoo.utils.utils import infinite_loader
from implicit_process_zoo.vip import VIP

UCI_REGRESSION_DATASETS = [
    "boston",
    "energy",
    "concrete",
    "naval",
    "power",
    "protein",
    "kin8nm",
    "yatch",
    "winered",
]

# Default per-dataset training-iteration budgets, mirroring the FTIP
# cold-start BayesLinear / alpha=1.0 sweep recorded under
# results/uci/ftip_*_alpha1.0_bayes_*. Used only when the user does
# not pass --iterations or --epochs explicitly.
DEFAULT_UCI_ITERS = {
    "boston": 30_000,
    "concrete": 30_000,
    "energy": 30_000,
    "protein": 30_000,
    "kin8nm": 60_000,
    "naval": 60_000,
    "power": 60_000,
    "winered": 60_000,
    "yatch": 60_000,
}

ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}

LAYER_MODELS = {
    "BayesLinear": BayesLinear,
}

REGRESSION_MODELS = [
    "vip",
    "ftip",
    "fbnn",
    "tfsvi",
    "mfvi",
    "gmvip",
    "sip",
    "map",
]


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return float(value)


def _float_or_median(value):
    if isinstance(value, str) and value.lower() == "median":
        return "median"
    return float(value)


def _float_or_prior_marginal(value):
    if isinstance(value, str) and value.lower() == "prior_marginal":
        return "prior_marginal"
    return float(value)


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
        mu_b = mean_pred[:, start:end, :]  # (S, B, D)
        std_b = std_pred[:, start:end, :]  # (S, B, D)
        B = end - start

        # Draw MC samples from the mixture
        comp_idx = torch.randint(S, (n_mc, B), device=device)
        n_idx = torch.arange(B, device=device).unsqueeze(0).expand(n_mc, B)
        mu_chosen = mu_b[comp_idx, n_idx, :]
        std_chosen = std_b[comp_idx, n_idx, :]
        y_samples = mu_chosen + std_chosen * torch.randn_like(mu_chosen)  # (n_mc, B, D)

        # Evaluate log p(y|x) under the full mixture
        y_exp = y_samples.unsqueeze(1)  # (n_mc, 1, B, D)
        mu_exp = mu_b.unsqueeze(0)  # (1, S, B, D)
        std_exp = std_b.unsqueeze(0)  # (1, S, B, D)
        var_exp = std_exp**2

        log_comp = -0.5 * (log2pi + var_exp.log() + (y_exp - mu_exp) ** 2 / var_exp)
        log_comp = log_comp.sum(-1)  # (n_mc, S, B)

        log_mix = torch.logsumexp(log_comp, dim=1) - log_S  # (n_mc, B)
        all_entropy.append(-log_mix.mean(dim=0))  # (B,)

    return torch.cat(all_entropy, dim=0)


def _batched_entropy(model, x, model_type, eval_samples, a=None, batch_size=2048):
    """Compute per-point predictive entropy without materializing full (S, N, D)."""
    all_entropy = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size]
        if model_type == "ftip" and a is not None:
            mean, std = model.forward_with_coefficients(xb, a)
            std = std.unsqueeze(-1).expand_as(mean)
        elif model_type == "vip":
            mean, std = model(xb)
        elif model_type == "fbnn":
            mean, std = _fbnn_pred_components(model, xb)
        elif model_type == "tfsvi" or model_type == "mfvi":
            mean, std = _tfsvi_pred_components(model, xb, eval_samples)
        elif model_type == "gmvip":
            mean, std = _gmvip_pred_components(model, xb, eval_samples)
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
        entropy_samples = (
            args.gmvip_num_eval_samples if model_type == "gmvip" else args.eval_samples
        )
        entropy_id = _batched_entropy(model, x_id, model_type, entropy_samples, a)
        entropy_ood = _batched_entropy(model, x_ood, model_type, entropy_samples, a)

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
        "n_id": len(entropy_id),
        "n_ood": len(entropy_ood),
    }


def parse_args(
    argv=None,
    *,
    description="UCI regression benchmark",
    dataset_names=None,
    dataset_group_label="UCI regression datasets",
    default_output_dir="results",
):
    dataset_names = list(UCI_REGRESSION_DATASETS if dataset_names is None else dataset_names)
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Experiment ---
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=REGRESSION_MODELS + ["all"],
        help="Model to train.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=dataset_names + ["all"],
        help=f"Dataset name or 'all' for all {dataset_group_label}.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--test_size", type=float, default=0.1, help="Fraction of data used for testing."
    )
    p.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"], help="Tensor dtype."
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    p.add_argument(
        "--test_ood",
        action="store_true",
        default=False,
        help="Evaluate OOD detection using predictive entropy as score.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Directory to save result JSON files.",
    )

    # --- BayesianNN (generative function / prior) ---
    p.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[10, 10],
        help="Hidden layer widths for the BayesianNN prior.",
    )
    p.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=list(ACTIVATIONS.keys()),
        help="Activation function for the BayesianNN.",
    )
    p.add_argument(
        "--layer_model",
        type=str,
        default="BayesLinear",
        choices=list(LAYER_MODELS.keys()),
        help="Bayesian layer type. Benchmarks use full BayesLinear.",
    )
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the BayesianNN.")
    p.add_argument(
        "--weight_log_sigma_init",
        type=float,
        default=0.0,
        help="Initial log std for shared BayesianNN weight/bias "
        "samples used by VIP/FTIP/MFVI/FBNN/TFSVI.",
    )

    # --- Model (shared VIP / FTIP) ---
    p.add_argument(
        "--regression_coeffs", type=int, default=20, help="Number of regression coefficients (S)."
    )
    p.add_argument(
        "--bb_alpha",
        type=float,
        default=None,
        help="BB-alpha parameter (0 = ELBO, 1 = BB-alpha energy). If unset: 0.0 for all models.",
    )
    p.add_argument(
        "--use_prior_regularizer",
        action="store_true",
        default=False,
        help="Enable the method's optional prior regularizer.",
    )
    p.add_argument("--no_prior_regularizer", action="store_true", help="Disable prior regularizer.")
    p.add_argument(
        "--regularizer_mode",
        type=str,
        default="evidence",
        choices=["evidence", "KL"],
        help="Prior regularizer mode.",
    )
    p.add_argument(
        "--prior_regularizer_scaler",
        type=float,
        default=1.0,
        help="Prior regularizer scaling factor.",
    )
    p.add_argument(
        "--vip_learn_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the VIP BayesianNN generator/prior parameters. "
        "Use --no-vip_learn_prior for a frozen standard BNN prior.",
    )
    p.add_argument(
        "--ftip_learn_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the FTIP BayesianNN generator/prior parameters. "
        "Use --no-ftip_learn_prior for a frozen standard BNN prior.",
    )

    # --- FTIP-specific ---
    p.add_argument(
        "--flow_type",
        type=str,
        default="spline_1x1",
        choices=["affine", "spline", "spline_1x1"],
        help="FTIP flow class. 'affine' = original CouplingFlow "
        "(affine coupling), 'spline' = SplineCouplingFlow "
        "(RQ spline coupling), 'spline_1x1' = "
        "SplineCoupling1x1Flow (spline coupling + Glow 1x1 LU "
        "mixing, default).",
    )
    p.add_argument(
        "--flow_num_bins",
        type=int,
        default=8,
        help="Bins per RQ-spline coupling layer (ignored if flow_type=affine).",
    )
    p.add_argument(
        "--flow_domain",
        type=float,
        default=3.0,
        help="Spline domain half-width B (ignored if flow_type=affine).",
    )
    p.add_argument(
        "--flow_depth",
        type=int,
        default=2,
        help="Number of coupling layers in the normalizing flow (FTIP only).",
    )
    p.add_argument(
        "--num_samples", type=int, default=200, help="Number of MC posterior samples (FTIP only)."
    )
    p.add_argument(
        "--eval_samples",
        type=int,
        default=1000,
        help="Number of MC samples used at evaluation time (FTIP only).",
    )
    p.add_argument(
        "--warm_start_from",
        type=str,
        default=None,
        help="Legacy model state or versioned checkpoint used to initialize FTIP.",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a full schema-v1 training checkpoint.",
    )
    p.add_argument(
        "--resume_step_offset",
        type=int,
        default=0,
        help="Deprecated; full checkpoints restore their own global step.",
    )
    p.add_argument(
        "--learnable_affine",
        action="store_true",
        default=True,
        help="Make the affine warm-start layer trainable.",
    )
    p.add_argument(
        "--no_learnable_affine",
        action="store_true",
        help="Fix the affine warm-start layer (not trainable).",
    )

    # --- FBNN-specific ---
    p.add_argument(
        "--fbnn_prior",
        type=str,
        default="gp",
        choices=["gp", "bnn"],
        help="fBNN prior family: 'gp' (RFF GP, paper default) "
        "or 'bnn' (Bayesian NN with SSGE prior score).",
    )
    p.add_argument(
        "--fbnn_freeze_prior",
        action="store_true",
        default=False,
        help="Freeze the prior's parameters. By default the GP "
        "kernel hyperparameters are LEARNED jointly, "
        "matching Sun et al. 2019.",
    )
    p.add_argument(
        "--fbnn_gp_inner_dim",
        type=int,
        default=10,
        help="Inner-layer dim of the RFF GP prior (number of random features).",
    )
    p.add_argument(
        "--fbnn_gp_kernel_amp",
        type=float,
        default=1.0,
        help="Initial amplitude of the FBNN RFF GP prior.",
    )
    p.add_argument(
        "--fbnn_gp_kernel_length",
        type=float,
        default=1.0,
        help="Initial length-scale of the FBNN RFF GP prior.",
    )
    p.add_argument(
        "--fbnn_num_measurement",
        type=int,
        default=20,
        help="# training-point measurements for the functional KL.",
    )
    p.add_argument(
        "--fbnn_num_context",
        type=int,
        default=20,
        help="# OOD context points sampled from N(0, context_std^2).",
    )
    p.add_argument(
        "--fbnn_context_std",
        type=float,
        default=2.0,
        help="Std of the Gaussian from which context points are sampled.",
    )
    p.add_argument(
        "--fbnn_lambda_kl",
        type=float,
        default=1.0,
        help="Weight on the functional KL term in the FBNN ELBO.",
    )
    p.add_argument(
        "--fbnn_num_eval_samples",
        type=int,
        default=200,
        help="MC posterior samples used at FBNN evaluation time.",
    )

    # --- TFSVI-specific (Rudner et al., 2022) ---
    p.add_argument(
        "--tfsvi_sigma_prior",
        type=float,
        default=1.0,
        help="Prior std for the parameter Gaussian p(theta) = N(0, sigma_prior^2 I).",
    )
    p.add_argument(
        "--tfsvi_S_ctx", type=int, default=5, help="# context sets in the max-KL estimator."
    )
    p.add_argument("--tfsvi_K_ctx", type=int, default=20, help="# points per context set.")
    p.add_argument(
        "--tfsvi_num_train_samples",
        type=int,
        default=20,
        help="MC parameter samples per TFSVI training step.",
    )
    p.add_argument(
        "--tfsvi_num_eval_samples",
        type=int,
        default=200,
        help="MC parameter samples used at TFSVI evaluation time.",
    )

    # --- MFVI-specific ---
    p.add_argument(
        "--mfvi_num_eval_samples",
        type=int,
        default=200,
        help="MC weight samples used at MFVI evaluation time. "
        "Training uses --regression_coeffs as the per-step "
        "MC count (matches the other methods in this script).",
    )

    # --- SIP-specific (Sparse Implicit Process) ---
    p.add_argument(
        "--sip_num_inducing", type=int, default=100, help="Number of sparse inducing inputs Z."
    )
    p.add_argument(
        "--sip_inducing_method",
        type=str,
        default="kmeans",
        choices=["random_subset", "kmeans", "grid_1d", "train_quantiles"],
        help="Initialization method for SIP inducing inputs.",
    )
    p.add_argument(
        "--sip_num_prior_samples",
        type=int,
        default=512,
        help="Prior samples used to estimate SIP sparse moments.",
    )
    p.add_argument(
        "--sip_num_train_samples",
        type=int,
        default=None,
        help=(
            "Posterior/prior inducing samples used in SIP's "
            "Monte Carlo likelihood and critic KL. Defaults to "
            "--sip_num_prior_samples."
        ),
    )
    p.add_argument(
        "--sip_num_eval_samples",
        type=int,
        default=200,
        help="Posterior samples used at SIP evaluation time.",
    )
    p.add_argument(
        "--sip_learn_inducing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimize SIP inducing inputs Z.",
    )
    p.add_argument(
        "--sip_learn_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Optimize SIP BNN-prior parameters. "
            "Use --no-sip_learn_prior for a frozen standard BNN prior."
        ),
    )
    p.add_argument(
        "--sip_detach_covariances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Detach SIP empirical covariance estimates from prior gradients.",
    )
    p.add_argument(
        "--sip_jitter", type=float, default=1e-5, help="Diagonal jitter added to SIP K_ZZ."
    )
    p.add_argument(
        "--sip_log_variance_init",
        type=float,
        default=-5.0,
        help="Initial log observation-noise variance for SIP regression.",
    )
    p.add_argument(
        "--sip_min_log_variance",
        type=float,
        default=None,
        help="Optional lower bound for SIP regression log observation-noise variance.",
    )
    p.add_argument(
        "--sip_fix_random_noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use cached BNN prior noise for SIP moment estimates "
            "and critic prior samples. Default False matches the "
            "paper's stochastic prior sampling."
        ),
    )
    p.add_argument(
        "--sip_beta",
        type=float,
        default=1.0,
        help="Weight on the SIP critic-estimated inducing KL.",
    )
    p.add_argument(
        "--sip_beta_warmup_steps",
        type=int,
        default=0,
        help="Linear warmup steps for the SIP KL weight.",
    )
    p.add_argument(
        "--sip_critic_hidden_dim",
        type=int,
        default=50,
        help="Hidden width of the SIP inducing-space critic.",
    )
    p.add_argument(
        "--sip_critic_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the SIP inducing-space critic.",
    )
    p.add_argument(
        "--sip_critic_steps", type=int, default=1, help="Critic updates per SIP variational update."
    )
    p.add_argument(
        "--sip_posterior_noise_dim",
        type=int,
        default=100,
        help=(
            "Noise dimension for the implicit SIP q_phi(u) sampler. "
            "The reference SIP implementation uses 100."
        ),
    )
    p.add_argument(
        "--sip_posterior_hidden_dim",
        type=int,
        default=50,
        help="Hidden width of the implicit SIP q_phi(u) sampler.",
    )
    p.add_argument(
        "--sip_posterior_depth",
        type=int,
        default=2,
        help="Hidden-layer count of the implicit SIP q_phi(u) sampler.",
    )

    # --- GMVIP-specific ---
    p.add_argument(
        "--gmvip_operator_type",
        choices=["empirical", "rbf"],
        default="rbf",
        help="GMVIP Matheron operator.",
    )
    p.add_argument(
        "--gmvip_posterior_type",
        choices=["gaussian", "realnvp"],
        default="gaussian",
        help="GMVIP latent coefficient posterior.",
    )
    p.add_argument(
        "--gmvip_num_inducing", type=int, default=32, help="Number of GMVIP inducing points."
    )
    p.add_argument(
        "--gmvip_inducing_method",
        type=str,
        default="kmeans",
        choices=["random_subset", "kmeans", "grid_1d", "train_quantiles"],
        help="Initialization rule for GMVIP inducing points.",
    )
    p.add_argument(
        "--gmvip_num_operator_bank_samples",
        type=int,
        default=256,
        help="Prior samples used to initialize GMVIP operator moments.",
    )
    p.add_argument(
        "--gmvip_num_train_samples",
        type=int,
        default=16,
        help="Posterior function samples per GMVIP training step.",
    )
    p.add_argument(
        "--gmvip_num_eval_samples",
        type=int,
        default=200,
        help="Posterior function samples used at GMVIP evaluation time.",
    )
    p.add_argument(
        "--gmvip_antithetic_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use antithetic base Gaussian pairs for Gaussian/RealNVP GMVIP coefficient samples.",
    )
    p.add_argument(
        "--gmvip_beta", type=float, default=1.0, help="Weight on the GMVIP latent coefficient KL."
    )
    p.add_argument(
        "--gmvip_beta_warmup_steps", type=int, default=0, help="Linear warmup steps for GMVIP beta."
    )
    p.add_argument(
        "--gmvip_data_alpha",
        type=float,
        default=0.0,
        help="Alpha data objective for GMVIP; 0 gives the standard ELBO data term.",
    )
    p.add_argument(
        "--gmvip_weight_log_sigma_init",
        type=float,
        default=0.0,
        help="Frozen BNN prior weight log sigma for GMVIP.",
    )
    p.add_argument(
        "--gmvip_learn_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the GMVIP BNN basis/prior parameters as in VIP.",
    )
    p.add_argument(
        "--gmvip_detach_operator_prior_grad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop GMVIP operator-statistics gradients from updating "
        "the BNN prior parameters while preserving gradients "
        "through residual prior samples and learnable Z.",
    )
    p.add_argument(
        "--gmvip_learn_noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn GMVIP Gaussian observation noise.",
    )
    p.add_argument(
        "--gmvip_init_log_noise",
        type=float,
        default=-2.5,
        help="Initial GMVIP log observation noise.",
    )
    p.add_argument(
        "--gmvip_min_log_noise",
        type=_optional_float,
        default=-5.0,
        help="Optional minimum GMVIP log observation noise; pass none to disable.",
    )
    p.add_argument(
        "--gmvip_max_log_noise",
        type=_optional_float,
        default=None,
        help="Optional maximum GMVIP log observation noise; pass none to disable.",
    )
    p.add_argument(
        "--gmvip_learn_Z",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn GMVIP inducing locations after initialization.",
    )
    p.add_argument(
        "--gmvip_learn_kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn the GMVIP RBF operator kernel hyperparameters.",
    )
    p.add_argument(
        "--gmvip_ard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ARD lengthscales in the GMVIP RBF operator.",
    )
    p.add_argument(
        "--gmvip_init_lengthscale",
        type=_float_or_median,
        default="median",
        help="Initial GMVIP RBF lengthscale or 'median'.",
    )
    p.add_argument(
        "--gmvip_init_outputscale",
        type=_float_or_prior_marginal,
        default="prior_marginal",
        help="Initial GMVIP RBF outputscale or 'prior_marginal'.",
    )
    p.add_argument(
        "--gmvip_inducing_scale",
        type=str,
        default="prior_cholesky",
        choices=["prior_cholesky", "rbf_cholesky", "prior_diag", "identity"],
        help="Map from whitened GMVIP coefficients a to inducing values u.",
    )
    p.add_argument(
        "--gmvip_mean_mode",
        type=str,
        default="prior_sample",
        choices=["prior_sample", "zero", "prior_api"],
        help="Mean initialization mode for GMVIP inducing values.",
    )
    p.add_argument("--gmvip_jitter", type=float, default=1e-5, help="GMVIP linear algebra jitter.")
    p.add_argument(
        "--gmvip_shrinkage",
        type=float,
        default=0.02,
        help="Empirical-operator covariance shrinkage.",
    )
    p.add_argument(
        "--gmvip_posterior_init_mean",
        type=float,
        default=0.0,
        help="Initial GMVIP Gaussian coefficient posterior mean.",
    )
    p.add_argument(
        "--gmvip_posterior_init_log_std",
        type=float,
        default=0.0,
        help="Initial GMVIP Gaussian coefficient posterior log std.",
    )
    p.add_argument(
        "--gmvip_posterior_min_log_std",
        type=_optional_float,
        default=-8.0,
        help="Optional minimum GMVIP posterior log std; pass none to disable.",
    )
    p.add_argument(
        "--gmvip_posterior_max_log_std",
        type=_optional_float,
        default=None,
        help="Optional maximum GMVIP posterior log std; pass none to disable.",
    )
    p.add_argument(
        "--gmvip_flow_depth",
        type=int,
        default=4,
        help="Number of affine coupling layers for GMVIP RealNVP q(a).",
    )
    p.add_argument(
        "--gmvip_flow_hidden_dim",
        type=int,
        default=128,
        help="Hidden width for GMVIP RealNVP coupling nets.",
    )
    p.add_argument(
        "--gmvip_flow_num_layers",
        type=int,
        default=2,
        help="MLP layer count for GMVIP RealNVP coupling nets.",
    )
    p.add_argument(
        "--gmvip_flow_dropout",
        type=float,
        default=0.0,
        help="Dropout for GMVIP RealNVP coupling nets.",
    )
    p.add_argument(
        "--gmvip_flow_scale_bound",
        type=float,
        default=2.0,
        help="Tanh bound for GMVIP RealNVP log-scales.",
    )
    p.add_argument(
        "--gmvip_max_grad_norm",
        type=_optional_float,
        default=None,
        help="Optional GMVIP gradient clipping norm; pass none to disable.",
    )

    # --- MAP-specific ---
    p.add_argument(
        "--map_l2",
        type=float,
        default=1e-4,
        help="L2 weight penalty for deterministic MAP baseline.",
    )
    p.add_argument(
        "--map_log_variance_init",
        type=float,
        default=-5.0,
        help="Initial log observation variance for MAP baseline.",
    )

    # --- Auto warm-start (VIP -> FTIP pipeline) ---
    p.add_argument(
        "--auto_warm_start",
        action="store_true",
        default=True,
        help="Automatically train VIP first, then warm-start FTIP (FTIP only).",
    )
    p.add_argument(
        "--no_auto_warm_start",
        action="store_true",
        help="Disable auto warm-start; train FTIP from scratch.",
    )
    p.add_argument(
        "--vip_epochs",
        type=int,
        default=None,
        help="Epochs for VIP pre-training phase (auto warm-start only).",
    )
    p.add_argument(
        "--vip_iterations",
        type=int,
        default=None,
        help="Iterations for VIP pre-training phase (auto warm-start only).",
    )
    p.add_argument(
        "--vip_lr", type=float, default=1e-3, help="Learning rate for VIP pre-training phase."
    )
    p.add_argument(
        "--ftip_lr",
        type=float,
        default=1e-4,
        help="Learning rate for FTIP fine-tuning phase (auto warm-start only). "
        "1e-4 lets the flow escape the VIP warm-start init; smaller values "
        "(e.g. 1e-5) leave the spline coupling layers frozen at the affine "
        "VIP posterior.",
    )

    # --- Checkpointing ---
    p.add_argument(
        "--save_checkpoint",
        action="store_true",
        default=True,
        help="Save model checkpoint after training.",
    )
    p.add_argument(
        "--no_save_checkpoint", action="store_true", help="Disable saving model checkpoint."
    )

    # --- Training ---
    p.add_argument("--batch_size", type=int, default=100, help="Training batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (mutually exclusive with --epochs).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (mutually exclusive with --iterations).",
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=1000,
        help="Compute light metrics on train/test every N iterations.",
    )
    p.add_argument(
        "--cosine_annealing",
        action="store_true",
        default=True,
        help="Use cosine annealing LR schedule.",
    )
    p.add_argument("--no_cosine_annealing", action="store_true", help="Disable cosine annealing.")
    p.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch.compile for faster training (requires Triton).",
    )
    add_wandb_args(p)
    args = p.parse_args(argv)

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

    if args.resume_step_offset < 0:
        p.error("--resume_step_offset must be non-negative.")
    if args.resume_from_checkpoint and args.warm_start_from:
        p.error("--resume_from_checkpoint cannot be combined with --warm_start_from.")
    if args.gmvip_num_inducing <= 0:
        p.error("--gmvip_num_inducing must be positive.")
    if args.model in {"gmvip", "all"} and args.gmvip_operator_type == "empirical":
        if args.gmvip_mean_mode != "prior_sample":
            p.error("--gmvip_mean_mode is only configurable for --gmvip_operator_type rbf.")
        if args.gmvip_inducing_scale != "prior_cholesky":
            p.error("--gmvip_inducing_scale is only configurable for --gmvip_operator_type rbf.")
        args.gmvip_learn_kernel = False
    # Disable auto_warm_start if an explicit checkpoint path is given
    if args.warm_start_from:
        args.auto_warm_start = False
    if args.resume_from_checkpoint:
        args.auto_warm_start = False

    # Track whether the user explicitly chose a training budget; if not,
    # main() picks per-dataset iters from DEFAULT_UCI_ITERS so the script
    # mirrors the FTIP cold-start sweep by default (30k for the small
    # group, 60k for the rest).
    args._iters_user_supplied = args.iterations is not None or args.epochs is not None
    # Detect whether the user passed --bb_alpha so wrappers can preserve an
    # explicit choice while defaulting all models to alpha=0.
    args._bb_alpha_user_supplied = args.bb_alpha is not None
    if args.bb_alpha is None:
        args.bb_alpha = 0.0

    # Default VIP pre-training length: same as main training length
    if args.vip_epochs is None and args.vip_iterations is None:
        args.vip_epochs = args.epochs
        args.vip_iterations = args.iterations

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
        if hasattr(layer, "fix_random_noise") and layer.fix_random_noise and old != S:
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

    if model_type == "sip":
        X_train_tensor = torch.tensor(
            train_dataset.inputs,
            dtype=dtype,
            device=device,
        )
        inducing_inputs = initialize_inducing_points(
            X_train_tensor,
            num_inducing=_arg("sip_num_inducing", 100),
            method=_arg("sip_inducing_method", "kmeans"),
            seed=args.seed,
        )
        learn_prior = bool(_arg("sip_learn_prior", True))
        prior_fn = BayesianNN(
            input_dim=train_dataset.input_dim,
            num_samples=_arg("sip_num_prior_samples", 512),
            structure=args.hidden_dims,
            activation=ACTIVATIONS[args.activation],
            output_dim=train_dataset.output_dim,
            layer_model=BayesLinear,
            dropout=args.dropout,
            fix_random_noise=_arg("sip_fix_random_noise", False),
            zero_mean_prior=not learn_prior,
            weight_log_sigma_init=0.0,
            device=device,
            seed=args.seed + 1,
            dtype=dtype,
        )
        if not learn_prior and hasattr(prior_fn, "freeze_parameters"):
            prior_fn.freeze_parameters()
        return SIP(
            generative_function=prior_fn,
            inducing_inputs=inducing_inputs,
            output_dim=train_dataset.output_dim,
            likelihood="regression",
            num_data=len(train_dataset),
            num_prior_samples=_arg("sip_num_prior_samples", 512),
            num_train_samples=_arg("sip_num_train_samples", None),
            num_eval_samples=_arg("sip_num_eval_samples", 200),
            bb_alpha=args.bb_alpha,
            beta=_arg("sip_beta", 1.0),
            beta_warmup_steps=_arg("sip_beta_warmup_steps", 0),
            learn_inducing=_arg("sip_learn_inducing", False),
            detach_covariances=_arg("sip_detach_covariances", False),
            critic_hidden_dim=_arg("sip_critic_hidden_dim", 50),
            critic_lr=_arg("sip_critic_lr", 1e-3),
            critic_steps=_arg("sip_critic_steps", 1),
            posterior_noise_dim=_arg("sip_posterior_noise_dim", 100),
            posterior_hidden_dim=_arg("sip_posterior_hidden_dim", 50),
            posterior_depth=_arg("sip_posterior_depth", 2),
            fresh_prior_samples=not _arg("sip_fix_random_noise", False),
            y_mean=train_dataset.targets_mean,
            y_std=train_dataset.targets_std,
            jitter=_arg("sip_jitter", 1e-5),
            log_variance_init=_arg("sip_log_variance_init", -5.0),
            min_log_variance=_arg("sip_min_log_variance", None),
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    if model_type == "gmvip":
        train_inputs = torch.as_tensor(
            train_dataset.inputs,
            dtype=dtype,
            device=device,
        )
        learn_prior = bool(_arg("gmvip_learn_prior", False))
        inducing_points = initialize_inducing_points(
            train_inputs,
            num_inducing=_arg("gmvip_num_inducing", 32),
            method=_arg("gmvip_inducing_method", "kmeans"),
            seed=args.seed + 31,
        )
        prior_samples = max(
            int(_arg("gmvip_num_operator_bank_samples", 256)),
            int(_arg("gmvip_num_train_samples", 16)),
            2,
        )
        prior = BayesianNN(
            input_dim=train_dataset.input_dim,
            num_samples=prior_samples,
            structure=args.hidden_dims,
            activation=ACTIVATIONS[args.activation],
            output_dim=train_dataset.output_dim,
            layer_model=BayesLinear,
            dropout=args.dropout,
            fix_random_noise=True,
            zero_mean_prior=not learn_prior,
            weight_log_sigma_init=_arg("gmvip_weight_log_sigma_init", 0.0),
            device=device,
            seed=args.seed + 1,
            dtype=dtype,
        )
        if not learn_prior and hasattr(prior, "freeze_parameters"):
            prior.freeze_parameters()
        model = GeneralizedMatheronVIP(
            base_prior=prior,
            inducing_points=inducing_points,
            operator_type=_arg("gmvip_operator_type", "rbf"),
            posterior_type=_arg("gmvip_posterior_type", "gaussian"),
            num_operator_bank_samples=_arg("gmvip_num_operator_bank_samples", 256),
            learn_noise=_arg("gmvip_learn_noise", True),
            init_log_noise=_arg("gmvip_init_log_noise", -2.5),
            min_log_noise=_arg("gmvip_min_log_noise", -5.0),
            max_log_noise=_arg("gmvip_max_log_noise", None),
            freeze_base_prior=not learn_prior,
            detach_prior_samples=not learn_prior,
            detach_operator_prior_grad=_arg("gmvip_detach_operator_prior_grad", False),
            jitter=_arg("gmvip_jitter", 1e-5),
            shrinkage=_arg("gmvip_shrinkage", 0.02),
            learn_Z=_arg("gmvip_learn_Z", False),
            learn_kernel=_arg("gmvip_learn_kernel", True),
            ard=_arg("gmvip_ard", True),
            init_lengthscale=_arg("gmvip_init_lengthscale", "median"),
            init_outputscale=_arg("gmvip_init_outputscale", "prior_marginal"),
            inducing_scale=_arg("gmvip_inducing_scale", "prior_cholesky"),
            mean_mode=_arg("gmvip_mean_mode", "prior_sample"),
            posterior_init_mean=_arg("gmvip_posterior_init_mean", 0.0),
            posterior_init_log_std=_arg("gmvip_posterior_init_log_std", 0.0),
            posterior_min_log_std=_arg("gmvip_posterior_min_log_std", -8.0),
            posterior_max_log_std=_arg("gmvip_posterior_max_log_std", None),
            flow_depth=_arg("gmvip_flow_depth", 4),
            flow_hidden_dim=_arg("gmvip_flow_hidden_dim", 128),
            flow_num_layers=_arg("gmvip_flow_num_layers", 2),
            flow_dropout=_arg("gmvip_flow_dropout", 0.0),
            flow_scale_bound=_arg("gmvip_flow_scale_bound", 2.0),
            antithetic_samples=_arg("gmvip_antithetic_samples", True),
            num_data=len(train_dataset),
            num_train_samples=_arg("gmvip_num_train_samples", 16),
            beta=_arg("gmvip_beta", 1.0),
            beta_warmup_steps=_arg("gmvip_beta_warmup_steps", 0),
            data_alpha=_arg("gmvip_data_alpha", 0.0),
            max_grad_norm=_arg("gmvip_max_grad_norm", None),
            operator_bank_seed=args.seed + 101,
        )
        model.register_buffer(
            "y_mean",
            torch.as_tensor(train_dataset.targets_mean, dtype=dtype, device=device),
        )
        model.register_buffer(
            "y_std",
            torch.as_tensor(train_dataset.targets_std, dtype=dtype, device=device),
        )
        return model

    train_generator_prior = True
    if model_type == "vip":
        train_generator_prior = bool(_arg("vip_learn_prior", True))
    elif model_type == "ftip":
        train_generator_prior = bool(_arg("ftip_learn_prior", True))

    gen_fn = BayesianNN(
        input_dim=train_dataset.input_dim,
        num_samples=args.regression_coeffs,
        structure=args.hidden_dims,
        activation=ACTIVATIONS[args.activation],
        output_dim=train_dataset.output_dim,
        layer_model=BayesLinear,
        dropout=args.dropout,
        zero_mean_prior=not train_generator_prior,
        weight_log_sigma_init=_arg("weight_log_sigma_init", 0.0),
        device=device,
        seed=args.seed,
        dtype=dtype,
    )
    if not train_generator_prior:
        gen_fn.freeze_parameters()

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
                kernel_amp=args.fbnn_gp_kernel_amp,
                kernel_length=args.fbnn_gp_kernel_length,
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
                weight_log_sigma_init=_arg("weight_log_sigma_init", 0.0),
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
            device=device,
            dtype=dtype,
        )
        model = FTIP(**common, flow=flow, num_samples=args.num_samples)

    return model


def _build_flow(args, input_dim, device, dtype):
    """Construct an FTIP flow based on ``args.flow_type``."""
    common = dict(
        depth=args.flow_depth, input_dim=input_dim, device=device, dtype=dtype, seed=args.seed
    )
    if args.flow_type == "affine":
        return CouplingFlow(**common)
    if args.flow_type == "spline":
        return SplineCouplingFlow(**common, num_bins=args.flow_num_bins, B=args.flow_domain)
    if args.flow_type == "spline_1x1":
        return SplineCoupling1x1Flow(**common, num_bins=args.flow_num_bins, B=args.flow_domain)
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
    F = model.predict_f_samples(xb, num_samples=S)  # (S, N, D)
    mean = F * model.y_std + model.y_mean
    sigma = torch.sqrt(torch.exp(model.log_variance)) * model.y_std
    std = sigma.expand_as(mean)
    return mean, std


def _tfsvi_pred_components(model, xb, S):
    """Same Gaussian-mixture form as FBNN: each parameter sample defines
    N(f_s(x)*y_std + y_mean, exp(log_var)*y_std^2) on the original scale.
    """
    F = model.predict_f_samples(xb, num_samples=S)  # (S, N, D)
    mean = F * model.y_std + model.y_mean
    sigma = torch.sqrt(torch.exp(model.log_variance)) * model.y_std
    std = sigma.expand_as(mean)
    return mean, std


def _gmvip_pred_components(model, xb, S):
    """GMVIP predictive mixture on the original target scale."""
    F = model.predict_samples(xb, num_samples=S, noisy=False)  # (S, N, 1)
    y_mean = getattr(model, "y_mean", torch.zeros(1, dtype=F.dtype, device=F.device))
    y_std = getattr(model, "y_std", torch.ones(1, dtype=F.dtype, device=F.device))
    mean = F * y_std + y_mean
    std = (model.noise_std * y_std).expand_as(mean)
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
        elif model_type == "gmvip":
            S = args.gmvip_num_eval_samples
        elif model_type == "sip":
            S = args.sip_num_eval_samples
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
            xb, yb = x[i : i + batch_size], y[i : i + batch_size]
            if model_type == "ftip":
                mean, std = model.forward_with_coefficients(xb, a)
                std = std.unsqueeze(-1).expand_as(mean)
                metrics.update(
                    yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False
                )
            elif model_type == "fbnn":
                mean, std = _fbnn_pred_components(model, xb)
                metrics.update(
                    yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False
                )
            elif model_type == "tfsvi":
                mean, std = _tfsvi_pred_components(model, xb, args.tfsvi_num_eval_samples)
                metrics.update(
                    yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False
                )
            elif model_type == "mfvi":
                mean, std = _tfsvi_pred_components(model, xb, args.mfvi_num_eval_samples)
                metrics.update(
                    yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False
                )
            elif model_type == "gmvip":
                mean, std = _gmvip_pred_components(model, xb, args.gmvip_num_eval_samples)
                metrics.update(
                    yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False
                )
            else:
                mean, std = model(xb)
                metrics.update(
                    yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=False
                )
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
            xb = x[i : i + batch_size]
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
            xb, yb = x[i : i + batch_size], y[i : i + batch_size]
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
            elif model_type == "gmvip":
                S_light = min(64, args.gmvip_num_eval_samples)
                mean, std = _gmvip_pred_components(model, xb, S_light)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
            else:
                mean, std = model(xb)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
    d = metrics.get_dict()
    model.train()
    return {"RMSE": d["RMSE"], "NLL": d["NLL"]}


def train_with_metrics(
    model,
    train_loader,
    train_test_dataset,
    validation_dataset,
    args,
    lr=None,
    epochs=None,
    iterations=None,
    model_type=None,
    desc="Training",
):
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
    optimizer_params = (
        model.vi_parameters() if hasattr(model, "vi_parameters") else model.parameters()
    )
    optimizer = torch.optim.Adam(optimizer_params, lr=lr, weight_decay=wd)

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
    step_offset = int(getattr(args, "resume_step_offset", 0) or 0)
    resume_checkpoint = getattr(model, "_resume_checkpoint", None)
    if resume_checkpoint is not None:
        step_offset = restore_training_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
        )

    # TFSVI samples its KL context set from `model._train_inputs`; populate
    # it here since we bypass `model.fit()` and call `_train_step` directly.
    if model_type == "tfsvi":
        all_X = [inp for inp, _ in train_loader]
        model._train_inputs = torch.cat(all_X, dim=0).to(device)

    if hasattr(model, "prepare_for_training"):
        model.prepare_for_training(train_loader)

    model.train()
    disable_tqdm = os.environ.get("IPZOO_DISABLE_TQDM", "").lower() in {
        "1",
        "true",
        "yes",
    }

    if iterations is not None:
        total = iterations
        data_stream = infinite_loader(train_loader)
        iters_per_epoch = len(train_loader)
        for _ in range(step_offset % iters_per_epoch):
            next(data_stream)
        loop = tqdm(range(total), unit=" iter", desc=desc, disable=disable_tqdm)

        for i in loop:
            global_step = step_offset + i + 1
            inputs, target = next(data_stream)
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = model._train_step(optimizer, inputs, target)
            losses.append(loss.item())
            wandb_log_train_step(
                args,
                global_step,
                loss,
                optimizer=optimizer,
                model=model,
                model_type=model_type,
            )

            if scheduler is not None and (i + 1) % iters_per_epoch == 0:
                scheduler.step()

            if (i + 1) % args.eval_every == 0:
                metrics_history["iterations"].append(global_step)
                train_eval = evaluate_light(model, train_test_dataset, args, model_type=model_type)
                validation_eval = evaluate_light(
                    model, validation_dataset, args, model_type=model_type
                )
                metrics_history["train"].append(train_eval)
                metrics_history["validation"].append(validation_eval)
                wandb_log_eval(global_step, train_eval, validation_eval)

            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    else:
        total_epochs = epochs
        loop = tqdm(range(total_epochs), unit=" epoch", desc=desc, disable=disable_tqdm)
        it = 0
        for _ in loop:
            for inputs, target in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                loss = model._train_step(optimizer, inputs, target)
                losses.append(loss.item())
                it += 1
                global_step = step_offset + it
                wandb_log_train_step(
                    args,
                    global_step,
                    loss,
                    optimizer=optimizer,
                    model=model,
                    model_type=model_type,
                )

                if it % args.eval_every == 0:
                    metrics_history["iterations"].append(global_step)
                    train_eval = evaluate_light(
                        model, train_test_dataset, args, model_type=model_type
                    )
                    validation_eval = evaluate_light(
                        model, validation_dataset, args, model_type=model_type
                    )
                    metrics_history["train"].append(train_eval)
                    metrics_history["validation"].append(validation_eval)
                    wandb_log_eval(global_step, train_eval, validation_eval)

            if scheduler is not None:
                scheduler.step()
            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    # Extract training diagnostics from model buffers (model-specific).
    diagnostics = {}
    for attr in (
        "KLs",
        "bb_alphas",
        "prior_regularizers",
        "data_terms",
        "function_terms",
        "raw_KLs",
        "kl_floors",
        "betas",
        "kinetics",
        "score_losses",
        "l2_terms",
        "kl_mins",
        "kl_maxs",
        "kl_stds",
        "divergence_means",
        "score_dot_means",
        "vector_field_norms",
        "mean_abs_scores",
        "mean_abs_vs",
        "mean_abs_score_dot_vs",
        "u0_abs_maxes",
        "u1_abs_maxes",
        "ut_abs_maxes",
        "transport_rel_changes",
        "sliced_flow_prior_nlls",
        "sliced_flow_posterior_nlls",
        "sliced_flow_prior_update_counts",
        "sliced_flow_posterior_update_counts",
        "sliced_flow_kl_raws",
    ):
        if hasattr(model, attr):
            diagnostics[attr] = [float(v) for v in getattr(model, attr)]
    if model_type == "ftip":
        diagnostics["base_KLs"] = [float(v) for v in model.base_KLs]
        diagnostics["flow_ldj"] = [float(v) for v in model.flow_ldj]

    return losses, metrics_history, diagnostics, optimizer, scheduler


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


def _compact_float_tag(value):
    if value is None:
        return "none"
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def _variant_tag(args, model_type):
    """Filename tag for variants that share the same top-level model name."""
    if model_type == "gmvip":
        tag = (
            f"_{getattr(args, 'gmvip_operator_type', 'rbf')}"
            f"_{getattr(args, 'gmvip_posterior_type', 'gaussian')}"
            f"_{getattr(args, 'gmvip_mean_mode', 'prior_sample')}"
            f"_{getattr(args, 'gmvip_inducing_scale', 'prior_cholesky')}"
            f"_Z{getattr(args, 'gmvip_num_inducing', 32)}"
            f"_{getattr(args, 'gmvip_inducing_method', 'kmeans')}"
            f"_S{getattr(args, 'gmvip_num_train_samples', 16)}"
            f"_b{_compact_float_tag(getattr(args, 'gmvip_beta', 1.0))}"
            f"_a{_compact_float_tag(getattr(args, 'gmvip_data_alpha', 0.0))}"
            f"_wls{_compact_float_tag(getattr(args, 'gmvip_weight_log_sigma_init', 0.0))}"
        )
        if getattr(args, "gmvip_learn_prior", False):
            tag = f"{tag}_learnprior"
        if getattr(args, "gmvip_learn_Z", False):
            tag = f"{tag}_learnZ"
        if not getattr(args, "gmvip_learn_kernel", True):
            tag = f"{tag}_fixedK"
        if getattr(args, "gmvip_learn_prior", False) and getattr(
            args, "gmvip_detach_operator_prior_grad", False
        ):
            tag = f"{tag}_opdetprior"
        if getattr(args, "gmvip_posterior_type", None) == "realnvp":
            tag = (
                f"{tag}_flowd{getattr(args, 'gmvip_flow_depth', 4)}"
                f"_flowh{getattr(args, 'gmvip_flow_hidden_dim', 128)}"
            )
        return tag
    if model_type == "vip":
        return "_learnprior" if getattr(args, "vip_learn_prior", True) else "_fixedprior"
    if model_type == "ftip":
        return "_learnprior" if getattr(args, "ftip_learn_prior", True) else "_fixedprior"
    if model_type == "sip":
        train_samples = getattr(args, "sip_num_train_samples", None)
        if train_samples is None:
            train_samples = getattr(args, "sip_num_prior_samples", 512)
        tag = (
            f"_Z{getattr(args, 'sip_num_inducing', 100)}"
            f"_{getattr(args, 'sip_inducing_method', 'kmeans')}"
            f"_S{getattr(args, 'sip_num_prior_samples', 512)}"
            f"_Strain{train_samples}"
            f"_critic{getattr(args, 'sip_critic_steps', 1)}"
            f"_beta{_compact_float_tag(getattr(args, 'sip_beta', 1.0))}"
        )
        tag = f"{tag}_learnZ" if getattr(args, "sip_learn_inducing", False) else f"{tag}_fixedZ"
        tag = f"{tag}_learnprior" if getattr(args, "sip_learn_prior", True) else f"{tag}_fixedprior"
        tag = (
            f"{tag}_fixednoise"
            if getattr(args, "sip_fix_random_noise", False)
            else f"{tag}_freshnoise"
        )
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


def _build_result(
    dataset_name,
    model_type,
    model,
    args,
    train_loader,
    train_test_dataset,
    test_dataset,
    lr=None,
    epochs=None,
    iterations=None,
    desc="Training",
):
    """Train a model, evaluate it, and return the result dict."""
    if lr is None:
        lr = args.lr

    if args.compile:
        try:
            if hasattr(model, "nelbo"):
                model.nelbo = torch.compile(model.nelbo)
        except Exception:
            print("  [warn] torch.compile unavailable, running without it")

    t0 = time.time()
    losses, metrics_history, diagnostics, optimizer, scheduler = train_with_metrics(
        model,
        train_loader,
        train_test_dataset,
        test_dataset,
        args,
        lr=lr,
        epochs=epochs,
        iterations=iterations,
        model_type=model_type,
        desc=desc,
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
        "weight_log_sigma_init": args.weight_log_sigma_init,
        "regression_coeffs": args.regression_coeffs,
        "bb_alpha": args.bb_alpha,
        "use_prior_regularizer": args.use_prior_regularizer,
        "regularizer_mode": args.regularizer_mode,
        "prior_regularizer_scaler": args.prior_regularizer_scaler,
        "cosine_annealing": args.cosine_annealing,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "resume_step_offset": args.resume_step_offset,
    }
    if model_type == "ftip":
        hyperparameters["flow_depth"] = args.flow_depth
        hyperparameters["flow_type"] = args.flow_type
        hyperparameters["flow_num_bins"] = args.flow_num_bins
        hyperparameters["flow_domain"] = args.flow_domain
        hyperparameters["num_samples"] = args.num_samples
        hyperparameters["eval_samples"] = args.eval_samples
        hyperparameters["ftip_learn_prior"] = args.ftip_learn_prior
        hyperparameters["auto_warm_start"] = args.auto_warm_start
    if model_type == "mfvi":
        hyperparameters["mfvi_num_eval_samples"] = args.mfvi_num_eval_samples
    if model_type == "vip":
        hyperparameters["vip_learn_prior"] = args.vip_learn_prior
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
        hyperparameters["fbnn_gp_kernel_amp"] = args.fbnn_gp_kernel_amp
        hyperparameters["fbnn_gp_kernel_length"] = args.fbnn_gp_kernel_length
        hyperparameters["fbnn_num_measurement"] = args.fbnn_num_measurement
        hyperparameters["fbnn_num_context"] = args.fbnn_num_context
        hyperparameters["fbnn_context_std"] = args.fbnn_context_std
        hyperparameters["fbnn_lambda_kl"] = args.fbnn_lambda_kl
        hyperparameters["fbnn_num_eval_samples"] = args.fbnn_num_eval_samples
    if model_type == "sip":
        hyperparameters.update(
            {
                "sip_layer_model": "BayesLinear",
                "sip_num_inducing": args.sip_num_inducing,
                "sip_inducing_method": args.sip_inducing_method,
                "sip_num_prior_samples": args.sip_num_prior_samples,
                "sip_num_train_samples": args.sip_num_train_samples,
                "sip_num_eval_samples": args.sip_num_eval_samples,
                "sip_beta": args.sip_beta,
                "sip_beta_warmup_steps": args.sip_beta_warmup_steps,
                "sip_learn_inducing": args.sip_learn_inducing,
                "sip_learn_prior": args.sip_learn_prior,
                "sip_detach_covariances": args.sip_detach_covariances,
                "sip_jitter": args.sip_jitter,
                "sip_log_variance_init": args.sip_log_variance_init,
                "sip_min_log_variance": args.sip_min_log_variance,
                "sip_fix_random_noise": args.sip_fix_random_noise,
                "sip_critic_hidden_dim": args.sip_critic_hidden_dim,
                "sip_critic_lr": args.sip_critic_lr,
                "sip_critic_steps": args.sip_critic_steps,
                "sip_posterior_noise_dim": args.sip_posterior_noise_dim,
                "sip_posterior_hidden_dim": args.sip_posterior_hidden_dim,
                "sip_posterior_depth": args.sip_posterior_depth,
            }
        )
    if model_type == "gmvip":
        hyperparameters.update(
            {
                "gmvip_layer_model": "BayesLinear",
                "gmvip_operator_type": args.gmvip_operator_type,
                "gmvip_posterior_type": args.gmvip_posterior_type,
                "gmvip_num_inducing": args.gmvip_num_inducing,
                "gmvip_inducing_method": args.gmvip_inducing_method,
                "gmvip_num_operator_bank_samples": args.gmvip_num_operator_bank_samples,
                "gmvip_num_train_samples": args.gmvip_num_train_samples,
                "gmvip_num_eval_samples": args.gmvip_num_eval_samples,
                "gmvip_antithetic_samples": args.gmvip_antithetic_samples,
                "gmvip_beta": args.gmvip_beta,
                "gmvip_beta_warmup_steps": args.gmvip_beta_warmup_steps,
                "gmvip_data_alpha": args.gmvip_data_alpha,
                "gmvip_weight_log_sigma_init": args.gmvip_weight_log_sigma_init,
                "gmvip_learn_prior": args.gmvip_learn_prior,
                "gmvip_detach_operator_prior_grad": args.gmvip_detach_operator_prior_grad,
                "gmvip_learn_noise": args.gmvip_learn_noise,
                "gmvip_init_log_noise": args.gmvip_init_log_noise,
                "gmvip_min_log_noise": args.gmvip_min_log_noise,
                "gmvip_max_log_noise": args.gmvip_max_log_noise,
                "gmvip_learn_Z": args.gmvip_learn_Z,
                "gmvip_learn_kernel": args.gmvip_learn_kernel,
                "gmvip_ard": args.gmvip_ard,
                "gmvip_init_lengthscale": args.gmvip_init_lengthscale,
                "gmvip_init_outputscale": args.gmvip_init_outputscale,
                "gmvip_inducing_scale": args.gmvip_inducing_scale,
                "gmvip_mean_mode": args.gmvip_mean_mode,
                "gmvip_jitter": args.gmvip_jitter,
                "gmvip_shrinkage": args.gmvip_shrinkage,
                "gmvip_posterior_init_mean": args.gmvip_posterior_init_mean,
                "gmvip_posterior_init_log_std": args.gmvip_posterior_init_log_std,
                "gmvip_posterior_min_log_std": args.gmvip_posterior_min_log_std,
                "gmvip_posterior_max_log_std": args.gmvip_posterior_max_log_std,
                "gmvip_flow_depth": args.gmvip_flow_depth,
                "gmvip_flow_hidden_dim": args.gmvip_flow_hidden_dim,
                "gmvip_flow_num_layers": args.gmvip_flow_num_layers,
                "gmvip_flow_dropout": args.gmvip_flow_dropout,
                "gmvip_flow_scale_bound": args.gmvip_flow_scale_bound,
                "gmvip_max_grad_norm": args.gmvip_max_grad_norm,
            }
        )
    if model_type == "map":
        hyperparameters["map_l2"] = args.map_l2
        hyperparameters["map_log_variance_init"] = args.map_log_variance_init

    restored_step = int(
        getattr(model, "_resume_checkpoint", {}).get(
            "global_step", getattr(args, "resume_step_offset", 0) or 0
        )
    )
    final_step = restored_step + len(losses)

    result = {
        "dataset": dataset_name,
        "model": model_type,
        "hyperparameters": hyperparameters,
        "final_step": final_step,
        "train_time_s": round(train_time, 2),
        "train": train_metrics,
        "test": test_metrics,
        "prior": prior_stats,
        "losses": losses,
        "metrics_history": metrics_history,
        "diagnostics": diagnostics,
    }

    for split, m in [("Train", train_metrics), ("Test", test_metrics)]:
        print(
            f"  {model_type.upper()} {split}: RMSE={m['RMSE']:.4f}  NLL={m['NLL']:.4f}"
            f"  CRPS={m['CRPS']:.4f}  CQM={m['CQM']:.4f}"
        )
    print(
        f"  Prior: mean(Var[f])={prior_stats['var_mean']:.6f}  "
        f"var(Var[f])={prior_stats['var_var']:.6f}"
    )
    print(f"  Time: {train_time:.1f}s")

    if args.test_ood:
        ood_metrics = evaluate_ood(model, test_dataset, args, model_type=model_type, seed=args.seed)
        result["test"].update(ood_metrics)
        result["ood"] = ood_metrics
        print(
            f"  OOD: AUROC={ood_metrics['AUROC']:.4f}  "
            f"H(in)={ood_metrics['entropy_id_mean']:.4f}+/-{ood_metrics['entropy_id_std']:.4f}  "
            f"H(ood)={ood_metrics['entropy_ood_mean']:.4f}+/-{ood_metrics['entropy_ood_std']:.4f}"
        )

    wandb_log_result(result, step=final_step)

    # Save checkpoint
    if args.save_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt = _ckpt_path(args, dataset_name, model_type)
        checkpoint = build_training_checkpoint(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=final_step,
            arguments=vars(args),
        )
        save_training_checkpoint(ckpt, checkpoint)
        print(f"  Checkpoint: {ckpt}")

    return result, model


def run_single(dataset_name, args):
    """Run benchmark on a single dataset. Returns a list of result dicts."""
    wandb_run = init_wandb_run(
        args,
        config={"dataset_name": dataset_name, "model_type": args.model},
    )

    try:
        return _run_single(dataset_name, args)
    finally:
        finish_wandb_run(wandb_run)


def _run_single(dataset_name, args):
    """Run benchmark on a single dataset. Returns a list of result dicts."""
    use_warm_start = args.model == "ftip" and args.auto_warm_start

    header = "FTIP (warm-start from VIP)" if use_warm_start else args.model.upper()
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}  |  Model: {header}")
    print(f"{'=' * 60}")

    dataset = get_dataset(dataset_name)
    train_dataset, train_test_dataset, test_dataset = dataset.get_split(args.test_size, args.seed)

    use_cuda = args.device and "cuda" in args.device
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=0,
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
            dataset_name,
            "vip",
            vip_model,
            args,
            train_loader,
            train_test_dataset,
            test_dataset,
            lr=args.vip_lr,
            epochs=vip_ep,
            iterations=vip_it,
            desc="VIP pre-training",
        )

        # Save warm-start checkpoint for FTIP (before continuing)
        vip_ws_state = {n: p.data.clone() for n, p in vip_model.named_parameters()}

        # Continue training VIP for the remaining budget (= ftip budget) to get fair baseline
        print(f"\n  VIP baseline: continuing for {ftip_phase_str} (total={budget_str})")
        vip_baseline_result, _ = _build_result(
            dataset_name,
            "vip",
            vip_model,
            args,
            train_loader,
            train_test_dataset,
            test_dataset,
            lr=args.vip_lr,
            epochs=ftip_ep,
            iterations=ftip_it,
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
            dataset_name,
            "ftip",
            ftip_model,
            args,
            train_loader,
            train_test_dataset,
            test_dataset,
            lr=args.ftip_lr,
            epochs=ftip_ep,
            iterations=ftip_it,
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
        print(
            f"\n  Delta (FTIP - VIP baseline): RMSE={ftip_rmse - vip_rmse:+.4f}  NLL={ftip_nll - vip_nll:+.4f}"
        )
        print(f"  FTIP total time: {ftip_result['total_time_s']:.1f}s")

    else:
        # ── Single-model training (VIP or cold-start FTIP) ──
        model = build_model(args, train_dataset)

        if args.resume_from_checkpoint:
            device = torch.device(args.device)
            checkpoint = load_training_checkpoint(
                args.resume_from_checkpoint,
                map_location=device,
            )
            state = checkpoint["model"]
            if "anchors" in state and hasattr(model, "anchors"):
                anchors = state["anchors"].to(
                    device=device,
                    dtype=getattr(model, "dtype", state["anchors"].dtype),
                )
                model.anchors = anchors.detach().clone()
                if hasattr(model, "n_anchors"):
                    model.n_anchors = int(anchors.shape[0])
                if hasattr(model, "_user_anchors"):
                    model._user_anchors = True
            model._resume_checkpoint = checkpoint
            print(
                f"  Resumed {args.model.upper()} from {args.resume_from_checkpoint} "
                f"(global step={checkpoint['global_step']})"
            )

        # Manual warm-start from external checkpoint
        if args.warm_start_from and args.model == "ftip":
            device = torch.device(args.device)
            vip_model = build_model(args, train_dataset, model_type="vip")
            vip_model.load_state_dict(
                load_warm_start_state(args.warm_start_from, map_location=device)
            )
            model.warm_start_from_vip(vip_model, learnable_affine=args.learnable_affine)
            del vip_model
            print(
                f"  Warm-started from {args.warm_start_from} "
                f"(affine learnable={args.learnable_affine})"
            )

        result, _ = _build_result(
            dataset_name,
            args.model,
            model,
            args,
            train_loader,
            train_test_dataset,
            test_dataset,
        )
        if args.warm_start_from:
            result["warm_start"] = {
                "enabled": True,
                "warm_start_from": args.warm_start_from,
                "learnable_affine": args.learnable_affine,
            }
        if args.resume_from_checkpoint:
            result["resume"] = {
                "enabled": True,
                "resume_from_checkpoint": args.resume_from_checkpoint,
                "resume_step_offset": result["final_step"] - len(result["losses"]),
            }
        results.append(result)

    return results


def run_from_args(args, *, dataset_names=None, default_iters=None):
    dataset_names = list(UCI_REGRESSION_DATASETS if dataset_names is None else dataset_names)
    default_iters = DEFAULT_UCI_ITERS if default_iters is None else default_iters
    torch.manual_seed(args.seed)

    if args.dataset == "all":
        datasets = dataset_names
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
                run_args.iterations = default_iters.get(ds, 30_000)
                run_args.epochs = None
                if run_args.vip_epochs is None and run_args.vip_iterations is None:
                    run_args.vip_iterations = run_args.iterations
                    run_args.vip_epochs = None
            # Keep alpha=0.0 in filenames/metadata unless the user explicitly
            # sets it.
            if not run_args._bb_alpha_user_supplied:
                run_args.bb_alpha = 0.0
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
                print(
                    f"\n  Skipping {run_args.model}/{ds} seed={run_args.seed}: {out_path} already exists"
                )
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

    return all_results


def main():
    args = parse_args()
    return run_from_args(args)


if __name__ == "__main__":
    main()
