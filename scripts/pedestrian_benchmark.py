"""Pedestrian-level split benchmark.

Runs the pedestrian trajectory benchmark with a group split by pedestrian id.
The default target is future path length. Each run writes a JSON result file.

Example:
    python -m scripts.pedestrian_benchmark --model ftip
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

from src.utils.dataset import Pedestrian_Dataset
from src.utils.metrics import MetricsRegression, TrajectoryMetrics
from src.utils.utils import infinite_loader
from src.priors.generative_functions import (
    BayesianLSTM, BayesLinear, GP,
)
from src.flows import CouplingFlow, SplineCouplingFlow, SplineCoupling1x1Flow
from src.vip import VIP
from src.ftip import FTIP
from src.mfvi import MFVI
from src.fbnn import FBNN
from src.tfsvi import TFSVI
from src.ap_fsvi import APFSVI
from scripts.benchmark_utils import (
    add_wandb_args,
    finish_wandb_run,
    init_wandb_run,
    print_comparison,
    save_comparison,
    wandb_log_eval,
    wandb_log_result,
    wandb_log_train_step,
    wandb_run_name,
)


MODELS = ["vip", "ftip", "mfvi", "fbnn", "tfsvi", "ap_fsvi"]
LAYER_MODELS = {
    "BayesLinear": BayesLinear,
}


# ---------------------------------------------------------------------------
# Data adapter
# ---------------------------------------------------------------------------

def _build_targets(obs, pred, target_mode, target_horizon="full"):
    """Compute flat target vector and origin from obs/pred trajectories.

    Parameters
    ----------
    obs : (T_obs, 2) tensor
    pred : (T_pred, 2) tensor
    target_mode : "absolute", "displacement", or "relative"
    target_horizon : "full" (predict all T_pred steps) or "last" (only the
        final position; output is 2-dim). Not supported for "displacement".

    Returns
    -------
    target : flat target vector (T_pred*2 for "full", 2 for "last")
    origin : (2,) tensor — last observed position (for reconstruction)
    """
    origin = obs[-1]  # (2,)
    if target_mode == "path_length":
        # 1-D scalar target: total length of the future trajectory, measured
        # from the last observed position through all future points.
        full = torch.cat([origin.unsqueeze(0), pred], dim=0)  # (T_pred+1, 2)
        steps = (full[1:] - full[:-1]).norm(dim=-1)           # (T_pred,)
        return steps.sum().reshape(1), origin                 # (1,)
    if target_mode == "speed":
        # T_pred-D target: per-step scalar speed ||delta|| from previous pos.
        full = torch.cat([origin.unsqueeze(0), pred], dim=0)  # (T_pred+1, 2)
        steps = (full[1:] - full[:-1]).norm(dim=-1)           # (T_pred,)
        return steps, origin
    if target_horizon == "last":
        if target_mode == "displacement":
            raise ValueError(
                "target_horizon='last' is not supported for target_mode='displacement'"
            )
        last = pred[-1]  # (2,)
        if target_mode == "relative":
            return (last - origin).reshape(-1), origin
        return last.reshape(-1), origin
    if target_mode == "displacement":
        # Displacements: first relative to last obs, rest relative to previous pred
        full = torch.cat([origin.unsqueeze(0), pred], dim=0)  # (T_pred+1, 2)
        deltas = full[1:] - full[:-1]  # (T_pred, 2)
        return deltas.reshape(-1), origin
    elif target_mode == "relative":
        # Offsets from last observed position
        return (pred - origin).reshape(-1), origin
    else:
        return pred.reshape(-1), origin


class PedestrianTrainAdapter:
    """Wraps Pedestrian_Dataset samples into the flat (input, target) format
    that VIP/FTIP expect, with per-dimension normalization.

    Parameters
    ----------
    target_mode : "absolute" or "displacement"
        "absolute" — targets are raw future positions.
        "displacement" — targets are per-step deltas from previous position.
    """

    def __init__(self, dataset, indices, t_obs, t_pred, use_vel=False,
                 target_mode="absolute", target_horizon="full"):
        # use_vel=True: replace 2-D positions with 1-D scalar SPEED per step
        # (||obs[t] - obs[t-1]||, padding s[0]=s[1]). Rotation- and
        # translation-invariant. feature_dim=1.
        self.target_mode = target_mode
        self.target_horizon = target_horizon
        all_inputs = []
        all_targets = []
        all_origins = []
        for idx in indices:
            sample = dataset[idx]
            obs = sample["obs_traj"]  # (T_obs, 2)
            pred = sample["pred_traj"]  # (T_pred, 2)
            target, origin = _build_targets(obs, pred, target_mode, target_horizon)
            if use_vel:
                vel = torch.zeros_like(obs)
                vel[1:] = obs[1:] - obs[:-1]
                vel[0] = vel[1]
                speed = vel.norm(dim=-1, keepdim=True)   # (T_obs, 1)
                obs_feat = speed
            elif target_mode == "relative":
                obs_feat = obs - origin  # positions relative to last obs
            else:
                obs_feat = obs
            feat = obs_feat
            all_inputs.append(feat.reshape(-1))
            all_targets.append(target)
            all_origins.append(origin)

        all_inputs = torch.stack(all_inputs).numpy()
        all_targets = torch.stack(all_targets).numpy()
        self.origins = torch.stack(all_origins).numpy()  # (N, 2)

        # Normalization stats
        self.inputs_mean = np.mean(all_inputs, axis=0, keepdims=True)
        self.inputs_std = np.std(all_inputs, axis=0, keepdims=True) + 1e-6
        self.targets_mean = np.mean(all_targets, axis=0, keepdims=True)
        self.targets_std = np.std(all_targets, axis=0, keepdims=True) + 1e-6

        self.inputs = (all_inputs - self.inputs_mean) / self.inputs_std
        self.targets = (all_targets - self.targets_mean) / self.targets_std

        self.input_dim = self.inputs.shape[1]
        self.output_dim = self.targets.shape[1]
        self.n_samples = len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return self.n_samples


class PedestrianTestAdapter:
    """Test adapter: normalizes inputs only, keeps targets raw (denormalized).

    Matches the UCI Test_Dataset convention: inputs are normalized using
    training stats, but targets stay in original scale so that model.forward()
    (which denormalizes) and targets are in the same space for evaluation.
    """

    def __init__(self, dataset, indices, t_obs, t_pred, use_vel,
                 inputs_mean, inputs_std, targets_mean, targets_std,
                 target_mode="absolute", target_horizon="full"):
        self.target_mode = target_mode
        self.target_horizon = target_horizon
        all_inputs = []
        all_targets = []
        all_origins = []
        for idx in indices:
            sample = dataset[idx]
            obs = sample["obs_traj"]
            pred = sample["pred_traj"]
            target, origin = _build_targets(obs, pred, target_mode, target_horizon)
            if use_vel:
                vel = torch.zeros_like(obs)
                vel[1:] = obs[1:] - obs[:-1]
                vel[0] = vel[1]
                speed = vel.norm(dim=-1, keepdim=True)   # (T_obs, 1)
                obs_feat = speed
            elif target_mode == "relative":
                obs_feat = obs - origin
            else:
                obs_feat = obs
            feat = obs_feat
            all_inputs.append(feat.reshape(-1))
            all_targets.append(target)
            all_origins.append(origin)

        all_inputs = torch.stack(all_inputs).numpy()
        all_targets = torch.stack(all_targets).numpy()
        self.origins = torch.stack(all_origins).numpy()  # (N, 2)

        self.inputs_mean = inputs_mean
        self.inputs_std = inputs_std
        self.targets_mean = targets_mean
        self.targets_std = targets_std

        self.inputs = (all_inputs - inputs_mean) / inputs_std
        self.targets = all_targets  # raw, NOT normalized (matches UCI Test_Dataset)

        self.input_dim = self.inputs.shape[1]
        self.output_dim = self.targets.shape[1]
        self.n_samples = len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return self.n_samples


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def _build_lstm_gen_fn(args, train_adapter, num_samples, fix_random_noise=True,
                       seed_offset=0):
    """Build a BayesianLSTM with the given posterior MC sample count.

    ``fix_random_noise=True`` caches per-sample weight noise across forwards
    (required by FTIP/FBNN/TFSVI for sample consistency); ``False`` redraws
    noise every forward (standard for MFVI's MC ELBO).
    """
    feature_dim = 1 if args.use_vel else 2
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    return BayesianLSTM(
        num_samples=num_samples,
        input_dim=train_adapter.input_dim,
        output_dim=train_adapter.output_dim,
        t_obs=args.t_obs,
        feature_dim=feature_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        head_dims=args.head_dims,
        layer_model=BayesLinear,
        dropout=args.dropout,
        device=device,
        fix_random_noise=fix_random_noise,
        seed=args.seed + seed_offset,
        dtype=dtype,
    )


def _set_lstm_num_samples(gen_fn, S):
    """Mutate BayesianLSTM's MC sample count and regenerate cached weight noise.

    Mirrors the helper in ``scripts.synthetic_benchmark`` but uses
    ``gen_fn.head`` (the LSTM's Bayesian head; also exposed via the
    ``layers`` property). Used to coerce the gen_fn into deterministic-
    in-theta mode (S=1) for TFSVI.
    """
    old = gen_fn.num_samples
    gen_fn.num_samples = S
    for layer in gen_fn.head:
        if hasattr(layer, "num_samples"):
            layer.num_samples = S
        if (hasattr(layer, "fix_random_noise") and layer.fix_random_noise
                and S != old):
            layer.noise = layer.get_noise(first_call=True)


def build_model(args, train_adapter, model_type=None):
    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if model_type == "ap_fsvi":
        gen_args = copy.copy(args)
        gen_args.layer_model = "BayesLinear"
        if args.ap_fsvi_hidden_dims is not None:
            gen_args.head_dims = args.ap_fsvi_hidden_dims
        gen_fn = _build_lstm_gen_fn(
            gen_args,
            train_adapter,
            num_samples=args.ap_fsvi_num_samples,
            fix_random_noise=False,
        )
        return APFSVI(
            generative_function=gen_fn,
            input_dim=train_adapter.input_dim,
            output_dim=train_adapter.output_dim,
            likelihood="regression",
            num_data=len(train_adapter),
            num_samples=args.ap_fsvi_num_samples,
            num_prior_samples=(
                args.ap_fsvi_num_prior_samples or args.ap_fsvi_num_samples
            ),
            num_measurement=args.ap_fsvi_num_measurement,
            adaptive_measure_points=args.ap_fsvi_adaptive_measure_points,
            adaptive_measure_steps=args.ap_fsvi_adaptive_measure_steps,
            adaptive_measure_lr=args.ap_fsvi_adaptive_measure_lr,
            adaptive_measure_domain_limit=(
                args.ap_fsvi_adaptive_measure_domain_limit
            ),
            beta=args.ap_fsvi_beta,
            beta_start=args.ap_fsvi_beta_start,
            beta_warmup_steps=args.ap_fsvi_beta_warmup_steps,
            data_pretrain_steps=args.ap_fsvi_data_pretrain_steps,
            data_loss=args.ap_fsvi_data_loss,
            measurement_weights=args.ap_fsvi_measurement_weights,
            near_data_noise=args.ap_fsvi_near_data_noise,
            domain_std=args.ap_fsvi_domain_std,
            function_discrepancy=args.ap_fsvi_discrepancy,
            discrepancy_num_projections=args.ap_fsvi_discrepancy_projections,
            sample_knn_k=args.ap_fsvi_sample_knn_k,
            quantile_transport_k=args.ap_fsvi_quantile_transport_k,
            sinkhorn_epsilon=args.ap_fsvi_sinkhorn_epsilon,
            sinkhorn_iterations=args.ap_fsvi_sinkhorn_iterations,
            y_mean=train_adapter.targets_mean,
            y_std=train_adapter.targets_std,
            log_variance_init=args.ap_fsvi_log_variance_init,
            max_grad_norm=args.ap_fsvi_max_grad_norm,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    if model_type == "mfvi":
        # MC over weight samples per forward, so noise is regenerated every
        # call (fix_random_noise=False).
        gen_fn = _build_lstm_gen_fn(
            args, train_adapter, num_samples=args.mfvi_num_samples,
            fix_random_noise=False,
        )
        return MFVI(
            generative_function=gen_fn,
            output_dim=train_adapter.output_dim,
            likelihood="regression",
            num_data=len(train_adapter),
            num_samples=args.mfvi_num_samples,
            bb_alpha=args.mfvi_bb_alpha,
            y_mean=train_adapter.targets_mean,
            y_std=train_adapter.targets_std,
            device=device, dtype=dtype,
        )

    if model_type == "fbnn":
        # FBNN requires fix_random_noise=True so each of the S function
        # samples is consistent across measurement + batch points.
        gen_fn = _build_lstm_gen_fn(
            args, train_adapter, num_samples=args.fbnn_num_samples,
            fix_random_noise=True,
        )
        if args.fbnn_prior == "gp":
            prior = GP(
                input_dim=train_adapter.input_dim,
                output_dim=train_adapter.output_dim,
                inner_layer_dim=args.fbnn_gp_features,
                seed=args.seed + 1, device=device, dtype=dtype,
            )
        else:
            # BNN prior: same LSTM architecture, different seed, frozen below.
            prior = _build_lstm_gen_fn(
                args, train_adapter, num_samples=args.fbnn_num_samples,
                fix_random_noise=True, seed_offset=1,
            )
        return FBNN(
            generative_function=gen_fn,
            prior_function=prior,
            output_dim=train_adapter.output_dim,
            likelihood="regression",
            num_data=len(train_adapter),
            num_samples=args.fbnn_num_samples,
            num_measurement=args.fbnn_num_measurement,
            num_context=args.fbnn_num_context,
            context_std=args.fbnn_context_std,
            bb_alpha=args.fbnn_bb_alpha,
            lambda_kl=args.fbnn_lambda_kl,
            freeze_prior=args.fbnn_freeze_prior,
            y_mean=train_adapter.targets_mean,
            y_std=train_adapter.targets_std,
            device=device, dtype=dtype,
        )

    if model_type == "tfsvi":
        # TFSVI puts q(theta) over ALL parameters of the supplied gen_fn
        # (LSTM weights + Bayesian head's mu/raw_log_sigma). The gen_fn
        # forward must be deterministic in those parameters, so we set
        # num_samples=1 and keep fix_random_noise=True so the head's
        # cached weight-noise is frozen across forwards.
        gen_fn = _build_lstm_gen_fn(
            args, train_adapter, num_samples=1,
            fix_random_noise=True,
        )
        _set_lstm_num_samples(gen_fn, 1)
        # `structure` and `activation` are unused when generative_function
        # is supplied, but the signature requires them.
        return TFSVI(
            input_dim=train_adapter.input_dim,
            output_dim=train_adapter.output_dim,
            structure=args.head_dims,
            activation=torch.nn.Tanh(),
            likelihood="regression",
            num_data=len(train_adapter),
            sigma_prior=args.tfsvi_sigma_prior,
            num_samples=args.tfsvi_num_samples,
            bb_alpha=args.tfsvi_bb_alpha,
            S_ctx=args.tfsvi_S_ctx,
            K_ctx=args.tfsvi_K_ctx,
            y_mean=train_adapter.targets_mean,
            y_std=train_adapter.targets_std,
            generative_function=gen_fn,
            device=device, dtype=dtype,
        )

    # ---- VIP / FTIP path (unchanged) ----
    gen_fn = _build_lstm_gen_fn(
        args, train_adapter, num_samples=args.regression_coeffs,
        fix_random_noise=True,
    )

    common = dict(
        generative_function=gen_fn,
        num_regression_coeffs=args.regression_coeffs,
        output_dim=train_adapter.output_dim,
        likelihood="regression",
        num_data=len(train_adapter),
        bb_alpha=args.bb_alpha,
        use_prior_regularizer=args.use_prior_regularizer,
        regularizer_mode=args.regularizer_mode,
        prior_regularizer_scaler=args.prior_regularizer_scaler,
        y_mean=train_adapter.targets_mean,
        y_std=train_adapter.targets_std,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    if model_type == "vip":
        model = VIP(**common,
                    use_full_cov_elbo=getattr(args, "use_full_cov_elbo", False))
    else:
        flow = _build_flow(
            args,
            input_dim=args.regression_coeffs * train_adapter.output_dim,
            device=device, dtype=dtype,
        )
        model = FTIP(**common, flow=flow, num_samples=args.num_samples)

    return model


def _build_flow(args, input_dim, device, dtype):
    """Construct an FTIP flow based on ``args.flow_type``."""
    flow_type = getattr(args, "flow_type", "spline_1x1")
    common_kw = dict(depth=args.flow_depth, input_dim=input_dim,
                     device=device, dtype=dtype, seed=args.seed)
    num_bins = getattr(args, "flow_num_bins", 8)
    B = getattr(args, "flow_domain", 3.0)
    if flow_type == "affine":
        return CouplingFlow(**common_kw)
    if flow_type == "spline":
        return SplineCouplingFlow(**common_kw, num_bins=num_bins, B=B)
    if flow_type == "spline_1x1":
        return SplineCoupling1x1Flow(**common_kw, num_bins=num_bins, B=B)
    raise ValueError(f"Unknown flow_type: {flow_type!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Models that produce S posterior function samples per call rather than
# (mean, std). They denormalize via model.y_mean / model.y_std and
# treat model.log_variance as a scalar observation-noise (regression).
_SAMPLE_BASED_TYPES = ("mfvi", "fbnn", "tfsvi", "ap_fsvi")


def _eval_samples_for(args, model_type):
    if model_type == "ap_fsvi":
        return args.ap_fsvi_num_eval_samples
    return args.eval_samples


def _sample_based_predict(model, model_type, x, num_samples):
    """Call the right ``predict_f_samples`` and return (mean, std) of shape
    (S, N, D) on the original (denormalized) target scale, so the existing
    MetricsRegression / TrajectoryMetrics paths work unchanged."""
    if model_type == "fbnn":
        # FBNN caches S weight-noise samples; switch to S, sample, restore.
        old_S = model.num_samples
        model._set_num_samples(num_samples)
        model.num_samples = num_samples
        try:
            F = model.predict_f_samples(x, num_samples)
        finally:
            model._set_num_samples(old_S)
            model.num_samples = old_S
    else:
        F = model.predict_f_samples(x, num_samples)  # (S, N, D), normalized
    mean = F * model.y_std + model.y_mean
    std_scalar = torch.sqrt(torch.exp(model.log_variance)) * model.y_std
    std = std_scalar.expand_as(mean)
    return mean, std


def evaluate_regression(model, dataset, args, model_type=None, batch_size=None,
                        compute_crps=True):
    """Final-eval regression metrics: RMSE/NLL plus CRPS/CQM by default.

    CRPS only supports output_dim=1 (path_length / speed). For multi-D
    targets pass ``compute_crps=False`` so the heavy metrics are skipped
    (``light=True`` path).
    """
    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    x = torch.tensor(dataset.inputs, dtype=dtype, device=device)
    y = torch.tensor(dataset.targets, dtype=dtype, device=device)

    if batch_size is None:
        batch_size = 256

    # MetricsRegression's CRPS path only supports output_dim=1; auto-disable
    # for multi-dim trajectory targets so callers don't need to.
    light = not (compute_crps and dataset.output_dim == 1)

    metrics = MetricsRegression(num_data=len(dataset), device=device)
    model.eval()
    with torch.no_grad():
        a = model.sample_flow_coefficients(args.eval_samples) if model_type == "ftip" else None
        for i in range(0, x.shape[0], batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            if model_type == "ftip":
                mean, std = model.forward_with_coefficients(xb, a)
                if std.dim() < mean.dim():
                    std = std.unsqueeze(1).expand_as(mean)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=light)
            elif model_type in _SAMPLE_BASED_TYPES:
                mean, std = _sample_based_predict(
                    model, model_type, xb, _eval_samples_for(args, model_type)
                )
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=light)
            else:
                mean, std = model(xb)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=light)

    d = metrics.get_dict()
    out = {"RMSE": d["RMSE"], "NLL": d["NLL"]}
    if not light:
        out["CRPS"] = d["CRPS"]
        out["CQM"] = d["CQM"]
    return out


def _displacements_to_absolute(preds, origins, t_pred):
    """Convert displacement predictions to absolute positions.

    Parameters
    ----------
    preds : (S, N, T_pred*2) or (N, T_pred*2) — denormalized displacements
    origins : (N, 2) — last observed position per sample
    t_pred : int

    Returns
    -------
    absolute : same shape as preds — absolute positions
    """
    leading = preds.shape[:-1]  # (S, N) or (N,)
    flat = preds.reshape(*leading, t_pred, 2)
    cumulative = flat.cumsum(dim=-2)  # cumulative sum along time
    # Add origin: broadcast (N, 2) or (1, N, 2) depending on dims
    if cumulative.dim() == 3:
        # (S, N, T, 2) + (N, 2) -> need (1, N, 1, 2)
        absolute = cumulative + origins.unsqueeze(0).unsqueeze(-2)
    else:
        absolute = cumulative + origins.unsqueeze(-2)
    return absolute.reshape(preds.shape)


def _to_absolute_last(preds, origins, target_mode):
    """For target_horizon='last': preds shape (..., 2). Returns absolute (..., 2)."""
    if target_mode == "absolute":
        return preds
    if target_mode == "relative":
        if preds.dim() == 3:  # (S, N, 2)
            return preds + origins.unsqueeze(0)
        return preds + origins  # (N, 2)
    raise ValueError(f"target_mode={target_mode!r} unsupported with last horizon")


def _relative_to_absolute(preds, origins, t_pred):
    """Convert relative (offset-from-origin) predictions to absolute positions.

    Parameters
    ----------
    preds : (S, N, T_pred*2) or (N, T_pred*2) — denormalized offsets
    origins : (N, 2) — last observed position per sample
    t_pred : int
    """
    leading = preds.shape[:-1]
    flat = preds.reshape(*leading, t_pred, 2)
    if flat.dim() == 3:
        absolute = flat + origins.unsqueeze(0).unsqueeze(-2)
    else:
        absolute = flat + origins.unsqueeze(-2)
    return absolute.reshape(preds.shape)


def evaluate_trajectory(model, dataset, args, model_type=None, batch_size=None):
    """Trajectory metrics (ADE, FDE, minADE, minFDE) in original coordinates."""
    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    target_mode = getattr(dataset, "target_mode", "absolute")
    target_horizon = getattr(dataset, "target_horizon", "full")

    x = torch.tensor(dataset.inputs, dtype=dtype, device=device)
    y_raw = torch.tensor(dataset.targets, dtype=dtype, device=device)  # already denormalized
    origins = torch.tensor(dataset.origins, dtype=dtype, device=device)

    # Convert to absolute positions for trajectory metrics
    if target_horizon == "last":
        y_abs = _to_absolute_last(y_raw, origins, target_mode)
        t_pred_eff = 1
    else:
        if target_mode == "displacement":
            y_abs = _displacements_to_absolute(y_raw, origins, args.t_pred)
        elif target_mode == "relative":
            y_abs = _relative_to_absolute(y_raw, origins, args.t_pred)
        else:
            y_abs = y_raw
        t_pred_eff = args.t_pred

    if batch_size is None:
        batch_size = 256

    traj_metrics = TrajectoryMetrics(t_pred=t_pred_eff, K=args.best_of_k)
    model.eval()
    with torch.no_grad():
        a = model.sample_flow_coefficients(args.eval_samples) if model_type == "ftip" else None
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            yb = y_abs[i:i + batch_size]
            ob = origins[i:i + batch_size]
            if model_type == "ftip":
                mean, std = model.forward_with_coefficients(xb, a)
            elif model_type in _SAMPLE_BASED_TYPES:
                mean, _ = _sample_based_predict(
                    model, model_type, xb, _eval_samples_for(args, model_type)
                )
                std = None
            else:
                # Diagonal (independent-per-output-dim) VIP predictive:
                # consistent with the model's KL factorization q(a) = Prod_d
                # N(q_mu_d, L_d^T L_d).  The full_covariance=True path in
                # predict_f produces spurious cross-dim correlations.
                Fmean, Fvar = model.predict_f(xb, full_covariance=False)
                samples = model._sample_posterior(Fmean, Fvar, args.eval_samples)
                mean = samples * model.y_std + model.y_mean  # (S, N, D)
                std = None
            # mean is (S, N, D) denormalized
            if target_horizon == "last":
                mean = _to_absolute_last(mean, ob, target_mode)
            elif target_mode == "displacement":
                mean = _displacements_to_absolute(mean, ob, args.t_pred)
            elif target_mode == "relative":
                mean = _relative_to_absolute(mean, ob, args.t_pred)
            traj_metrics.update(mean, yb)

    return traj_metrics.get_dict()


def evaluate_light(model, dataset, args, model_type=None, batch_size=2048):
    """Fast eval for periodic training monitoring."""
    if model_type is None:
        model_type = args.model

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    x = torch.tensor(dataset.inputs, dtype=dtype, device=device)
    y = torch.tensor(dataset.targets, dtype=dtype, device=device)

    metrics = MetricsRegression(num_data=len(dataset), device=device)
    model.eval()
    with torch.no_grad():
        a = model.sample_flow_coefficients(args.eval_samples) if model_type == "ftip" else None
        for i in range(0, x.shape[0], batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            if model_type == "ftip":
                mean, std = model.forward_with_coefficients(xb, a)
                if std.dim() < mean.dim():
                    std = std.unsqueeze(1).expand_as(mean)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
            elif model_type in _SAMPLE_BASED_TYPES:
                # Light eval uses fewer MC samples to keep training-time
                # monitoring cheap.
                S_light = max(50, _eval_samples_for(args, model_type) // 4)
                mean, std = _sample_based_predict(model, model_type, xb, S_light)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
            else:
                mean, std = model(xb)
                metrics.update(yb, loss=torch.tensor(0.0), mean_pred=mean, std_pred=std, light=True)
    d = metrics.get_dict()
    model.train()
    return {"RMSE": d["RMSE"], "NLL": d["NLL"]}


def evaluate_all(model, dataset, args, model_type=None):
    """Combined regression + trajectory evaluation.

    For target_mode='path_length' only regression metrics are defined (the
    target is a scalar, not positions), so trajectory metrics are skipped.
    """
    reg = evaluate_regression(model, dataset, args, model_type=model_type)
    target_mode = getattr(dataset, "target_mode", "absolute")
    if target_mode in ("path_length", "speed"):
        return reg
    traj = evaluate_trajectory(model, dataset, args, model_type=model_type)
    return {**reg, **traj}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_with_metrics(model, train_loader, train_test_dataset, validation_dataset, args,
                       lr=None, epochs=None, iterations=None, model_type=None, desc="Training"):
    if lr is None:
        lr = args.lr
    if model_type is None:
        model_type = args.model
    if epochs is None and iterations is None:
        epochs = args.epochs
        iterations = args.iterations

    device = torch.device(args.device)
    weight_decay = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
    model.train()

    if iterations is not None:
        total = iterations
        data_stream = infinite_loader(train_loader)
        iters_per_epoch = len(train_loader)
        loop = tqdm(range(total), unit=" iter", desc=desc)

        last_val_nll = float("nan")
        last_val_rmse = float("nan")
        last_tr_nll = float("nan")
        for i in loop:
            inputs, target = next(data_stream)
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = model._train_step(optimizer, inputs, target)
            losses.append(loss.item())
            step = i + 1
            wandb_log_train_step(args, step, loss, optimizer, model, model_type)

            if scheduler is not None and step % iters_per_epoch == 0:
                scheduler.step()

            if step % args.eval_every == 0:
                tr = evaluate_light(model, train_test_dataset, args, model_type=model_type)
                val = evaluate_light(model, validation_dataset, args, model_type=model_type)
                metrics_history["iterations"].append(step)
                metrics_history["train"].append(tr)
                metrics_history["validation"].append(val)
                wandb_log_eval(step, tr, val)
                last_val_nll = val["NLL"]
                last_val_rmse = val["RMSE"]
                last_tr_nll = tr["NLL"]

            loop.set_postfix(
                val_NLL=f"{last_val_nll:.2f}",
                val_RMSE=f"{last_val_rmse:.3f}",
                tr_NLL=f"{last_tr_nll:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
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
                wandb_log_train_step(args, it, loss, optimizer, model, model_type)

                if it % args.eval_every == 0:
                    train_metrics = evaluate_light(
                        model, train_test_dataset, args, model_type=model_type
                    )
                    validation_metrics = evaluate_light(
                        model, validation_dataset, args, model_type=model_type
                    )
                    metrics_history["iterations"].append(it)
                    metrics_history["train"].append(train_metrics)
                    metrics_history["validation"].append(validation_metrics)
                    wandb_log_eval(it, train_metrics, validation_metrics)

            if scheduler is not None:
                scheduler.step()
            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    # KLs / bb_alphas exist on VIP, FTIP, MFVI, FBNN, TFSVI;
    # prior_regularizers is VIP/FTIP-specific.
    diagnostics = {
        "KLs": [float(v) for v in getattr(model, "KLs", [])],
        "bb_alphas": [float(v) for v in getattr(model, "bb_alphas", [])],
    }
    if hasattr(model, "prior_regularizers"):
        diagnostics["prior_regularizers"] = [
            float(v) for v in model.prior_regularizers
        ]
    if hasattr(model, "data_terms"):
        diagnostics["data_terms"] = [float(v) for v in model.data_terms]
    if hasattr(model, "betas"):
        diagnostics["betas"] = [float(v) for v in model.betas]
    if model_type == "ftip":
        diagnostics["base_KLs"] = [float(v) for v in model.base_KLs]
        diagnostics["flow_ldj"] = [float(v) for v in model.flow_ldj]
    if hasattr(model, "log_variance"):
        log_variance = model.log_variance.detach().flatten().cpu()
        noise_std = torch.exp(0.5 * log_variance)
        diagnostics["final_log_variance"] = [float(v) for v in log_variance]
        diagnostics["final_noise_std_normalized"] = [float(v) for v in noise_std]
        if hasattr(model, "y_std"):
            y_std = model.y_std.detach().flatten().cpu()
            if y_std.numel() == 1:
                y_std = y_std.expand_as(noise_std)
            diagnostics["final_noise_std_original"] = [
                float(v) for v in noise_std * y_std
            ]

    return losses, metrics_history, diagnostics


def _build_result(model_type, model, args, train_loader,
                  train_test_adapter, test_adapter, lr=None,
                  epochs=None, iterations=None, desc="Training"):
    """Train a model, evaluate it, and return the result dict."""
    if lr is None:
        lr = args.lr
    wandb_run = init_wandb_run(
        args,
        name=wandb_run_name("Pedestrian", model=model_type, seed=args.seed),
        group="pedestrian",
        tags=["pedestrian", model_type, args.target_mode],
        config={
            "dataset": "pedestrian",
            "model": model_type,
            "lr": lr,
            "epochs": epochs if epochs is not None else args.epochs,
            "iterations": iterations if iterations is not None else args.iterations,
            "desc": desc,
        },
    )

    t0 = time.time()
    losses, metrics_history, diagnostics = train_with_metrics(
        model, train_loader, train_test_adapter, test_adapter, args,
        lr=lr, epochs=epochs, iterations=iterations,
        model_type=model_type, desc=desc,
    )
    train_time = time.time() - t0

    train_metrics = evaluate_all(model, train_test_adapter, args, model_type=model_type)
    test_metrics = evaluate_all(model, test_adapter, args, model_type=model_type)

    hyperparameters = {
        "lr": lr,
        "batch_size": args.batch_size,
        "iterations": iterations if iterations is not None else args.iterations,
        "epochs": epochs if epochs is not None else args.epochs,
        "t_obs": args.t_obs,
        "t_pred": args.t_pred,
        "use_vel": args.use_vel,
        "cosine_annealing": args.cosine_annealing,
        "seed": args.seed,
        "dtype": args.dtype,
        "split_mode": args.split_mode,
        "test_seq": args.test_seq,
        "target_mode": args.target_mode,
        "target_horizon": args.target_horizon,
        "best_of_k": args.best_of_k,
    }
    hyperparameters.update({
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "head_dims": args.head_dims,
        "layer_model": args.layer_model,
        "dropout": args.dropout,
        "regression_coeffs": args.regression_coeffs,
        "bb_alpha": args.bb_alpha,
    })
    if model_type == "ftip":
        hyperparameters["flow_depth"] = args.flow_depth
        hyperparameters["num_samples"] = args.num_samples
        hyperparameters["eval_samples"] = args.eval_samples
    if model_type == "ap_fsvi":
        hyperparameters["ap_fsvi_layer_model"] = "BayesLinear"
        hyperparameters["ap_fsvi_num_samples"] = args.ap_fsvi_num_samples
        hyperparameters["ap_fsvi_num_prior_samples"] = (
            args.ap_fsvi_num_prior_samples or args.ap_fsvi_num_samples
        )
        hyperparameters["ap_fsvi_num_eval_samples"] = args.ap_fsvi_num_eval_samples
        hyperparameters["ap_fsvi_num_measurement"] = args.ap_fsvi_num_measurement
        hyperparameters["ap_fsvi_adaptive_measure_points"] = (
            args.ap_fsvi_adaptive_measure_points
        )
        hyperparameters["ap_fsvi_adaptive_measure_steps"] = (
            args.ap_fsvi_adaptive_measure_steps
        )
        hyperparameters["ap_fsvi_adaptive_measure_lr"] = (
            args.ap_fsvi_adaptive_measure_lr
        )
        hyperparameters["ap_fsvi_adaptive_measure_domain_limit"] = (
            args.ap_fsvi_adaptive_measure_domain_limit
        )
        hyperparameters["ap_fsvi_beta"] = args.ap_fsvi_beta
        hyperparameters["ap_fsvi_beta_start"] = args.ap_fsvi_beta_start
        hyperparameters["ap_fsvi_beta_warmup_steps"] = args.ap_fsvi_beta_warmup_steps
        hyperparameters["ap_fsvi_data_pretrain_steps"] = args.ap_fsvi_data_pretrain_steps
        hyperparameters["ap_fsvi_data_loss"] = args.ap_fsvi_data_loss
        hyperparameters["ap_fsvi_discrepancy"] = args.ap_fsvi_discrepancy
        hyperparameters["ap_fsvi_discrepancy_projections"] = args.ap_fsvi_discrepancy_projections
        hyperparameters["ap_fsvi_sample_knn_k"] = args.ap_fsvi_sample_knn_k
        hyperparameters["ap_fsvi_sinkhorn_epsilon"] = args.ap_fsvi_sinkhorn_epsilon
        hyperparameters["ap_fsvi_sinkhorn_iterations"] = args.ap_fsvi_sinkhorn_iterations
        hyperparameters["ap_fsvi_log_variance_init"] = args.ap_fsvi_log_variance_init
        hyperparameters["ap_fsvi_measurement_weights"] = args.ap_fsvi_measurement_weights
        hyperparameters["ap_fsvi_near_data_noise"] = args.ap_fsvi_near_data_noise
        hyperparameters["ap_fsvi_domain_std"] = args.ap_fsvi_domain_std
        hyperparameters["ap_fsvi_hidden_dims"] = args.ap_fsvi_hidden_dims
        hyperparameters["ap_fsvi_max_grad_norm"] = args.ap_fsvi_max_grad_norm

    result = {
        "dataset": "pedestrian",
        "model": model_type,
        "hyperparameters": hyperparameters,
        "train_time_s": round(train_time, 2),
        "train": train_metrics,
        "test": test_metrics,
        "losses": losses,
        "metrics_history": metrics_history,
        "diagnostics": diagnostics,
    }

    has_traj = "ADE" in test_metrics
    for split, m in [("Train", train_metrics), ("Test", test_metrics)]:
        line = f"  {model_type.upper()} {split}: RMSE={m['RMSE']:.4f}  NLL={m['NLL']:.4f}"
        if has_traj:
            line += (f"  ADE={m['ADE']:.4f}  FDE={m['FDE']:.4f}"
                     f"  minADE={m[f'minADE_{args.best_of_k}']:.4f}"
                     f"  minFDE={m[f'minFDE_{args.best_of_k}']:.4f}")
        print(line)
    print(f"  Time: {train_time:.1f}s")
    wandb_log_result(result)
    finish_wandb_run(wandb_run)

    return result, model


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

ETH_UCY_BENCHMARKS = {
    "eth":   ["biwi_eth"],
    "hotel": ["biwi_hotel"],
    "univ":  ["students001", "students003"],
    "zara1": ["crowds_zara01"],
    "zara2": ["crowds_zara02"],
}


# Per-scene first val frame in EigenTrajectory's canonical splits (the same
# across benchmarks). Derived by inspecting the first line of each
# <benchmark>/val/<scene>_val.txt. Rows with frame < cutoff belong to
# ET's "train" file; rows with frame >= cutoff belong to "val".
ET_VAL_FIRST_FRAME = {
    "biwi_eth":      10240,
    "biwi_hotel":    14400,
    "crowds_zara01":  7110,
    "crowds_zara02":  8420,
    "crowds_zara03":  6030,
    "students001":    3550,
    "students003":    4320,
    "uni_examples":   5940,
}


def build_splits(args, return_val=False):
    """Build train/test adapters from Pedestrian_Dataset.

    Supported split_mode values:
      - "leave_one_out": load EigenTrajectory's canonical train/val/test
        splits for the benchmark named in args.test_seq (one of
        ``eth``, ``hotel``, ``univ``, ``zara1``, ``zara2``).  Data lives in
        ``args.data_dir/<benchmark>/<split>/``.  No custom splitting — we
        just load the files.
      - "ped_split":    GroupShuffleSplit across the combined pool of scenes
        in ``raw/all_data``, carving test first, then val from the train pool.

    Returns
    -------
    If return_val=False (default, backward-compatible):
        (train_adapter, train_test_adapter, test_adapter)
    If return_val=True:
        (train_adapter, train_test_adapter, val_adapter, test_adapter)
    """
    val_frac = getattr(args, "val_frac", 0.2)
    target_mode = getattr(args, "target_mode", "absolute")
    target_horizon = getattr(args, "target_horizon", "full")

    if args.split_mode == "leave_one_out":
        benchmark = args.test_seq
        if benchmark not in ETH_UCY_BENCHMARKS:
            raise ValueError(
                f"split_mode='leave_one_out' expects args.test_seq to be one "
                f"of {list(ETH_UCY_BENCHMARKS)}; got {benchmark!r}."
            )

        train_ds = Pedestrian_Dataset(
            data_dir=args.data_dir, benchmark=benchmark, split="train",
            t_obs=args.t_obs, t_pred=args.t_pred,
        )
        val_ds = Pedestrian_Dataset(
            data_dir=args.data_dir, benchmark=benchmark, split="val",
            t_obs=args.t_obs, t_pred=args.t_pred,
        )
        test_ds = Pedestrian_Dataset(
            data_dir=args.data_dir, benchmark=benchmark, split="test",
            t_obs=args.t_obs, t_pred=args.t_pred,
        )
        print(f"Leave-one-out benchmark={benchmark}:  "
              f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

        # Build train adapter from train_ds, others share its normalization.
        train_adapter = PedestrianTrainAdapter(
            train_ds, list(range(len(train_ds))),
            args.t_obs, args.t_pred, args.use_vel,
            target_mode=target_mode, target_horizon=target_horizon,
        )
        train_test_adapter = PedestrianTestAdapter(
            train_ds, list(range(len(train_ds))),
            args.t_obs, args.t_pred, args.use_vel,
            train_adapter.inputs_mean, train_adapter.inputs_std,
            train_adapter.targets_mean, train_adapter.targets_std,
            target_mode=target_mode, target_horizon=target_horizon,
        )
        val_adapter = PedestrianTestAdapter(
            val_ds, list(range(len(val_ds))),
            args.t_obs, args.t_pred, args.use_vel,
            train_adapter.inputs_mean, train_adapter.inputs_std,
            train_adapter.targets_mean, train_adapter.targets_std,
            target_mode=target_mode, target_horizon=target_horizon,
        )
        test_adapter = PedestrianTestAdapter(
            test_ds, list(range(len(test_ds))),
            args.t_obs, args.t_pred, args.use_vel,
            train_adapter.inputs_mean, train_adapter.inputs_std,
            train_adapter.targets_mean, train_adapter.targets_std,
            target_mode=target_mode, target_horizon=target_horizon,
        )
        if return_val:
            return train_adapter, train_test_adapter, val_adapter, test_adapter
        return train_adapter, train_test_adapter, test_adapter

    # Non-leave-one-out: ped_split on the combined all-scenes pool.
    sequences = getattr(args, "sequences", None) or Pedestrian_Dataset.DEFAULT_SEQUENCES
    full_ds = Pedestrian_Dataset(
        data_dir=args.data_dir,
        sequences=sequences,
        t_obs=args.t_obs,
        t_pred=args.t_pred,
    )
    print(f"Total samples: {len(full_ds)}  (sequences: {list(sequences)})")
    # Pedestrian-level split to avoid sliding-window leakage.
    # Hierarchy: first carve test from full; then carve val from train.
    from sklearn.model_selection import GroupShuffleSplit
    seq_to_id = {name: idx for idx, name in enumerate(sorted(set(
        s["seq_name"] for s in full_ds.samples)))}
    groups = np.array([
        full_ds.samples[i]["ped_id"] * 1000
        + seq_to_id[full_ds.samples[i]["seq_name"]]
        for i in range(len(full_ds))
    ])
    test_size = getattr(args, "test_size", 0.2)
    sp_test = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=args.seed)
    trainval_idx, test_idx = next(sp_test.split(range(len(full_ds)),
                                                groups=groups))
    test_indices = list(test_idx)
    # Val carved from the train pool
    sp_val = GroupShuffleSplit(n_splits=1, test_size=val_frac,
                               random_state=args.seed + 1)
    rel_train, rel_val = next(sp_val.split(range(len(trainval_idx)),
                                           groups=groups[trainval_idx]))
    train_indices = [int(trainval_idx[i]) for i in rel_train]
    val_indices = [int(trainval_idx[i]) for i in rel_val]
    print(f"Ped-level split (test_size={test_size}, val_frac={val_frac}):  "
          f"train={len(train_indices)}, val={len(val_indices)}, "
          f"test={len(test_indices)}")

    train_adapter = PedestrianTrainAdapter(
        full_ds, train_indices, args.t_obs, args.t_pred, args.use_vel,
        target_mode=target_mode, target_horizon=target_horizon,
    )
    # Test adapter uses training normalization stats
    test_adapter = PedestrianTestAdapter(
        full_ds, test_indices, args.t_obs, args.t_pred, args.use_vel,
        train_adapter.inputs_mean, train_adapter.inputs_std,
        train_adapter.targets_mean, train_adapter.targets_std,
        target_mode=target_mode, target_horizon=target_horizon,
    )
    # Train-test adapter (training data with training stats, for eval)
    train_test_adapter = PedestrianTestAdapter(
        full_ds, train_indices, args.t_obs, args.t_pred, args.use_vel,
        train_adapter.inputs_mean, train_adapter.inputs_std,
        train_adapter.targets_mean, train_adapter.targets_std,
        target_mode=target_mode, target_horizon=target_horizon,
    )
    # Validation adapter (from the disjoint val indices, training stats)
    val_adapter = PedestrianTestAdapter(
        full_ds, val_indices, args.t_obs, args.t_pred, args.use_vel,
        train_adapter.inputs_mean, train_adapter.inputs_std,
        train_adapter.targets_mean, train_adapter.targets_std,
        target_mode=target_mode, target_horizon=target_horizon,
    )

    if return_val:
        return train_adapter, train_test_adapter, val_adapter, test_adapter
    return train_adapter, train_test_adapter, test_adapter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    use_warm_start = args.model == "ftip" and args.auto_warm_start

    header = "FTIP (warm-start from VIP)" if use_warm_start else args.model.upper()
    print(f"\n{'='*60}")
    print(f"  Pedestrian Trajectory Prediction  |  Model: {header}")
    print(f"{'='*60}")

    train_adapter, train_test_adapter, test_adapter = build_splits(args)

    use_cuda = args.device and "cuda" in args.device
    train_loader = DataLoader(
        train_adapter, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_cuda, num_workers=0,
    )

    results = []

    if use_warm_start:
        vip_it = args.vip_iterations
        ftip_it = args.iterations

        # --- Phase 1: VIP pre-training ---
        print(f"\n  Phase 1: Training VIP ({vip_it} iters, lr={args.vip_lr})")
        vip_model = build_model(args, train_adapter, model_type="vip")
        vip_ws_result, vip_model = _build_result(
            "vip", vip_model, args, train_loader,
            train_test_adapter, test_adapter, lr=args.vip_lr,
            iterations=vip_it, desc="VIP pre-training",
        )
        vip_ws_state = {n: p.data.clone() for n, p in vip_model.named_parameters()}

        # Continue VIP for remaining budget (fair baseline)
        print(f"\n  VIP baseline: continuing for {ftip_it} iters")
        vip_baseline_result, _ = _build_result(
            "vip", vip_model, args, train_loader,
            train_test_adapter, test_adapter, lr=args.vip_lr,
            iterations=ftip_it, desc="VIP baseline (continued)",
        )
        vip_baseline_result["train_time_s"] = round(
            vip_ws_result["train_time_s"] + vip_baseline_result["train_time_s"], 2
        )
        results.append(vip_baseline_result)

        # Restore warm-start checkpoint
        for n, p in vip_model.named_parameters():
            p.data.copy_(vip_ws_state[n])
        del vip_ws_state

        # --- Phase 2: FTIP fine-tuning ---
        print(f"\n  Phase 2: Fine-tuning FTIP ({ftip_it} iters, lr={args.ftip_lr})")
        ftip_model = build_model(args, train_adapter, model_type="ftip")
        ftip_model.warm_start_from_vip(vip_model, learnable_affine=args.learnable_affine)
        del vip_model

        ftip_result, _ = _build_result(
            "ftip", ftip_model, args, train_loader,
            train_test_adapter, test_adapter, lr=args.ftip_lr,
            iterations=ftip_it, desc="FTIP fine-tuning",
        )
        ftip_result["warm_start"] = {
            "enabled": True,
            "vip_iterations": vip_it,
            "vip_lr": args.vip_lr,
            "learnable_affine": args.learnable_affine,
        }
        ftip_result["total_time_s"] = round(
            vip_ws_result["train_time_s"] + ftip_result["train_time_s"], 2
        )
        results.append(ftip_result)

    else:
        model = build_model(args, train_adapter)
        # FBNN's measurement reservoir + TFSVI's context-sampling pool are
        # populated inside their .fit(); we drive training through
        # _train_step instead, so populate them upfront here.
        if args.model == "fbnn":
            model._fill_reservoir(train_loader)
        elif args.model == "ap_fsvi":
            model._fill_reservoir(train_loader)
        elif args.model == "tfsvi":
            all_X = [inputs for inputs, _ in train_loader]
            model._train_inputs = torch.cat(all_X, dim=0).to(
                torch.float64 if args.dtype == "float64" else torch.float32
            )
        if args.model == "mfvi":
            single_lr = args.mfvi_lr
        elif args.model == "fbnn":
            single_lr = args.fbnn_lr
        elif args.model == "tfsvi":
            single_lr = args.tfsvi_lr
        elif args.model == "ap_fsvi":
            single_lr = args.ap_fsvi_lr
        else:
            single_lr = args.lr
        result, _ = _build_result(
            args.model, model, args, train_loader,
            train_test_adapter, test_adapter, lr=single_lr,
            desc=f"{args.model.upper()} training",
        )
        results.append(result)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"pedestrian_{args.model}_split-{args.split_mode}_seed{args.seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Pedestrian trajectory prediction benchmark")

    # Data
    p.add_argument("--data_dir", default="./data/pedestrian")
    p.add_argument("--t_obs", type=int, default=8)
    p.add_argument("--t_pred", type=int, default=12)
    p.add_argument("--use_vel", action="store_true", default=False,
                   help="Unsupported with the EigenTrajectory 4-col format; "
                        "kept only for backwards compatibility.")
    p.add_argument("--no_vel", dest="use_vel", action="store_false")
    p.add_argument("--split_mode", choices=["ped_split"], default="ped_split")
    p.add_argument("--test_seq", default="eth",
                   help="Scene to hold out. Either a benchmark shortcut "
                        "(eth|hotel|univ|zara1|zara2), a raw scene name "
                        "(e.g. biwi_eth, crowds_zara02), or a comma-separated "
                        "list of scene names.")
    p.add_argument("--target_mode",
                   choices=["absolute", "displacement", "relative", "path_length"],
                   default="path_length",
                   help="absolute/relative/displacement predict positions; "
                        "'path_length' predicts the scalar future path length.")
    p.add_argument("--target_horizon", choices=["full", "last"], default="full",
                   help="'full' predicts the whole T_pred trajectory; "
                        "'last' predicts only the final position (2D). "
                        "'last' is incompatible with target_mode='displacement'.")

    # LSTM encoder
    p.add_argument("--lstm_hidden", type=int, default=64)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--head_dims", type=int, nargs="+", default=[64, 32])

    # Model
    p.add_argument("--model",
                   choices=MODELS + ["all"],
                   default="ftip")
    p.add_argument("--layer_model", choices=list(LAYER_MODELS.keys()), default="BayesLinear")
    p.add_argument("--dropout", type=float, default=0.0)
    add_wandb_args(p)
    p.add_argument("--regression_coeffs", type=int, default=40)
    p.add_argument("--bb_alpha", type=float, default=1.0)
    p.add_argument("--use_prior_regularizer", action="store_true", default=False)
    p.add_argument("--regularizer_mode", default="evidence")
    p.add_argument("--prior_regularizer_scaler", type=float, default=1.0)
    p.add_argument("--use_full_cov_elbo", action="store_true", default=False,
                   help="VIP only: compute ELBO/BB-alpha with full predictive "
                        "covariance instead of diagonal (only changes the "
                        "objective when bb_alpha != 0).")

    # Flow (FTIP)
    p.add_argument("--flow_depth", type=int, default=2)
    p.add_argument("--flow_type", type=str, default="affine",
                   choices=["affine", "spline", "spline_1x1"],
                   help="FTIP flow class. 'affine' = original CouplingFlow. "
                        "'spline' = "
                        "SplineCouplingFlow (RQ coupling). 'spline_1x1' = "
                        "SplineCoupling1x1Flow (spline + Glow 1x1 LU mixing).")
    p.add_argument("--flow_num_bins", type=int, default=8,
                   help="Bins per RQ-spline coupling layer (ignored if flow_type=affine).")
    p.add_argument("--flow_domain", type=float, default=3.0,
                   help="Spline domain half-width B (ignored if flow_type=affine).")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--eval_samples", type=int, default=1000)
    p.add_argument("--auto_warm_start", action="store_true", default=False)
    p.add_argument("--no_auto_warm_start", dest="auto_warm_start", action="store_false")
    p.add_argument("--learnable_affine", action="store_true", default=True)
    p.add_argument("--no_learnable_affine", dest="learnable_affine", action="store_false")

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vip_lr", type=float, default=1e-3)
    p.add_argument("--ftip_lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--vip_iterations", type=int, default=30000)
    p.add_argument("--cosine_annealing", action="store_true", default=True)
    p.add_argument("--no_cosine_annealing", dest="cosine_annealing", action="store_false")
    p.add_argument("--eval_every", type=int, default=500)

    # MFVI
    p.add_argument("--mfvi_num_samples", type=int, default=40,
                   help="MC weight samples per forward (training and eval).")
    p.add_argument("--mfvi_bb_alpha", type=float, default=1.0,
                   help="BB-alpha energy parameter; 1.0 matches the recipe "
                        "used for the FBNN/TFSVI runs in the same table.")
    p.add_argument("--mfvi_lr", type=float, default=1e-3)

    # FBNN (functional Bayesian NN with measurement-set KL via SSGE)
    p.add_argument("--fbnn_num_samples", type=int, default=40,
                   help="MC function samples for the SSGE score estimate.")
    p.add_argument("--fbnn_num_measurement", type=int, default=20,
                   help="Training-pool points sampled into each measurement set.")
    p.add_argument("--fbnn_num_context", type=int, default=20,
                   help="OOD context points (Gaussian in input space) per step.")
    p.add_argument("--fbnn_context_std", type=float, default=2.0,
                   help="Std of the Gaussian from which OOD context points are drawn.")
    p.add_argument("--fbnn_bb_alpha", type=float, default=1.0)
    p.add_argument("--fbnn_lambda_kl", type=float, default=1.0,
                   help="Weight on the functional-KL term.")
    p.add_argument("--fbnn_prior", choices=["gp", "bnn"], default="gp")
    p.add_argument("--fbnn_freeze_prior", action="store_true", default=True)
    p.add_argument("--fbnn_no_freeze_prior", dest="fbnn_freeze_prior",
                   action="store_false")
    p.add_argument("--fbnn_gp_features", type=int, default=100,
                   help="Inner-layer dim (RFF count) for the GP prior.")
    p.add_argument("--fbnn_lr", type=float, default=1e-3)

    # TFSVI (linearized function-space VI; q over LSTM gen_fn parameters)
    p.add_argument("--tfsvi_num_samples", type=int, default=40,
                   help="MC parameter samples for the expected log-likelihood.")
    p.add_argument("--tfsvi_S_ctx", type=int, default=5,
                   help="Number of context sets sampled for the max-KL estimator.")
    p.add_argument("--tfsvi_K_ctx", type=int, default=20,
                   help="Number of points per context set.")
    p.add_argument("--tfsvi_sigma_prior", type=float, default=1.0,
                   help="Isotropic prior std on theta: p(theta)=N(0, sigma^2 I).")
    p.add_argument("--tfsvi_bb_alpha", type=float, default=1.0)
    p.add_argument("--tfsvi_lr", type=float, default=1e-3)

    # AP-FSVI
    p.add_argument("--ap_fsvi_num_samples", type=int, default=32)
    p.add_argument("--ap_fsvi_num_prior_samples", type=int, default=None)
    p.add_argument("--ap_fsvi_num_eval_samples", type=int, default=200)
    p.add_argument("--ap_fsvi_num_measurement", type=int, default=64)
    p.add_argument("--ap_fsvi_adaptive_measure_points", action="store_true",
                   default=False)
    p.add_argument("--ap_fsvi_adaptive_measure_steps", type=int, default=3)
    p.add_argument("--ap_fsvi_adaptive_measure_lr", type=float, default=0.05)
    p.add_argument("--ap_fsvi_adaptive_measure_domain_limit", type=float,
                   default=None)
    p.add_argument("--ap_fsvi_beta", type=float, default=0.05)
    p.add_argument("--ap_fsvi_beta_start", type=float, default=0.0)
    p.add_argument("--ap_fsvi_beta_warmup_steps", type=int, default=5000)
    p.add_argument("--ap_fsvi_data_pretrain_steps", type=int, default=1000)
    p.add_argument("--ap_fsvi_data_loss",
                   choices=["expected_nll", "predictive_nll"],
                   default="expected_nll")
    p.add_argument("--ap_fsvi_discrepancy", type=str, default="mmd",
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
                   ])
    p.add_argument("--ap_fsvi_discrepancy_projections", type=int, default=64)
    p.add_argument("--ap_fsvi_sample_knn_k", type=int, default=3)
    p.add_argument("--ap_fsvi_quantile_transport_k", type=int, default=3)
    p.add_argument("--ap_fsvi_sinkhorn_epsilon", type=float, default=1.0)
    p.add_argument("--ap_fsvi_sinkhorn_iterations", type=int, default=50)
    p.add_argument("--ap_fsvi_log_variance_init", type=float, default=-2.0)
    p.add_argument("--ap_fsvi_measurement_weights", type=float, nargs=3,
                   default=[0.2, 0.2, 0.6],
                   metavar=("DATA", "NEAR", "DOMAIN"))
    p.add_argument("--ap_fsvi_near_data_noise", type=float, default=0.1)
    p.add_argument("--ap_fsvi_domain_std", type=float, default=2.5)
    p.add_argument("--ap_fsvi_hidden_dims", type=int, nargs="+", default=None)
    p.add_argument("--ap_fsvi_max_grad_norm", type=float, default=None)
    p.add_argument("--ap_fsvi_lr", type=float, default=1e-3)

    # Evaluation
    p.add_argument("--best_of_k", type=int, default=20)

    # System
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="./results/pedestrian_3way_pedsplit")

    args = p.parse_args()
    args.layer_model = "BayesLinear"
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.model == "all":
        all_results = []
        for model_type in MODELS:
            model_args = copy.copy(args)
            model_args.model = model_type
            all_results.extend(run(model_args))
        print_comparison(all_results, split="test", primary="RMSE")
        json_path, csv_path = save_comparison(
            all_results,
            args.output_dir,
            f"pedestrian_seed{args.seed}",
            split="test",
        )
        if json_path is not None:
            print(f"\nComparison saved to {json_path}")
            print(f"Comparison CSV saved to {csv_path}")
    else:
        results = run(args)
        print_comparison(results, split="test", primary="RMSE")
