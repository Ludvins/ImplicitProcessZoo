"""Label-free comparison of BNN, VIP, and GMVIP prior distributions.

The experiment uses the one-dimensional Bayesian neural-network architecture
from the synthetic regression benchmark, but it does not load observations or
optimize any model parameters.

Examples
--------
python -m experiments.synthetic.prior_fidelity
python -m experiments.synthetic.prior_fidelity --smoke --output-dir results/prior_smoke
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.common import write_csv_rows, write_json
from experiments.synthetic.prior_fidelity_metrics import (
    as_curves,
    estimate_rbf_bandwidth,
    fidelity_metrics,
    fit_pointwise_standardizer,
    projection_directions,
    standardize_curves,
)
from experiments.synthetic.prior_fidelity_plots import (
    plot_gmvip_bank_sensitivity,
    plot_matched_sweeps,
    plot_pointwise_w1,
    plot_prior_samples,
)
from implicit_process_zoo.gmvip import GeneralizedMatheronVIP
from implicit_process_zoo.priors.function_bank import CoherentPriorFunctionSampler
from implicit_process_zoo.priors.generative_functions import BayesianNN, BayesLinear

METRIC_NAMES = (
    "joint_sw2",
    "marginal_w1_mean",
    "marginal_w1_max",
    "mean_rmse",
    "covariance_rel_fro",
    "energy_distance",
    "rbf_mmd2",
)


@dataclass(frozen=True)
class ExperimentSetting:
    method: str
    coefficient_dim: int
    operator_bank_size: int | None
    is_published_default: bool = False
    is_display_default: bool = False
    in_matched_sweep: bool = False
    in_bank_sweep: bool = False


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _make_bnn_prior(
    *,
    num_samples: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BayesianNN:
    prior = BayesianNN(
        input_dim=1,
        num_samples=max(2, int(num_samples)),
        structure=[10, 10],
        activation=torch.tanh,
        output_dim=1,
        layer_model=BayesLinear,
        dropout=0.0,
        fix_random_noise=True,
        zero_mean_prior=True,
        weight_log_sigma_init=0.0,
        device=device,
        seed=int(seed),
        dtype=dtype,
    )
    prior.freeze_parameters()
    prior.eval()
    return prior


def sample_true_prior(
    x_grid: torch.Tensor,
    num_samples: int,
    *,
    prior_seed: int,
    sample_seed: int,
) -> torch.Tensor:
    """Draw coherent functions from the frozen BNN prior."""
    prior = _make_bnn_prior(
        num_samples=2,
        seed=prior_seed,
        device=x_grid.device,
        dtype=x_grid.dtype,
    )
    sampler = CoherentPriorFunctionSampler(prior)
    with torch.no_grad():
        values = sampler.sample_values(x_grid, int(num_samples), seed=int(sample_seed))
    return as_curves(values).detach()


def sample_vip_surrogate(
    x_grid: torch.Tensor,
    *,
    basis_size: int,
    num_samples: int,
    prior_seed: int,
    basis_seed: int,
    coefficient_seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample the untrained VIP surrogate with standard-normal coefficients."""
    basis_size = int(basis_size)
    if basis_size < 2:
        raise ValueError("VIP basis_size must be at least 2.")
    prior = _make_bnn_prior(
        num_samples=basis_size,
        seed=prior_seed,
        device=x_grid.device,
        dtype=x_grid.dtype,
    )
    sampler = CoherentPriorFunctionSampler(prior)
    with torch.no_grad():
        basis = as_curves(
            sampler.sample_values(x_grid, basis_size, seed=int(basis_seed))
        )
        empirical_mean = basis.mean(dim=0)
        features = (basis - empirical_mean) / math.sqrt(float(basis_size - 1))
        generator = torch.Generator(device=x_grid.device)
        generator.manual_seed(int(coefficient_seed))
        coefficients = torch.randn(
            int(num_samples),
            basis_size,
            generator=generator,
            device=x_grid.device,
            dtype=x_grid.dtype,
        )
        samples = empirical_mean + coefficients @ features
        covariance = features.T @ features
    return samples.detach(), {
        "basis": basis.detach(),
        "mean": empirical_mean.detach(),
        "covariance": covariance.detach(),
        "coefficient_prior_kl": x_grid.new_zeros(()),
    }


def sample_gmvip_surrogate(
    x_grid: torch.Tensor,
    *,
    num_inducing: int,
    operator_bank_size: int,
    num_samples: int,
    prior_seed: int,
    operator_seed: int,
    sample_seed: int,
    jitter: float,
    shrinkage: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample the production GMVIP surrogate prior without optimization."""
    inducing_points = torch.linspace(
        float(x_grid[0, 0]),
        float(x_grid[-1, 0]),
        int(num_inducing),
        dtype=x_grid.dtype,
        device=x_grid.device,
    ).unsqueeze(-1)
    prior = _make_bnn_prior(
        num_samples=max(2, int(operator_bank_size)),
        seed=prior_seed,
        device=x_grid.device,
        dtype=x_grid.dtype,
    )
    model = GeneralizedMatheronVIP(
        base_prior=prior,
        inducing_points=inducing_points,
        operator_type="empirical",
        posterior_type="gaussian",
        likelihood="regression",
        num_operator_bank_samples=int(operator_bank_size),
        learn_noise=False,
        init_log_noise=-5.0,
        freeze_base_prior=True,
        detach_prior_samples=True,
        detach_operator_prior_grad=True,
        jitter=float(jitter),
        shrinkage=float(shrinkage),
        learn_Z=False,
        learn_kernel=False,
        inducing_scale="prior_cholesky",
        mean_mode="prior_sample",
        posterior_init_mean=0.0,
        posterior_init_log_std=0.0,
        antithetic_samples=False,
        num_train_samples=1,
        operator_bank_seed=int(operator_seed),
        output_dim=1,
        path_mode="full",
    )
    model.eval()
    with torch.no_grad():
        samples = model.sample_prior_values(
            x_grid,
            int(num_samples),
            seed=int(sample_seed),
        )
        coefficient_prior_kl = model.kl_divergence()
    return as_curves(samples).detach(), {
        "coefficient_prior_kl": coefficient_prior_kl.detach(),
        "inducing_points": inducing_points.detach(),
    }


def _setting_key(
    method: str,
    coefficient_dim: int,
    operator_bank_size: int | None,
) -> tuple[str, int, int | None]:
    return method, int(coefficient_dim), operator_bank_size


def _build_settings(args: argparse.Namespace) -> list[ExperimentSetting]:
    settings: dict[tuple[str, int, int | None], dict] = {}

    def add(
        method: str,
        coefficient_dim: int,
        operator_bank_size: int | None,
        **flags: bool,
    ) -> None:
        key = _setting_key(method, coefficient_dim, operator_bank_size)
        if key not in settings:
            settings[key] = {
                "method": method,
                "coefficient_dim": int(coefficient_dim),
                "operator_bank_size": operator_bank_size,
                "is_published_default": False,
                "is_display_default": False,
                "in_matched_sweep": False,
                "in_bank_sweep": False,
            }
        for name, enabled in flags.items():
            settings[key][name] = bool(settings[key][name] or enabled)

    for dimension in args.matched_dims:
        add("vip", dimension, None, in_matched_sweep=True)
        add(
            "gmvip",
            dimension,
            args.gmvip_default_b,
            in_matched_sweep=True,
        )
    for bank_size in args.gmvip_bank_sizes:
        add(
            "gmvip",
            args.gmvip_default_m,
            bank_size,
            in_bank_sweep=True,
        )

    add(
        "vip",
        args.vip_default_s,
        None,
        is_published_default=not args.smoke,
        is_display_default=True,
    )
    add(
        "gmvip",
        args.gmvip_default_m,
        args.gmvip_default_b,
        is_published_default=not args.smoke,
        is_display_default=True,
    )
    return [
        ExperimentSetting(**setting)
        for setting in sorted(
            settings.values(),
            key=lambda value: (
                value["method"],
                value["coefficient_dim"],
                value["operator_bank_size"] or 0,
            ),
        )
    ]


def _seed_map(seed: int, setting: ExperimentSetting | None = None) -> dict[str, int]:
    base = 1_000_000 + int(seed) * 100_000
    if setting is None:
        return {
            "prior_constructor": base + 1,
            "calibration": base + 11,
            "reference": base + 12,
            "null": base + 13,
            "projections": base + 14,
        }
    bank = int(setting.operator_bank_size or 0)
    method_offset = 10_000 if setting.method == "vip" else 20_000
    setting_offset = method_offset + 17 * int(setting.coefficient_dim) + 3 * bank
    return {
        "prior_constructor": base + setting_offset + 1,
        "basis_or_operator": base + setting_offset + 2,
        "coefficients_or_samples": base + setting_offset + 3,
    }


def _row_for_setting(
    *,
    seed: int,
    setting: ExperimentSetting,
    metrics: dict[str, float],
    coefficient_prior_kl: float,
    args: argparse.Namespace,
) -> dict:
    return {
        "seed": int(seed),
        "method": setting.method,
        "coefficient_dim": int(setting.coefficient_dim),
        "operator_bank_size": (
            "" if setting.operator_bank_size is None else int(setting.operator_bank_size)
        ),
        "is_published_default": setting.is_published_default,
        "is_display_default": setting.is_display_default,
        "in_matched_sweep": setting.in_matched_sweep,
        "in_bank_sweep": setting.in_bank_sweep,
        "coefficient_prior_kl": float(coefficient_prior_kl),
        "num_samples": int(args.num_samples),
        "num_projections": int(args.num_projections),
        **metrics,
    }


def _null_row(seed: int, metrics: dict[str, float], args: argparse.Namespace) -> dict:
    return {
        "seed": int(seed),
        "method": "true_null",
        "coefficient_dim": "",
        "operator_bank_size": "",
        "is_published_default": False,
        "is_display_default": True,
        "in_matched_sweep": False,
        "in_bank_sweep": False,
        "coefficient_prior_kl": "",
        "num_samples": int(args.num_samples),
        "num_projections": int(args.num_projections),
        **metrics,
    }


def _summarize(rows: list[dict]) -> list[dict]:
    group_fields = (
        "method",
        "coefficient_dim",
        "operator_bank_size",
        "is_published_default",
        "is_display_default",
        "in_matched_sweep",
        "in_bank_sweep",
    )
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group in groups.items():
        item = dict(zip(group_fields, key))
        item["num_seeds"] = len(group)
        for metric in METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_stderr"] = (
                float(values.std(ddof=1) / math.sqrt(values.size))
                if values.size > 1
                else 0.0
            )
        summary.append(item)
    return sorted(
        summary,
        key=lambda row: (
            str(row["method"]),
            int(row["coefficient_dim"] or 0),
            int(row["operator_bank_size"] or 0),
        ),
    )


def _summarize_profiles(
    x_grid: torch.Tensor,
    profiles: dict[str, list[np.ndarray]],
) -> list[dict]:
    x = x_grid[:, 0].detach().cpu().numpy()
    rows = []
    for method in ("true_null", "vip", "gmvip"):
        values = np.stack(profiles[method], axis=0)
        mean = values.mean(axis=0)
        stderr = (
            values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
            if values.shape[0] > 1
            else np.zeros_like(mean)
        )
        rows.extend(
            {
                "x": float(x_value),
                "method": method,
                "mean": float(mean_value),
                "stderr": float(stderr_value),
            }
            for x_value, mean_value, stderr_value in zip(x, mean, stderr)
        )
    return rows


def _effective_config(args: argparse.Namespace, device: torch.device) -> dict:
    return {
        "experiment": "synthetic_prior_fidelity",
        "label_free": True,
        "smoke": bool(args.smoke),
        "architecture": {
            "input_dim": 1,
            "hidden_dims": [10, 10],
            "activation": "tanh",
            "output_dim": 1,
            "layer": "BayesLinear",
            "weight_mean": 0.0,
            "weight_std": 1.0,
            "bias_mean": 0.0,
            "bias_std": 1.0,
            "frozen": True,
        },
        "domain": {
            "x_min": float(args.x_min),
            "x_max": float(args.x_max),
            "grid_points": int(args.grid_points),
            "inducing_placement": "uniform",
        },
        "sampling": {
            "seeds": [int(seed) for seed in args.seeds],
            "num_samples_per_distribution": int(args.num_samples),
            "num_projections": int(args.num_projections),
            "robustness_samples": int(args.robustness_samples),
            "pairwise_chunk_size": int(args.pairwise_chunk_size),
            "independent_calibration_reference_null": True,
            "antithetic_samples": False,
        },
        "comparison": {
            "vip_published_default_s": 20,
            "gmvip_published_default_m": 256,
            "gmvip_published_default_b": 1024,
            "effective_vip_default_s": int(args.vip_default_s),
            "effective_gmvip_default_m": int(args.gmvip_default_m),
            "effective_gmvip_default_b": int(args.gmvip_default_b),
            "matched_dims": [int(value) for value in args.matched_dims],
            "gmvip_bank_sizes": [int(value) for value in args.gmvip_bank_sizes],
            "gmvip_jitter": float(args.gmvip_jitter),
            "gmvip_shrinkage": float(args.gmvip_shrinkage),
        },
        "runtime": {
            "device": str(device),
            "dtype": str(args.dtype),
        },
        "seed_policy": {
            "description": "Disjoint deterministic streams for calibration, reference, "
            "null, basis/operator banks, coefficients, and residual paths.",
        },
    }


def _apply_smoke_preset(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.seeds = [0]
    args.num_samples = 32
    args.num_projections = 16
    args.robustness_samples = 32
    args.grid_points = 31
    args.matched_dims = [4, 8]
    args.gmvip_bank_sizes = [16, 32]
    args.vip_default_s = 8
    args.gmvip_default_m = 8
    args.gmvip_default_b = 32
    args.saved_samples = 32
    args.pairwise_chunk_size = 16


def run_experiment(args: argparse.Namespace) -> dict:
    _apply_smoke_preset(args)
    device = _resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = _build_settings(args)
    config = _effective_config(args, device)
    config["settings"] = [asdict(setting) for setting in settings]
    write_json(output_dir / "run_config.json", config)

    x_grid = torch.linspace(
        float(args.x_min),
        float(args.x_max),
        int(args.grid_points),
        device=device,
        dtype=dtype,
    ).unsqueeze(-1)
    rows: list[dict] = []
    profiles: dict[str, list[np.ndarray]] = {
        "true_null": [],
        "vip": [],
        "gmvip": [],
    }
    sample_payload: dict[str, np.ndarray] = {
        "x_grid": x_grid[:, 0].detach().cpu().numpy()
    }
    display_seed = int(args.seeds[0])

    for seed in args.seeds:
        common_seeds = _seed_map(int(seed))
        calibration = sample_true_prior(
            x_grid,
            args.num_samples,
            prior_seed=common_seeds["prior_constructor"],
            sample_seed=common_seeds["calibration"],
        )
        reference = sample_true_prior(
            x_grid,
            args.num_samples,
            prior_seed=common_seeds["prior_constructor"],
            sample_seed=common_seeds["reference"],
        )
        null = sample_true_prior(
            x_grid,
            args.num_samples,
            prior_seed=common_seeds["prior_constructor"],
            sample_seed=common_seeds["null"],
        )
        standardizer_mean, standardizer_std = fit_pointwise_standardizer(calibration)
        reference_standardized = standardize_curves(
            reference, standardizer_mean, standardizer_std
        )
        null_standardized = standardize_curves(null, standardizer_mean, standardizer_std)
        directions = projection_directions(
            x_grid.shape[0],
            args.num_projections,
            seed=common_seeds["projections"],
            device=device,
            dtype=dtype,
        )
        bandwidth = estimate_rbf_bandwidth(
            reference_standardized[: args.robustness_samples]
        )
        null_metrics, null_profile = fidelity_metrics(
            reference_standardized,
            null_standardized,
            directions=directions,
            mmd_bandwidth=bandwidth,
            chunk_size=args.pairwise_chunk_size,
            robustness_max_samples=args.robustness_samples,
        )
        rows.append(_null_row(int(seed), null_metrics, args))
        profiles["true_null"].append(null_profile.cpu().numpy())

        if int(seed) == display_seed:
            saved = min(int(args.saved_samples), reference.shape[0])
            sample_payload.update(
                {
                    "true_reference": reference[:saved].cpu().numpy(),
                    "true_null": null[:saved].cpu().numpy(),
                    "standardizer_mean": standardizer_mean.cpu().numpy(),
                    "standardizer_std": standardizer_std.cpu().numpy(),
                    "pointwise_w1_true_null": null_profile.cpu().numpy(),
                }
            )

        for setting in settings:
            setting_seeds = _seed_map(int(seed), setting)
            if setting.method == "vip":
                candidate, diagnostics = sample_vip_surrogate(
                    x_grid,
                    basis_size=setting.coefficient_dim,
                    num_samples=args.num_samples,
                    prior_seed=setting_seeds["prior_constructor"],
                    basis_seed=setting_seeds["basis_or_operator"],
                    coefficient_seed=setting_seeds["coefficients_or_samples"],
                )
            else:
                candidate, diagnostics = sample_gmvip_surrogate(
                    x_grid,
                    num_inducing=setting.coefficient_dim,
                    operator_bank_size=int(setting.operator_bank_size),
                    num_samples=args.num_samples,
                    prior_seed=setting_seeds["prior_constructor"],
                    operator_seed=setting_seeds["basis_or_operator"],
                    sample_seed=setting_seeds["coefficients_or_samples"],
                    jitter=args.gmvip_jitter,
                    shrinkage=args.gmvip_shrinkage,
                )
            candidate_standardized = standardize_curves(
                candidate, standardizer_mean, standardizer_std
            )
            metrics, profile = fidelity_metrics(
                reference_standardized,
                candidate_standardized,
                directions=directions,
                mmd_bandwidth=bandwidth,
                chunk_size=args.pairwise_chunk_size,
                robustness_max_samples=args.robustness_samples,
            )
            coefficient_prior_kl = float(
                diagnostics["coefficient_prior_kl"].detach().cpu()
            )
            rows.append(
                _row_for_setting(
                    seed=int(seed),
                    setting=setting,
                    metrics=metrics,
                    coefficient_prior_kl=coefficient_prior_kl,
                    args=args,
                )
            )
            if setting.is_display_default:
                profiles[setting.method].append(profile.cpu().numpy())
                if int(seed) == display_seed:
                    saved = min(int(args.saved_samples), candidate.shape[0])
                    sample_payload[f"{setting.method}_default"] = (
                        candidate[:saved].cpu().numpy()
                    )
                    sample_payload[f"pointwise_w1_{setting.method}"] = profile.cpu().numpy()
            write_csv_rows(output_dir / "metrics.csv", rows)
            del candidate, candidate_standardized, diagnostics
        del calibration, reference, null
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = _summarize(rows)
    profile_rows = _summarize_profiles(x_grid, profiles)
    write_csv_rows(output_dir / "metrics.csv", rows)
    write_csv_rows(output_dir / "summary.csv", summary_rows)
    write_csv_rows(output_dir / "pointwise_w1.csv", profile_rows)
    np.savez_compressed(output_dir / "default_samples_and_profiles.npz", **sample_payload)

    if not args.skip_plots:
        plot_prior_samples(output_dir, sample_payload)
        plot_pointwise_w1(output_dir, profile_rows)
        plot_matched_sweeps(output_dir, summary_rows)
        plot_gmvip_bank_sensitivity(output_dir, summary_rows)

    return {
        "output_dir": str(output_dir),
        "metrics": rows,
        "summary": summary_rows,
        "profiles": profile_rows,
        "config": config,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a frozen one-dimensional BNN prior with VIP and GMVIP surrogates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="results/synthetic_prior_fidelity")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--grid-points", type=int, default=301)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--num-projections", type=int, default=512)
    parser.add_argument(
        "--robustness-samples",
        type=int,
        default=512,
        help="Deterministic sample prefix used for pairwise energy and MMD metrics.",
    )
    parser.add_argument("--pairwise-chunk-size", type=int, default=256)
    parser.add_argument(
        "--matched-dims",
        type=int,
        nargs="+",
        default=[8, 20, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--gmvip-bank-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
    )
    parser.add_argument("--vip-default-s", type=int, default=20)
    parser.add_argument("--gmvip-default-m", type=int, default=256)
    parser.add_argument("--gmvip-default-b", type=int, default=1024)
    parser.add_argument("--gmvip-jitter", type=float, default=1e-5)
    parser.add_argument("--gmvip-shrinkage", type=float, default=0.02)
    parser.add_argument("--saved-samples", type=int, default=256)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny deterministic configuration suitable for CI.",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.x_min < args.x_max:
        parser.error("--x-min must be smaller than --x-max.")
    positive_names = (
        "grid_points",
        "num_samples",
        "num_projections",
        "robustness_samples",
        "pairwise_chunk_size",
        "vip_default_s",
        "gmvip_default_m",
        "gmvip_default_b",
        "saved_samples",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.vip_default_s < 2 or any(value < 2 for value in args.matched_dims):
        parser.error("VIP basis sizes and matched dimensions must be at least 2.")
    if any(value <= 0 for value in args.gmvip_bank_sizes):
        parser.error("GMVIP bank sizes must be positive.")


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    result = run_experiment(args)
    print(f"Prior-fidelity artifacts written to {result['output_dir']}")
    return result


if __name__ == "__main__":
    main()
