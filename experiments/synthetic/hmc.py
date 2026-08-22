"""Hamiltonian Monte Carlo reference for the synthetic regression experiment.

The synthetic benchmark uses a small, fully specified Bayesian neural network,
so its weight-space posterior has a tractable density even though the induced
function-space prior is implicit.  This module keeps the Hamiltorch dependency lazy:
the rest of the repository remains usable without the optional ``hmc`` extra.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

WEIGHT_SITE_SHAPES = {
    "w1": (1, 10),
    "b1": (10,),
    "w2": (10, 10),
    "b2": (10,),
    "w3": (10, 1),
    "b3": (1,),
}
WEIGHT_SITE_NAMES = tuple(WEIGHT_SITE_SHAPES)
NOISE_SITE_NAME = "log_sigma_y"
HMC_PARAMETER_COUNT = sum(math.prod(shape) for shape in WEIGHT_SITE_SHAPES.values())
DEFAULT_MAX_RHAT = 1.01
DEFAULT_MIN_ESS = 400.0
DEFAULT_MAX_DIVERGENCES = 0
DEFAULT_DIAGNOSTIC_GRID_POINTS = 31
HAMILTORCH_REVISION = "19b627b2aabc77c1b4b78db0f860372eb1bf9778"
DEFAULT_MIN_ACCEPTANCE_RATE = 0.60


@dataclass(frozen=True)
class HMCConfig:
    """Configuration for the standard synthetic HMC reference."""

    chains: int = 1
    warmup_steps: int = 0
    num_samples: int = 1000
    num_predictive_samples: int = 1000
    step_size: float = 0.0005
    num_steps: int = 500
    inverse_mass: float = 0.1
    map_warmstart_steps: int = 1000
    map_warmstart_lr: float = 0.003
    initialization_jitter: float = 0.01
    device: str = "cuda"
    noise_log_loc: float = -2.5
    noise_log_scale: float = 1.0
    seed: int = 0
    divergence_energy_threshold: float = 1000.0
    diagnostic_grid_points: int = DEFAULT_DIAGNOSTIC_GRID_POINTS
    disable_progress: bool = False

    def validate(self) -> None:
        if self.chains <= 0:
            raise ValueError("HMC requires at least one chain.")
        if self.warmup_steps < 0:
            raise ValueError("HMC warm-up steps must be non-negative.")
        if self.num_samples <= 0:
            raise ValueError("HMC requires at least one retained draw per chain.")
        total_draws = self.chains * self.num_samples
        if not 0 < self.num_predictive_samples <= total_draws:
            raise ValueError(
                "HMC predictive draws must be positive and no larger than the "
                f"{total_draws} retained draws."
            )
        if self.step_size <= 0.0:
            raise ValueError("HMC step size must be positive.")
        if self.num_steps <= 0:
            raise ValueError("HMC leapfrog steps must be positive.")
        if self.inverse_mass <= 0.0:
            raise ValueError("HMC inverse mass must be positive.")
        if self.map_warmstart_steps < 0:
            raise ValueError("HMC MAP warm-start steps must be non-negative.")
        if self.map_warmstart_lr <= 0.0:
            raise ValueError("HMC MAP warm-start learning rate must be positive.")
        if self.initialization_jitter < 0.0:
            raise ValueError("HMC initialization jitter must be non-negative.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("HMC device must be either 'cpu' or 'cuda'.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("HMC device 'cuda' was requested but CUDA is not available.")
        if self.noise_log_scale <= 0.0:
            raise ValueError("HMC noise-prior log scale must be positive.")
        if self.divergence_energy_threshold <= 0.0:
            raise ValueError("HMC divergence energy threshold must be positive.")
        if self.diagnostic_grid_points <= 0:
            raise ValueError("HMC diagnostic grid size must be positive.")


@dataclass(frozen=True)
class HMCPredictive:
    """Predictive mixture arrays in the original target units."""

    means: np.ndarray
    stds: np.ndarray
    mixture_mean: np.ndarray
    mixture_std: np.ndarray


def hamiltorch_available() -> bool:
    """Return whether the pinned optional Hamiltorch backend can be imported."""

    return importlib.util.find_spec("hamiltorch") is not None


def hamiltorch_missing_reason() -> str | None:
    """Return an actionable installation message when Hamiltorch is unavailable."""

    if hamiltorch_available():
        return None
    return (
        "HMC requires the optional pinned Hamiltorch dependency. Install it with "
        '`python -m pip install -e ".[experiments,hmc]"`.'
    )


def bnn_forward(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Evaluate the 1-10-10-1 tanh BNN for scalar or batched parameters."""

    missing = set(WEIGHT_SITE_NAMES).difference(params)
    if missing:
        raise KeyError(f"Missing HMC parameter sites: {sorted(missing)}")
    if x.ndim != 2 or x.shape[-1] != 1:
        raise ValueError(f"HMC inputs must have shape [N,1], got {tuple(x.shape)}.")

    h1 = torch.tanh(torch.einsum("ni,...ih->...nh", x, params["w1"]) + params["b1"].unsqueeze(-2))
    h2 = torch.tanh(
        torch.einsum("...ni,...ih->...nh", h1, params["w2"]) + params["b2"].unsqueeze(-2)
    )
    return torch.einsum("...ni,...io->...no", h2, params["w3"]) + params["b3"].unsqueeze(-2)


def hmc_log_joint(
    params: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    noise_log_loc: float = -2.5,
    noise_log_scale: float = 1.0,
) -> torch.Tensor:
    """Evaluate the normalized-data log joint used by the Hamiltorch sampler."""

    if NOISE_SITE_NAME not in params:
        raise KeyError(f"Missing HMC parameter site {NOISE_SITE_NAME!r}.")
    if y.shape != (x.shape[0], 1):
        raise ValueError(f"HMC targets must have shape [N,1], got {tuple(y.shape)}.")

    log_two_pi = math.log(2.0 * math.pi)
    log_prob = sum(-0.5 * (params[name].square() + log_two_pi).sum() for name in WEIGHT_SITE_NAMES)
    log_sigma = params[NOISE_SITE_NAME]
    standardized_log_sigma = (log_sigma - float(noise_log_loc)) / float(noise_log_scale)
    log_prob = (
        log_prob
        - 0.5 * standardized_log_sigma.square()
        - math.log(float(noise_log_scale))
        - 0.5 * log_two_pi
    )
    standardized_residual = (y - bnn_forward(params, x)) * torch.exp(-log_sigma)
    likelihood = (
        -0.5 * standardized_residual.square().sum()
        - y.numel() * log_sigma
        - 0.5 * y.numel() * log_two_pi
    )
    return log_prob + likelihood


def select_draw_indices(chains: int, draws: int, count: int) -> np.ndarray:
    """Select deterministic, approximately chain-balanced posterior draws."""

    total = int(chains) * int(draws)
    if chains <= 0 or draws <= 0 or not 0 < count <= total:
        raise ValueError("Requested predictive draw count must lie in [1, chains * draws].")
    flat = np.linspace(0, total - 1, num=count, dtype=np.int64)
    chain, draw = np.unravel_index(flat, (chains, draws))
    return np.stack([chain, draw], axis=1)


def posterior_predictive(
    samples: dict[str, torch.Tensor],
    x_grid: torch.Tensor,
    *,
    y_mean: float,
    y_std: float,
    num_predictive_samples: int,
) -> tuple[HMCPredictive, np.ndarray]:
    """Convert posterior parameter draws to GMVIP-compatible mixture components."""

    _validate_sample_shapes(samples)
    chains, draws = samples[NOISE_SITE_NAME].shape[:2]
    selected_indices = select_draw_indices(chains, draws, num_predictive_samples)
    chain_idx = torch.as_tensor(selected_indices[:, 0], dtype=torch.long, device=x_grid.device)
    draw_idx = torch.as_tensor(selected_indices[:, 1], dtype=torch.long, device=x_grid.device)
    selected = {
        name: value.to(x_grid.device)[chain_idx, draw_idx]
        for name, value in samples.items()
        if name in {*WEIGHT_SITE_NAMES, NOISE_SITE_NAME}
    }

    function_norm = bnn_forward(selected, x_grid)[..., 0]
    means = function_norm * float(y_std) + float(y_mean)
    noise_std = torch.exp(selected[NOISE_SITE_NAME]) * float(y_std)
    stds = noise_std[:, None].expand_as(means)
    mixture_mean = means.mean(dim=0)
    mixture_var = (stds.square() + means.square()).mean(dim=0) - mixture_mean.square()

    predictive = HMCPredictive(
        means=means.detach().cpu().numpy(),
        stds=stds.detach().cpu().numpy(),
        mixture_mean=mixture_mean.detach().cpu().numpy(),
        mixture_std=torch.sqrt(torch.clamp(mixture_var, min=1e-12)).detach().cpu().numpy(),
    )
    return predictive, selected_indices


def convergence_diagnostics(
    samples: dict[str, torch.Tensor],
    x_grid: torch.Tensor,
    *,
    divergence_count: int,
    acceptance_rates: list[float] | None = None,
    max_rhat: float = DEFAULT_MAX_RHAT,
    min_ess: float = DEFAULT_MIN_ESS,
    max_divergences: int = DEFAULT_MAX_DIVERGENCES,
    min_acceptance_rate: float = DEFAULT_MIN_ACCEPTANCE_RATE,
    diagnostic_grid_points: int = DEFAULT_DIAGNOSTIC_GRID_POINTS,
) -> dict[str, Any]:
    """Compute function-space diagnostics and a strict convergence decision."""

    _validate_sample_shapes(samples)

    grid_count = min(int(diagnostic_grid_points), int(x_grid.shape[0]))
    grid_indices = (
        torch.linspace(
            0,
            x_grid.shape[0] - 1,
            steps=grid_count,
            dtype=torch.float64,
            device=x_grid.device,
        )
        .round()
        .to(torch.long)
    )
    x_diagnostic = x_grid[grid_indices]
    function_values = _posterior_function_grid(samples, x_diagnostic)
    log_noise = samples[NOISE_SITE_NAME].to(x_grid.device)
    diagnostic_values = torch.cat([function_values, log_noise[..., None]], dim=-1)

    rhat, ess = _split_rhat_and_ess(diagnostic_values)
    finite_rhat = rhat[torch.isfinite(rhat)]
    finite_ess = ess[torch.isfinite(ess)]
    all_diagnostics_finite = bool(torch.isfinite(rhat).all() and torch.isfinite(ess).all())
    observed_max_rhat = (
        float(finite_rhat.max().item())
        if finite_rhat.numel() and all_diagnostics_finite
        else math.inf
    )
    observed_min_ess = (
        float(finite_ess.min().item()) if finite_ess.numel() and all_diagnostics_finite else 0.0
    )

    failures = []
    if not all_diagnostics_finite:
        failures.append("function/noise diagnostics contain non-finite values")
    if divergence_count > max_divergences:
        failures.append(f"divergences={divergence_count} exceeds the allowed {max_divergences}")
    acceptance_rates = (
        [] if acceptance_rates is None else [float(rate) for rate in acceptance_rates]
    )
    observed_min_acceptance = min(acceptance_rates, default=1.0)
    if observed_min_acceptance < min_acceptance_rate:
        failures.append(
            f"minimum chain acceptance={observed_min_acceptance:.3f} "
            f"is below {min_acceptance_rate:.3f}"
        )
    if observed_max_rhat > max_rhat:
        failures.append(f"max function/noise R-hat={observed_max_rhat:.4f} exceeds {max_rhat}")
    if observed_min_ess < min_ess:
        failures.append(f"min function/noise ESS={observed_min_ess:.1f} is below {min_ess:.1f}")

    flat_parameters = torch.cat(
        [
            samples[name].reshape(samples[name].shape[0], samples[name].shape[1], -1)
            for name in (*WEIGHT_SITE_NAMES, NOISE_SITE_NAME)
        ],
        dim=-1,
    )
    parameter_rhat, parameter_ess = _split_rhat_and_ess(flat_parameters)
    return {
        "converged": not failures,
        "failures": failures,
        "thresholds": {
            "max_function_noise_rhat": float(max_rhat),
            "min_function_noise_ess": float(min_ess),
            "max_divergences": int(max_divergences),
            "min_chain_acceptance": float(min_acceptance_rate),
        },
        "function_noise": {
            "grid_indices": grid_indices.detach().cpu().tolist(),
            "r_hat": _jsonable(rhat),
            "effective_sample_size": _jsonable(ess),
            "max_r_hat": observed_max_rhat,
            "min_effective_sample_size": observed_min_ess,
            "all_finite": all_diagnostics_finite,
            "includes_log_noise_as_last_entry": True,
        },
        "parameters": {
            "r_hat": _jsonable(parameter_rhat),
            "effective_sample_size": _jsonable(parameter_ess),
            "max_r_hat": float(torch.nan_to_num(parameter_rhat, nan=math.inf).max().item()),
            "min_effective_sample_size": float(
                torch.nan_to_num(parameter_ess, nan=0.0).min().item()
            ),
        },
        "divergences": int(divergence_count),
        "acceptance_rates": acceptance_rates,
        "minimum_acceptance_rate": observed_min_acceptance,
        "parameter_diagnostics_are_gating": False,
    }


def save_hmc_artifacts(
    output_dir: Path,
    *,
    samples: dict[str, torch.Tensor],
    selected_indices: np.ndarray,
    x_grid: np.ndarray,
    predictive: HMCPredictive,
    result: dict[str, Any],
) -> dict[str, str]:
    """Save portable posterior/predictive arrays and an auditable JSON summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "hmc_posterior_samples.npz"
    summary_path = output_dir / "hmc_summary.json"
    arrays: dict[str, Any] = {
        f"posterior_{name}": value.detach().cpu().numpy() for name, value in samples.items()
    }
    arrays.update(
        {
            "selected_chain_draw_indices": np.asarray(selected_indices, dtype=np.int64),
            "x_grid": np.asarray(x_grid, dtype=np.float64),
            "predictive_component_means": predictive.means,
            "predictive_component_stds": predictive.stds,
            "predictive_mixture_mean": predictive.mixture_mean,
            "predictive_mixture_std": predictive.mixture_std,
            "configuration_json": np.asarray(
                json.dumps(result["inference"], sort_keys=True), dtype=np.str_
            ),
        }
    )
    np.savez_compressed(sample_path, **arrays)
    result["artifacts"] = {
        "posterior_predictive": str(sample_path),
        "summary": str(summary_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(result), handle, indent=2)
    return result["artifacts"]


def run_hmc_reference(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    x_grid: np.ndarray,
    y_mean: float,
    y_std: float,
    output_dir: Path,
    config: HMCConfig,
    dataset_name: str = "synthetic",
) -> tuple[dict[str, Any], HMCPredictive]:
    """Run sequential HMC-family chains and return a plot-ready predictive mixture."""

    config.validate()
    missing = hamiltorch_missing_reason()
    if missing is not None:
        raise ImportError(missing)

    import hamiltorch

    device = torch.device(config.device)
    x = _as_column_tensor(train_x, name="train_x", device=device)
    y = _as_column_tensor(train_y, name="train_y", device=device)
    grid = _as_column_tensor(x_grid, name="x_grid", device=device)
    if x.shape != y.shape:
        raise ValueError(f"HMC train inputs and targets must align, got {x.shape} and {y.shape}.")

    started = time.perf_counter()
    map_initialization, map_log_joint = _fit_map_initialization(
        x,
        y,
        config=config,
        device=device,
    )
    inverse_mass = torch.full_like(map_initialization, config.inverse_mass)
    chain_samples: list[torch.Tensor] = []
    sampler_diagnostics = []
    divergence_count = 0
    acceptance_rates = []
    chain_seeds = []
    for chain_idx in range(config.chains):
        chain_seed = int(config.seed) + 100_003 * chain_idx
        chain_seeds.append(chain_seed)
        hamiltorch.set_random_seed(chain_seed)
        flat_initial = map_initialization + config.initialization_jitter * torch.randn_like(
            map_initialization
        )

        def log_prob_func(flat_params: torch.Tensor) -> torch.Tensor:
            return hmc_log_joint(
                _unflatten_params(flat_params),
                x,
                y,
                noise_log_loc=config.noise_log_loc,
                noise_log_scale=config.noise_log_scale,
            )

        current = flat_initial
        warmup_acceptance_rates = []
        warmup_energy_errors = []
        if config.warmup_steps:
            current, _, window_acceptance, window_errors = _hamiltorch_sample_block(
                hamiltorch,
                log_prob_func,
                current,
                proposals=config.warmup_steps,
                num_steps=config.num_steps,
                step_size=config.step_size,
                inverse_mass=inverse_mass,
                show_progress=False,
            )
            warmup_acceptance_rates.append(window_acceptance)
            warmup_energy_errors.extend(window_errors)

        _, retained, acceptance_rate, energy_errors = _hamiltorch_sample_block(
            hamiltorch,
            log_prob_func,
            current,
            proposals=config.num_samples,
            num_steps=config.num_steps,
            step_size=config.step_size,
            inverse_mass=inverse_mass,
            show_progress=not config.disable_progress,
        )
        if retained.shape != (config.num_samples, HMC_PARAMETER_COUNT + 1):
            raise RuntimeError(
                f"Hamiltorch returned an unexpected posterior shape: {tuple(retained.shape)}."
            )
        chain_samples.append(retained)
        acceptance_rate = float(acceptance_rate)
        acceptance_rates.append(acceptance_rate)
        chain_divergences = _count_energy_divergences(
            energy_errors,
            threshold=config.divergence_energy_threshold,
        )
        divergence_count += chain_divergences
        sampler_diagnostics.append(
            {
                "chain": chain_idx,
                "seed": chain_seed,
                "acceptance_rate": acceptance_rate,
                "step_size": config.step_size,
                "warmup_acceptance_rates": warmup_acceptance_rates,
                "warmup_divergences": _count_energy_divergences(
                    warmup_energy_errors,
                    threshold=config.divergence_energy_threshold,
                ),
                "divergences": chain_divergences,
                "max_absolute_energy_error": max(
                    (abs(error) for error in energy_errors if math.isfinite(error)),
                    default=math.inf,
                ),
                "proposals": len(energy_errors),
            }
        )

    flat_samples = torch.stack(chain_samples, dim=0)
    samples = _unflatten_sample_batch(flat_samples)
    if config.chains >= 2 and config.num_samples >= 4:
        diagnostics = convergence_diagnostics(
            samples,
            grid,
            divergence_count=divergence_count,
            acceptance_rates=acceptance_rates,
            diagnostic_grid_points=config.diagnostic_grid_points,
        )
        diagnostics["assessed"] = True
    else:
        diagnostics = {
            "assessed": False,
            "converged": None,
            "reason": (
                "The BayesiPy notebook protocol uses one chain, so split-R-hat "
                "and cross-chain ESS are unavailable."
            ),
            "divergences": int(divergence_count),
            "acceptance_rates": acceptance_rates,
            "minimum_acceptance_rate": min(acceptance_rates, default=None),
            "parameter_diagnostics_are_gating": False,
        }
    diagnostics["sampler_by_chain"] = sampler_diagnostics
    predictive, selected_indices = posterior_predictive(
        samples,
        grid,
        y_mean=float(y_mean),
        y_std=float(y_std),
        num_predictive_samples=config.num_predictive_samples,
    )
    runtime = time.perf_counter() - started
    inference = asdict(config)
    inference.update(
        {
            "sampler": "hamiltorch_hmc",
            "hamiltorch_version": str(hamiltorch.__version__),
            "hamiltorch_revision": HAMILTORCH_REVISION,
            "source_protocol": (
                "https://github.com/Ludvins/BayesiPy/blob/main/"
                "examples/Synthetic_1D_regression.ipynb"
            ),
            "burn": -1,
            "dtype": "float64",
            "chain_execution": "sequential",
            "gradient_execution": (
                "cuda_graph_exact_autograd" if device.type == "cuda" else "eager_autograd"
            ),
            "chain_seeds": chain_seeds,
            "weight_prior": "independent_standard_normal",
            "mass_matrix": {
                "type": "fixed_diagonal",
                "inverse_mass": float(config.inverse_mass),
            },
            "initialization": {
                "type": "shared_map_with_independent_gaussian_jitter",
                "map_steps": config.map_warmstart_steps,
                "map_learning_rate": config.map_warmstart_lr,
                "jitter_standard_deviation": config.initialization_jitter,
                "map_log_joint": map_log_joint,
            },
            "noise_prior": {
                "distribution": "log_normal",
                "log_location": config.noise_log_loc,
                "log_scale": config.noise_log_scale,
            },
            "network": [1, 10, 10, 1],
            "weight_bias_parameter_count": HMC_PARAMETER_COUNT,
            "joint_parameter_count": HMC_PARAMETER_COUNT + 1,
            "posterior_prediction": (
                "one function path and sampled observation-noise standard "
                "deviation per retained draw"
            ),
        }
    )
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "model": "hmc",
        "train_time_s": round(runtime, 2),
        "inference": inference,
        "diagnostics": diagnostics,
        "converged": diagnostics["converged"],
    }
    save_hmc_artifacts(
        Path(output_dir),
        samples=samples,
        selected_indices=selected_indices,
        x_grid=np.asarray(x_grid, dtype=np.float64),
        predictive=predictive,
        result=result,
    )

    return result, predictive


def _initial_values(config: HMCConfig, *, device: torch.device) -> dict[str, torch.Tensor]:
    values = {
        name: 0.05 * torch.randn(shape, dtype=torch.float64, device=device)
        for name, shape in WEIGHT_SITE_SHAPES.items()
    }
    values[NOISE_SITE_NAME] = torch.tensor(
        config.noise_log_loc,
        dtype=torch.float64,
        device=device,
    ) + 0.05 * torch.randn((), dtype=torch.float64, device=device)
    return values


def _fit_map_initialization(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    config: HMCConfig,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    initial = _flatten_params(_initial_values(config, device=device))
    if config.map_warmstart_steps == 0:
        log_joint = hmc_log_joint(
            _unflatten_params(initial),
            x,
            y,
            noise_log_loc=config.noise_log_loc,
            noise_log_scale=config.noise_log_scale,
        )
        return initial.detach(), float(log_joint.detach().cpu())

    flat_params = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([flat_params], lr=config.map_warmstart_lr)
    for _ in range(config.map_warmstart_steps):
        optimizer.zero_grad(set_to_none=True)
        negative_log_joint = -hmc_log_joint(
            _unflatten_params(flat_params),
            x,
            y,
            noise_log_loc=config.noise_log_loc,
            noise_log_scale=config.noise_log_scale,
        )
        if not torch.isfinite(negative_log_joint):
            raise RuntimeError("HMC MAP warm start produced a non-finite objective.")
        negative_log_joint.backward()
        torch.nn.utils.clip_grad_norm_([flat_params], max_norm=100.0)
        optimizer.step()

    log_joint = hmc_log_joint(
        _unflatten_params(flat_params),
        x,
        y,
        noise_log_loc=config.noise_log_loc,
        noise_log_scale=config.noise_log_scale,
    )
    return flat_params.detach(), float(log_joint.detach().cpu())


def _hamiltorch_sample_block(
    hamiltorch,
    log_prob_func,
    initial: torch.Tensor,
    *,
    proposals: int,
    num_steps: int,
    step_size: float,
    inverse_mass: torch.Tensor,
    show_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, float, list[float]]:
    energy_errors: list[float] = []
    original_acceptance = hamiltorch.samplers.acceptance
    original_leapfrog = hamiltorch.samplers.leapfrog
    original_empty_cache = torch.cuda.empty_cache

    def recording_acceptance(old_energy, new_energy):
        delta = float((new_energy - old_energy).detach().cpu())
        energy_errors.append(delta)
        return original_acceptance(old_energy, new_energy)

    hamiltorch.samplers.acceptance = recording_acceptance
    if initial.device.type == "cuda":
        static_parameters = initial.detach().clone().requires_grad_(True)
        capture_stream = torch.cuda.Stream(device=initial.device)
        capture_stream.wait_stream(torch.cuda.current_stream(initial.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                torch.autograd.grad(log_prob_func(static_parameters), static_parameters)
        torch.cuda.current_stream(initial.device).wait_stream(capture_stream)

        gradient_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gradient_graph):
            static_gradient = torch.autograd.grad(
                log_prob_func(static_parameters),
                static_parameters,
            )[0]

        def graph_gradient(parameters):
            with torch.no_grad():
                static_parameters.copy_(parameters)
            gradient_graph.replay()
            return static_gradient

        def graph_leapfrog(parameters, momentum, inner_log_prob_func, **kwargs):
            if (
                inner_log_prob_func is not log_prob_func
                or kwargs.get("sampler") != hamiltorch.Sampler.HMC
            ):
                return original_leapfrog(
                    parameters,
                    momentum,
                    inner_log_prob_func,
                    **kwargs,
                )
            steps = int(kwargs["steps"])
            epsilon = float(kwargs["step_size"])
            block_inverse_mass = kwargs["inv_mass"]
            if not isinstance(block_inverse_mass, torch.Tensor) or block_inverse_mass.ndim != 1:
                raise ValueError("CUDA-graph HMC requires a diagonal inverse-mass tensor.")

            position = parameters.detach()
            updated_momentum = momentum + 0.5 * epsilon * graph_gradient(position)
            for _ in range(steps):
                position = position + epsilon * block_inverse_mass * updated_momentum
                position_gradient = graph_gradient(position)
                updated_momentum = updated_momentum + epsilon * position_gradient
            updated_momentum = updated_momentum - 0.5 * epsilon * position_gradient
            return [position], [updated_momentum]

        hamiltorch.samplers.leapfrog = graph_leapfrog
        # Hamiltorch calls this after every gradient evaluation. Retaining the
        # allocator cache leaves the transition kernel unchanged and avoids
        # hundreds of device synchronizations per notebook-style trajectory.
        torch.cuda.empty_cache = lambda: None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Converting a tensor with requires_grad=True to a scalar",
            )
            returned, acceptance_rate = hamiltorch.sample(
                log_prob_func,
                initial,
                num_samples=proposals,
                burn=-1,
                num_steps_per_sample=num_steps,
                step_size=step_size,
                inv_mass=inverse_mass,
                sampler=hamiltorch.Sampler.HMC,
                store_on_GPU=True,
                debug=2,
                verbose=show_progress,
            )
    finally:
        hamiltorch.samplers.acceptance = original_acceptance
        hamiltorch.samplers.leapfrog = original_leapfrog
        torch.cuda.empty_cache = original_empty_cache

    retained = torch.stack(returned[1:], dim=0)
    return returned[-1], retained, float(acceptance_rate), energy_errors


def _count_energy_divergences(energy_errors: list[float], *, threshold: float) -> int:
    return sum(not math.isfinite(error) or abs(error) > threshold for error in energy_errors)


def _flatten_params(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [params[name].reshape(-1) for name in (*WEIGHT_SITE_NAMES, NOISE_SITE_NAME)],
        dim=0,
    )


def _unflatten_params(flat_params: torch.Tensor) -> dict[str, torch.Tensor]:
    if flat_params.ndim != 1 or flat_params.numel() != HMC_PARAMETER_COUNT + 1:
        raise ValueError(
            f"Flat HMC parameters must have shape [{HMC_PARAMETER_COUNT + 1}], "
            f"got {tuple(flat_params.shape)}."
        )
    result = {}
    start = 0
    for name, shape in WEIGHT_SITE_SHAPES.items():
        stop = start + math.prod(shape)
        result[name] = flat_params[start:stop].reshape(shape)
        start = stop
    result[NOISE_SITE_NAME] = flat_params[start]
    return result


def _unflatten_sample_batch(flat_samples: torch.Tensor) -> dict[str, torch.Tensor]:
    if flat_samples.ndim != 3 or flat_samples.shape[-1] != HMC_PARAMETER_COUNT + 1:
        raise ValueError(
            "Flat HMC samples must have shape [chains,draws,parameters], got "
            f"{tuple(flat_samples.shape)}."
        )
    result = {}
    start = 0
    for name, shape in WEIGHT_SITE_SHAPES.items():
        stop = start + math.prod(shape)
        result[name] = flat_samples[..., start:stop].reshape(*flat_samples.shape[:2], *shape)
        start = stop
    result[NOISE_SITE_NAME] = flat_samples[..., start]
    return result


def _as_column_tensor(values: np.ndarray, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != 1 or tensor.shape[0] == 0:
        raise ValueError(f"{name} must have nonempty shape [N,1], got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def _validate_sample_shapes(samples: dict[str, torch.Tensor]) -> None:
    required = {*WEIGHT_SITE_NAMES, NOISE_SITE_NAME}
    missing = required.difference(samples)
    if missing:
        raise KeyError(f"Missing HMC posterior sites: {sorted(missing)}")
    noise_shape = samples[NOISE_SITE_NAME].shape
    if len(noise_shape) != 2:
        raise ValueError(
            f"HMC posterior log noise must have shape [chains,draws], got {noise_shape}."
        )
    leading = noise_shape
    for name, event_shape in WEIGHT_SITE_SHAPES.items():
        expected = (*leading, *event_shape)
        if tuple(samples[name].shape) != expected:
            raise ValueError(
                f"HMC posterior site {name!r} must have shape {expected}, "
                f"got {tuple(samples[name].shape)}."
            )


def _posterior_function_grid(
    samples: dict[str, torch.Tensor],
    x_grid: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    chains, draws = samples[NOISE_SITE_NAME].shape
    flat_samples = {
        name: value.to(x_grid.device).reshape(chains * draws, *value.shape[2:])
        for name, value in samples.items()
        if name in WEIGHT_SITE_NAMES
    }
    chunks = []
    for start in range(0, chains * draws, chunk_size):
        stop = min(start + chunk_size, chains * draws)
        chunk = {name: value[start:stop] for name, value in flat_samples.items()}
        chunks.append(bnn_forward(chunk, x_grid)[..., 0])
    return torch.cat(chunks, dim=0).reshape(chains, draws, x_grid.shape[0])


def _split_rhat_and_ess(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return split-R-hat and bulk autocorrelation ESS along the last dimension."""

    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("Diagnostics require values shaped [chains,draws,variables].")
    chains, draws, variables = values.shape
    half = draws // 2
    split = values[:, : 2 * half].reshape(chains, 2, half, variables)
    split = split.reshape(chains * 2, half, variables)
    split_chains = split.shape[0]

    chain_means = split.mean(dim=1)
    within = split.var(dim=1, unbiased=True).mean(dim=0)
    between = half * chain_means.var(dim=0, unbiased=True)
    variance = ((half - 1) * within + between) / half
    rhat = torch.sqrt(variance / within)

    centered = split - chain_means[:, None, :]
    fft_size = 1 << (2 * half - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=fft_size, dim=1)
    autocov = torch.fft.irfft(spectrum * spectrum.conj(), n=fft_size, dim=1).real[:, :half]
    autocov = autocov / half
    mean_autocov = autocov.mean(dim=0)
    variance = torch.clamp(variance, min=torch.finfo(values.dtype).tiny)
    rho = 1.0 - (within[None, :] - mean_autocov) / variance[None, :]
    rho[0] = 1.0

    pair_count = (half - 1) // 2
    if pair_count:
        pair_sums = rho[1 : 2 * pair_count + 1].reshape(pair_count, 2, variables).sum(dim=1)
        positive = pair_sums > 0
        still_positive = torch.cumprod(positive.to(torch.int64), dim=0).to(torch.bool)
        tau = -1.0 + 2.0 * (pair_sums * still_positive).sum(dim=0)
    else:
        tau = torch.ones(variables, dtype=values.dtype, device=values.device)
    tau = torch.clamp(tau, min=1.0)
    ess = torch.clamp(split_chains * half / tau, max=float(split_chains * half))
    return rhat, ess


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
