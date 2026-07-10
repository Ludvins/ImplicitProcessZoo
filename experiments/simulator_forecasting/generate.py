from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .datasets.damped_oscillator import normalize_time
from .priors import DampedOscillatorPrior


def generate_dataset(
    out: str | Path,
    *,
    n_targets: int = 100,
    n_prior: int = 4096,
    n_test: int = 500,
    t_max: float = 30.0,
    forcing_delta: float = 0.1,
    rho: float = 0.98,
    sigma_u: float = 0.05,
    sigma_y: float = 0.05,
    misspecified: bool = False,
    seed: int = 0,
) -> dict[str, str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_grid = np.linspace(0.0, float(t_max), int(n_test), dtype=np.float64)
    prior = DampedOscillatorPrior(
        t_grid,
        y_mean=0.0,
        y_std=1.0,
        num_samples=int(n_targets),
        reference_bank_size=int(n_prior),
        seed=int(seed) + 1_000_003,
        forcing_delta=float(forcing_delta),
        rho=float(rho),
        sigma_u=float(sigma_u),
        sample_drag=bool(misspecified),
        dtype=torch.float64,
    )
    latents = prior.sample_latents(
        int(n_targets), seed=int(seed) + 1_000_003, sample_drag=bool(misspecified)
    )
    X = torch.as_tensor(
        normalize_time(t_grid, t_max=float(t_max)).reshape(-1, 1), dtype=torch.float64
    )
    with torch.no_grad():
        y = prior.evaluate_raw(X, latents).detach().cpu().numpy()

    target_path = out / "target_paths.npz"
    metadata_path = out / "metadata.json"
    np.savez_compressed(
        target_path,
        t=t_grid,
        y=y,
        latents=latents.detach().cpu().numpy(),
    )
    metadata = {
        "experiment": "simulator_forecasting",
        "prior_type": "randomly_forced_damped_oscillator",
        "t_obs": 8.0,
        "t_max": float(t_max),
        "n_test": int(n_test),
        "n_targets": int(n_targets),
        "n_prior": int(n_prior),
        "forcing_delta": float(forcing_delta),
        "rho": float(rho),
        "sigma_u": float(sigma_u),
        "sigma_y": float(sigma_y),
        "misspecified": bool(misspecified),
        "target_drag_distribution": "Uniform(0.02, 0.08)" if misspecified else "point_mass_0",
        "seed_targets": int(seed) + 1_000_003,
        "latent_schema": list(DampedOscillatorPrior.theta_names)
        + [f"u_{i}" for i in range(prior.forcing_count)],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"target_paths": str(target_path), "metadata": str(metadata_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate damped-oscillator forecasting targets.")
    parser.add_argument("--out", default="data/simprior/simulator_forecasting")
    parser.add_argument("--n-targets", type=int, default=100)
    parser.add_argument("--n-prior", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--t-max", type=float, default=30.0)
    parser.add_argument("--forcing-delta", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.98)
    parser.add_argument("--sigma-u", type=float, default=0.05)
    parser.add_argument("--sigma-y", type=float, default=0.05)
    parser.add_argument("--misspecified", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = parse_args(argv)
    return generate_dataset(
        args.out,
        n_targets=args.n_targets,
        n_prior=args.n_prior,
        n_test=args.n_test,
        t_max=args.t_max,
        forcing_delta=args.forcing_delta,
        rho=args.rho,
        sigma_u=args.sigma_u,
        sigma_y=args.sigma_y,
        misspecified=args.misspecified,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
