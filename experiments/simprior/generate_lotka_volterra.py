from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_THETA_NAMES = ("alpha", "beta", "delta", "gamma", "x0", "y0")


def _sample_theta(rng: np.random.Generator) -> np.ndarray:
    alpha = rng.lognormal(mean=math.log(1.50), sigma=0.15)
    beta = rng.lognormal(mean=math.log(1.00), sigma=0.15)
    delta = rng.lognormal(mean=math.log(0.75), sigma=0.15)
    gamma = rng.lognormal(mean=math.log(1.00), sigma=0.15)
    x0 = rng.uniform(0.8, 1.2)
    y0 = rng.uniform(0.8, 1.2)
    return np.array([alpha, beta, delta, gamma, x0, y0], dtype=np.float64)


def _rhs(_t: float, state: np.ndarray, theta: np.ndarray) -> list[float]:
    alpha, beta, delta, gamma, _, _ = theta
    prey, predator = state
    return [
        alpha * prey - beta * prey * predator,
        delta * prey * predator - gamma * predator,
    ]


def _simulate_one(theta: np.ndarray, t_grid: np.ndarray) -> np.ndarray | None:
    _, _, _, _, x0, y0 = theta
    solution = solve_ivp(
        lambda t, state: _rhs(t, state, theta),
        (float(t_grid[0]), float(t_grid[-1])),
        np.array([x0, y0], dtype=np.float64),
        t_eval=t_grid,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        return None
    y = solution.y.T.astype(np.float64, copy=False)
    if not np.isfinite(y).all():
        return None
    if np.any(y < 0.0):
        return None
    if float(np.max(y)) > 20.0:
        return None
    return y


def generate_bank(
    n_paths: int,
    *,
    t_grid: np.ndarray,
    seed: int,
    max_attempt_multiplier: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    paths: list[np.ndarray] = []
    thetas: list[np.ndarray] = []
    max_attempts = max(int(n_paths) * int(max_attempt_multiplier), int(n_paths) + 100)
    attempts = 0
    while len(paths) < int(n_paths) and attempts < max_attempts:
        attempts += 1
        theta = _sample_theta(rng)
        path = _simulate_one(theta, t_grid)
        if path is None:
            continue
        paths.append(path)
        thetas.append(theta)
    if len(paths) < int(n_paths):
        raise RuntimeError(
            f"Only generated {len(paths)} valid Lotka-Volterra paths after {attempts} attempts."
        )
    return np.stack(paths, axis=0), np.stack(thetas, axis=0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Lotka-Volterra simulator-prior banks.")
    parser.add_argument("--out", default="data/simprior/lotka_volterra")
    parser.add_argument("--n-prior", type=int, default=4096)
    parser.add_argument("--n-targets", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_grid = np.arange(0.0, float(args.t_max) + 0.5 * float(args.dt), float(args.dt), dtype=np.float64)

    prior_y, prior_theta = generate_bank(args.n_prior, t_grid=t_grid, seed=args.seed)
    target_y, target_theta = generate_bank(args.n_targets, t_grid=t_grid, seed=args.seed + 1_000_003)

    prior_path = out / "prior_paths.npz"
    target_path = out / "target_paths.npz"
    metadata_path = out / "metadata.json"
    np.savez_compressed(prior_path, t=t_grid, y=prior_y, theta=prior_theta)
    np.savez_compressed(target_path, t=t_grid, y=target_y, theta=target_theta)
    metadata = {
        "experiment": "lotka_volterra",
        "dt": float(args.dt),
        "t_max": float(args.t_max),
        "n_prior": int(args.n_prior),
        "n_targets": int(args.n_targets),
        "seed_prior": int(args.seed),
        "seed_targets": int(args.seed + 1_000_003),
        "theta_names": list(DEFAULT_THETA_NAMES),
        "reject_if_negative": True,
        "reject_if_max_exceeds": 20.0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "prior_paths": str(prior_path),
        "target_paths": str(target_path),
        "metadata": str(metadata_path),
    }


if __name__ == "__main__":
    main()
