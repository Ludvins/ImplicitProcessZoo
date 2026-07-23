import numpy as np
import torch
from scipy.integrate import solve_ivp

from experiments.volterra.benchmark import LotkaVolterraPrior, normalize_time


def test_lotka_volterra_prior_samples_theta_deterministically():
    t = np.linspace(0.0, 1.0, 5)
    prior = LotkaVolterraPrior(t, y_mean=np.zeros((1, 2)), y_std=np.ones((1, 2)), seed=123)

    first = prior.sample_latents(4, seed=7)
    second = prior.sample_latents(4, seed=7)
    other = prior.sample_latents(4, seed=8)

    assert torch.allclose(first, second)
    assert not torch.allclose(first, other)


def test_lotka_volterra_prior_is_deterministic_for_fixed_theta():
    t = np.linspace(0.0, 1.0, 5)
    prior = LotkaVolterraPrior(t, y_mean=np.zeros((1, 2)), y_std=np.ones((1, 2)))
    X = torch.tensor(normalize_time(t[[1, 3, 4]], t_max=1.0), dtype=torch.float64).unsqueeze(-1)
    theta = torch.tensor(
        [
            [1.0, 0.2, 0.1, 0.8, 1.0, 1.1],
            [1.2, 0.3, 0.2, 0.9, 0.9, 1.2],
        ],
        dtype=torch.float64,
    )

    first = prior.evaluate(X, theta)
    second = prior.evaluate(X, theta)

    assert torch.allclose(first, second)


def test_lotka_volterra_prior_returns_initial_condition_at_zero():
    t = np.linspace(0.0, 1.0, 5)
    prior = LotkaVolterraPrior(t, y_mean=np.zeros((1, 2)), y_std=np.ones((1, 2)))
    X = torch.tensor([[normalize_time(np.array([0.0]), t_max=1.0)[0]]], dtype=torch.float64)
    theta = torch.tensor([[1.5, 1.0, 0.75, 1.0, 0.85, 1.15]], dtype=torch.float64)

    values = prior.evaluate_raw(X, theta)

    assert torch.allclose(values[0, 0], theta[0, 4:6])


def test_lotka_volterra_prior_rejects_invalid_trajectories():
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, -0.1], [3.0, 4.0]],
            [[1.0, float("inf")], [3.0, 4.0]],
            [[1.0, 2.0], [20.1, 4.0]],
        ],
        dtype=torch.float64,
    )

    valid = LotkaVolterraPrior._valid_trajectories(values)

    assert torch.equal(valid, torch.tensor([True, False, False, False]))


def test_lotka_volterra_prior_banks_are_nested():
    t = np.linspace(0.0, 4.0, 81)
    prior = LotkaVolterraPrior(t, seed=123)

    small = prior.sample_latents(20, seed=91)
    large = prior.sample_latents(64, seed=91)

    assert torch.equal(small, large[:20])


def test_lotka_volterra_rk4_matches_high_accuracy_reference():
    t = np.linspace(0.0, 5.0, 101)
    theta = np.array([1.5, 1.0, 0.75, 1.0, 0.9, 1.1], dtype=np.float64)
    prior = LotkaVolterraPrior(t)
    X = torch.tensor(normalize_time(t, t_max=5.0), dtype=torch.float64).unsqueeze(-1)
    actual = prior.evaluate_raw(X, torch.tensor(theta).unsqueeze(0))[0].numpy()

    def rhs(_time, state):
        prey, predator = state
        alpha, beta, delta, gamma = theta[:4]
        return [
            alpha * prey - beta * prey * predator,
            delta * prey * predator - gamma * predator,
        ]

    expected = solve_ivp(
        rhs,
        (0.0, 5.0),
        theta[4:],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    ).y.T

    assert np.allclose(actual, expected, rtol=1e-4, atol=1e-6)
