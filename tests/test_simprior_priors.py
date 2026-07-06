import numpy as np
import torch

from experiments.volterra.datasets.lotka_volterra import normalize_time
from experiments.volterra.priors import LotkaVolterraPrior


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
