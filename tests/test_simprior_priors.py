import numpy as np
import torch

from experiments.simprior.datasets.lotka_volterra import normalize_time
from experiments.simprior.priors import LotkaVolterraPrior


def test_lotka_volterra_prior_is_deterministic_for_fixed_ids():
    t = np.linspace(0.0, 30.0, 7)
    y = np.stack(
        [
            np.stack([1.0 + t, 2.0 + t], axis=-1),
            np.stack([3.0 + t, 4.0 + t], axis=-1),
        ],
        axis=0,
    )
    prior = LotkaVolterraPrior(t, y, y_mean=np.zeros((1, 2)), y_std=np.ones((1, 2)))
    X = torch.tensor(normalize_time(t[[1, 3, 5]]), dtype=torch.float64).unsqueeze(-1)
    ids = torch.tensor([1, 0])

    first = prior.evaluate(X, ids)
    second = prior.evaluate(X, ids)

    assert torch.allclose(first, second)


def test_lotka_volterra_prior_interpolation_exact_grid_points():
    t = np.linspace(0.0, 30.0, 11)
    y = np.stack([np.stack([t, t * t], axis=-1)], axis=0)
    prior = LotkaVolterraPrior(t, y, y_mean=np.zeros((1, 2)), y_std=np.ones((1, 2)))
    chosen = np.array([0, 4, 10])
    X = torch.tensor(normalize_time(t[chosen]), dtype=torch.float64).unsqueeze(-1)

    values = prior.evaluate(X, torch.tensor([0]))

    assert torch.allclose(values[0], torch.tensor(y[0, chosen], dtype=torch.float64))
