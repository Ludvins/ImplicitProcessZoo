import numpy as np
import torch

from experiments.simulator_forecasting.datasets import normalize_time
from experiments.simulator_forecasting.priors import DampedOscillatorPrior


def test_damped_oscillator_prior_samples_latents_deterministically():
    t = np.linspace(0.0, 1.0, 6)
    prior = DampedOscillatorPrior(t, seed=123, forcing_delta=0.1)

    first = prior.sample_latents(4, seed=7)
    second = prior.sample_latents(4, seed=7)
    other = prior.sample_latents(4, seed=8)

    assert torch.allclose(first, second)
    assert not torch.allclose(first, other)


def test_damped_oscillator_prior_returns_initial_position_at_zero():
    t = np.linspace(0.0, 1.0, 6)
    prior = DampedOscillatorPrior(t, y_mean=0.0, y_std=1.0, forcing_delta=0.1)
    latent = torch.zeros(1, 8 + prior.forcing_count, dtype=torch.float64)
    latent[0, 0] = 1.0
    latent[0, 1] = 0.1
    latent[0, 2] = 0.5
    latent[0, 3] = 1.0
    latent[0, 5] = 0.37
    latent[0, 6] = -0.2
    X = torch.tensor([[normalize_time(np.array([0.0]), t_max=1.0)[0]]], dtype=torch.float64)

    values = prior.evaluate_raw(X, latent)

    assert torch.allclose(values[0, 0, 0], latent[0, 5])


def test_damped_oscillator_prior_misspecified_drag_sampling():
    t = np.linspace(0.0, 1.0, 6)
    matched = DampedOscillatorPrior(t, sample_drag=False, seed=10)
    misspecified = DampedOscillatorPrior(t, sample_drag=True, seed=10)

    matched_latents = matched.sample_latents(8, seed=3)
    misspecified_latents = misspecified.sample_latents(8, seed=3)

    assert torch.allclose(matched_latents[:, 7], torch.zeros(8, dtype=torch.float64))
    assert torch.all(misspecified_latents[:, 7] >= 0.02)
    assert torch.all(misspecified_latents[:, 7] <= 0.08)


def test_damped_oscillator_prior_output_shapes_and_normalization():
    t = np.linspace(0.0, 2.0, 11)
    prior = DampedOscillatorPrior(t, y_mean=2.0, y_std=4.0, num_samples=3, seed=44)
    X = torch.tensor(normalize_time(t[[0, 3, 10]], t_max=2.0), dtype=torch.float64).unsqueeze(-1)
    latents = prior.sample_latents(3, seed=5)

    raw = prior.evaluate_raw(X, latents)
    normalized = prior.evaluate(X, latents)

    assert raw.shape == (3, 3, 1)
    assert normalized.shape == (3, 3, 1)
    assert torch.allclose(normalized, (raw - 2.0) / 4.0)
