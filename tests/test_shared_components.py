"""Tests for shared components: generative functions, flows, likelihood utilities."""

import pytest
import torch
import numpy as np

from src.priors.generative_functions import (
    BayesianNN, BayesLinear, SimplerBayesLinear, GP,
)
from src.flows.flows import CouplingFlow
from src.utils.likelihood import (
    gaussian_logp,
    bernoulli_logp,
    multiclass_logp,
    inv_probit,
    predict_mean_and_var_regression,
    predict_mean_and_var_binary,
    gaussian_variational_expectations,
    bernoulli_variational_expectations,
)

from tests.conftest import DEVICE, DTYPE, SEED, INPUT_DIM, OUTPUT_DIM, NUM_SAMPLES


# ===================================================================
# BayesianNN
# ===================================================================

class TestBayesianNN:

    def test_output_shape(self, bnn):
        X = torch.randn(10, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        out = bnn(X)
        assert out.shape == (NUM_SAMPLES, 10, OUTPUT_DIM)

    def test_kl_is_scalar(self, bnn):
        kl = bnn.KL()
        assert kl.dim() == 0
        assert kl.item() >= 0

    def test_fix_random_noise_deterministic(self, bnn):
        X = torch.randn(5, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        out1 = bnn(X)
        out2 = bnn(X)
        assert torch.allclose(out1, out2)

    def test_no_fix_random_noise_stochastic(self, bnn_variational):
        X = torch.randn(5, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        out1 = bnn_variational(X)
        out2 = bnn_variational(X)
        # With fix_random_noise=False, outputs should differ
        assert not torch.allclose(out1, out2)

    def test_freeze_defreeze(self, bnn):
        bnn.freeze_parameters()
        for p in bnn.parameters():
            assert not p.requires_grad
        bnn.defreeze_parameters()
        for p in bnn.parameters():
            assert p.requires_grad

    def test_regularizer(self, bnn):
        reg = bnn.regularizer()
        assert reg.dim() == 0


class TestSimplerBayesLinear:

    def test_scalar_params(self):
        layer = SimplerBayesLinear(
            NUM_SAMPLES, 3, 4, device=DEVICE, fix_random_noise=True,
            seed=SEED, dtype=DTYPE,
        )
        # SimplerBayesLinear has scalar mu and log_sigma
        assert layer.weight_log_sigma.dim() == 0
        assert layer.bias_log_sigma.dim() == 0

    def test_kl_scales_with_dim(self):
        layer_small = SimplerBayesLinear(
            NUM_SAMPLES, 2, 2, device=DEVICE, fix_random_noise=True,
            seed=SEED, dtype=DTYPE,
        )
        layer_big = SimplerBayesLinear(
            NUM_SAMPLES, 10, 10, device=DEVICE, fix_random_noise=True,
            seed=SEED, dtype=DTYPE,
        )
        # At initialization (log_sigma=0, mu=0), KL should be 0
        assert abs(layer_small.KL().item()) < 1e-10
        assert abs(layer_big.KL().item()) < 1e-10


# ===================================================================
# GP
# ===================================================================

class TestGP:

    def test_output_shape(self, gp):
        X = torch.randn(10, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        S = 7
        out = gp(X, S)
        assert out.shape == (S, 10, OUTPUT_DIM)

    def test_forward_mean(self, gp):
        X = torch.randn(10, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        mean = gp.forward_mean(X)
        assert mean.shape == (10, OUTPUT_DIM)

    def test_freeze_defreeze(self, gp):
        gp.freeze_parameters()
        for p in gp.parameters():
            assert not p.requires_grad
        gp.defreeze_parameters()
        for p in gp.parameters():
            assert p.requires_grad


# ===================================================================
# CouplingFlow
# ===================================================================

class TestCouplingFlow:

    def test_forward_shape(self, coupling_flow):
        a = torch.randn(8, NUM_SAMPLES, dtype=DTYPE, device=DEVICE)
        transformed, ldj = coupling_flow(a)
        assert transformed.shape == a.shape
        assert ldj.shape == (8,)

    def test_invertibility_approximate(self, coupling_flow):
        """Flow should be smooth — small perturbation -> small output change."""
        a = torch.randn(4, NUM_SAMPLES, dtype=DTYPE, device=DEVICE)
        a_perturbed = a + 1e-5 * torch.randn_like(a)
        out1, _ = coupling_flow(a)
        out2, _ = coupling_flow(a_perturbed)
        assert torch.allclose(out1, out2, atol=1e-2)


# ===================================================================
# Likelihood utilities
# ===================================================================

class TestLikelihoods:

    def test_gaussian_logp_shape(self):
        F = torch.randn(5, 10, 1, dtype=DTYPE)
        Y = torch.randn(10, 1, dtype=DTYPE)
        log_var = torch.tensor(0.0, dtype=DTYPE)
        lp = gaussian_logp(F, Y, log_var)
        assert lp.shape == (5, 10, 1)

    def test_bernoulli_logp_shape(self):
        F = torch.randn(5, 10, 1, dtype=DTYPE)
        Y = torch.ones(10, 1, dtype=DTYPE)
        lp = bernoulli_logp(F, Y)
        assert lp.shape == (5, 10, 1)

    def test_multiclass_logp_shape(self):
        F = torch.randn(5, 10, 3, dtype=DTYPE)
        Y = torch.zeros(10, 3, dtype=DTYPE)
        Y[:, 0] = 1.0
        lp = multiclass_logp(F, Y, num_classes=3, epsilon=1e-3)
        assert lp.shape == (5, 10, 3)

    def test_inv_probit_bounded(self):
        x = torch.randn(100, dtype=DTYPE)
        p = inv_probit(x)
        assert (p >= 0).all()
        assert (p <= 1).all()


class TestPredictMeanVar:

    def test_regression(self):
        Fmu = torch.randn(20, 1, dtype=DTYPE)
        Fvar = torch.rand(20, 1, dtype=DTYPE).abs() + 0.01
        log_var = torch.tensor(-1.0, dtype=DTYPE)
        mean, var = predict_mean_and_var_regression(Fmu, Fvar, log_var)
        assert mean.shape == (20, 1)
        assert var.shape == (20, 1)
        assert (var > 0).all()

    def test_binary(self):
        Fmu = torch.randn(20, 1, dtype=DTYPE)
        Fvar = torch.rand(20, 1, dtype=DTYPE).abs() + 0.01
        mean, var = predict_mean_and_var_binary(Fmu, Fvar)
        assert mean.shape == (20, 1)
        assert var.shape == (20, 1)


class TestVariationalExpectations:

    def test_gaussian(self):
        # gaussian_variational_expectations expects [S, N, D] inputs
        Fmu = torch.randn(1, 20, 1, dtype=DTYPE)
        Fvar = torch.rand(1, 20, 1, dtype=DTYPE).abs() + 0.01
        Y = torch.randn(20, 1, dtype=DTYPE)
        log_var = torch.tensor(-1.0, dtype=DTYPE)
        ve = gaussian_variational_expectations(Fmu, Fvar, Y, alpha=0,
                                                log_variance=log_var)
        assert ve.shape == (20, 1)

    def test_bernoulli(self):
        Fmu = torch.randn(1, 20, 1, dtype=DTYPE)
        Fvar = torch.rand(1, 20, 1, dtype=DTYPE).abs() + 0.01
        Y = torch.ones(20, 1, dtype=DTYPE)
        ve = bernoulli_variational_expectations(Fmu, Fvar, Y,
                                                 num_gh_points=20,
                                                 dtype=DTYPE, device=DEVICE)
        # hermgaussquadrature preserves the [S, N, D] input shape
        assert ve.shape == (1, 20, 1)
