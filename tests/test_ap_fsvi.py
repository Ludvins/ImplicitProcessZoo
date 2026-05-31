"""Tests for AP-FSVI (Adaptive Projective Function-Space VI)."""

from argparse import Namespace

import pytest
import torch

from src.ap_fsvi import APFSVI, FunctionDiscrepancy, MMDivergence
from src.priors.generative_functions import BayesianNN, BayesLinear, ExactGP
from tests.conftest import (
    BATCH_SIZE,
    DEVICE,
    DTYPE,
    INPUT_DIM,
    NUM_DATA,
    NUM_SAMPLES,
    OUTPUT_DIM,
    SEED,
)


def _make_generator(output_dim=OUTPUT_DIM, num_samples=NUM_SAMPLES,
                    fix_random_noise=False, seed=SEED):
    return BayesianNN(
        structure=[8],
        activation=torch.tanh,
        num_samples=num_samples,
        input_dim=INPUT_DIM,
        output_dim=output_dim,
        layer_model=BayesLinear,
        seed=seed,
        fix_random_noise=fix_random_noise,
        device=DEVICE,
        dtype=DTYPE,
    )


class TestAPFSVIConstruction:

    def test_regression(self):
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_measurement=8,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        assert model.likelihood_type == "regression"
        assert hasattr(model, "log_variance")
        assert torch.allclose(
            model.log_variance,
            torch.full_like(model.log_variance, -2.0),
        )

    def test_log_variance_init_is_configurable(self):
        model = APFSVI(
            generative_function=_make_generator(output_dim=2),
            input_dim=INPUT_DIM,
            output_dim=2,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_measurement=8,
            log_variance_init=[-1.5, -0.5],
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        expected = torch.tensor([-1.5, -0.5], dtype=DTYPE, device=DEVICE)
        assert torch.allclose(model.log_variance, expected)

    def test_direct_posterior_has_no_coefficient_parameters(self):
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        parameter_names = {name for name, _ in model.named_parameters()}
        assert "q_mu" not in parameter_names
        assert "q_sqrt_tri" not in parameter_names

    def test_direct_posterior_uses_full_bayeslinear_parameters(self):
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        assert all(
            type(layer) is BayesLinear
            for layer in model.generative_function.layers
        )
        assert model.generative_function.layers[0].weight_mu.ndim == 2
        assert model.generative_function.layers[0].weight_log_sigma.ndim == 2

    def test_reuse_adaptive_measurement_points_prefers_cached_set(self):
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_measurement=4,
            adaptive_measure_points=True,
            reuse_adaptive_measure_points=True,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        fresh = torch.randn(4, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        cached = torch.randn_like(fresh)
        model._last_adaptive_measure = cached

        reused = model._reuse_or_fresh_measurement_set(fresh)

        assert torch.allclose(reused, cached)
        model._last_adaptive_measure = cached[:2]
        assert torch.allclose(model._reuse_or_fresh_measurement_set(fresh), fresh)

    def test_partial_reuse_adaptive_measurement_points_mixes_cached_and_fresh(self):
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_measurement=4,
            adaptive_measure_points=True,
            reuse_adaptive_measure_points=True,
            adaptive_measure_reuse_fraction=0.5,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        fresh = torch.zeros(4, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        cached = torch.ones_like(fresh)
        model._last_adaptive_measure = cached

        mixed = model._reuse_or_fresh_measurement_set(fresh)

        assert torch.count_nonzero(mixed == 1).item() == 2 * INPUT_DIM
        assert torch.count_nonzero(mixed == 0).item() == 2 * INPUT_DIM

    def test_adaptive_measurement_reuse_fraction_validates_bounds(self):
        with pytest.raises(ValueError, match="adaptive_measure_reuse_fraction"):
            APFSVI(
                generative_function=_make_generator(),
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                likelihood="regression",
                num_data=NUM_DATA,
                num_samples=NUM_SAMPLES,
                adaptive_measure_reuse_fraction=1.5,
                device=DEVICE,
                dtype=DTYPE,
                seed=SEED,
            )

    def test_uci_builder_forces_full_bayeslinear_for_ap_fsvi(self):
        from scripts.uci_benchmark import build_model

        class _Dataset:
            input_dim = 13
            output_dim = 1
            targets_mean = 0.0
            targets_std = 1.0

            def __len__(self):
                return 100

        args = Namespace(
            model="ap_fsvi",
            device=str(DEVICE),
            dtype="float64" if DTYPE == torch.float64 else "float32",
            hidden_dims=[10, 10],
            activation="tanh",
            layer_model="SimplerBayesLinear",
            dropout=0.0,
            seed=SEED,
            ap_fsvi_prior="gp",
            ap_fsvi_weight_log_sigma_init=0.0,
            ap_fsvi_num_samples=NUM_SAMPLES,
            ap_fsvi_num_prior_samples=None,
            ap_fsvi_num_measurement=8,
            ap_fsvi_adaptive_measure_points=False,
            ap_fsvi_adaptive_measure_steps=3,
            ap_fsvi_adaptive_measure_lr=0.05,
            ap_fsvi_adaptive_measure_domain_limit=None,
            ap_fsvi_beta=0.05,
            ap_fsvi_beta_start=0.0,
            ap_fsvi_beta_warmup_steps=0,
            ap_fsvi_data_pretrain_steps=0,
            ap_fsvi_data_loss="expected_nll",
            ap_fsvi_measurement_weights=[0.2, 0.2, 0.6],
            ap_fsvi_near_data_noise=0.1,
            ap_fsvi_domain_std=2.5,
            ap_fsvi_discrepancy="mmd",
            ap_fsvi_discrepancy_projections=64,
            ap_fsvi_sinkhorn_epsilon=1.0,
            ap_fsvi_sinkhorn_iterations=50,
            ap_fsvi_log_variance_init=-5.0,
            ap_fsvi_max_grad_norm=None,
        )

        model = build_model(args, _Dataset())
        assert all(
            type(layer) is BayesLinear
            for layer in model.generative_function.layers
        )

    def test_uci_builder_can_use_matching_standard_bnn_prior(self):
        from scripts.uci_benchmark import build_model

        class _Dataset:
            input_dim = 13
            output_dim = 1
            targets_mean = 0.0
            targets_std = 1.0

            def __len__(self):
                return 100

        args = Namespace(
            model="ap_fsvi",
            device=str(DEVICE),
            dtype="float64" if DTYPE == torch.float64 else "float32",
            hidden_dims=[10, 10],
            activation="tanh",
            layer_model="SimplerBayesLinear",
            dropout=0.0,
            seed=SEED,
            ap_fsvi_prior="bnn",
            ap_fsvi_weight_log_sigma_init=0.0,
            ap_fsvi_num_samples=NUM_SAMPLES,
            ap_fsvi_num_prior_samples=None,
            ap_fsvi_num_measurement=8,
            ap_fsvi_adaptive_measure_points=False,
            ap_fsvi_adaptive_measure_steps=3,
            ap_fsvi_adaptive_measure_lr=0.05,
            ap_fsvi_adaptive_measure_domain_limit=None,
            ap_fsvi_beta=0.05,
            ap_fsvi_beta_start=0.0,
            ap_fsvi_beta_warmup_steps=0,
            ap_fsvi_data_pretrain_steps=0,
            ap_fsvi_data_loss="expected_nll",
            ap_fsvi_measurement_weights=[0.2, 0.2, 0.6],
            ap_fsvi_near_data_noise=0.1,
            ap_fsvi_domain_std=2.5,
            ap_fsvi_discrepancy="mmd",
            ap_fsvi_discrepancy_projections=64,
            ap_fsvi_sinkhorn_epsilon=1.0,
            ap_fsvi_sinkhorn_iterations=50,
            ap_fsvi_log_variance_init=-5.0,
            ap_fsvi_max_grad_norm=None,
        )

        model = build_model(args, _Dataset())
        assert isinstance(model.prior_function, BayesianNN)
        assert [layer.input_dim for layer in model.prior_function.layers] == [13, 10, 10]
        assert [layer.output_dim for layer in model.prior_function.layers] == [10, 10, 1]
        assert all(type(layer) is BayesLinear for layer in model.prior_function.layers)
        assert all(not param.requires_grad for param in model.prior_function.parameters())
        assert all(layer.zero_mean_prior for layer in model.prior_function.layers)
        assert all(
            torch.allclose(layer.weight_log_sigma, torch.zeros_like(layer.weight_log_sigma))
            for layer in model.prior_function.layers
        )
        X = torch.randn(5, _Dataset.input_dim, dtype=DTYPE, device=DEVICE)
        prior_values = model.forward_prior(X, 7)
        assert prior_values.shape == (7, X.shape[0], _Dataset.output_dim)

    def test_stein_warns_when_prior_has_no_score(self):
        prior = _make_generator(fix_random_noise=False, seed=SEED + 1)
        with pytest.warns(UserWarning, match="Stein discrepancy needs a prior score"):
            APFSVI(
                generative_function=_make_generator(),
                prior_function=prior,
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                likelihood="regression",
                num_data=NUM_DATA,
                num_samples=NUM_SAMPLES,
                num_measurement=8,
                function_discrepancy="stein",
                device=DEVICE,
                dtype=DTYPE,
                seed=SEED,
            )

    def test_binary_classification_outputs_probabilities_and_trains(self):
        model = APFSVI(
            generative_function=_make_generator(output_dim=1),
            input_dim=INPUT_DIM,
            output_dim=1,
            likelihood="binary",
            num_classes=2,
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_measurement=8,
            beta=0.1,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        X = torch.randn(BATCH_SIZE, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        y = torch.randint(0, 2, (BATCH_SIZE, 1), device=DEVICE).to(DTYPE)
        probs = model.predict_y_samples(X, 4)
        assert probs.shape == (4, BATCH_SIZE, 1)
        assert torch.all((probs >= 0.0) & (probs <= 1.0))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = model._train_step(optimizer, X, y)
        assert torch.isfinite(loss)

    def test_multiclass_classification_outputs_logits_and_trains(self):
        num_classes = 3
        model = APFSVI(
            generative_function=_make_generator(output_dim=num_classes),
            input_dim=INPUT_DIM,
            output_dim=num_classes,
            likelihood="multiclass",
            num_classes=num_classes,
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_measurement=8,
            beta=0.1,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        X = torch.randn(BATCH_SIZE, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        y = torch.randint(0, num_classes, (BATCH_SIZE,), device=DEVICE)
        logits = model.predict_y_samples(X, 4)
        assert logits.shape == (4, BATCH_SIZE, num_classes)

        centered = model._regularizer_values(logits)
        assert torch.allclose(
            centered.mean(dim=-1),
            torch.zeros(4, BATCH_SIZE, dtype=DTYPE, device=DEVICE),
            atol=1e-10,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = model._train_step(optimizer, X, y)
        assert torch.isfinite(loss)


class TestAPFSVIShapes:

    @pytest.fixture
    def model(self):
        return APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_measurement=8,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

    def test_predict_f_samples(self, model, regression_data):
        X, _ = regression_data
        F = model.predict_f_samples(X, 5)
        assert F.shape == (5, X.shape[0], OUTPUT_DIM)

    def test_forward(self, model, regression_data):
        X, _ = regression_data
        samples, std = model(X)
        assert samples.shape == (NUM_SAMPLES, X.shape[0], OUTPUT_DIM)
        assert std.shape == samples.shape

    def test_forward_prior(self, model, regression_data):
        X, _ = regression_data
        prior = model.forward_prior(X, 7)
        assert prior.shape == (7, X.shape[0], OUTPUT_DIM)

    def test_supplied_generator_produces_coherent_functions(self, regression_data):
        X, _ = regression_data
        generator = _make_generator(num_samples=4, fix_random_noise=True)
        assert torch.allclose(
            generator(X[:BATCH_SIZE]),
            generator(X[:BATCH_SIZE]),
        )


class TestAPFSVILoss:

    @pytest.fixture
    def model(self):
        return APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_prior_samples=NUM_SAMPLES,
            num_measurement=8,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

    def test_nelbo_is_scalar(self, model, regression_data):
        X, y = regression_data
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_predictive_nll_data_loss_mode(self, regression_data):
        X, y = regression_data
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_measurement=8,
            data_loss="predictive_nll",
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0

    def test_mmd_detects_shift(self):
        torch.manual_seed(SEED)
        z = torch.randn(12, 5, 1, dtype=DTYPE)
        shifted = z + 1.0
        mmd = MMDivergence()
        assert mmd(z, shifted) > mmd(z, z)

    @pytest.mark.parametrize(
        "kind",
        [
            "mmd",
            "energy",
            "sliced_wasserstein",
            "sinkhorn",
            "sample_sliced_kl",
            "sample_sliced_knn_kl",
            "sample_sliced_gaussian_kl",
            "sample_sliced_quantile_transport_kl",
        ],
    )
    def test_sample_discrepancies_detect_shift(self, kind):
        torch.manual_seed(SEED)
        z = torch.randn(12, 4, 1, dtype=DTYPE)
        shifted = z + 1.0
        discrepancy = FunctionDiscrepancy(
            kind=kind,
            num_projections=16,
            sinkhorn_iterations=10,
        )
        assert discrepancy(z, shifted) > discrepancy(z, z)

    def test_prior_whitened_sliced_kl_detects_gp_shift(self):
        torch.manual_seed(SEED)
        gp = ExactGP(
            num_samples=16,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            fix_random_noise=False,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        gp.freeze_parameters()
        X = torch.linspace(-1.0, 1.0, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
        X = X.repeat(1, INPUT_DIM)
        prior = gp(X, num_samples=16)
        shifted = prior + 1.0
        discrepancy = FunctionDiscrepancy(
            kind="prior_whitened_sliced_kl",
            num_projections=32,
        )

        assert discrepancy(
            shifted, measurement_inputs=X, prior_function=gp
        ) > discrepancy(prior, measurement_inputs=X, prior_function=gp)

    def test_prior_whitened_gaussian_kl_is_finite(self):
        torch.manual_seed(SEED)
        gp = ExactGP(
            num_samples=16,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            fix_random_noise=False,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        gp.freeze_parameters()
        X = torch.randn(5, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        values = gp(X, num_samples=16)
        discrepancy = FunctionDiscrepancy(kind="prior_whitened_gaussian_kl")
        value = discrepancy(values, measurement_inputs=X, prior_function=gp)
        assert torch.isfinite(value)
        assert value >= 0

    def test_spectral_sliced_kl_small_for_matching_gp_and_larger_for_shift(self):
        torch.manual_seed(SEED)
        gp = ExactGP(
            num_samples=128,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            fix_random_noise=False,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        gp.freeze_parameters()
        X = torch.linspace(-1.0, 1.0, 16, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
        X = X.repeat(1, INPUT_DIM)
        prior = gp(X, num_samples=128)
        shifted = prior + 1.0
        discrepancy = FunctionDiscrepancy(
            kind="spectral_sliced_kl",
            spectral_num_modes=8,
            spectral_estimator="gaussian",
        )

        matching_value = discrepancy(
            prior, measurement_inputs=X, prior_function=gp
        )
        shifted_value = discrepancy(
            shifted, measurement_inputs=X, prior_function=gp
        )
        assert torch.isfinite(matching_value)
        assert torch.isfinite(shifted_value)
        assert matching_value < 0.25
        assert shifted_value > matching_value

    def test_spectral_sliced_cumulant_estimator_is_finite(self):
        torch.manual_seed(SEED)
        gp = ExactGP(
            num_samples=128,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            fix_random_noise=False,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        gp.freeze_parameters()
        X = torch.linspace(-1.0, 1.0, 16, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
        X = X.repeat(1, INPUT_DIM)
        prior = gp(X, num_samples=128)
        non_gaussian = prior + 0.3 * prior.pow(3)
        discrepancy = FunctionDiscrepancy(
            kind="spectral_sliced_kl",
            spectral_num_modes=8,
            spectral_estimator="cumulant",
        )
        value = discrepancy(non_gaussian, measurement_inputs=X, prior_function=gp)
        assert torch.isfinite(value)
        assert value >= 0

    def test_spectral_sliced_kl_empirical_prior_fallback(self):
        torch.manual_seed(SEED)
        prior = torch.randn(128, 12, 1, dtype=DTYPE, device=DEVICE)
        shifted = prior + 0.5
        discrepancy = FunctionDiscrepancy(
            kind="spectral_sliced_kl",
            spectral_num_modes=6,
            spectral_estimator="gaussian",
        )
        matching_value = discrepancy(prior, prior_values=prior)
        shifted_value = discrepancy(shifted, prior_values=prior)
        assert torch.isfinite(matching_value)
        assert torch.isfinite(shifted_value)
        assert shifted_value > matching_value

    def test_spectral_projected_kl_small_for_matching_gp_and_larger_for_shift(self):
        torch.manual_seed(SEED)
        gp = ExactGP(
            num_samples=128,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            fix_random_noise=False,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        gp.freeze_parameters()
        X = torch.linspace(-1.0, 1.0, 16, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
        X = X.repeat(1, INPUT_DIM)
        prior = gp(X, num_samples=128)
        shifted = prior + 1.0
        discrepancy = FunctionDiscrepancy(
            kind="spectral_projected_kl",
            spectral_num_modes=8,
            spectral_estimator="full_gaussian",
            spectral_cov_shrinkage=0.05,
        )

        matching_value = discrepancy(
            prior, measurement_inputs=X, prior_function=gp
        )
        shifted_value = discrepancy(
            shifted, measurement_inputs=X, prior_function=gp
        )
        assert torch.isfinite(matching_value)
        assert torch.isfinite(shifted_value)
        assert matching_value < 0.75
        assert shifted_value > matching_value

    def test_spectral_projected_kl_detects_cross_mode_correlation(self):
        class IdentityPrior:
            jitter = 1e-6

            def marginal(self, X):
                m = X.shape[0]
                mean = torch.zeros(m, dtype=X.dtype, device=X.device)
                covariance = torch.eye(m, dtype=X.dtype, device=X.device)
                return mean, covariance

        torch.manual_seed(SEED)
        X = torch.linspace(-1.0, 1.0, 4, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
        iid = torch.randn(512, 4, 1, dtype=DTYPE, device=DEVICE)
        correlated = torch.randn(512, 4, 1, dtype=DTYPE, device=DEVICE)
        shared = torch.randn(512, dtype=DTYPE, device=DEVICE)
        residual = torch.randn(512, dtype=DTYPE, device=DEVICE)
        rho = 0.85
        correlated[:, 0, 0] = shared
        correlated[:, 1, 0] = rho * shared + (1.0 - rho**2) ** 0.5 * residual
        discrepancy = FunctionDiscrepancy(
            kind="spectral_projected_kl",
            spectral_num_modes=4,
            spectral_estimator="full_gaussian",
            spectral_cov_shrinkage=0.0,
        )

        iid_value = discrepancy(iid, measurement_inputs=X, prior_function=IdentityPrior())
        correlated_value = discrepancy(
            correlated, measurement_inputs=X, prior_function=IdentityPrior()
        )

        assert torch.isfinite(iid_value)
        assert torch.isfinite(correlated_value)
        assert correlated_value > iid_value + 0.05

    def test_spectral_projected_knn_estimator_is_finite(self):
        class IdentityPrior:
            jitter = 1e-6

            def marginal(self, X):
                m = X.shape[0]
                mean = torch.zeros(m, dtype=X.dtype, device=X.device)
                covariance = torch.eye(m, dtype=X.dtype, device=X.device)
                return mean, covariance

        torch.manual_seed(SEED)
        X = torch.linspace(-1.0, 1.0, 4, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
        values = torch.randn(64, 4, 1, dtype=DTYPE, device=DEVICE)
        discrepancy = FunctionDiscrepancy(
            kind="spectral_projected_kl",
            spectral_num_modes=4,
            spectral_estimator="knn_entropy",
            spectral_knn_k=3,
        )
        value = discrepancy(values, measurement_inputs=X, prior_function=IdentityPrior())
        assert torch.isfinite(value)
        assert value >= 0

    @pytest.mark.parametrize("mode", ["prior_pca", "discrepancy_pca"])
    def test_sample_sliced_kl_deterministic_projection_modes_are_sensitive(self, mode):
        torch.manual_seed(SEED)
        prior = torch.randn(128, 8, 1, dtype=DTYPE, device=DEVICE)
        shifted = prior + 0.5
        discrepancy = FunctionDiscrepancy(
            kind="sample_sliced_kl",
            num_projections=8,
            sample_projection_mode=mode,
        )
        matching_value = discrepancy(prior, prior_values=prior)
        shifted_value = discrepancy(shifted, prior_values=prior)
        assert torch.isfinite(matching_value)
        assert torch.isfinite(shifted_value)
        assert shifted_value > matching_value

    def test_sample_sliced_knn_kl_detects_shift_and_has_gradient(self):
        torch.manual_seed(SEED)
        prior = torch.randn(128, 8, 1, dtype=DTYPE, device=DEVICE)
        posterior = (prior + 0.75).detach().clone().requires_grad_(True)
        discrepancy = FunctionDiscrepancy(
            kind="sample_sliced_knn_kl",
            num_projections=16,
            sample_knn_k=3,
        )
        matching_value = discrepancy(prior, prior_values=prior)
        shifted_value = discrepancy(posterior, prior_values=prior)

        assert torch.isfinite(matching_value)
        assert torch.isfinite(shifted_value)
        assert matching_value < 1e-6
        assert shifted_value > matching_value

        shifted_value.backward()
        assert posterior.grad is not None
        assert torch.isfinite(posterior.grad).all()
        assert posterior.grad.abs().sum() > 0

    def test_sample_sliced_quantile_transport_kl_detects_shift(self):
        torch.manual_seed(SEED)
        prior = torch.randn(128, 8, 1, dtype=DTYPE, device=DEVICE)
        shifted = prior + 0.75
        discrepancy = FunctionDiscrepancy(
            kind="sample_sliced_quantile_transport_kl",
            num_projections=16,
            quantile_transport_k=3,
        )
        matching_value = discrepancy(prior, prior_values=prior)
        shifted_value = discrepancy(shifted, prior_values=prior)
        assert torch.isfinite(matching_value)
        assert torch.isfinite(shifted_value)
        assert matching_value < 1e-6
        assert shifted_value > matching_value

    def test_sinkhorn_defaults_to_non_debiased(self):
        discrepancy = FunctionDiscrepancy(kind="sinkhorn")
        assert discrepancy.sinkhorn_debiased is False

    def test_exact_gp_cholesky_cache_skips_adaptive_points(self):
        gp = ExactGP(
            num_samples=2,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            fix_random_noise=False,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        gp.freeze_parameters()
        X = torch.randn(5, INPUT_DIM, dtype=DTYPE, device=DEVICE)
        gp(X)
        cached = gp._cached_cholesky
        assert cached is not None
        gp(X)
        assert gp._cached_cholesky is cached

        X_adaptive = X.detach().clone().requires_grad_(True)
        gp(X_adaptive)
        assert gp._cached_cholesky is None

    @pytest.mark.parametrize(
        "kind",
        [
            "mmd",
            "energy",
            "sliced_wasserstein",
            "sinkhorn",
            "stein",
            "prior_whitened_gaussian_kl",
            "prior_whitened_sliced_kl",
            "spectral_sliced_kl",
            "spectral_projected_kl",
            "sample_sliced_kl",
            "sample_sliced_knn_kl",
            "sample_sliced_gaussian_kl",
            "sample_sliced_quantile_transport_kl",
        ],
    )
    def test_function_discrepancy_options_train(self, regression_data, kind):
        X, y = regression_data
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_prior_samples=NUM_SAMPLES,
            num_measurement=8,
            function_discrepancy=kind,
            discrepancy_num_projections=16,
            sinkhorn_iterations=10,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    @pytest.mark.parametrize(
        "mode",
        ["fixed_random", "prior_pca", "discrepancy_pca", "fixed_orthogonal"],
    )
    def test_sample_sliced_projection_modes_train(self, regression_data, mode):
        X, y = regression_data
        model = APFSVI(
            generative_function=_make_generator(),
            prior_function=_make_generator(num_samples=NUM_SAMPLES, seed=SEED + 1),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_prior_samples=NUM_SAMPLES,
            num_measurement=8,
            function_discrepancy="sample_sliced_kl",
            discrepancy_num_projections=8,
            sample_projection_mode=mode,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model.prior_function.freeze_parameters()
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_fixed_measurement_points_reuse_context(self, regression_data):
        X, y = regression_data
        model = APFSVI(
            generative_function=_make_generator(),
            prior_function=_make_generator(num_samples=NUM_SAMPLES, seed=SEED + 1),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_prior_samples=NUM_SAMPLES,
            num_measurement=8,
            function_discrepancy="sample_sliced_gaussian_kl",
            sample_projection_mode="fixed_orthogonal",
            fixed_measure_points=True,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model.prior_function.freeze_parameters()
        model._reservoir = X.detach()
        model._initialize_fixed_measurement_set([(X, y)])
        first = model._sample_measurement_set(X[:BATCH_SIZE])
        second = model._sample_measurement_set(X[BATCH_SIZE:])
        assert torch.allclose(first, second)

    def test_beta_schedule_honors_pretrain_and_warmup(self):
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            beta=0.2,
            beta_start=0.05,
            beta_warmup_steps=10,
            data_pretrain_steps=2,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model._step = 2
        assert model._scheduled_beta() == 0.0
        model._step = 7
        assert model._scheduled_beta() == pytest.approx(0.125)
        model._step = 20
        assert model._scheduled_beta() == pytest.approx(0.2)

    def test_measurement_sampler_respects_domain_bounds(self, regression_data):
        X, _ = regression_data
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_measurement=30,
            measurement_weights=(0.2, 0.4, 0.4),
            near_data_noise=5.0,
            domain_bounds=(-1.0, 1.0),
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model._reservoir = torch.linspace(
            -0.5, 0.5, 10, dtype=DTYPE
        ).unsqueeze(-1).repeat(1, INPUT_DIM)
        X_measure = model._sample_measurement_set(X[:BATCH_SIZE])
        assert X_measure.shape == (30, INPUT_DIM)
        assert torch.all(X_measure >= -1.0)
        assert torch.all(X_measure <= 1.0)

    def test_adaptive_measurement_points_move_up_discrepancy(self, regression_data):
        X, _ = regression_data
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_measurement=4,
            adaptive_measure_points=True,
            adaptive_measure_steps=2,
            adaptive_measure_lr=0.2,
            adaptive_measure_normalize_grad=False,
            adaptive_measure_domain_limit=10.0,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )

        class QuadraticDiscrepancy:
            requires_prior_samples = False

            def __call__(
                self,
                posterior_values,
                prior_values=None,
                measurement_inputs=None,
                prior_function=None,
            ):
                return measurement_inputs.square().sum()

        model.divergence = QuadraticDiscrepancy()
        X_measure = torch.full((4, INPUT_DIM), 0.1, dtype=DTYPE, device=DEVICE)
        X_adapted = model._adapt_measurement_points(X_measure)
        assert not X_adapted.requires_grad
        assert X_adapted.square().sum() > X_measure.square().sum()
        assert all(param.grad is None for param in model.parameters())

    def test_adaptive_measurement_points_train_with_real_discrepancy(
        self, regression_data
    ):
        X, y = regression_data
        model = APFSVI(
            generative_function=_make_generator(num_samples=4),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=4,
            num_prior_samples=4,
            num_measurement=4,
            adaptive_measure_points=True,
            adaptive_measure_steps=1,
            adaptive_measure_lr=0.01,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model._step = 1
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    @pytest.mark.parametrize(
        "kind,prior",
        [
            ("prior_whitened_sliced_kl", "gp"),
            ("spectral_sliced_kl", "gp"),
            ("spectral_projected_kl", "gp"),
            ("sample_sliced_kl", "bnn"),
        ],
    )
    def test_candidate_then_one_step_adaptive_measurement_train(
        self, regression_data, kind, prior
    ):
        X, y = regression_data
        prior_function = None
        if prior == "bnn":
            prior_function = _make_generator(num_samples=4, seed=SEED + 1)
            prior_function.freeze_parameters()
        model = APFSVI(
            generative_function=_make_generator(num_samples=4),
            prior_function=prior_function,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=4,
            num_prior_samples=4,
            num_measurement=4,
            adaptive_measure_points=True,
            adaptive_measure_mode="candidate_then_one_step",
            adaptive_measure_steps=1,
            adaptive_measure_lr=0.01,
            adaptive_measure_every=1,
            adaptive_candidate_pool_multiplier=2,
            adaptive_num_samples=2,
            adaptive_num_prior_samples=2,
            adaptive_num_projections=4,
            function_discrepancy=kind,
            discrepancy_num_projections=8,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        model._step = 1
        loss = model.nelbo(X[:BATCH_SIZE], y[:BATCH_SIZE])
        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)
        assert len(model.adaptive_measure_displacement_means) == 1
        assert len(model.adaptive_measure_relative_displacement_means) == 1
        assert torch.isfinite(model.adaptive_measure_displacement_means[-1])
        assert torch.isfinite(model.adaptive_measure_relative_displacement_means[-1])


class TestAPFSVITraining:

    def test_fit_iterations(self, regression_loader):
        model = APFSVI(
            generative_function=_make_generator(),
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            likelihood="regression",
            num_data=NUM_DATA,
            num_samples=NUM_SAMPLES,
            num_measurement=8,
            device=DEVICE,
            dtype=DTYPE,
            seed=SEED,
        )
        losses = model.fit(regression_loader, iterations=3, return_loss=True)
        assert len(losses) == 3
        assert len(model.KLs) == 3
