import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.gmvip import (
    CholeskyGaussianCoefficientPosterior,
    GeneralizedMatheronVIP,
    RBFKernel,
    RealNVPCoefficientPosterior,
)
from src.priors.function_bank import CoherentPriorFunctionSampler, PriorFunctionBank
from src.priors.generative_functions import BayesianNN, BayesLinear
from src.utils.empirical_covariance import empirical_cross_cov, empirical_mean
from src.utils.linalg import right_cholesky_solve, safe_cholesky


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _make_prior(num_samples=48, seed=0, input_dim=1, output_dim=1):
    prior = BayesianNN(
        structure=[8],
        activation=torch.nn.Tanh(),
        num_samples=num_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        layer_model=BayesLinear,
        seed=seed,
        fix_random_noise=True,
        zero_mean_prior=True,
        weight_log_sigma_init=-1.0,
        device=DEVICE,
        dtype=DTYPE,
    )
    prior.freeze_parameters()
    return prior


def _toy_data(num_points=32):
    X = torch.linspace(-1.8, 1.8, num_points, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    y = torch.sin(2.0 * X[:, 0]) + 0.15 * X[:, 0]
    return X, y


def _toy_classification_data(num_points=12, num_classes=3):
    X = torch.linspace(-1.8, 1.8, num_points, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    y = (torch.arange(num_points, device=DEVICE) % num_classes).long()
    return X, y


def _make_model(
    operator_type="rbf",
    posterior_type="gaussian",
    num_inducing=6,
    num_operator_bank_samples=40,
    mean_mode="prior_sample",
    inducing_scale="prior_cholesky",
    seed=0,
):
    Z = torch.linspace(-1.5, 1.5, num_inducing, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    model = GeneralizedMatheronVIP(
        base_prior=_make_prior(num_samples=num_operator_bank_samples, seed=seed),
        inducing_points=Z,
        operator_type=operator_type,
        posterior_type=posterior_type,
        num_operator_bank_samples=num_operator_bank_samples,
        jitter=1e-5,
        shrinkage=0.02,
        init_log_noise=-1.5,
        mean_mode=mean_mode,
        inducing_scale=inducing_scale,
        operator_bank_seed=seed + 100,
        flow_depth=2,
        flow_hidden_dim=32,
        flow_num_layers=2,
    )
    return model


def _make_vector_model(
    operator_type="rbf",
    num_inducing=5,
    num_operator_bank_samples=16,
    seed=0,
    learn_noise=True,
):
    Z = torch.linspace(-1.5, 1.5, num_inducing, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    return GeneralizedMatheronVIP(
        base_prior=_make_prior(
            num_samples=num_operator_bank_samples,
            seed=seed,
            output_dim=2,
        ),
        inducing_points=Z,
        operator_type=operator_type,
        posterior_type="gaussian",
        num_operator_bank_samples=num_operator_bank_samples,
        learn_noise=learn_noise,
        init_log_noise=torch.tensor([-2.0, -1.5], dtype=DTYPE, device=DEVICE),
        jitter=1e-5,
        shrinkage=0.02,
        init_lengthscale=0.7,
        learn_kernel=False,
        mean_mode="prior_sample",
        inducing_scale="prior_cholesky",
        operator_bank_seed=seed + 100,
        output_dim=2,
        num_train_samples=4,
        max_grad_norm=None,
    )


def _make_multiclass_model(
    operator_type="rbf",
    posterior_type="gaussian",
    num_inducing=5,
    num_operator_bank_samples=12,
    num_classes=3,
    seed=0,
):
    Z = torch.linspace(-1.5, 1.5, num_inducing, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    if operator_type == "empirical":
        mean_mode = "prior_sample"
        inducing_scale = "prior_cholesky"
    else:
        mean_mode = "zero"
        inducing_scale = "rbf_cholesky"
    return GeneralizedMatheronVIP(
        base_prior=_make_prior(
            num_samples=num_operator_bank_samples,
            seed=seed,
            output_dim=num_classes,
        ),
        inducing_points=Z,
        operator_type=operator_type,
        posterior_type=posterior_type,
        likelihood="multiclass",
        output_dim=num_classes,
        num_classes=num_classes,
        num_operator_bank_samples=num_operator_bank_samples,
        jitter=1e-5,
        shrinkage=0.02,
        mean_mode=mean_mode,
        inducing_scale=inducing_scale,
        learn_kernel=False,
        operator_bank_seed=seed + 100,
        num_train_samples=4,
        max_grad_norm=None,
    )


def test_supported_configs_instantiate_and_removed_posterior_raises():
    empirical = _make_model(operator_type="empirical", posterior_type="gaussian", seed=1)
    rbf = _make_model(operator_type="rbf", posterior_type="gaussian", seed=2)
    empirical_flow = _make_model(operator_type="empirical", posterior_type="realnvp", seed=4)
    rbf_flow = _make_model(operator_type="rbf", posterior_type="realnvp", seed=5)

    assert empirical.operator_type == "empirical"
    assert rbf.operator_type == "rbf"
    assert empirical_flow.posterior_type == "realnvp"
    assert rbf_flow.posterior_type == "realnvp"
    assert rbf.antithetic_samples is True
    assert rbf_flow.antithetic_samples is True

    with pytest.raises(ValueError):
        _make_model(operator_type="rbf", posterior_type="unsupported", seed=6)


def test_deprecated_gaussian_likelihood_type_maps_to_regression():
    Z = torch.linspace(-1.5, 1.5, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    model = GeneralizedMatheronVIP(
        base_prior=_make_prior(num_samples=12, seed=3),
        inducing_points=Z,
        operator_type="rbf",
        posterior_type="gaussian",
        likelihood_type="gaussian",
        num_operator_bank_samples=12,
        operator_bank_seed=103,
    )

    assert model.likelihood_type == "regression"


@pytest.mark.parametrize("operator_type", ["empirical", "rbf"])
def test_multiclass_supported_configs_sample_shapes_and_zero_kl(operator_type):
    model = _make_multiclass_model(operator_type=operator_type, seed=11)
    X, _ = _toy_classification_data(num_points=7, num_classes=3)

    posterior = model.sample_posterior_values(X, num_samples=5)
    prior = model.sample_prior_values(X, num_samples=6)

    assert model.likelihood_type == "multiclass"
    assert posterior.shape == (5, X.shape[0], 3)
    assert prior.shape == (6, X.shape[0], 3)
    assert torch.isfinite(posterior).all()
    assert torch.isfinite(prior).all()
    assert torch.allclose(model.kl_divergence(), torch.zeros((), dtype=DTYPE), atol=1e-12)


def test_multiclass_realnvp_is_not_supported():
    with pytest.raises(NotImplementedError, match="RealNVP"):
        _make_multiclass_model(operator_type="rbf", posterior_type="realnvp", seed=12)


def test_vector_regression_realnvp_is_not_supported():
    Z = torch.linspace(-1.5, 1.5, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    with pytest.raises(NotImplementedError, match="RealNVP"):
        GeneralizedMatheronVIP(
            base_prior=_make_prior(num_samples=12, seed=15, output_dim=2),
            inducing_points=Z,
            operator_type="rbf",
            posterior_type="realnvp",
            num_operator_bank_samples=12,
            output_dim=2,
        )


def test_cholesky_gaussian_posterior_shapes_and_zero_kl():
    posterior = CholeskyGaussianCoefficientPosterior(16, device=DEVICE, dtype=DTYPE)
    samples = posterior.rsample(num_samples=7)
    kl = posterior.kl_to_standard_normal()

    assert samples.shape == (7, 16)
    assert posterior.scale_tril.shape == (16, 16)
    assert kl.ndim == 0
    assert torch.allclose(kl, torch.zeros((), dtype=DTYPE), atol=1e-12)


def test_multiclass_cholesky_gaussian_posterior_shapes_and_zero_kl():
    posterior = CholeskyGaussianCoefficientPosterior(5, output_dim=3, device=DEVICE, dtype=DTYPE)
    samples = posterior.rsample(num_samples=7)
    prior = posterior.sample_prior(num_samples=6)
    kl = posterior.kl_to_standard_normal()

    assert samples.shape == (7, 5, 3)
    assert prior.shape == (6, 5, 3)
    assert posterior.scale_tril.shape == (3, 5, 5)
    assert posterior.std.shape == (5, 3)
    assert kl.ndim == 0
    assert torch.allclose(kl, torch.zeros((), dtype=DTYPE), atol=1e-12)


@pytest.mark.parametrize("operator_type", ["empirical", "rbf"])
def test_vector_regression_gmvip_sample_shapes_are_finite(operator_type):
    torch.manual_seed(41)
    model = _make_vector_model(operator_type=operator_type, seed=41)
    X = torch.linspace(-1.0, 1.0, 7, dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    posterior = model.sample_posterior_values(X, num_samples=5, seed=411)
    prior = model.sample_prior_values(X, num_samples=6, seed=412)
    pred_samples = model.predict_samples(X, num_samples=5)

    assert posterior.shape == (5, 7, 2)
    assert prior.shape == (6, 7, 2)
    assert pred_samples.shape == (5, 7, 2)
    assert torch.isfinite(posterior).all()
    assert torch.isfinite(prior).all()


@pytest.mark.parametrize("operator_type", ["empirical", "rbf"])
def test_vector_regression_elbo_backward_reaches_parameters(operator_type):
    torch.manual_seed(42)
    model = _make_vector_model(operator_type=operator_type, seed=42)
    X = torch.linspace(-1.0, 1.0, 9, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    y = torch.stack([torch.sin(X[:, 0]), torch.cos(X[:, 0])], dim=-1)

    loss, diagnostics = model.elbo_loss(X, y, num_samples=4, num_data=X.shape[0], beta=0.1)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["kl"])
    assert model.coefficients.loc.grad is not None
    assert model.coefficients.raw_scale_tril.grad is not None
    assert torch.isfinite(model.coefficients.loc.grad).all()
    assert torch.isfinite(model.coefficients.raw_scale_tril.grad).all()


def test_vector_regression_fixed_noise_likelihood_is_finite_and_not_trainable():
    model = _make_vector_model(operator_type="rbf", seed=43, learn_noise=False)
    X = torch.linspace(-1.0, 1.0, 6, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    y = torch.stack([X[:, 0], X[:, 0].square()], dim=-1)

    loss, diagnostics = model.elbo_loss(X, y, num_samples=4, num_data=X.shape[0])

    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["noise"]).all()
    assert model.likelihood.log_noise.requires_grad is False


def test_coefficient_posteriors_support_antithetic_base_samples():
    gaussian = CholeskyGaussianCoefficientPosterior(6, device=DEVICE, dtype=DTYPE)
    g_samples = gaussian.rsample(num_samples=8, antithetic=True)

    flow = RealNVPCoefficientPosterior(
        6,
        num_flows=2,
        hidden_dim=16,
        device=DEVICE,
        dtype=DTYPE,
    )
    f_samples, f_kl_terms, _ = flow.rsample_with_kl(num_samples=8, antithetic=True)

    assert torch.allclose(g_samples[:4] + g_samples[4:], torch.zeros_like(g_samples[:4]))
    assert torch.allclose(f_samples[:4] + f_samples[4:], torch.zeros_like(f_samples[:4]), atol=1e-12)
    assert torch.allclose(f_kl_terms, torch.zeros_like(f_kl_terms), atol=1e-10)


def test_realnvp_posterior_shapes_and_identity_initial_kl():
    posterior = RealNVPCoefficientPosterior(
        8,
        num_flows=2,
        hidden_dim=16,
        device=DEVICE,
        dtype=DTYPE,
    )
    samples, kl_terms, diagnostics = posterior.rsample_with_kl(num_samples=9)
    prior = posterior.sample_prior(num_samples=7)
    kl = posterior.kl_to_standard_normal(num_samples=16)

    assert samples.shape == (9, 8)
    assert prior.shape == (7, 8)
    assert kl_terms.shape == (9,)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(kl_terms).all()
    assert torch.isfinite(diagnostics["flow_logdet_mean"])
    assert torch.allclose(kl_terms, torch.zeros_like(kl_terms), atol=1e-10)
    assert torch.allclose(kl, torch.zeros((), dtype=DTYPE), atol=1e-10)


def test_rbf_kernel_and_cardinal_identity():
    X = torch.linspace(-2.0, 2.0, 7, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    Z = torch.linspace(-1.5, 1.5, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    kernel = RBFKernel(input_dim=1, lengthscale=0.7, outputscale=1.2, device=DEVICE, dtype=DTYPE)

    K = kernel(X, Z)
    model = _make_model(operator_type="rbf", posterior_type="gaussian", num_inducing=5, seed=5)
    Psi_Z = model.operator.psi(model.Z)

    assert K.shape == (7, 5)
    assert torch.isfinite(K).all()
    assert torch.allclose(Psi_Z, torch.eye(5, dtype=DTYPE), atol=1e-12, rtol=1e-12)


def test_multiclass_operator_identity_shapes():
    empirical = _make_multiclass_model(operator_type="empirical", num_inducing=4, seed=13)
    rbf = _make_multiclass_model(operator_type="rbf", num_inducing=4, seed=14)

    empirical_psi = empirical.operator.psi(empirical.Z)
    rbf_psi = rbf.operator.psi(rbf.Z)

    assert empirical_psi.shape == (3, 4, 4)
    assert rbf_psi.shape == (4, 4)
    assert torch.allclose(
        empirical_psi,
        torch.eye(4, dtype=DTYPE, device=DEVICE).unsqueeze(0).expand(3, -1, -1),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(rbf_psi, torch.eye(4, dtype=DTYPE, device=DEVICE), atol=1e-12, rtol=1e-12)


def test_empirical_operator_and_linalg_shapes_are_finite():
    values_x = torch.randn(20, 7, dtype=DTYPE)
    values_z = torch.randn(20, 5, dtype=DTYPE)
    mean_x = empirical_mean(values_x)
    mean_z = empirical_mean(values_z)
    K_xz = empirical_cross_cov(values_x, values_z, mean_x, mean_z)
    K_zz = empirical_cross_cov(values_z, values_z, mean_z, mean_z)
    L = safe_cholesky(K_zz + 1e-4 * torch.eye(5, dtype=DTYPE), initial_jitter=1e-6)
    Psi = right_cholesky_solve(K_xz, L)
    model = _make_model(operator_type="empirical", posterior_type="gaussian", seed=6)

    model_Psi = model.operator.psi(torch.linspace(-1.0, 1.0, 4, dtype=DTYPE).unsqueeze(-1))

    assert Psi.shape == (7, 5)
    assert model_Psi.shape == (4, model.num_inducing)
    assert torch.isfinite(model_Psi).all()


def test_vector_empirical_covariance_and_linalg_shapes_are_finite():
    values_x = torch.randn(20, 7, 3, dtype=DTYPE)
    values_z = torch.randn(20, 5, 3, dtype=DTYPE)
    mean_x = empirical_mean(values_x)
    mean_z = empirical_mean(values_z)
    K_xz = empirical_cross_cov(values_x, values_z, mean_x, mean_z)
    K_zz = empirical_cross_cov(values_z, values_z, mean_z, mean_z)
    L = safe_cholesky(
        K_zz + 1e-4 * torch.eye(5, dtype=DTYPE).unsqueeze(0),
        initial_jitter=1e-6,
    )
    Psi = right_cholesky_solve(K_xz, L)

    assert mean_x.shape == (7, 3)
    assert K_xz.shape == (3, 7, 5)
    assert L.shape == (3, 5, 5)
    assert Psi.shape == (3, 7, 5)
    assert torch.isfinite(Psi).all()


def test_prior_bank_evaluates_same_functions_across_inputs():
    prior = _make_prior(num_samples=24, seed=7)
    bank = PriorFunctionBank(prior, num_bank_samples=24, seed=88)
    X = torch.linspace(-2.0, -0.5, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    Z = torch.linspace(0.2, 1.2, 4, dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    separate_x = bank.evaluate(X)
    separate_z = bank.evaluate(Z)
    joint = bank.evaluate(torch.cat([X, Z], dim=0))

    assert torch.allclose(separate_x, joint[:, : X.shape[0]])
    assert torch.allclose(separate_z, joint[:, X.shape[0] :])


def test_prior_bank_preserves_vector_outputs():
    prior = _make_prior(num_samples=10, seed=7, output_dim=3)
    bank = PriorFunctionBank(prior, num_bank_samples=10, seed=88)
    X = torch.linspace(-2.0, -0.5, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    values = bank.evaluate(X)

    assert values.shape == (10, X.shape[0], 3)
    assert torch.isfinite(values).all()


def test_coherent_sampler_offsets_same_shaped_layer_noise():
    prior = BayesianNN(
        structure=[5, 5, 5],
        activation=torch.nn.Tanh(),
        num_samples=4,
        input_dim=1,
        output_dim=1,
        layer_model=BayesLinear,
        seed=12,
        fix_random_noise=True,
        zero_mean_prior=True,
        weight_log_sigma_init=-1.0,
        device=DEVICE,
        dtype=DTYPE,
    )
    sampler = CoherentPriorFunctionSampler(prior)

    latents = sampler.sample_latents(4)
    noises = [noise for _, noise in latents.module_noises]

    assert noises[1][0].shape == noises[2][0].shape
    assert not torch.allclose(noises[1][0], noises[2][0])


def test_rbf_apply_uses_rbf_cholesky_once_for_interpolation():
    torch.manual_seed(20)
    model = _make_model(operator_type="rbf", posterior_type="gaussian", num_inducing=5, seed=20)
    X, _ = _toy_data(num_points=8)
    coefficients = model.posterior.rsample(num_samples=3)
    g_X, g_Z = model.sample_residual_prior_values(X, num_samples=3, seed=120)
    calls = 0
    original_l_zz = model.operator._L_ZZ

    def counted_l_zz():
        nonlocal calls
        calls += 1
        return original_l_zz()

    model.operator._L_ZZ = counted_l_zz
    values = model.operator.apply(X, g_X, g_Z, coefficients)

    assert values.shape == (3, X.shape[0])
    assert torch.isfinite(values).all()
    assert calls == 1


def test_rbf_inducing_scale_is_empirical_prior_cholesky():
    torch.manual_seed(24)
    model = _make_model(operator_type="rbf", posterior_type="gaussian", num_inducing=5, seed=24)
    bank_Z = model.operator.moment_bank.evaluate(model.Z)
    mu_Z = empirical_mean(bank_Z)
    K_ZZ_raw = empirical_cross_cov(bank_Z, bank_Z, mu_Z, mu_Z)
    from src.utils.empirical_covariance import stabilize_covariance

    K_ZZ = stabilize_covariance(
        K_ZZ_raw,
        jitter=model.operator.jitter,
        shrinkage=model.operator.shrinkage,
    )
    expected = safe_cholesky(K_ZZ, initial_jitter=model.operator.jitter)

    assert model.operator.inducing_scale == "prior_cholesky"
    assert torch.allclose(model.operator.inducing_scale_matrix(), expected)


def test_empirical_operator_rejects_noncanonical_mean_or_scale():
    with pytest.raises(ValueError, match="mean_mode='prior_sample'"):
        _make_model(
            operator_type="empirical",
            mean_mode="zero",
            inducing_scale="prior_cholesky",
            seed=30,
        )
    with pytest.raises(ValueError, match="inducing_scale='prior_cholesky'"):
        _make_model(
            operator_type="empirical",
            mean_mode="prior_sample",
            inducing_scale="identity",
            seed=31,
        )


def test_empirical_inducing_map_is_empirical_mean_and_cholesky():
    torch.manual_seed(32)
    model = _make_model(operator_type="empirical", posterior_type="gaussian", num_inducing=5, seed=32)
    bank_Z = model.operator.bank.evaluate(model.Z)
    mu_Z = empirical_mean(bank_Z)
    K_ZZ_raw = empirical_cross_cov(bank_Z, bank_Z, mu_Z, mu_Z)
    from src.utils.empirical_covariance import stabilize_covariance

    K_ZZ = stabilize_covariance(
        K_ZZ_raw,
        jitter=model.operator.jitter,
        shrinkage=model.operator.shrinkage,
    )
    expected_scale = safe_cholesky(K_ZZ, initial_jitter=model.operator.jitter)

    assert torch.allclose(model.operator.inducing_mean(), mu_Z)
    assert torch.allclose(model.operator.inducing_scale_matrix(), expected_scale)


def test_empirical_cross_covariance_uses_joint_xz_bank_evaluation():
    torch.manual_seed(33)
    model = _make_model(operator_type="empirical", posterior_type="gaussian", num_inducing=5, seed=33)
    X, _ = _toy_data(num_points=7)
    calls = []
    original_evaluate = model.operator.bank.evaluate

    def counted_evaluate(X_arg):
        calls.append(int(X_arg.shape[0]))
        return original_evaluate(X_arg)

    model.operator.bank.evaluate = counted_evaluate
    psi = model.compute_interpolation_matrix(X)

    assert psi.shape == (X.shape[0], model.num_inducing)
    assert torch.isfinite(psi).all()
    assert calls == [X.shape[0] + model.num_inducing]


def test_rbf_identity_inducing_scale_and_zero_mean():
    torch.manual_seed(28)
    model = _make_model(
        operator_type="rbf",
        posterior_type="gaussian",
        num_inducing=5,
        mean_mode="zero",
        inducing_scale="identity",
        seed=28,
    )
    expected = torch.eye(model.num_inducing, dtype=DTYPE, device=DEVICE)

    assert model.operator.inducing_scale == "identity"
    assert torch.allclose(model.operator.inducing_mean(), torch.zeros(model.num_inducing, dtype=DTYPE))
    assert torch.allclose(model.operator.inducing_scale_matrix(), expected)


def test_rbf_kernel_inducing_scale_and_zero_mean():
    torch.manual_seed(29)
    model = _make_model(
        operator_type="rbf",
        posterior_type="gaussian",
        num_inducing=5,
        mean_mode="zero",
        inducing_scale="rbf_cholesky",
        seed=29,
    )
    K_ZZ = model.operator.kernel(model.Z, model.Z)
    K_ZZ = 0.5 * (K_ZZ + K_ZZ.T)
    expected = safe_cholesky(K_ZZ, initial_jitter=model.operator.jitter)

    assert model.operator.inducing_scale == "rbf_cholesky"
    assert torch.allclose(model.operator.inducing_mean(), torch.zeros(model.num_inducing, dtype=DTYPE))
    assert torch.allclose(model.operator.inducing_scale_matrix(), expected)


@pytest.mark.parametrize("operator_type", ["empirical", "rbf"])
def test_inducing_scale_gradients_reach_learnable_Z(operator_type):
    torch.manual_seed(25)
    Z = torch.linspace(-1.4, 1.4, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    model = GeneralizedMatheronVIP(
        base_prior=_make_prior(num_samples=40, seed=25),
        inducing_points=Z,
        operator_type=operator_type,
        posterior_type="gaussian",
        num_operator_bank_samples=40,
        learn_Z=True,
        learn_kernel=False,
        jitter=1e-5,
        shrinkage=0.02,
        operator_bank_seed=125,
    )

    scale = model.operator.inducing_scale_matrix()
    scale.square().sum().backward()

    assert model.operator.Z_param.grad is not None
    assert torch.isfinite(model.operator.Z_param.grad).all()
    assert model.operator.Z_param.grad.abs().sum() > 0.0


@pytest.mark.parametrize("operator_type", ["empirical", "rbf"])
def test_inducing_scale_gradients_reach_tunable_prior(operator_type):
    torch.manual_seed(26)
    Z = torch.linspace(-1.4, 1.4, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    prior = _make_prior(num_samples=40, seed=26)
    prior.defreeze_parameters()
    model = GeneralizedMatheronVIP(
        base_prior=prior,
        inducing_points=Z,
        operator_type=operator_type,
        posterior_type="gaussian",
        num_operator_bank_samples=40,
        freeze_base_prior=False,
        detach_prior_samples=False,
        detach_operator_prior_grad=False,
        learn_Z=False,
        learn_kernel=False,
        jitter=1e-5,
        shrinkage=0.02,
        operator_bank_seed=126,
    )

    scale = model.operator.inducing_scale_matrix()
    scale.square().sum().backward()
    grad_sum = sum(
        param.grad.abs().sum()
        for param in prior.parameters()
        if param.grad is not None
    )

    assert torch.isfinite(grad_sum)
    assert grad_sum > 0.0


def test_detached_operator_prior_grad_keeps_z_and_residual_prior_gradients():
    torch.manual_seed(28)
    Z = torch.linspace(-1.4, 1.4, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    prior = _make_prior(num_samples=32, seed=28)
    prior.defreeze_parameters()
    model = GeneralizedMatheronVIP(
        base_prior=prior,
        inducing_points=Z,
        operator_type="empirical",
        posterior_type="gaussian",
        num_operator_bank_samples=32,
        freeze_base_prior=False,
        detach_prior_samples=False,
        detach_operator_prior_grad=True,
        learn_Z=True,
        jitter=1e-5,
        shrinkage=0.02,
        init_log_noise=-1.5,
        operator_bank_seed=128,
    )
    X, y = _toy_data(num_points=10)

    operator_objective = (
        model.operator.inducing_mean().square().sum()
        + model.operator.inducing_scale_matrix().square().sum()
        + model.compute_interpolation_matrix(X).square().sum()
    )
    operator_objective.backward()
    prior_operator_grad = sum(
        (
            param.grad.abs().sum()
            for param in prior.parameters()
            if param.grad is not None
        ),
        torch.tensor(0.0, dtype=DTYPE, device=DEVICE),
    )

    assert prior_operator_grad.item() == 0.0
    assert model.operator.Z_param.grad is not None
    assert torch.isfinite(model.operator.Z_param.grad).all()
    assert model.operator.Z_param.grad.abs().sum() > 0.0

    model.zero_grad(set_to_none=True)
    loss, _ = model.elbo_loss(X, y, num_samples=4, num_data=X.shape[0])
    loss.backward()
    prior_residual_grad = sum(
        (
            param.grad.abs().sum()
            for param in prior.parameters()
            if param.grad is not None
        ),
        torch.tensor(0.0, dtype=DTYPE, device=DEVICE),
    )

    assert torch.isfinite(loss)
    assert prior_residual_grad > 0.0
    assert model.operator.Z_param.grad is not None
    assert model.operator.Z_param.grad.abs().sum() > 0.0


@pytest.mark.parametrize(
    ("operator_type", "posterior_type"),
    [
        ("empirical", "gaussian"),
        ("empirical", "realnvp"),
        ("rbf", "gaussian"),
        ("rbf", "realnvp"),
    ],
)
def test_posterior_and_prior_sample_shapes_are_finite(operator_type, posterior_type):
    torch.manual_seed(8)
    model = _make_model(operator_type=operator_type, posterior_type=posterior_type, seed=8)
    X, _ = _toy_data(num_points=9)

    posterior = model.sample_posterior_values(X, num_samples=5)
    prior = model.sample_prior_values(X, num_samples=7)

    assert posterior.shape == (5, 9)
    assert prior.shape == (7, 9)
    assert torch.isfinite(posterior).all()
    assert torch.isfinite(prior).all()
    assert prior.var(dim=0).mean() > 1e-8


def test_gaussian_elbo_backward_reaches_variational_parameters():
    torch.manual_seed(9)
    model = _make_model(operator_type="rbf", posterior_type="gaussian", seed=9)
    X, y = _toy_data(num_points=12)

    loss, diagnostics = model.elbo_loss(X, y, num_samples=6, num_data=X.shape[0])
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["kl"])
    assert model.coefficients.loc.grad is not None
    assert model.coefficients.raw_scale_tril.grad is not None
    assert model.likelihood.log_noise.grad is not None
    assert torch.isfinite(model.coefficients.loc.grad).all()
    assert torch.isfinite(model.coefficients.raw_scale_tril.grad).all()


@pytest.mark.parametrize("operator_type", ["empirical", "rbf"])
def test_multiclass_elbo_backward_reaches_variational_and_prior_parameters(operator_type):
    torch.manual_seed(39)
    Z = torch.linspace(-1.4, 1.4, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    prior = _make_prior(num_samples=16, seed=39, output_dim=3)
    prior.defreeze_parameters()
    mean_mode = "prior_sample" if operator_type == "empirical" else "zero"
    inducing_scale = "prior_cholesky" if operator_type == "empirical" else "rbf_cholesky"
    model = GeneralizedMatheronVIP(
        base_prior=prior,
        inducing_points=Z,
        operator_type=operator_type,
        posterior_type="gaussian",
        likelihood="multiclass",
        output_dim=3,
        num_classes=3,
        num_operator_bank_samples=16,
        freeze_base_prior=False,
        detach_prior_samples=False,
        learn_kernel=False,
        jitter=1e-5,
        shrinkage=0.02,
        mean_mode=mean_mode,
        inducing_scale=inducing_scale,
        operator_bank_seed=139,
        max_grad_norm=None,
    )
    X, y = _toy_classification_data(num_points=9, num_classes=3)

    loss, diagnostics = model.elbo_loss(X, y, num_samples=4, num_data=X.shape[0])
    loss.backward()
    prior_grad = sum(
        (
            param.grad.abs().sum()
            for param in prior.parameters()
            if param.grad is not None
        ),
        torch.tensor(0.0, dtype=DTYPE, device=DEVICE),
    )

    assert torch.isfinite(loss)
    assert diagnostics["kl"].ndim == 0
    assert "noise" not in diagnostics
    assert model.coefficients.loc.grad is not None
    assert model.coefficients.raw_scale_tril.grad is not None
    assert torch.isfinite(model.coefficients.loc.grad).all()
    assert torch.isfinite(model.coefficients.raw_scale_tril.grad).all()
    assert torch.isfinite(prior_grad)
    assert prior_grad > 0.0


def test_realnvp_elbo_backward_reaches_flow_parameters():
    torch.manual_seed(27)
    model = _make_model(operator_type="rbf", posterior_type="realnvp", seed=27)
    X, y = _toy_data(num_points=12)

    loss, diagnostics = model.elbo_loss(X, y, num_samples=6, num_data=X.shape[0])
    loss.backward()
    grads = [
        param.grad
        for param in model.posterior.parameters()
        if param.requires_grad and param.grad is not None
    ]

    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["kl"])
    assert torch.isfinite(diagnostics["flow_logdet_mean"])
    assert len(grads) > 0
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert sum(grad.abs().sum() for grad in grads) > 0.0


@pytest.mark.parametrize("operator_type", ["empirical", "rbf"])
def test_learnable_inducing_locations_receive_gradients(operator_type):
    torch.manual_seed(19)
    Z = torch.linspace(-1.4, 1.4, 5, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    model = GeneralizedMatheronVIP(
        base_prior=_make_prior(num_samples=40, seed=19),
        inducing_points=Z,
        operator_type=operator_type,
        posterior_type="gaussian",
        num_operator_bank_samples=40,
        learn_Z=True,
        jitter=1e-5,
        shrinkage=0.02,
        init_log_noise=-1.5,
        operator_bank_seed=119,
    )
    X, y = _toy_data(num_points=10)

    loss, _ = model.elbo_loss(X, y, num_samples=4, num_data=X.shape[0])
    loss.backward()

    assert model.operator.learn_Z
    assert model.operator.Z_param.grad is not None
    assert torch.isfinite(model.operator.Z_param.grad).all()
    assert model.operator.Z_param.grad.abs().sum() > 0.0


@pytest.mark.parametrize("posterior_type", ["gaussian", "realnvp"])
def test_alpha_one_data_objective_is_finite_and_differentiable(posterior_type):
    torch.manual_seed(17)
    model = _make_model(operator_type="rbf", posterior_type=posterior_type, seed=17)
    X, y = _toy_data(num_points=12)

    loss, diagnostics = model.elbo_loss(
        X,
        y,
        num_samples=4,
        num_data=X.shape[0],
        beta=0.1,
        data_alpha=1.0,
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]

    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["data_nll"])
    assert torch.allclose(diagnostics["data_alpha"], torch.tensor(1.0, dtype=DTYPE))
    assert len(grads) > 0
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_train_step_accepts_alpha_objective():
    torch.manual_seed(22)
    model = _make_model(operator_type="rbf", posterior_type="realnvp", seed=22)
    model.data_alpha = 1.0
    model.beta = 0.1
    model.num_train_samples = 2
    model.max_grad_norm = None
    X, y = _toy_data(num_points=16)
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)
    optimizer = torch.optim.Adam(model.vi_parameters(), lr=1e-3)
    model.prepare_for_training(loader)

    losses = []
    for X_batch, y_batch in loader:
        loss = model._train_step(optimizer, X_batch, y_batch)
        losses.append(float(loss.detach()))

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in losses)
    assert len(model.data_terms) == 2
    assert len(model.function_terms) == 2
    assert model.betas == [0.1, 0.1]
    assert torch.allclose(model.last_train_metrics["data_alpha"], torch.tensor(1.0, dtype=DTYPE))


def test_runner_batched_full_train_eval_matches_direct_gaussian_objective():
    pytest.skip("legacy GMVIP gap runner is not part of the experiments package")


def test_predict_shapes_and_finite_values():
    torch.manual_seed(11)
    model = _make_model(operator_type="rbf", posterior_type="gaussian", seed=11)
    X, _ = _toy_data(num_points=11)

    pred = model.predict(X, num_samples=16)

    assert pred["f_mean"].shape == (11,)
    assert pred["f_var"].shape == (11,)
    assert pred["y_mean"].shape == (11,)
    assert pred["y_var"].shape == (11,)
    assert torch.isfinite(pred["f_mean"]).all()
    assert torch.isfinite(pred["f_var"]).all()
    assert torch.all(pred["y_var"] >= pred["f_var"])


def test_toy_regression_data_nll_improves():
    torch.manual_seed(12)
    X, y = _toy_data(num_points=28)
    model = _make_model(
        operator_type="rbf",
        posterior_type="gaussian",
        num_inducing=8,
        num_operator_bank_samples=64,
        seed=12,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        _, initial = model.elbo_loss(X, y, num_samples=32, num_data=X.shape[0], beta=0.1)
    initial_nll = initial["data_nll"]

    for _ in range(45):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.elbo_loss(X, y, num_samples=8, num_data=X.shape[0], beta=0.1)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    with torch.no_grad():
        _, final = model.elbo_loss(X, y, num_samples=32, num_data=X.shape[0], beta=0.1)

    assert final["data_nll"] < initial_nll
