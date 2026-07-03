# Generalized Matheron VIP: Current Specification

This document is the active specification for `src/gmvip`.

The canonical construction is

```text
f(X) = g(X) + Psi_Z(X) [mu_Z + D_Z a - g(Z)]
```

where `g` is a coherent prior-function sample, `Z` are inducing inputs,
`Psi_Z` is a Matheron/cardinal interpolation operator, and `a` is a latent
coefficient vector with prior `p(a) = N(0, I)`.

## Supported Methods

The method family is intentionally small:

| Method | `operator_type` | `posterior_type` |
| --- | --- | --- |
| Empirical GMVIP | `empirical` | `gaussian` |
| Empirical GMVIP + flow | `empirical` | `realnvp` |
| RBF GMVIP | `rbf` | `gaussian` |
| RBF GMVIP + flow | `rbf` | `realnvp` |

No extra discrete latent selector, context-dependent coefficient posterior,
functional KL, MMD KL, adversarial KL, classification head, or direct
data-conditioned generator is in scope for this implementation.

## Components

### Prior Sample `g`

Use the existing coherent prior-function sampler/bank. For a posterior draw, the
same stochastic prior identity must be evaluated at all required inputs,
including `X` and `Z`.

When the prior bank is used to estimate moments, bank identities are fixed by
seed. If the BNN prior is tunable, the bank noise identities stay fixed while the
prior parameters can receive gradients.

### Operator `Psi_Z`

Two operators are supported:

- `empirical`: estimate prior covariance from a bank of coherent BNN prior
  samples and form the empirical Matheron interpolation operator.
- `rbf`: use an RBF cardinal kernel on `Z` for interpolation.

Both paths avoid `torch.inverse`; linear solves use stabilized Cholesky factors.
All covariance solves must include jitter, and the RBF solve on `K_ZZ` must also
use jitter.

### Inducing Scale `D_Z`

`D_Z` maps whitened coefficients into inducing values. Supported RBF-operator
choices are:

- `prior_cholesky`: Cholesky factor of the stabilized empirical BNN-prior
  covariance at `Z`.
- `prior_diag`: diagonal scale from the stabilized empirical BNN-prior
  covariance at `Z`.
- `rbf_cholesky`: Cholesky factor of the RBF operator kernel on `Z`.
- `identity`: the identity matrix.

`prior_cholesky` is the most prior-calibrated option. `prior_diag`,
`rbf_cholesky`, and `identity` are scalability/ablation options that reduce or
remove dependence on full empirical BNN-prior covariance moments.

### Coefficient Posterior `q(a)`

Two coefficient posteriors are supported:

- `gaussian`: full-covariance Gaussian parameterized by a Cholesky factor.
- `realnvp`: affine-coupling RealNVP flow applied to a standard Gaussian base.

The KL is always the latent-variable KL against `N(0, I)`. The Gaussian KL is
closed-form. The RealNVP KL is estimated with reparameterized samples:

```text
KL(q(a) || p(a)) = E_q[log q(a) - log N(a; 0, I)]
```

Antithetic base sampling is enabled by default for Gaussian and RealNVP
coefficient samples.

### Likelihood

Only scalar-output Gaussian regression is supported:

```text
y | f(X) ~ N(f(X), sigma_y^2)
```

The likelihood noise is tunable unless explicitly frozen. Optional lower/upper
log-noise clamps are guardrails, not part of the core method.

## Public API

`GeneralizedMatheronVIP` exposes:

- `sample_posterior_values(X, num_samples, seed=None)`
- `sample_posterior_values_with_kl(X, num_samples, seed=None)`
- `sample_prior_values(X, num_samples, seed=None)`
- `kl_divergence()`
- `elbo(X_batch, y_batch, num_samples, num_data=None, beta=1.0, data_alpha=0.0, seed=None)`
- `elbo_loss(...)`
- `nelbo(...)`
- `predict(X, num_samples=100, include_noise=True, seed=None)`
- `predict_samples(X, num_samples=100, include_noise=True, seed=None)`
- `_train_step(optimizer, X, y)` for framework-level training loops

There is no context-setting API.

## Runners

The Gap and UCI runners should expose only:

- `--posterior_type gaussian`
- `--posterior_type realnvp`

Flow options are the RealNVP depth, hidden width, MLP layer count, dropout, and
scale bound. There are no additional selector-posterior options.

## Tests

Tests should cover:

- supported configurations instantiate
- unsupported posterior names raise `ValueError`
- posterior/prior sample shapes are `[R, N]` and finite
- Gaussian KL is zero at initialization
- RealNVP starts near identity with finite sampled KL terms
- `Psi_Z(Z)` is close to identity for RBF
- empirical interpolation matrices are finite
- ELBO is finite and gradients reach posterior/noise parameters
- RealNVP gradients reach flow parameters
- alpha data objectives are finite
- `_train_step` works with minibatches
- simple 1D smoke training improves the data objective
