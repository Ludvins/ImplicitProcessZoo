# Implicit Process Zoo

This repository collects methods that perform function-space variational
inference to learn implicit processes. The focus is on Bayesian models whose
priors are easy to sample from but whose finite-dimensional densities are not
available in closed form, such as Bayesian neural networks and other implicit
function generators.

The codebase provides a shared benchmark framework for comparing these
approaches on regression and classification tasks.

The main regression benchmark currently supports:

- `map`: deterministic MAP baseline, optimizing a single network and
  observation-noise estimate instead of a posterior.
- `mfvi`: mean-field variational inference, using independent Gaussian
  marginals over weights.
- `fbnn`: functional Bayesian neural network baseline, regularizing finite
  measurement-set function samples with a score-based functional KL surrogate.
- `tfsvi`: tractable function-space variational inference, linearizing the
  network around the variational mean so posterior and prior functions become
  Gaussian.
- `vip`: variational implicit processes, representing posterior functions with
  a finite coefficient posterior over sampled prior-function features.
- `ftip`: flow-transformed implicit processes, keeping VIP's finite coefficient
  space while replacing the Gaussian coefficient posterior with an invertible
  flow.
- `gmvip`: generalized Matheron VIP, using an inducing-point Matheron update and
  keeping the KL in latent coefficient space.
- `sip`: sparse implicit process, summarizing the implicit process with
  inducing variables and estimating the inducing-space KL with a separate
  critic.

Supported Python versions are 3.10 through 3.12. CI exercises Python 3.10 and
3.12 on CPU.

## Installation

Install the complete editable research environment:

```bash
pip install -r requirements.txt
```

`requirements.txt` delegates to the dependency groups in `pyproject.toml`. For
a reproducible Python 3.12 CPU environment, install
`requirements/lock-cpu-py312.txt` instead. Select the platform-appropriate
PyTorch wheel before installing the project when CUDA or another accelerator is
required.

## Repository Layout

```text
implicit_process_zoo/
  priors/                 Bayesian neural network and GP prior samplers
  flows/                  affine, spline, and mixing layers for flow posteriors
  utils/                  datasets, metrics, likelihoods, linalg, helpers

  mfvi/                   mean-field variational implicit process
  fbnn/                   functional BNN
  tfsvi/                  tractable function-space VI
  vip/                    variational implicit process
  ftip/                   flow-transformed implicit process
  gmvip/                  generalized Matheron VIP
  sip/                    sparse implicit process
  map_baseline.py         deterministic MAP baseline

experiments/
  benchmark_utils.py      shared reporting, W&B, and comparison helpers
  uci/                    UCI-style scalar regression benchmark
  regression/             Year/Airline/Taxi large regression benchmark
  classification/         image classification benchmark
  synthetic/              Variational-LLA synthetic plotting runner
  volterra/               Lotka-Volterra simulator-prior experiment

tests/                    unit and smoke tests
```

The seven probabilistic methods live under their own `implicit_process_zoo/<method>/` packages.
`implicit_process_zoo/map_baseline.py` is kept separate because it is a deterministic reference
model rather than an implicit-process approximation.

## Methods

This section summarizes the eight approaches exposed by
`experiments.uci.benchmark`. The notation below uses scalar regression for clarity.
Let $D = \{(x_i, y_i)\}_{i=1}^N$, let $B$ be a minibatch, and let the
likelihood be

$$
p(y_i | f_i) = N(y_i; f_i, \sigma_y^2).
$$

Most variational methods optimize a minibatch-scaled objective of the form

$$
L = -\frac{N}{|B|} A_B + R,
$$

where $A_B$ is a data-fit term and $R$ is the method-specific regularizer or
KL. When the BB-alpha parameter is zero (`--bb_alpha 0`), the data-fit term is
the usual expected log-likelihood. When it is nonzero, the code uses the
BB-alpha energy

$$
A_B = \sum_{i \in B} \frac{1}{\alpha} \log \left[ \frac{1}{S} \sum_{s=1}^S \exp\{\alpha \log p(y_i | f_i^{(s)})\} \right],
$$

with the $\alpha \to 0$ limit equal to
$\sum_i \mathbb{E}_q[\log p(y_i \mid f_i)]$. Classification uses the same
structure with Bernoulli or multiclass likelihoods where implemented.

### MAP (`map`)

MAP is the deterministic baseline. It uses a standard MLP
$f_\theta(x)$ and learns a Gaussian observation variance $\sigma_y^2$.
There is no posterior distribution over functions or weights.

The optimized objective is

$$
L_{\mathrm{MAP}}(\theta, \sigma_y^2) = -\frac{N}{|B|} \sum_{i \in B} \log N(y_i; f_\theta(x_i), \sigma_y^2) + \frac{\lambda}{2} ||\theta||_2^2.
$$

The predictive distribution is

$$
p(y_{\ast} \mid x_{\ast}, D) \approx N(f_{\theta}(x_{\ast}), \sigma_y^2).
$$

This is useful as a calibration point: it tests how much the Bayesian or
implicit-process methods improve over a point estimate with learned noise.

### Mean-Field VI (`mfvi`)

MFVI performs weight-space variational inference. The Bayesian neural network
parameters have a diagonal Gaussian variational posterior

$$
q_\lambda(\theta) = N(\theta; \mu, \mathrm{diag}(\sigma^2)), \quad \theta^{(s)} = \mu + \sigma \odot \epsilon^{(s)}, \quad \epsilon^{(s)} \sim N(0, I).
$$

The prior is a standard normal over the same weight vector:

$$
p(\theta) = N(0, I).
$$

The ELBO-style objective is

$$
L_{\mathrm{MFVI}} = -\frac{N}{|B|} \sum_{i \in B} E_{q(\theta)}[\log p(y_i | f_{\theta}(x_i))] + KL(q_\lambda(\theta) || p(\theta)).
$$

In code, the BNN uses fresh Monte Carlo weight noise during training. The
posterior uncertainty is therefore entirely weight-space diagonal Gaussian
uncertainty propagated through the network.

### Functional BNN (`fbnn`)

Reference: Sun et al. (ICLR 2019), "Functional Variational Bayesian Neural
Networks" [1].

FBNN keeps a trainable Bayesian neural network posterior but regularizes it in
function space rather than directly with a weight-space KL. It compares the
posterior function distribution $q(f)$ to a prior process $p_0(f)$, which can be
a GP or a BNN prior.

At each step, the code builds a measurement set

$$
X_M = X_{\mathrm{context}} \cup X_{\mathrm{train-subset}}.
$$

The posterior BNN gives function samples $f_M^{(s)} = f^{(s)}(X_M)$. The
functional KL is optimized through a score-matching surrogate:

$$
KL(q(f_M) || p_0(f_M)) \quad \Longrightarrow \quad E_{q(f_M)} \left[ f_M^T \left( \nabla_{f_M} \log q(f_M) - \nabla_{f_M} \log p_0(f_M) \right) \right].
$$

The code estimates the posterior score with a spectral Stein gradient estimator
(SSGE). For GP priors, the prior score is analytic:

$$
\nabla_f \log p_0(f_M) = -(K_{MM} + \gamma^2 I)^{-1}(f_M - m_M).
$$

The training objective is

$$
L_{\mathrm{FBNN}} = -\frac{N}{|B|} A_B + \lambda_{\mathrm{KL}} \widehat{KL}_{\mathrm{func}}.
$$

The context points are important: they push the posterior back toward the prior
away from observed training inputs.

### Tractable FSVI (`tfsvi`)

Reference: Rudner et al. (NeurIPS 2022), "Tractable Function-Space Variational
Inference in Bayesian Neural Networks" [2].

TFSVI uses a mean-field Gaussian posterior over parameters,

$$
q(\theta) = N(\mu, \mathrm{diag}(\sigma^2)), \quad p(\theta) = N(0, \sigma_p^2 I),
$$

but regularizes the induced function distribution. The key approximation is a
first-order Taylor expansion around $\theta = \mu$:

$$
f(X_C; \theta) \approx f(X_C; \mu) + J_C(\theta - \mu),
$$

where $J_C$ is the Jacobian of the network outputs at a context set $X_C$.
This makes the induced posterior and prior Gaussian in function space:

$$
\tilde q(f_C) = N( f(X_C; \mu), J_C \mathrm{diag}(\sigma^2) J_C^T ),
$$

$$
\tilde p(f_C) = N( f(X_C; \mu) - J_C \mu, \sigma_p^2 J_C J_C^T ).
$$

The objective is

$$
L_{\mathrm{TFSVI}} = -\frac{N}{|B|} A_B + \max_{c=1,\ldots,S_{\mathrm{ctx}}} KL( \tilde q(f_{C_c}) || \tilde p(f_{C_c}) ).
$$

The implementation sums scalar-output Gaussian KLs over output dimensions. The
base network is treated as an architecture template; TFSVI's trainable
parameters are the flat variational mean and log standard deviation.

### VIP (`vip`)

Reference: Ma et al. (ICML 2019), "Variational Implicit Processes" [3].

VIP constructs a finite-rank implicit process from coherent prior-function
samples. Given prior samples $g_s(\cdot)$, define

$$
m(X) = \frac{1}{S}\sum_{s=1}^S g_s(X), \quad \Phi_s(X) = \frac{g_s(X) - m(X)}{\sqrt{S-1}}.
$$

The coefficient posterior is a full-covariance Gaussian:

$$
q(a) = N(q_\mu, L_q L_q^T), \quad p(a) = N(0, I).
$$

The posterior process is

$$
f(X) = m(X) + \Phi(X)^T a.
$$

Therefore, for regression, the marginal $q(f(X))$ is Gaussian and can be
handled analytically:

$$
E_q[f(X)] = m(X) + \Phi(X)^T q_\mu,
$$

$$
Cov_q[f(X)] = \Phi(X)^T L_q L_q^T \Phi(X).
$$

The objective is

$$
L_{\mathrm{VIP}} = -\frac{N}{|B|} E_{q(f_B)}[\log p(y_B | f_B)] + KL(q(a) || N(0, I)).
$$

VIP is finite-dimensional in coefficient space but function-valued after it is
composed with the sampled prior basis. The prior BNN can be frozen or trained,
depending on the experiment flags.

### FTIP (`ftip`)

Reference: Ortega et al. (2026 preprint), "Flow-Transformed Implicit Processes
for Function-Space Variational Inference" [4].

FTIP keeps the same finite-rank prior basis as VIP but replaces the Gaussian
coefficient posterior with a normalizing-flow posterior. Let

$$
z \sim N(0, I), \quad a = T_\phi(z),
$$

where $T_\phi$ is an invertible flow. The induced coefficient density is

$$
\log q_\phi(a) = \log p(z) - \log \left|\det \frac{\partial T_\phi(z)}{\partial z}\right|.
$$

The posterior process is still

$$
f(X) = m(X) + \Phi(X)^T a,
$$

but $q(a)$ can now be non-Gaussian and multimodal if the flow is expressive
enough. The code supports affine coupling, spline coupling, and spline coupling
with Glow-style $1 \times 1$ mixing.

The objective is estimated with Monte Carlo samples:

$$
L_{\mathrm{FTIP}} = -\frac{N}{|B|} A_B + E_{a \sim q_\phi} [ \log q_\phi(a) - \log N(a;0,I) ].
$$

FTIP can optionally warm-start from a trained VIP model by initializing an
affine flow layer from VIP's Gaussian coefficient posterior.

### Generalized Matheron VIP (`gmvip`)

GMVIP rewrites the finite-rank implicit process through a Matheron-style update
at inducing inputs $Z$. For a coherent prior sample $g(\cdot)$, whitened
coefficients $a$, inducing mean $\mu_Z$, inducing scale $D_Z$, and
interpolation operator $\Psi_Z$, the sampled function is

$$
f(X) = g(X) + \Psi_Z(X) [ \mu_Z + D_Z a - g(Z) ].
$$

The inducing values are

$$
u = f(Z) = \mu_Z + D_Z a.
$$

The latent coefficient prior is always

$$
p(a) = N(0, I).
$$

The Gaussian posterior option uses

$$
q(a) = N(m_a, L_a L_a^T),
$$

and the RealNVP option uses $a = T_\phi(z)$, $z \sim N(0,I)$, with the same
latent-variable KL form as FTIP.

GMVIP supports two operators:

- Empirical operator:

  $$
  \Psi_Z(X) = K_{XZ}^{\mathrm{emp}} (K_{ZZ}^{\mathrm{emp}})^{-1},
  $$

  where the mean, cross-covariance, and inducing covariance are estimated from a
  coherent bank of prior functions evaluated on $[X; Z]$. For this operator,
  $\mu_Z$ is the empirical prior mean and $D_Z$ is the stabilized empirical
  Cholesky factor.

- RBF operator:

  $$
  \Psi_Z(X) = K_{\mathrm{RBF}}(X,Z) K_{\mathrm{RBF}}(Z,Z)^{-1},
  $$

  with learnable or fixed RBF lengthscales/output scale. The inducing mean and
  scale are configurable; the canonical RBF setting is $\mu_Z = 0$ and
  $D_Z = \mathrm{chol}(K_{\mathrm{RBF}}(Z,Z))$.

The training objective is

$$
ELBO = E_{q(a),g} [ \log p(y_B | f_B) ] - \beta KL(q(a) || N(0,I)),
$$

with minibatch scaling on the likelihood and an optional BB-alpha data term.
Only the latent-variable KL is used.

### Sparse Implicit Process (`sip`)

Reference: Rodriguez Santana et al. (ICML 2022), "Function-space Inference with
Sparse Implicit Processes" [5].

SIP summarizes an implicit process with inducing inputs $Z$ and inducing
variables

$$
u = f(Z),
$$

where $Z$ can be fixed or learned. The variational family keeps the posterior
over inducing values implicit:

$$
\xi \sim N(0,I), \quad \epsilon = m_\epsilon + \exp(\tfrac{1}{2}\ell_\epsilon) \odot \xi, \quad u = h_\phi(\epsilon).
$$

Here $h_\phi$ is a neural sampler and $(m_\epsilon,\ell_\epsilon)$ are trainable
input-noise moments. The full process posterior is then defined through the
sparse factorization

$$
q_\phi(f_X, u) = p_\theta(f_X \mid u) q_\phi(u).
$$

The prior over inducing values is also implicit:

$$
u_p = f_\theta(Z), \quad f_\theta \sim p_\theta.
$$

For each batch, coherent prior samples are evaluated jointly at $[X; Z]$. This
single joint evaluation estimates

$$
m_X,\quad m_Z,\quad K_{ZZ},\quad K_{XZ}, \quad \mathrm{diag}(K_{XX}),
$$

and, for prediction, the full $K_{XX}$. These moments define the GP-style sparse
prior conditional used inside the variational family:

$$
\mu_{X|u} = m_X + K_{XZ}K_{ZZ}^{-1}(u - m_Z),
$$

$$
\Sigma_{X|u} = K_{XX} - K_{XZ}K_{ZZ}^{-1}K_{ZX}.
$$

The training objective uses the diagonal conditional
$N(\mu_{X|u}, \mathrm{diag}(\Sigma_{X|u}))$ for minibatch likelihood terms,
matching the efficient training path in the released SIP code. Prediction uses
the full sparse conditional covariance $N(\mu_{X|u}, \Sigma_{X|u})$. In both
paths $K_{ZZ}$ and conditional covariances are stabilized with jittered
Cholesky solves rather than explicit matrix inverses.

The functional ELBO optimized by the primal parameters is

$$
\mathcal{L}_{\mathrm{SIP}} = \mathbb{E}_{q_\phi(f_X, u)} [ \log p(y \mid f_X) ] - \beta \mathcal{R}_{\mathrm{KL}},
$$

with minibatch scaling on the likelihood. The current implementation does not
use a Gaussian closed-form KL for $q_\phi(u)$. Since both $q_\phi(u)$ and
$p_\theta(u)$ are implicit, SIP trains a separate critic $T_\omega(u)$ to
estimate the inducing-space log density ratio. The critic is trained with the
binary classification objective

$$
\min_\omega - \frac{1}{2} \left[ \mathbb{E}_{q_\phi(u)} [ \log \sigma(T_\omega(u)) ] + \mathbb{E}_{p_\theta(u)} [ \log(1 - \sigma(T_\omega(u))) ] \right].
$$

At the optimum,

$$
T_{\omega^{\ast}}(u) = \log q_\phi(u) - \log p_\theta(u),
$$

so the forward inducing-space KL is estimated as

$$
KL(q_\phi(u) \| p_\theta(u)) = \mathbb{E}_{q_\phi(u)}[T_{\omega^{\ast}}(u)].
$$

The implementation uses the symmetrized inducing regularizer from the SIP
training code:

$$
\mathcal{R}_{\mathrm{KL}} \approx \frac{1}{2} \left( \mathbb{E}_{q_\phi(u)}[T_{\omega^{\ast}}(u)] - \mathbb{E}_{p_\theta(u)}[T_{\omega^{\ast}}(u)] \right).
$$

The critic is optimized separately before each primal update and is not part of
the variational/prior optimizer step. The code logs the critic loss, critic
accuracy, critic saturation fraction, forward KL estimate, reverse KL estimate,
and the active $\beta$ value. The tractability comes from applying the
adversarial KL only to $u$, whose dimension is controlled by $|Z|$, while
recovering predictions over arbitrary inputs through the sparse conditional.

## UCI Regression

Run a single method:

```bash
python -m experiments.uci.benchmark --model gmvip --dataset concrete
```

Run all tracked regression methods on one dataset:

```bash
python -m experiments.uci.benchmark --model all --dataset boston
```

Supported UCI regression datasets:

```text
boston, concrete, energy, kin8nm, naval, power, protein, winered, yacht
```

The benchmark also exposes synthetic or diagnostic regression datasets through
the shared dataset loader, including `gap`, `bimodal`, `skewed`,
`heteroscedastic`, `snelson`, and `variational_lla`. The former misspellings
`yatch` and `heterocedastic` remain deprecated command-line aliases.

Common experiment flags:

```bash
python -m experiments.uci.benchmark \
  --model vip \
  --dataset boston \
  --seed 0 \
  --iterations 30000 \
  --bb_alpha 0 \
  --batch_size 100 \
  --lr 0.001 \
  --hidden_dims 10 10 \
  --activation tanh \
  --layer_model BayesLinear \
  --device cuda
```

Prior-learning variants are controlled explicitly for methods where that
comparison is meaningful:

```bash
# VIP
--vip_learn_prior
--no-vip_learn_prior

# FTIP
--ftip_learn_prior
--no-ftip_learn_prior

# GMVIP
--gmvip_learn_prior
--no-gmvip_learn_prior

# SIP
--sip_learn_prior
--no-sip_learn_prior
```

FTIP supports VIP warm-starting through `--auto_warm_start`, which is enabled by
default for direct `--model ftip` runs. For equal-budget comparisons where FTIP
should receive exactly the requested number of optimizer steps, pass:

```bash
--no_auto_warm_start
```

GMVIP's Matheron form is:

$$
f(X) = g(X) + \Psi_Z(X) [\mu_Z + D_Z a - g(Z)].
$$

The UCI runner supports empirical and RBF Matheron operators, Gaussian and
RealNVP coefficient posteriors, learnable inducing locations, antithetic
coefficient sampling, and tunable prior BNNs. A representative empirical GMVIP
run is:

```bash
python -m experiments.uci.benchmark \
  --model gmvip \
  --dataset concrete \
  --iterations 30000 \
  --bb_alpha 0 \
  --gmvip_operator_type empirical \
  --gmvip_posterior_type gaussian \
  --gmvip_num_inducing 100 \
  --gmvip_inducing_method kmeans \
  --gmvip_learn_Z \
  --gmvip_num_operator_bank_samples 512 \
  --gmvip_num_train_samples 512 \
  --gmvip_num_eval_samples 512 \
  --gmvip_mean_mode prior_sample \
  --gmvip_inducing_scale prior_cholesky \
  --gmvip_jitter 0.001 \
  --gmvip_max_log_noise none \
  --gmvip_learn_prior
```

## Large Regression

`experiments.regression.benchmark` reuses the same model/training code as the UCI
runner, but restricts the dataset set to the larger Variational-LLA-style
regression tasks:

```text
year, airline, taxi
```

Example:

```bash
python -m experiments.regression.benchmark \
  --model gmvip \
  --dataset year \
  --iterations 30000 \
  --bb_alpha 0 \
  --batch_size 100 \
  --lr 0.001 \
  --hidden_dims 10 10 \
  --activation tanh \
  --layer_model BayesLinear \
  --device cuda
```

`year` downloads `YearPredictionMSD.txt` through the dataset loader when needed.
`airline` expects the preprocessed Variational-LLA Airline file at
`data/airline.csv`. `taxi` uses `data/taxi.csv` when present, otherwise it can
create it from the NYC yellow taxi parquet source if `pyarrow` is installed.

## Electricity Load Forecasting

The canonical corrected ELD experiment is documented in
[`experiments/eld_forecasting/README.md`](experiments/eld_forecasting/README.md).
That guide freezes the original 24-hour-context/24-hour-forecast protocol and
target identities, and covers data preparation, corrected half-open index
regions, reporting, resource expectations, and exact three-seed and figure
reproduction commands.

## Synthetic Plot Runner

`experiments.synthetic.plot` trains selected regression methods on the
Variational-LLA synthetic dataset and writes publication-style predictive plots:

```bash
python -m experiments.synthetic.plot --models mfvi fbnn vip tfsvi ftip gmvip
python -m experiments.synthetic.plot --models all --iterations 2000 --device cuda
```

All model/training flags accepted by `experiments.uci.benchmark` can be forwarded to
this runner. The dataset is fixed to `variational_lla`.

## Classification

The classification runner supports `FashionMNIST` and `CIFAR10` with `map`,
`mfvi`, `fbnn`, `tfsvi`, `vip`, `ftip`, `gmvip`, and `sip`:

```bash
python -m experiments.classification.benchmark --dataset FashionMNIST --model vip
python -m experiments.classification.benchmark --dataset CIFAR10 --model all
```

Use `--help` on any benchmark entrypoint for the complete CLI.

## Outputs

Benchmark runs write JSON result files under `--output_dir` (`results` by
default). When several methods or datasets are run together, comparison JSON and
CSV summaries are also written. Regression metrics include RMSE, NLL, CRPS, and
CQM when available.

Checkpoints are saved by default; pass `--no_save_checkpoint` to disable them.
Full version-1 training checkpoints support exact optimizer/scheduler/RNG
resume. Use `--warm-start-from` for legacy model-only state dictionaries;
passing one to `--resume-from-checkpoint` fails with migration guidance.

## Tests

Run the full test suite:

```bash
ruff format --check .
ruff check .
python -m pytest -q --cov
```

Useful focused checks:

```bash
python -m pytest tests/test_gmvip.py -q
python -m pytest tests/test_ftip.py -q
python -m pytest tests/test_sip.py -q
```

The measured publication baseline is 54% statement coverage across the
library and experiment packages. Pull requests may raise this floor but must not
lower it.

## License and citation

The software is released under the [MIT License](LICENSE). Citation metadata is
provided in [`CITATION.cff`](CITATION.cff). Individual datasets retain their
own licenses; in particular, ElectricityLoadDiagrams20112014 is CC BY 4.0 and
requires attribution as described in the ELD guide.

## References

[1] Shengyang Sun, Guodong Zhang, Jiaxin Shi, and Roger Grosse. "Functional
Variational Bayesian Neural Networks." International Conference on Learning
Representations (ICLR), 2019. https://openreview.net/forum?id=rkxacs0qY7

[2] Tim G. J. Rudner, Zonghao Chen, Yee Whye Teh, and Yarin Gal. "Tractable
Function-Space Variational Inference in Bayesian Neural Networks."
Advances in Neural Information Processing Systems 35 (NeurIPS), 2022.
https://arxiv.org/abs/2312.17199

[3] Chao Ma, Yingzhen Li, and Jose Miguel Hernandez-Lobato. "Variational
Implicit Processes." Proceedings of the 36th International Conference on
Machine Learning, PMLR 97:4222-4233, 2019.
https://proceedings.mlr.press/v97/ma19b.html

[4] Luis A. Ortega, Andres R. Masegosa, and Thomas D. Nielsen.
"Flow-Transformed Implicit Processes for Function-Space Variational Inference."
Preprint submitted for revision, 2026. https://arxiv.org/abs/2606.01954

[5] Simon Rodriguez Santana, Bryan Zaldivar, and Daniel Hernandez-Lobato.
"Function-space Inference with Sparse Implicit Processes." International
Conference on Machine Learning (ICML), 2022.
https://arxiv.org/abs/2110.07618

## Notes

This is research code. Many methods expose experimental knobs through the CLI,
and several benchmarks are intended for controlled comparisons rather than
general-purpose model selection. Prefer explicit flags in experiment scripts so
prior learning, inducing-point learning, and sample counts are unambiguous.
