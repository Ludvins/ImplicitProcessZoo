# Synthetic plots

`experiments.synthetic.plot` trains scalar-regression methods on the fixed
Variational-LLA synthetic dataset and writes publication-style predictive
plots.

The dataset is always `variational_lla`. The runner reuses model/training flags
from the UCI benchmark and adds plot ordering, panels, density bands,
predictive intervals, and output-format controls.

```bash
python -m experiments.synthetic.plot \
  --models mfvi fbnn vip tfsvi ftip gmvip

python -m experiments.synthetic.plot \
  --models all --iterations 2000 --device cuda
```

Outputs default to `results/synthetic_plot`. The runner saves method results,
prediction grids, and rendered figures for the selected model set.

## Label-free prior-fidelity diagnostic

`experiments.synthetic.prior_fidelity` compares the original one-dimensional
BNN prior with the untrained VIP and empirical GMVIP surrogate priors. It uses
the synthetic experiment's frozen `1-10-10-1` tanh architecture with
unit-Gaussian weights and biases, but it does not load a dataset, use labels,
or optimize model parameters.

```bash
python -m experiments.synthetic.prior_fidelity

python -m experiments.synthetic.prior_fidelity \
  --smoke \
  --output-dir results/synthetic_prior_fidelity_smoke
```

The full run uses five seeds, 2,048 samples per distribution, 301 inputs on
`[-5, 5]`, and 512 random projections. It compares the published defaults
(VIP `S=20`; GMVIP `M=256`, `B=1024`), performs a matched coefficient-dimension
sweep, and varies the GMVIP operator-bank size at `M=256`. The core metrics use
all 2,048 samples; the pairwise energy and MMD robustness checks use a
deterministic 512-sample prefix to keep their quadratic cost bounded.

The output directory contains:

- `metrics.csv` with seed-level sliced Wasserstein, marginal Wasserstein,
  moment-error, energy-distance, and MMD diagnostics;
- `summary.csv` with means and standard errors across seeds;
- `pointwise_w1.csv` and `default_samples_and_profiles.npz` for the
  one-dimensional fidelity profiles and displayed sample paths;
- `run_config.json` with the architecture, effective smoke/full settings, and
  seed policy;
- PNG and PDF figures for the prior samples, pointwise distance, matched
  dimension sweep, moment errors, bank-size sensitivity, and robustness
  metrics.

An independent BNN sample fits the pointwise standardization. Separate samples
form the reference and true-prior split baseline, so that the latter provides
the finite-sample noise floor for every reported distance.
