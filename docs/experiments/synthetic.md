# Synthetic plots

`experiments.synthetic.plot` trains scalar-regression methods on the fixed
synthetic dataset and writes publication-style predictive plots.

The dataset is always `synthetic`. The runner reuses model/training flags
from the UCI benchmark and adds plot ordering, panels, density bands,
predictive intervals, and output-format controls.

```bash
python -m experiments.synthetic.plot \
  --models mfvi fbnn vip tfsvi ftip gmvip

python -m experiments.synthetic.plot \
  --models all --iterations 2000 --device cuda
```

## HMC reference

The synthetic runner also provides an opt-in weight-space HMC reference. Install
the pinned Hamiltorch backend and compare it with GMVIP using:

```bash
python -m pip install -e ".[experiments,hmc]"

python -m experiments.synthetic.plot \
  --models gmvip hmc
```

Plain `--models all` retains the eight standard methods. Use `--models all hmc`
to add the more expensive HMC panel explicitly.

HMC follows the transition and prediction settings in BayesiPy's
`Synthetic_1D_regression.ipynb`: one float64 Hamiltorch chain on CUDA, 1,000
draws with `burn=-1`, step size \(5\times10^{-4}\), 500 leapfrog steps per
draw, and a fixed diagonal inverse mass of \(0.1\). The
\(1\)-\(10\)-\(10\)-\(1\) BNN
weights and biases have independent standard-normal priors. As in the other
synthetic runs, HMC uses raw inputs and normalized targets. The normalized
observation-noise scale is sampled fully Bayesianly through
\(\log\sigma_y\sim\mathcal N(-2.5,1)\); the variational methods instead optimize
their noise parameters.

The existing synthetic BNN architecture, posterior, and MAP-based
initialization are retained; only the notebook's HMC transition settings and
posterior-prediction construction are copied.

The HMC panel uses the same representation as GMVIP: every posterior function
draw is plotted with its corresponding \(\pm2\sigma_y\) observation-noise band.
The runner uses all 1,000 draws and saves the raw
posterior, full-grid mixture components, chain/draw indices, configuration, and
diagnostics in `hmc_posterior_samples.npz` and `hmc_summary.json`.
On CUDA, the exact autograd calculation is captured in a CUDA graph to remove
per-leapfrog launch overhead; the HMC trajectory and Metropolis decision are
unchanged.

The notebook protocol has only one chain, so split-\(\hat R\) and cross-chain
ESS cannot be computed. Acceptance and energy-divergence information are saved
for auditing, but the HMC panel is presented as a visual reference rather than
a convergence-certified posterior.

Outputs default to `results`. The runner saves method results,
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
