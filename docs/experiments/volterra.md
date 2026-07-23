# Lotka--Volterra benchmark

The experiment is intentionally contained in two modules: `benchmark.py`
generates targets, trains methods, and records metrics; `plot.py` renders the
validated result artifacts. The input is normalized time and the output is
`[prey, predator]`.

## Run the benchmark

```bash
python -m experiments.volterra.benchmark \
  --methods analog_prior,gmvip_surrogate_prior,vip,ftip,gmvip_empirical \
  --vip-basis-size 20 --seed 0 --target-ids 0:20
```

Missing targets are generated deterministically. Use `--regenerate-targets`
to replace them explicitly. Results are written below
`results/volterra/seed_0/S_20/<method>/`.

This is the standard protocol. Trainable methods learn one Gaussian
observation-noise scale for prey and one for predator, initialized at the
simulator noise. The training-free prior baselines continue to use the known
simulator noise.

## Additional methods

```bash
python -m experiments.volterra.benchmark \
  --methods empirical_gp,map,mfvi,sip,gmvip_rbf,oracle_prior_bank \
  --vip-basis-size 20 --seed 0 --target-ids 0:20
```

All eleven methods are selected explicitly as a comma-separated list. A
one-target verification uses the same interface with `--smoke --target-ids 0`.

## VIP and FTIP basis sizes

Run each value of \(S\) separately. VIP always receives 800 optimization
steps. FTIP uses a separate 400-step VIP warm start on the same basis followed
by 400 fine-tuning steps.

```bash
python -m experiments.volterra.benchmark \
  --methods vip,ftip --vip-basis-size 64 \
  --seed 0 --target-ids 0:20
```

Repeat with `--vip-basis-size 20`, `64`, `128`, and `256`.

## Fixed observation-noise sensitivity

To run the nonstandard fixed-noise sensitivity, disable noise learning and use
a separate output root:

```bash
python -m experiments.volterra.benchmark \
  --methods vip,ftip,gmvip_empirical \
  --vip-basis-size 20 --no-learn-observation-noise \
  --seed 0 --target-ids 0:20 \
  --output-root results/volterra_fixed_noise
```

In the standard run, the learned physical prey and predator noise scales are
saved in every
target-level metric row, prediction archive, checkpoint state, and protocol
manifest. Mixture NLL then uses the learned scale rather than the known fixed
simulator scale.

## Render figures and tables

```bash
python -m experiments.volterra.plot \
  --methods analog_prior,gmvip_surrogate_prior,vip,ftip,gmvip_empirical \
  --vip-basis-size 20 --vip-basis-sizes 20,64,128,256 \
  --target-ids 9 --aggregate-target-ids 0:20
```

The renderer accepts only complete manifests produced from the same target
dataset. It writes PNG/PDF figures and both LaTeX tables under
`outputs/volterra/`.
Use `--aggregate-target-ids 0:19` to save a target-19 leave-one-out table
without altering the underlying 20-target results.
Pass `--fixed-noise-results-root results/volterra_fixed_noise` to additionally
render the fixed-versus-learned noise sensitivity table.

## Package layout and outputs

- `benchmark.py` contains the complete experiment and CLI.
- `plot.py` validates results and renders artifacts.

Every method directory contains a protocol manifest, target-level metrics,
runtime details, checkpoints, and all 1,024 full-grid posterior samples.
Gaussian-mixture NLL is evaluated in physical units from those exact samples.
Means and sample standard errors are computed across the selected target
trajectories. The reported metrics are RMSE, Gaussian-mixture NLL, CRPS,
90% coverage, and the fitted Lotka--Volterra ODE residual.

Training uses \(t\le15\), the gap \(15<t\le20\) is deliberately unused, and all
reported metrics use only \(20<t\le30\). There is no validation split,
validation loss, early stopping, or checkpoint selection: every trained
method is evaluated at its exact final scheduled optimization step. GMVIP uses
512 operator-bank trajectories and 96 shared inducing times with a full joint
covariance over the 192 prey--predator inducing values.
