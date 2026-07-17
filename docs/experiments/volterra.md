# Lotka--Volterra simulator prior

This package implements vector-output Lotka--Volterra simulator-prior
regression for GMVIP and related baselines. The input is normalized time and
the output is `[prey, predator]`. Presets live directly in `run.py`; no external
configuration directory is required.

## Generate targets

```bash
python -m experiments.volterra.generate \
  --out data/simprior/lotka_volterra \
  --n-prior 4096 \
  --n-targets 100 \
  --dt 0.05 \
  --t-max 30.0 \
  --seed 0
```

`--n-prior` records the reference Monte Carlo size for diagnostics. The prior
itself is sampled live from the ODE parameter distribution; no prior trajectory
bank is stored.

## Run presets

```bash
# Default paper-style preset.
python -m experiments.volterra.run --seed 0

# Fast one-target verification.
python -m experiments.volterra.run \
  --preset lotka_volterra_smoke \
  --method gmvip_empirical --seed 0 --skip-plots
```

The default preset uses 20 targets, 80 noisy observations per target, a
512-function empirical operator bank, 96 inducing locations, 800 optimization
steps for GMVIP, and 256 evaluation samples. VIP and FTIP each use 20 sampled
prior basis functions. VIP is trained for 400 steps; FTIP is warm-started from
a 400-step VIP fit and then fine-tuned for 400 additional steps. SIP uses the
generic implementation with a simulator adapter that draws fresh ODE prior
latents by default.

Supported methods include the training-free `analog_prior` (raw simulator
prior), `gmvip_surrogate_prior` (GMVIP with its coefficient law fixed to
standard normal), and `empirical_gp`, together with `map`, `mfvi`, `vip`,
`ftip`, `sip`, `gmvip_empirical`, `gmvip_rbf`, and `oracle_prior_bank`. Use
`--method all` to run the complete supported set.

## Comparison plots

```bash
python -m experiments.volterra.compare \
  --results-root results/simprior/lotka_volterra \
  --seed 0 --target-ids 0,8 --metric rmse --n-win-loss 3
```

## Reproduce the ordered 20-target figure

```bash
python -m experiments.volterra.run \
  --method vip --seed 0 --target-start 0 --target-stop 20 \
  --output-dir results/volterra_coeff_ablation/s20 \
  --skip-plots --disable-tqdm

python -m experiments.volterra.run \
  --method ftip --seed 0 --target-start 0 --target-stop 20 \
  --output-dir results/volterra_coeff_ablation/s20 \
  --skip-plots --disable-tqdm

python -m experiments.volterra.run \
  --method gmvip_empirical --seed 0 --target-start 0 --target-stop 20 \
  --output-dir results/simprior_joint_output_z96 \
  --skip-plots --disable-tqdm

python -m experiments.volterra.run \
  --method gmvip_surrogate_prior --seed 0 --target-start 0 --target-stop 20 \
  --output-dir results/simprior_joint_output_z96 \
  --skip-plots --disable-tqdm

python -m experiments.volterra.plot
```

The FTIP run trains a 400-step VIP source, warm-starts the flow, and performs
400 additional fine-tuning steps. Empirical GMVIP uses a 512-sample operator
bank, 96 inducing locations, a joint 192-dimensional output covariance, and
800 steps. The default plot reads the (S=20) VIP and FTIP runs above, compares
them with the raw prior predictive, GMVIP surrogate prior, and trained GMVIP on
target 9, and writes matching PNG/PDF files. Empirical GP remains available as
an optional diagnostic but is not in the default paper comparison.

Select other methods, roots, or targets with:

```bash
python -m experiments.volterra.plot \
  --methods vip,ftip,gmvip_empirical --target-ids 5,9,12

python -m experiments.volterra.plot \
  --results-root results/my_run/lotka_volterra \
  --methods vip,gmvip_empirical --target-ids all \
  --out-dir results/my_run/lotka_volterra/figures

python -m experiments.volterra.plot \
  --methods vip,ftip,gmvip_empirical \
  --method-root ftip=results/other_ftip/lotka_volterra \
  --target-ids 9
```

## Package layout and outputs

- `generate.py` integrates and validates target trajectories.
- `run.py` defines presets, model factories, training, evaluation, and saves.
- `compare.py` selects metric wins/losses and creates shared-axis comparisons.
- `plot.py` builds compact trajectory grids; `plots.py` provides primitives.
- `interfaces.py` contains shared dataclasses/protocols.
- `metrics.py` contains probabilistic and simulator residual diagnostics.
- `datasets/lotka_volterra.py` creates train/validation/test regions.
- `priors/lotka_volterra_prior.py` integrates live parameter draws.

Generated targets default to `data/simprior/lotka_volterra/`. Each run writes
under `results/simprior/lotka_volterra/<method>/seed_<seed>/`, including
`metrics.json`, `metrics_per_target.csv`, `runtime.json`, compressed prediction
arrays, and optional figures. Shared-axis comparisons go under `shared_axes/`.

Paper reporting uses 90% interval coverage for calibration. All reported
metrics are evaluated only on the designated test partition (`t > 20`), while
`15 < t <= 20` remains validation-only. Dynamics metrics include absolute prey
and predator first-local-peak-time errors, mean absolute oscillation-period
error across the two species, absolute prey-to-predator phase-lag error, and
the fraction of posterior sample values below zero. Timing metrics are in the
time units of the simulation grid and use the posterior mean; positivity uses
all posterior samples on the test partition.

The default empirical GMVIP is joint-output. Its empirical inducing covariance
and Gaussian coefficient posterior each operate on the flattened prey--predator
array, so 96 temporal inducing locations produce one full covariance over 192
inducing variables. Set `gmvip.joint_output_covariance: false` to recover the
older output-wise block-diagonal parameterization.

Three training-free controls are available. `analog_prior` (shown as *Prior
predictive*) draws unconditional trajectories directly from the ODE parameter
prior. `gmvip_surrogate_prior` uses the same joint-output empirical GMVIP
operator as the trained method but fixes
\(q(a)=p(a)=\mathcal N(0,I)\), so it isolates the effect of the surrogate
construction without posterior adaptation. `empirical_gp` is an optional
diagnostic that estimates a joint prey--predator mean and covariance from 512
simulator trajectories and conditions the resulting finite-grid Gaussian
process analytically on the noisy observations. Its sampler uses an equivalent
low-rank Matheron update rather than factorizing the full 1202-dimensional grid
covariance.
