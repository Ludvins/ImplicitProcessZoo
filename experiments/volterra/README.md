# Lotka-Volterra Simulator-Prior Experiment

This package implements the Lotka-Volterra simulator-prior regression
experiment for GM-VIP and related baselines. The input is normalized time and
the output is the vector `[prey, predator]`.

The runner is self-contained. Standard settings live directly in `run.py` as
built-in presets, so no `configs/` folder is required for normal runs.

## Commands

Generate the full Lotka-Volterra target set:

```bash
python -m experiments.volterra.generate \
  --out data/simprior/lotka_volterra \
  --n-prior 4096 \
  --n-targets 100 \
  --dt 0.05 \
  --t-max 30.0 \
  --seed 0
```

`--n-prior` records the reference Monte Carlo size used for prior diagnostics;
the implicit-process prior itself is sampled live from the Lotka-Volterra ODE
parameter distribution and no prior trajectory bank is stored.

Run a method with the built-in paper preset:

```bash
python -m experiments.volterra.run --seed 0
```

The default `lotka_volterra` preset runs GM-VIP empirical with 256 prior
function samples for the empirical operator, 64 inducing points, 80 noisy
training observations per target, and 400 optimization steps. Running the same
preset with `--method vip` uses 512 VIP regression coefficients and the same
400-step budget. Running with `--method sip` uses the generic Sparse Implicit
Process implementation with a Lotka-Volterra adapter that draws fresh ODE prior
latents by default for SIP prior and critic calls.

Run the smoke preset:

```bash
python -m experiments.volterra.run \
  --preset lotka_volterra_smoke \
  --method gmvip_empirical \
  --seed 0 \
  --skip-plots
```

Build shared-axis comparison plots from saved predictions:

```bash
python -m experiments.volterra.compare \
  --results-root results/simprior/lotka_volterra \
  --seed 0 \
  --target-ids 0,8 \
  --metric rmse \
  --n-win-loss 3
```

Reproduce the ordered 20-target VIP/FTIP/GMVIP trajectory figure:

```bash
python -m experiments.volterra.run \
  --method vip \
  --seed 0 \
  --output-dir results/simprior_paper_ready_defaults \
  --skip-plots \
  --disable-tqdm

python -m experiments.volterra.run \
  --method ftip \
  --seed 0 \
  --target-start 0 \
  --target-stop 20 \
  --output-dir results/simprior_search_ordering/ftip_steps625_mc8_coeff128 \
  --skip-plots \
  --disable-tqdm

python -m experiments.volterra.run \
  --method gmvip_empirical \
  --seed 0 \
  --target-start 0 \
  --target-stop 20 \
  --output-dir results/simprior_search_ordering/gmvip_bank512_z96_beta1_steps800 \
  --skip-plots \
  --disable-tqdm

python -m experiments.volterra.plot
```

The FTIP command trains a VIP source model, warm-starts FTIP from it, and then
uses the built-in intermediate fine-tuning budget so the 20-target mean
summaries land between VIP and empirical GMVIP. The empirical GMVIP method uses
the built-in 512-sample operator bank, 96 inducing points, and 800 optimization
steps. The plot command defaults to the result roots above, displays target 9
without metric boxes, and writes matching `.png` and `.pdf` files.

The plot command also accepts alternate methods, method-specific result roots,
and target selections:

```bash
python -m experiments.volterra.plot \
  --methods vip,ftip,gmvip_empirical \
  --target-ids 5,9,12

python -m experiments.volterra.plot \
  --results-root results/my_run/lotka_volterra \
  --methods vip,gmvip_empirical \
  --target-ids all \
  --out-dir results/my_run/lotka_volterra/figures

python -m experiments.volterra.plot \
  --methods vip,ftip,gmvip_empirical \
  --method-root ftip=results/other_ftip/lotka_volterra \
  --target-ids 9
```

## Files

- `generate.py`: samples held-out target parameters, integrates target
  trajectories with `scipy.integrate.solve_ivp`, rejects invalid paths, and
  writes `target_paths.npz` plus `metadata.json`.
- `run.py`: main experiment runner. It defines the built-in presets, builds
  MAP/MFVI/VIP/FTIP/GM-VIP/oracle-bank methods, trains each target, evaluates
  metrics, saves predictions, and optionally writes per-target figures.
- `compare.py`: post-processing script for saved result directories. It creates
  shared-axis visual comparisons and selects GM-VIP empirical win/loss targets
  by a chosen metric.
- `plot.py`: builds compact no-legend trajectory grids with selected methods as
  columns and stacked prey/predator trajectory rows.
- `interfaces.py`: dataclass/protocol definitions shared by datasets, priors,
  and runners.
- `metrics.py`: probabilistic and simulator-prior metrics, including RMSE,
  CRPS, sample-based Gaussian NLL, coverage/width, nearest-prior MSE, and
  Lotka-Volterra residual diagnostics.
- `plots.py`: plotting utilities. Shared-axis Lotka-Volterra comparison plots
  are the standard visual format for method comparison.
- `datasets/lotka_volterra.py`: turns generated target trajectories into tasks.
  It chooses sparse noisy training observations in `t <= 15`, clean validation
  points in `15 < t <= 20`, clean test points in `t > 20`, and full plot grids.
- `priors/simulator_prior.py`: common simulator-prior interface.
- `priors/lotka_volterra_prior.py`: live Lotka-Volterra ODE prior. Each latent
  is `[alpha, beta, delta, gamma, x0, y0]`, sampled from the parameter prior and
  integrated on demand at requested times.

## Built-In Presets

`run.py` defines:

- `lotka_volterra`: full paper-style run, 20 target trajectories, 80 noisy
  training observations per target, GM-VIP empirical with a 256-function
  empirical prior bank and 64 inducing points, 400 optimization steps, and 256
  posterior samples for evaluation. VIP comparisons use 512 regression
  coefficients under the same training budget.
- `lotka_volterra_smoke`: tiny one-target run for fast verification.

Use `--method all` to run every supported method in the selected preset.

Supported methods:

- `map`
- `mfvi`
- `vip`
- `ftip`
- `sip`
- `gmvip_empirical`
- `gmvip_rbf`
- `oracle_prior_bank`

## Outputs

Artifact roots intentionally remain `data/simprior` and `results/simprior` so
existing generated datasets and result folders do not need to be moved.

Generated target data defaults to:

```text
data/simprior/lotka_volterra/
```

Each method writes to:

```text
results/simprior/lotka_volterra/<method>/seed_<seed>/
```

Important artifacts:

- `metrics.json`: aggregate summary.
- `metrics_per_target.csv`: target-level metrics.
- `runtime.json`: train-time and early-stopping information.
- `predictions/target_<id>.npz`: saved posterior samples and truth arrays.
- `figures/`: per-target trajectory, phase, and prior-vs-posterior plots when
  plotting is enabled.

Shared-axis comparison plots are written by `compare.py` under:

```text
results/simprior/lotka_volterra/shared_axes/
```
