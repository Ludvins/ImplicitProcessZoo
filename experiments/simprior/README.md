# Simulator-Prior Experiments

This package implements simulator-prior regression experiments for GM-VIP and
related baselines. The active milestone is Lotka-Volterra trajectory regression:
the input is normalized time and the output is the vector `[prey, predator]`.

The runner is self-contained. Standard settings live directly in
`run_simprior.py` as built-in presets, so no `configs/` folder is required for
normal runs.

## Commands

Generate the full Lotka-Volterra banks:

```bash
python -m experiments.simprior.generate_lotka_volterra \
  --out data/simprior/lotka_volterra \
  --n-prior 4096 \
  --n-targets 100 \
  --dt 0.05 \
  --t-max 30.0 \
  --seed 0
```

Run a method with the built-in paper preset:

```bash
python -m experiments.simprior.run_simprior --method gmvip_rbf --seed 0
```

Run the smoke preset:

```bash
python -m experiments.simprior.run_simprior \
  --preset lotka_volterra_smoke \
  --method gmvip_empirical \
  --seed 0 \
  --skip-plots
```

Build shared-axis comparison plots from saved predictions:

```bash
python -m experiments.simprior.compare_lotka_volterra \
  --results-root results/simprior/lotka_volterra \
  --seed 0 \
  --target-ids 0,8 \
  --metric rmse \
  --n-win-loss 3
```

An optional `--config path/to/override.yaml` still exists for local one-off
overrides. It is merged into the selected built-in preset and is not needed for
standard experiments.

## Files

- `generate_lotka_volterra.py`: samples Lotka-Volterra parameters, integrates
  trajectories with `scipy.integrate.solve_ivp`, rejects invalid paths, and
  writes `prior_paths.npz`, `target_paths.npz`, and `metadata.json`.
- `run_simprior.py`: main experiment runner. It contains the built-in
  Lotka-Volterra presets, builds MAP/MFVI/VIP/FTIP/GM-VIP/oracle-bank methods,
  trains each target, evaluates metrics, saves predictions, and optionally
  writes per-target figures.
- `compare_lotka_volterra.py`: post-processing script for saved result
  directories. It creates shared-axis visual comparisons and selects GM-VIP
  empirical win/loss targets by a chosen metric.
- `interfaces.py`: dataclass/protocol definitions shared by datasets, priors,
  and runners.
- `metrics.py`: probabilistic and simulator-prior metrics, including RMSE,
  CRPS, sample-based Gaussian NLL, coverage/width, nearest-prior MSE, and
  Lotka-Volterra residual diagnostics.
- `plots.py`: plotting utilities. Shared-axis Lotka-Volterra comparison plots
  are the standard visual format for method comparison.
- `datasets/lotka_volterra.py`: turns generated trajectory banks into target
  tasks. It chooses sparse noisy training observations in `t <= 15`, clean
  validation points in `15 < t <= 20`, clean test points in `t > 20`, and full
  plot grids.
- `priors/simulator_prior.py`: common simulator-prior interface.
- `priors/lotka_volterra_prior.py`: bank-backed Lotka-Volterra prior with
  coherent path IDs and linear interpolation over normalized time.

## Built-In Presets

`run_simprior.py` defines:

- `lotka_volterra`: full paper-style run, 20 target trajectories, 40 noisy
  training observations per target, prior bank size 512 for learned implicit
  methods, and 256 posterior samples for evaluation.
- `lotka_volterra_smoke`: tiny one-target run for fast verification.

Use `--method all` to run every supported method in the selected preset.

Supported methods:

- `map`
- `mfvi`
- `vip`
- `ftip`
- `gmvip_empirical`
- `gmvip_rbf`
- `oracle_prior_bank`

## Outputs

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

Shared-axis comparison plots are written by `compare_lotka_volterra.py` under:

```text
results/simprior/lotka_volterra/shared_axes/
```
