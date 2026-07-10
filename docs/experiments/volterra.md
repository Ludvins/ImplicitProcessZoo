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
256-function empirical operator bank, 64 inducing points, 400 optimization
steps, and 256 evaluation samples. VIP uses 512 coefficients under the same
step budget. SIP uses the generic implementation with a simulator adapter that
draws fresh ODE prior latents by default.

Supported methods are `map`, `mfvi`, `vip`, `ftip`, `sip`,
`gmvip_empirical`, `gmvip_rbf`, and `oracle_prior_bank`. Use `--method all` to
run the preset's complete supported set.

## Comparison plots

```bash
python -m experiments.volterra.compare \
  --results-root results/simprior/lotka_volterra \
  --seed 0 --target-ids 0,8 --metric rmse --n-win-loss 3
```

## Reproduce the ordered 20-target figure

```bash
python -m experiments.volterra.run \
  --method vip --seed 0 \
  --output-dir results/simprior_paper_ready_defaults \
  --skip-plots --disable-tqdm

python -m experiments.volterra.run \
  --method ftip --seed 0 --target-start 0 --target-stop 20 \
  --output-dir results/simprior_search_ordering/ftip_steps625_mc8_coeff128 \
  --skip-plots --disable-tqdm

python -m experiments.volterra.run \
  --method gmvip_empirical --seed 0 --target-start 0 --target-stop 20 \
  --output-dir results/simprior_search_ordering/gmvip_bank512_z96_beta1_steps800 \
  --skip-plots --disable-tqdm

python -m experiments.volterra.plot
```

The FTIP run trains a VIP source, warm-starts the flow, and uses the built-in
intermediate fine-tuning budget. Empirical GMVIP uses a 512-sample operator
bank, 96 inducing points, and 800 steps. The default plot shows target 9 and
writes matching PNG/PDF files.

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
