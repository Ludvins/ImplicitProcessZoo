# Simulator-prior forecasting

This package implements a randomly forced damped-oscillator forecasting
benchmark for GMVIP, VIP, FTIP, and neural variational baselines.

## Basic commands

```bash
python -m experiments.simulator_forecasting.generate

python -m experiments.simulator_forecasting.run \
  --preset simulator_forecasting_smoke \
  --method all --seed 0 \
  --skip-plots --disable-tqdm

python -m experiments.simulator_forecasting.run \
  --preset simulator_forecasting_paper \
  --method gmvip --seed 0
```

Outputs are written under
`results/simprior/simulator_forecasting/<method>/seed_<seed>/`. The main table,
`metrics_per_target_region.csv`, has one row per target, training-set size, and
horizon region.

## Reproduce the T_obs=15 figure

From a checkout without saved simulator-forecasting artifacts, run:

```bash
python -m experiments.simulator_forecasting.run \
  --preset simulator_forecasting_tobs15_vip_ftip_gmvip_figure \
  --method vip,ftip,gmvip \
  --seed 0 \
  --output-dir results/simulator_forecasting_tobs15_figure \
  --skip-plots --disable-tqdm
```

The preset fixes `T_obs=15`, `n_train=64`, one target, a 1024-sample prior
bank, 3000 training steps, 1000 posterior evaluation samples, and regions
`[0,15]`, `(15,20]`, and `(20,30]`. FTIP trains a VIP source model for 3000
steps, warm-starts the flow, and fine-tunes for another 3000 steps.

```bash
python -m experiments.simulator_forecasting.plot \
  --results-root results/simulator_forecasting_tobs15_figure/simulator_forecasting \
  --methods vip,ftip,gmvip \
  --target-id 0 --n-train 64 \
  --out results/simulator_forecasting_tobs15_figure/simulator_forecasting/figures/posterior_grid_ntrain64_vip_ftip_gmvip
```

This writes matching `.png` and `.pdf` files. Target `0` is the intended
figure target. Missing target data are generated below
`data/simprior/simulator_forecasting_tobs15`.

## Reproduce the 20-target result

```bash
python -m experiments.simulator_forecasting.run \
  --preset simulator_forecasting_tobs15_vip_ftip_gmvip_20targets \
  --method vip,ftip,gmvip --seed 0 \
  --output-dir results/simulator_forecasting_tobs15_20targets \
  --skip-plots --disable-tqdm

python -m experiments.simulator_forecasting.compare \
  --results-root results/simulator_forecasting_tobs15_20targets/simulator_forecasting

python -m experiments.simulator_forecasting.plot \
  --results-root results/simulator_forecasting_tobs15_20targets/simulator_forecasting \
  --methods vip,ftip,gmvip --target-id 0 --n-train 64 \
  --out results/simulator_forecasting_tobs15_20targets/simulator_forecasting/figures/forecast
```

The last command writes `forecast.png` and `forecast.pdf`. Alternative targets
can be selected with ranges such as `--target-ids 0,3-5`, together with
`--seed`, `--n-train`, and `--out-dir`.
