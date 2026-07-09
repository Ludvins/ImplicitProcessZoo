# Simulator-Prior Forecasting Experiment

This package implements the randomly forced damped-oscillator forecasting
benchmark for GM-VIP, VIP, and neural variational baselines.

Generate targets:

```bash
python -m experiments.simulator_forecasting.generate
```

Run the smoke preset:

```bash
python -m experiments.simulator_forecasting.run \
  --preset simulator_forecasting_smoke \
  --method all \
  --seed 0 \
  --skip-plots \
  --disable-tqdm
```

Run a paper-style method:

```bash
python -m experiments.simulator_forecasting.run \
  --preset simulator_forecasting_paper \
  --method gmvip \
  --seed 0
```

Outputs are written under:

```text
results/simprior/simulator_forecasting/<method>/seed_<seed>/
```

The main metric file is `metrics_per_target_region.csv`, with one row per
target, training-set size, and horizon region.

## Reproduce the T_obs=15 VIP/FTIP/GMVIP figure

From a fresh checkout with no saved simulator-forecasting artifacts, run the
three plotted methods with the figure preset:

```bash
python -m experiments.simulator_forecasting.run \
  --preset simulator_forecasting_tobs15_vip_ftip_gmvip_figure \
  --method vip,ftip,gmvip \
  --seed 0 \
  --output-dir results/simulator_forecasting_tobs15_figure \
  --skip-plots \
  --disable-tqdm
```

The preset fixes the figure settings: `T_obs=15`, `n_train=64`, one target,
a 1024-sample simulator prior bank, 3000 training steps, 1000 posterior
evaluation samples, and regions `[0, 15]`, `(15, 20]`, and `(20, 30]`. FTIP
trains a VIP source model for 3000 steps, warm-starts the flow, and then
fine-tunes for 3000 steps.

Generate the trajectory figure for the intended target:

```bash
python -m experiments.simulator_forecasting.plot \
  --results-root results/simulator_forecasting_tobs15_figure/simulator_forecasting \
  --methods vip,ftip,gmvip \
  --target-id 0 \
  --n-train 64 \
  --out results/simulator_forecasting_tobs15_figure/simulator_forecasting/figures/posterior_grid_ntrain64_vip_ftip_gmvip
```

This writes:

```text
results/simulator_forecasting_tobs15_figure/simulator_forecasting/figures/posterior_grid_ntrain64_vip_ftip_gmvip.png
results/simulator_forecasting_tobs15_figure/simulator_forecasting/figures/posterior_grid_ntrain64_vip_ftip_gmvip.pdf
```

The plotted target is `target_id=0`. The runner auto-generates the simulator
target bank under `data/simprior/simulator_forecasting_tobs15` if it is not
already present.

## Reproduce the broader T_obs=15 20-target result

These commands reproduce the 20-target run used for the ordered VIP, FTIP, and
GMVIP trajectory figure. The built-in preset fixes `T_obs=15`, `n_train=64`,
20 targets, a 1024-sample simulator prior bank, 3000 training steps, 1000
posterior evaluation samples, and the three regions `[0, 15]`, `(15, 20]`, and
`(20, 30]`. FTIP trains a VIP source model for 3000 steps, warm-starts the
flow, and then fine-tunes for 3000 steps.

```bash
python -m experiments.simulator_forecasting.run \
  --preset simulator_forecasting_tobs15_vip_ftip_gmvip_20targets \
  --method vip,ftip,gmvip \
  --seed 0 \
  --output-dir results/simulator_forecasting_tobs15_20targets \
  --skip-plots \
  --disable-tqdm

python -m experiments.simulator_forecasting.compare \
  --results-root results/simulator_forecasting_tobs15_20targets/simulator_forecasting

python -m experiments.simulator_forecasting.plot \
  --results-root results/simulator_forecasting_tobs15_20targets/simulator_forecasting \
  --methods vip,ftip,gmvip \
  --target-id 0 \
  --n-train 64 \
  --out results/simulator_forecasting_tobs15_20targets/simulator_forecasting/figures/forecast
```

The final command uses defaults aligned with the run commands above and writes
both:

```text
results/simulator_forecasting_tobs15_20targets/simulator_forecasting/figures/forecast.png
results/simulator_forecasting_tobs15_20targets/simulator_forecasting/figures/forecast.pdf
```

The plotting command can also select different methods and targets:

```bash
python -m experiments.simulator_forecasting.plot \
  --methods vip,ftip,gmvip \
  --target-ids 0,3-5 \
  --n-train 64 \
  --seed 0 \
  --out-dir results/simulator_forecasting_tobs15_20targets/simulator_forecasting/figures
```
