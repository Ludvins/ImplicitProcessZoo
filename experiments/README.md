# Experiments

This package contains the runnable benchmark, reporting, and plotting modules.
Reusable inference implementations live in `implicit_process_zoo/`.

The canonical experiment index, commands, outputs, and reproducibility guidance
are in the [online documentation](https://ludvins.github.io/ImplicitProcessZoo/experiments/).

Run any module from the repository root with `python -m`, for example:

```bash
python -m experiments.uci.benchmark --help
python -m experiments.volterra.benchmark --help
python -m experiments.simulator_forecasting.benchmark --help
python -m experiments.simulator_forecasting.plot --help
python -m experiments.eld_forecasting.benchmark --help
python -m experiments.eld_forecasting.plot --help
```

The standard Lotka--Volterra run learns separate prey and predator observation
noise scales for trainable methods:

```bash
python -m experiments.volterra.benchmark \
  --methods analog_prior,gmvip_surrogate_prior,vip,ftip,gmvip_empirical \
  --vip-basis-size 20 --seed 0 --target-ids 0:20
```

Use `--no-learn-observation-noise` only for the fixed-noise sensitivity.

The standard electricity run uses GMVIP with \(M=96\), learns scalar
observation noise for VIP, FTIP, and GMVIP, and saves 1,024 predictive
samples. Run this command for seeds 0, 1, and 2:

```bash
python -m experiments.eld_forecasting.benchmark \
  --methods analog,vip,ftip,empirical_gaussian,gmvip_empirical \
  --vip-basis-size 20 --seed 0 --target-ids 0:25
```

The standard oscillator run uses a shared \(S=256\) VIP/FTIP basis, an exact
VIP warm start for FTIP, learned scalar noise, and 1,024 predictive samples:

```bash
python -m experiments.simulator_forecasting.benchmark \
  --methods vip,ftip,gmvip \
  --vip-basis-size 256 --seed 0 --target-ids 0:20 \
  --regenerate-targets
```

After the runs complete, reproduce the reported tables and figures with:

```bash
python -m experiments.eld_forecasting.plot
python -m experiments.simulator_forecasting.plot
```

Generated data, results, checkpoints, plots, logs, and W&B files are runtime
artifacts and do not belong inside this package.
