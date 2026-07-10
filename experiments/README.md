# Experiments

This package contains the runnable benchmark, reporting, and plotting modules.
Reusable inference implementations live in `implicit_process_zoo/`.

The canonical experiment index, commands, outputs, and reproducibility guidance
are in the [online documentation](https://ludvins.github.io/ImplicitProcessZoo/experiments/).

Run any module from the repository root with `python -m`, for example:

```bash
python -m experiments.uci.benchmark --help
python -m experiments.volterra.run --help
python -m experiments.simulator_forecasting.run --help
python -m experiments.eld_forecasting.run --help
```

Generated data, results, checkpoints, plots, logs, and W&B files are runtime
artifacts and do not belong inside this package.
