# Experiments

This package contains runnable experiment entrypoints and experiment-specific
helpers. Reusable model implementations live under `implicit_process_zoo/`; experiment packages
compose those models with datasets, command-line interfaces, reporting, plots,
and artifact conventions.

## Layout

- `benchmark_utils.py`: shared reporting helpers, W&B integration, comparison
  table builders, and training-diagnostic logging.
- `uci/`: UCI-style scalar regression benchmark for the shared method suite.
- `regression/`: large scalar regression benchmark for `year`, `airline`, and
  `taxi`.
- `classification/`: FashionMNIST and CIFAR10 image classification benchmark.
- `synthetic/`: Variational-LLA synthetic regression plotting experiment.
- `volterra/`: Lotka-Volterra simulator-prior regression experiment.
- `simulator_forecasting/`: damped-oscillator simulator-prior forecasting
  experiment.
- `eld_forecasting/`: ElectricityLoadDiagrams20112014 empirical-prior
  forecasting experiment.

## Commands

Run entrypoints with `python -m` from the repository root:

```bash
python -m experiments.uci.benchmark --help
python -m experiments.regression.benchmark --help
python -m experiments.classification.benchmark --help
python -m experiments.synthetic.plot --help
python -m experiments.volterra.generate --help
python -m experiments.volterra.run --help
python -m experiments.volterra.compare --help
python -m experiments.volterra.plot --help
python -m experiments.simulator_forecasting.run --help
python -m experiments.simulator_forecasting.plot --help
python -m experiments.eld_forecasting.run --help
python -m experiments.eld_forecasting.valbank --help
```

Generated data, model checkpoints, metrics, plots, logs, and W&B files are
runtime artifacts. They are written under locations such as `data/`, `results/`,
`outputs/`, `logs/`, and `wandb/`, not inside this package.
