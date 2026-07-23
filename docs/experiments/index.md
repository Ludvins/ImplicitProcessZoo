# Experiments

Reusable inference code lives in `implicit_process_zoo/`. The `experiments/`
packages compose those models with datasets, command-line interfaces,
reporting, plots, and artifact conventions.

| Package | Task |
| --- | --- |
| `uci` | UCI-style scalar regression |
| `regression` | Year, airline, and taxi large regression |
| `classification` | FashionMNIST and CIFAR10 classification |
| `synthetic` | Variational-LLA scalar regression plots |
| `volterra` | Lotka--Volterra simulator-prior regression with learned per-output noise |
| `simulator_forecasting` | Damped-oscillator forecasting |
| `eld_forecasting` | Electricity-load empirical-prior forecasting |

Run every entry point from the repository root with `python -m`:

```bash
python -m experiments.uci.benchmark --help
python -m experiments.regression.benchmark --help
python -m experiments.classification.benchmark --help
python -m experiments.synthetic.plot --help
python -m experiments.volterra.benchmark --help
python -m experiments.volterra.plot --help
python -m experiments.simulator_forecasting.benchmark --help
python -m experiments.simulator_forecasting.plot --help
python -m experiments.eld_forecasting.benchmark --help
python -m experiments.eld_forecasting.plot --help
```

Generated data, checkpoints, metrics, plots, logs, and W&B files are runtime
artifacts written under gitignored locations such as `data/`, `results/`,
`outputs/`, `logs/`, and `wandb/`.
