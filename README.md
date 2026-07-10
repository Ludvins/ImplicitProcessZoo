# Implicit Process Zoo

[![CI](https://github.com/Ludvins/ImplicitProcessZoo/actions/workflows/ci.yml/badge.svg)](https://github.com/Ludvins/ImplicitProcessZoo/actions/workflows/ci.yml)
[![Documentation](https://github.com/Ludvins/ImplicitProcessZoo/actions/workflows/docs.yml/badge.svg)](https://ludvins.github.io/ImplicitProcessZoo/)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implicit Process Zoo is a PyTorch research library for function-space
variational inference with implicit stochastic-process priors. It provides a
shared model API and reproducible runners for regression, classification, and
simulator-prior forecasting experiments.

**[Read the documentation](https://ludvins.github.io/ImplicitProcessZoo/)**

## Installation

Python 3.10 through 3.12 is supported. Install the complete editable research
environment:

```bash
python -m pip install -r requirements.txt
```

For only the model library:

```bash
python -m pip install -e .
```

PyTorch wheels are platform-specific. Select the appropriate CPU or accelerator
build from the [official PyTorch installer](https://pytorch.org/get-started/locally/)
before installing the project. A reproducible Python 3.12 CPU environment is
also recorded in `requirements/lock-cpu-py312.txt`.

See the [installation guide](https://ludvins.github.io/ImplicitProcessZoo/getting-started/installation/)
for dependency groups and environment verification.

## Models

| CLI name | Method | Posterior/approximation |
| --- | --- | --- |
| `map` | MAP | Deterministic network and learned observation noise |
| `mfvi` | Mean-field VI | Diagonal Gaussian over network weights |
| `fbnn` | Functional BNN | Score-based functional KL |
| `tfsvi` | Tractable FSVI | Linearized Gaussian function distribution |
| `vip` | Variational Implicit Process | Gaussian sampled-basis coefficients |
| `ftip` | Flow-Transformed Implicit Process | Flow-transformed basis coefficients |
| `gmvip` | Generalized Matheron VIP | Inducing-point Matheron update |
| `sip` | Sparse Implicit Process | Implicit inducing posterior and KL critic |

The [model guide](https://ludvins.github.io/ImplicitProcessZoo/models/) contains
the theory, objectives, trade-offs, and source-backed API reference.

## Quickstart

Run experiment entry points with `python -m` from the repository root:

```bash
python -m experiments.uci.benchmark \
  --model gmvip \
  --dataset concrete \
  --iterations 30000 \
  --seed 0
```

Use `--help` for the complete current configuration:

```bash
python -m experiments.uci.benchmark --help
python -m experiments.classification.benchmark --help
python -m experiments.eld_forecasting.run --help
```

The public tensor prediction contract is:

```python
model.predict_f_samples(x, num_samples, seed=0)  # [S, N, D]
model.predict_y_samples(x, num_samples, seed=0)  # [S, N, D]
```

Detailed commands and output schemas are documented for
[all experiments](https://ludvins.github.io/ImplicitProcessZoo/experiments/),
including the canonical
[electricity-load forecasting protocol](https://ludvins.github.io/ImplicitProcessZoo/experiments/eld/).

## Experiment families

| Module | Scope |
| --- | --- |
| `experiments.uci.benchmark` | UCI scalar regression |
| `experiments.regression.benchmark` | Year, airline, and taxi regression |
| `experiments.classification.benchmark` | FashionMNIST and CIFAR10 |
| `experiments.synthetic.plot` | Variational-LLA synthetic plots |
| `experiments.volterra` | Lotka--Volterra simulator prior |
| `experiments.simulator_forecasting` | Damped-oscillator forecasting |
| `experiments.eld_forecasting` | Electricity-load forecasting |

## Repository layout

```text
implicit_process_zoo/  model, prior, flow, data, and shared utility code
experiments/           runnable benchmarks, reporting, and plots
docs/                  MkDocs source and executable examples
tests/                 unit, regression, smoke, and publish-readiness tests
```

Generated data, results, checkpoints, logs, plots, and W&B files are excluded
from version control.

Full training checkpoints include optimizer, scheduler, step, normalization,
arguments, and random-generator states. Seeded predictions preserve model RNG
state, and verified dataset downloads use atomic writes and SHA-256 checks.

## Development

```bash
ruff format --check .
ruff check .
python -m pytest -q --cov
mkdocs build --strict
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[documentation workflow](https://ludvins.github.io/ImplicitProcessZoo/development/documentation/).

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). Please also cite
the original method and dataset papers relevant to your work.

Copyright 2026 Luis A. Ortega. Released under the [MIT License](LICENSE).
