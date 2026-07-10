# Implicit Process Zoo

Implicit Process Zoo is a PyTorch research library for function-space
variational inference with implicit stochastic-process priors. It brings eight
inference approaches, shared training and prediction utilities, and
reproducible benchmark runners into one codebase.

[Install the project](getting-started/installation.md){ .md-button .md-button--primary }
[Choose a model](getting-started/model-selection.md){ .md-button }

## Included methods

| Method | Representation | Function-space ingredient |
| --- | --- | --- |
| [MAP](models/map.md) | Point estimate | Deterministic reference |
| [MFVI](models/mfvi.md) | Diagonal Gaussian over weights | Uncertainty propagated through the network |
| [FBNN](models/fbnn.md) | Bayesian neural network | Score-based functional KL |
| [TFSVI](models/tfsvi.md) | Gaussian over weights | Linearized functional KL |
| [VIP](models/vip.md) | Gaussian coefficients | Sampled prior-function basis |
| [FTIP](models/ftip.md) | Flow-transformed coefficients | Non-Gaussian sampled-function basis |
| [GMVIP](models/gmvip.md) | Inducing-point coefficients | Matheron update |
| [SIP](models/sip.md) | Implicit inducing distribution | Critic-estimated inducing KL |

## Start with a benchmark

```bash
python -m experiments.uci.benchmark --model gmvip --dataset concrete
```

The module-based command interface is stable across the experiment packages.
Use `--help` on any runner to see its complete, source-of-truth configuration.

## Research scope

The repository includes scalar and large regression, classification,
synthetic-data visualization, Lotka--Volterra and damped-oscillator
simulator-prior forecasting, and the canonical electricity-load forecasting
study. Generated datasets and result artifacts remain outside version control.

The software is released under the [MIT License](development/citation.md#license).
If it supports published work, use the repository's
[citation metadata](development/citation.md).
