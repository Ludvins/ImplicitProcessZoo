# Experiments

This directory contains experiment-specific entry points and helpers that sit on
top of the reusable models under `src/`.

## Layout

- `simprior/`: simulator-prior regression experiments. The current implemented
  milestone is Lotka-Volterra trajectory regression with vector outputs
  `[prey, predator]`.
- `__init__.py`: marks this directory as an importable Python package so scripts
  can be run with `python -m experiments.<module>`.

Generated data and result artifacts are written outside this package:

- `data/simprior/...`: generated simulator banks and prepared datasets.
- `results/simprior/...`: metrics, predictions, runtimes, and figures.

Those directories are intentionally runtime artifacts, not source files.
