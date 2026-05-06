
FTIP Benchmarks
===============

This repository contains the code used to run the benchmark experiments for
FTIP and the baseline methods.

Python version: 3.10.11

Benchmark entrypoints:

- `python -m scripts.uci_benchmark --model ftip --dataset boston`
- `python -m scripts.synthetic_benchmark --models ftip --datasets bimodal`
- `python -m scripts.pedestrian_benchmark --model ftip`
- `python -m scripts.classification_benchmark --model ftip --dataset MNIST`
- `python -m scripts.binary_classification_benchmark --model ftip --dataset HIGGS`

Use `--help` on each entrypoint for the full set of command-line options.
