# Quickstart

## Run an experiment

All experiment entry points use `python -m experiments...`:

```bash
python -m experiments.uci.benchmark \
  --model vip \
  --dataset boston \
  --iterations 30000 \
  --seed 0 \
  --device cpu
```

The runner normalizes the data, constructs the requested model, trains it,
evaluates probabilistic metrics, and writes structured artifacts below
`results/uci` by default.

The executable entry-point check used by the documentation is:

```py
--8<-- "docs/examples/benchmark_cli.py"
```

## Use the Python API

The shortest self-contained training example uses the deterministic MAP
reference and the same fit/prediction contracts as the probabilistic models:

```py
--8<-- "docs/examples/train_and_predict.py"
```

For the implicit-process methods, begin by constructing a coherent prior
function generator:

```py
--8<-- "docs/examples/prior_and_model.py"
```

See [training](../guides/training.md), [prediction](../guides/prediction.md),
and the [curated API](../api/index.md) for the supported library interface.
