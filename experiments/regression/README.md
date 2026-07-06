# Large Regression

`experiments.regression.benchmark` reuses the UCI training and evaluation code
for the larger scalar regression tasks used in Variational-LLA-style
experiments.

## Datasets

Supported datasets are:

```text
year, airline, taxi
```

Default training budgets are 60,000 iterations for `year` and `airline`, and
120,000 for `taxi`. Default hidden dimensions are `[50, 50]` for `year` and
`[100, 100]` for `airline` and `taxi`, unless `--hidden_dims` is supplied.

`year` downloads `YearPredictionMSD.txt` through the dataset loader when needed.
`airline` expects `data/airline.csv`. `taxi` uses `data/taxi.csv` when present
or can build it from the NYC yellow taxi parquet source when `pyarrow` is
installed.

## Commands

```bash
python -m experiments.regression.benchmark --model gmvip --dataset year
python -m experiments.regression.benchmark --model all --dataset all --device cuda
```

Results default to `results/regression` and follow the same JSON, comparison,
W&B, and checkpoint conventions as `experiments.uci.benchmark`.
