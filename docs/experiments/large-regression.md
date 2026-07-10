# Large regression

`experiments.regression.benchmark` reuses the UCI model, training, evaluation,
checkpoint, and reporting code for larger scalar tasks:

```text
year, airline, taxi
```

Default budgets are 60,000 iterations for `year` and `airline`, and 120,000
for `taxi`. Default hidden widths are `[50, 50]` for `year` and `[100, 100]`
for `airline` and `taxi`, unless `--hidden_dims` is supplied.

```bash
python -m experiments.regression.benchmark \
  --model gmvip \
  --dataset year \
  --iterations 30000 \
  --bb_alpha 0 \
  --batch_size 100 \
  --lr 0.001 \
  --hidden_dims 50 50 \
  --activation tanh \
  --layer_model BayesLinear \
  --device cuda

python -m experiments.regression.benchmark \
  --model all --dataset all --device cuda
```

`year` downloads `YearPredictionMSD.txt` through the verified loader when
needed. `airline` expects `data/airline.csv`. `taxi` uses `data/taxi.csv` when
present or can build it from the configured NYC yellow-taxi parquet source
when the optional parquet dependency is installed.

Results default to `results/regression` and follow the UCI JSON, comparison,
W&B, and checkpoint conventions.
