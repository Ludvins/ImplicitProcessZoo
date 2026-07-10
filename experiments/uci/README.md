# UCI Regression

`experiments.uci.benchmark` is the scalar regression benchmark for the shared
method suite. It builds models from `implicit_process_zoo/`, loads datasets through
`implicit_process_zoo.utils.dataset`, trains one or more methods, and writes JSON metrics plus
optional checkpoints.

## Datasets

The UCI set is:

```text
boston, concrete, energy, kin8nm, naval, power, protein, winered, yacht
```

The same runner can also load diagnostic regression datasets exposed by the
dataset loader, including `gap`, `bimodal`, `skewed`, `heteroscedastic`,
`snelson`, and `variational_lla`.

The historical spellings `yatch` and `heterocedastic` are accepted as
deprecated CLI aliases.

## Commands

```bash
python -m experiments.uci.benchmark --model gmvip --dataset concrete
python -m experiments.uci.benchmark --model all --dataset boston
```

Common options include `--iterations`, `--batch_size`, `--lr`,
`--hidden_dims`, `--activation`, `--layer_model`, `--bb_alpha`, and `--device`.
Use `--help` for method-specific flags such as prior-learning, inducing-point,
flow, and evaluation-sample settings.

## Outputs

Results default to `results/uci`. Each run writes a method/dataset JSON result,
and multi-method or multi-dataset runs also write comparison JSON/CSV summaries.
Checkpoints are saved by default unless `--no_save_checkpoint` is passed.
