# Synthetic Plot Runner

`experiments.synthetic.plot` trains selected scalar-regression methods on the
fixed Variational-LLA synthetic dataset and writes publication-style predictive
plots.

## Behavior

The dataset is always `variational_lla`. The runner forwards model and training
flags from `experiments.uci.benchmark`, then adds plotting controls such as
model ordering, figure panels, density bands, predictive intervals, and output
format.

## Commands

```bash
python -m experiments.synthetic.plot --models mfvi fbnn vip tfsvi ftip gmvip
python -m experiments.synthetic.plot --models all --iterations 2000 --device cuda
```

Outputs default to `results/synthetic_plot`. The runner saves method results,
prediction grids, and rendered figures for the selected model set.
