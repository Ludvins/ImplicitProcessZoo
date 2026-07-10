# Synthetic plots

`experiments.synthetic.plot` trains scalar-regression methods on the fixed
Variational-LLA synthetic dataset and writes publication-style predictive
plots.

The dataset is always `variational_lla`. The runner reuses model/training flags
from the UCI benchmark and adds plot ordering, panels, density bands,
predictive intervals, and output-format controls.

```bash
python -m experiments.synthetic.plot \
  --models mfvi fbnn vip tfsvi ftip gmvip

python -m experiments.synthetic.plot \
  --models all --iterations 2000 --device cuda
```

Outputs default to `results/synthetic_plot`. The runner saves method results,
prediction grids, and rendered figures for the selected model set.
