
GMVIP Benchmarks
================

This repository contains the code used to run the benchmark experiments for
GMVIP, FTIP, and baseline methods.

Python version: 3.10.11

Benchmark entrypoints:

- `python -m scripts.uci_benchmark --model ftip --dataset boston`
- `python -m scripts.synthetic_benchmark --models ftip --datasets bimodal`
- `python -m scripts.synthetic_benchmark --models all --datasets bimodal`
- `python -m scripts.uci_benchmark --model all --dataset boston`
- `python -m scripts.pedestrian_benchmark --model all`
- `python -m scripts.pedestrian_benchmark --model ftip`
- `python -m scripts.classification_benchmark --model ftip --dataset MNIST`
- `python -m scripts.binary_classification_benchmark --model ftip --dataset HIGGS`

Use `--help` on each entrypoint for the full set of command-line options.

Weights & Biases tracking is available for the regression benchmark entrypoints.
Add `--wandb` to log training loss, learning rate, periodic train/test metrics,
and final run summaries. By default runs go to
`https://wandb.ai/ludvins/gmvip`; override the destination with
`--wandb_entity` or `--wandb_project`, group related runs with `--wandb_group`,
and use `--wandb_mode offline` when running without a network session. Run
names are formatted for scanning, e.g.
`UCI | Boston | GMVIP | RBF | Gaussian | seed 42`; pass `--wandb_name`
to override a run name manually.

The training stream logs the total loss plus its decomposition:
`train/data_fit`, `train/regularizer`, and `train/reconstructed_loss`.
Method-specific terms are also logged when available, including `train/kl`,
`train/prior_regularizer`, `train/ftip_base_kl`, `train/ftip_flow_ldj`,
`train/gmvip_kl`, and `train/gmvip_beta`.

Examples:

- `python -m scripts.uci_benchmark --model gmvip --dataset boston --wandb`
- `python -m scripts.synthetic_benchmark --models all --datasets bimodal --wandb --wandb_group synthetic-bimodal`
The regression benchmarks write per-run JSON files and also emit comparison
tables plus JSON/CSV comparison summaries when several models are run together.
