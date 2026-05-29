
FTIP Benchmarks
===============

This repository contains the code used to run the benchmark experiments for
FTIP, the baseline methods, and an added AP-FSVI prototype.

Python version: 3.10.11

Benchmark entrypoints:

- `python -m scripts.uci_benchmark --model ftip --dataset boston`
- `python -m scripts.synthetic_benchmark --models ftip --datasets bimodal`
- `python -m scripts.synthetic_benchmark --models ap_fsvi --datasets bimodal`
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
`https://wandb.ai/ludvins/apfsvi`; override the destination with
`--wandb_entity` or `--wandb_project`, group related runs with `--wandb_group`,
and use `--wandb_mode offline` when running without a network session. Run
names are formatted for scanning, e.g.
`UCI | Boston | AP-FSVI | Stein | Beta=1 | seed 42`; pass `--wandb_name`
to override a run name manually.

The training stream logs the total loss plus its decomposition:
`train/data_fit`, `train/regularizer`, and `train/reconstructed_loss`.
Method-specific terms are also logged when available, including `train/kl`,
`train/prior_regularizer`, `train/ftip_base_kl`, `train/ftip_flow_ldj`,
`train/ap_fsvi_discrepancy`, and `train/ap_fsvi_beta`.

Examples:

- `python -m scripts.uci_benchmark --model ap_fsvi --dataset boston --wandb`
- `python -m scripts.synthetic_benchmark --models all --datasets bimodal --wandb --wandb_group synthetic-bimodal`
- `python -m scripts.ap_fsvi_uci_sweep --datasets boston energy concrete --wandb --wandb_group bayeslinear-convergence`

AP-FSVI is implemented under `src/ap_fsvi` as a regression MVP with a supplied
posterior `generative_function`, an RBF GP function prior, data/near-data/domain
measurement points, and a configurable function-space discrepancy. Available
AP-FSVI discrepancies are `mmd`, `energy`, `sliced_wasserstein`, `stein`, and
`sinkhorn`; `mmd` remains the default. The benchmark
scripts pass the same generator-family objects used by the other methods
(for example `BayesianNN` on UCI/synthetic regression) so the posterior
architecture is defined outside AP-FSVI. It is wired into the synthetic
benchmark as the `ap_fsvi` model so it can be compared directly against VIP,
FTIP, MFVI, FBNN, and TFSVI on the included synthetic datasets.
The regression benchmarks write per-run JSON files and also emit comparison
tables plus JSON/CSV comparison summaries when several models are run together.
