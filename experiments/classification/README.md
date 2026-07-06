# Classification

`experiments.classification.benchmark` runs image classification experiments on
FashionMNIST and CIFAR10 using the shared method suite.

## Datasets And Models

Datasets:

```text
FashionMNIST, CIFAR10
```

Models:

```text
map, mfvi, fbnn, tfsvi, vip, ftip, gmvip, sip
```

The default backbone is LeNet. `--backbone resnet18` is available for CIFAR10.
Bayesian classifier heads are used by default; `--full_bayes_cnn` also makes
convolutional layers Bayesian where supported.

## Commands

```bash
python -m experiments.classification.benchmark --dataset FashionMNIST --model vip
python -m experiments.classification.benchmark --dataset CIFAR10 --model all
```

For smoke checks, use `--limit_train`, `--limit_test`, small sample counts, and
`--no_save_checkpoint`. Results default to `results/classification` and include
per-run JSON plus comparison summaries when multiple jobs are expanded.
