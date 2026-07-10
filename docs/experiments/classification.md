# Classification

`experiments.classification.benchmark` runs FashionMNIST and CIFAR10 image
classification using the shared method suite.

```text
Datasets: FashionMNIST, CIFAR10
Models:   map, mfvi, fbnn, tfsvi, vip, ftip, gmvip, sip
```

The default backbone is LeNet. `--backbone resnet18` is available for CIFAR10.
Bayesian classifier heads are used by default; `--full_bayes_cnn` also makes
convolutional layers Bayesian where supported.

```bash
python -m experiments.classification.benchmark \
  --dataset FashionMNIST --model vip

python -m experiments.classification.benchmark \
  --dataset CIFAR10 --model all
```

For smoke checks, use `--limit_train`, `--limit_test`, small posterior sample
counts, and `--no_save_checkpoint`. Results default to
`results/classification` and include per-run JSON plus comparison summaries
when multiple jobs are expanded.
