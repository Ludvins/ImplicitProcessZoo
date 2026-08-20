# Classification

`benchmark.py` evaluates the shared model suite on FashionMNIST and CIFAR10.

See the [classification guide](https://ludvins.github.io/ImplicitProcessZoo/experiments/classification/)
for backbones, smoke settings, commands, and outputs.

```bash
python -m experiments.classification.benchmark --help
```

## Frozen CLIP on CIFAR-10

`clip_benchmark.py` compares MAP, MFVI, FBNN, TFSVI, VIP, FTIP, GMVIP, and SIP
using a single linear head over frozen, normalized CLIP ViT-B/32 image embeddings.
The default run independently tunes every method at training sizes 500, 1,000,
5,000, 10,000, and 45,000.

The CIFAR-10 training set is divided into nested class-balanced tuning and
selection sets. A fixed 5,000-image split is reserved for temperature
calibration, and the 10,000 test images are evaluated only after model selection.
Final results use seeds 0, 1, and 2 with 100 predictive samples.

Install the vision and experiment dependencies together with Transformers:

```bash
python -m pip install -e ".[experiments,vision]" transformers
```

Run a smoke test before starting the full benchmark:

```bash
python -m experiments.classification.clip_benchmark --stage smoke
python -m experiments.classification.clip_benchmark --stage all
```

Stages can also be run separately and restricted to selected methods or sizes:

```bash
python -m experiments.classification.clip_benchmark \
  --stage tune \
  --methods map gmvip \
  --sizes 500 1000 5000

python -m experiments.classification.clip_benchmark \
  --stage final \
  --methods map gmvip \
  --sizes 500 1000 5000
```

Embeddings, resumable checkpoints, diagnostic state, and result tables are
written to `results/classification/clip_cifar10/` by default. The headline table
contains calibrated accuracy, NLL, ECE, temperature, trainable parameter count,
training time, and peak GPU memory; raw metrics remain in the JSON diagnostics.

```bash
python -m experiments.classification.clip_benchmark --help
```
