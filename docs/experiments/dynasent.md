# DynaSent frozen-feature classification

`experiments.dynasent.benchmark` compares the eight classification-capable
methods with linear heads over frozen CLIP text embeddings.

```text
Dataset: DynaSent v1.1
Labels:  negative, neutral, positive
Models:  map, mfvi, fbnn, tfsvi, vip, ftip, gmvip, sip
Encoder: openai/clip-vit-base-patch32
```

## Data and embeddings

The runner downloads the official DynaSent v1.1 archive, validates unique text
IDs across the official files, and records archive, record, embedding, and split
hashes. Sentences are whitespace-normalized without changing their content,
tokenized to CLIP's 77-token limit, projected to 512 dimensions, and
L2-normalized. The encoder is frozen and removed before head training.

| Role | Source | Examples |
| --- | --- | ---: |
| Tuning train | Round-1 train | 80,488 |
| Model selection | Stratified half of Round-1 dev | 1,800 |
| Temperature calibration | Other half of Round-1 dev | 1,800 |
| Final head training | Tuning train plus selection | 82,288 |
| In-distribution test | Round-1 test | 3,600 |
| Shifted test | Round-2 Dynabench test | 720 |

The development split is partitioned deterministically with `--split-seed`.
Round-1 and Round-2 test examples are not used for fitting, model selection, or
temperature calibration.

## Model selection

All methods use the same batch size, cosine learning-rate schedule, maximum
epoch count, early-stopping policy, and 100-sample final predictive estimate.
Each method is tuned independently with the repository's frozen-feature grid.
Candidates are ranked by five-fold cross-fitted, temperature-scaled selection
NLL, with accuracy, parameter count, runtime, and candidate order as tie
breakers.

The final winner is retrained on 82,288 examples for its selected epoch count
with seeds 0, 1, and 2. A positive scalar temperature is fitted on the untouched
calibration set. FBNN's prior is fixed; VIP, FTIP, GMVIP, and SIP train their
linear priors; GMVIP and SIP train inducing locations in the 512-dimensional
embedding space.

## Commands

Install the experiment dependencies and Transformers:

```bash
python -m pip install -e ".[experiments]" transformers
```

Run smoke checks and the complete experiment:

```bash
python -m experiments.dynasent.benchmark --stage smoke
python -m experiments.dynasent.benchmark --stage all
```

Run selected stages or methods:

```bash
python -m experiments.dynasent.benchmark \
  --stage tune \
  --methods map vip gmvip

python -m experiments.dynasent.benchmark \
  --stage final \
  --methods map vip gmvip
```

The runner resumes by default and records individual OOM, NaN, and numerical
failures without dropping them silently. Use `--retry-failures` to retry failed
records and `--overwrite` to replace the state in an explicitly selected output
directory.

## Outputs

Results default to `results/dynasent/clip_multiclass/` and include embedding
metadata, checkpoints, the resumable JSON state, failure diagnostics, winner
tables, headline CSV/Markdown tables, and raw per-seed metrics.

The headline table contains calibrated accuracy, NLL, ECE, macro-F1, Brier
score, predictive entropy, and AURC for the in-distribution and shifted tests.
AURC is a selective-prediction uncertainty metric rather than a standard
DynaSent metric; lower values indicate that confidence ranks correct predictions
ahead of errors. Entropy AUROC is an additional diagnostic for separating the
Round-2 shift from Round 1.
