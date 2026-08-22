# DynaSent Classification

`benchmark.py` compares MAP, MFVI, FBNN, TFSVI, VIP, FTIP, GMVIP, and SIP
using a linear three-class head over frozen CLIP ViT-B/32 text embeddings.

```bash
python -m pip install -e ".[experiments]" transformers
python -m experiments.dynasent.benchmark --help
```

## Protocol

The runner downloads DynaSent v1.1 and retains the `negative`, `neutral`, and
`positive` examples. Each sentence is tokenized to CLIP's 77-token limit,
encoded with `openai/clip-vit-base-patch32`, projected to 512 dimensions, and
L2-normalized. CLIP is frozen and released before the classification heads are
trained.

| Role | Source | Examples |
| --- | --- | ---: |
| Tuning train | Round-1 train | 80,488 |
| Model selection | Stratified half of Round-1 dev | 1,800 |
| Temperature calibration | Other half of Round-1 dev | 1,800 |
| Final head training | Tuning train plus selection | 82,288 |
| In-distribution test | Round-1 test | 3,600 |
| Shifted test | Round-2 Dynabench test | 720 |

Every method is tuned independently with the frozen-feature candidate grid.
Candidates are selected by five-fold cross-fitted, temperature-scaled NLL.
The winner is retrained for its selected epoch count with seeds 0, 1, and 2.
FBNN uses a fixed prior; VIP, FTIP, GMVIP, and SIP train their linear priors;
GMVIP and SIP also train their inducing locations in the embedding space.

## Running

Run the one-epoch smoke checks before the full benchmark:

```bash
python -m experiments.dynasent.benchmark --stage smoke
python -m experiments.dynasent.benchmark --stage all
```

Stages and methods can be restricted independently:

```bash
python -m experiments.dynasent.benchmark \
  --stage tune \
  --methods map vip gmvip

python -m experiments.dynasent.benchmark \
  --stage final \
  --methods map vip gmvip
```

Runs resume by default. Use `--retry-failures` to retry failed configurations,
or `--overwrite` to start a new state in the selected output directory. For an
offline run, provide an existing archive and compatible embedding cache with
`--no-download --archive ... --embedding-cache ...`.

## Outputs and metrics

Artifacts default to `results/dynasent/clip_multiclass/`. The directory contains
the embedding cache and its signature, resumable checkpoints,
`benchmark_state.json`, `failures.json`, winner tables, headline tables, and a
raw per-seed CSV.

The headline table reports temperature-calibrated accuracy, NLL, ECE,
macro-F1, Brier score, predictive entropy, and AURC for both test sets, together
with temperature, parameter count, runtime, peak GPU memory, and completed seed
count. AURC is an uncertainty/selective-prediction metric, not a standard
DynaSent metric; lower AURC means confidence ranks correct predictions ahead of
errors more effectively. Entropy AUROC measures separation between Round 1 and
Round 2 and is reported as an additional shift-detection diagnostic.
