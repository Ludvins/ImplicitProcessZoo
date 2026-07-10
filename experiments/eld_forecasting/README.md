# Electricity Load Diagram Forecasting

This experiment evaluates empirical function priors on the UCI
ElectricityLoadDiagrams20112014 (ELD) data. All corrected outputs carry
`methodology_version: 2` and belong under new `*_v2` result roots. Existing
methodology-version 1 artifacts are historical records: do not overwrite them
or aggregate them with version 2.

## Data source and preparation

The dataset was created by Artur Trindade and is distributed by the UCI Machine
Learning Repository under CC BY 4.0:

- Dataset: ElectricityLoadDiagrams20112014
- DOI: `10.24432/C58C86`
- UCI record: <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>
- Archive SHA-256: `F6C4D0E0DF12ECDB9EA008DD6EEF3518ADB52C559D04A9BAC2E1B81DCFC8D4E1`

Prepare the verified archive with:

```bash
python -m experiments.eld_forecasting.prepare \
  --download \
  --root data/electricity_load_diagrams
```

The downloader uses HTTPS, retries and a timeout, writes atomically, verifies
the archive hash, and extracts only the expected member. Preparation writes
float32 values, nanosecond timestamps, client names, and source metadata below
`data/electricity_load_diagrams/processed/`.

UCI reports each value as client demand in kW at 15-minute intervals using
Portuguese time. Every recorded day contains exactly 96 measurements. On the
23-hour March clock-change day, values from 01:00 to 02:00 are zero; on the
25-hour October clock-change day, that interval aggregates both hours. The
experiment preserves this documented convention instead of inserting or
removing samples.

## Presets and leakage rules

`eld_smoke`, `eld_pilot`, `eld_validation`, and `eld_paper` are defined in
`experiments.eld_forecasting.run`. Windows and metric regions use half-open
integer index intervals:

- observed prefix: `[0, prefix_points)`
- forecast: `[prefix_points, window_points)`
- validation-bank training context: `[0, train_points)`
- validation: `[train_points, context_points)`
- final test: `[context_points, window_points)`

Same-day and next-day regions meet at index 96 and never overlap. Regions are
generated from each preset and validated to be nonempty, in bounds,
nonoverlapping, and complete for the intended partition.

By default, empirical-prior windows come from 2011–2013 and targets from 2014.
Target observations after the active prefix are never used to construct the
prior bank, normalize a prefix, rank stress, or choose a validation-bank rule.
Targets and target ranges are resolved before prior banks are built. One task
blueprint is then reused across methods so every method sees the same target,
normalization, prior candidates, and seed.

The validation-bank protocol uses 15 training hours, 5 validation hours, and a
28-hour final test horizon by default. Every candidate rule for one target uses
the same target seed. The chosen rule minimizes the configured validation
metric, then the model is rebuilt with the 20-hour context and evaluated only
on the final test. `selection_decisions.csv` records the candidate count, seed,
fallback tier, actual calendar/client constraints, scores, and selected rule.

Real-data stress labels are based only on the observed prefix. The two
dimensionless components are prefix coefficient of variation and standardized
prefix ramp. Their percentile ranks are averaged; the stored 80th-percentile
threshold determines the stress flag. Raw components, ranks, combined score,
and threshold are all written to result rows.

## Exact seed-0 reruns

All supported methods are `analog`, `seasonal_naive`, `vip`, `vip_512`, `ftip`,
and `gmvip_empirical` (`--method all`). Run these commands from the repository
root after preparing the data:

```bash
# Synthetic smoke, all methods
python -m experiments.eld_forecasting.run \
  --preset eld_smoke --synthetic-smoke --method all --seed 0 \
  --output-dir results/eld_forecasting_v2_smoke --disable-tqdm

# Real-data validation preset, all methods
python -m experiments.eld_forecasting.run \
  --preset eld_validation --method all --seed 0 \
  --output-dir results/eld_forecasting_v2_validation

# Validation-bank selection with documented defaults
python -m experiments.eld_forecasting.valbank \
  --seed 0 \
  --output-dir results/eld_forecasting_v2_valbank_context15_val5_test28

# Paper/test preset, all methods
python -m experiments.eld_forecasting.run \
  --preset eld_paper --method all --seed 0 \
  --output-dir results/eld_forecasting_v2_paper

# Aggregate each v2 root independently
python -m experiments.eld_forecasting.compare \
  --results-root results/eld_forecasting_v2_validation \
  --output results/eld_forecasting_v2_validation/comparison.csv
python -m experiments.eld_forecasting.compare \
  --results-root results/eld_forecasting_v2_paper \
  --output results/eld_forecasting_v2_paper/comparison.csv

# Prediction figures from a v2 root only
python -m experiments.eld_forecasting.plot_predictions \
  --results-root results/eld_forecasting_v2_paper \
  --output-dir results/eld_forecasting_v2_paper/figures \
  --seed 0 --formats png,pdf
```

`compare` and `plot_predictions` reject roots that lack methodology-version 2
metadata. They cannot silently combine legacy and corrected results.

The completed seed-0 regeneration environment, artifact counts, runtimes, and
SHA-256 checksums are recorded in `regeneration_v2_seed0.json`. Large generated
results remain gitignored and should be stored as release/research artifacts,
not committed to the source history.

Long runs may be split with `--target-start`/`--target-stop`. Write each shard
to its own root, then merge one method only after every shard succeeds:

```bash
python -m experiments.eld_forecasting.merge_shards \
  --shard-root results/eld_v2_shards/000_025 \
  --shard-root results/eld_v2_shards/025_050 \
  --shard-root results/eld_v2_shards/050_075 \
  --shard-root results/eld_v2_shards/075_100 \
  --output-root results/eld_forecasting_v2_validation \
  --method vip --seed 0
```

The merger rejects differing configs, non-v2 rows, duplicate targets, missing
predictions, and mismatched row/runtime target sets. It rebuilds summaries from
all target rows and records the source shard roots in `metrics.json`.

The base runner also flushes CSV, runtime, and summary JSON atomically after
every target. Pass `--resume-artifacts` with the same method/config/output root
to skip targets that have both metric rows and runtime entries. A changed config
is rejected. A prediction written immediately before an interruption but not
present in both indexes is regenerated instead of being trusted.

## Outputs

Each method writes `config.yaml`, `metrics.json`, `runtime.json`,
`metrics_per_target_region.csv`, and compressed per-target prediction files
below `<root>/<method>/seed_<seed>/`. Rows include:

- method, seed, target/client identity, start time, and `methodology_version`
- `last_observed_hour`, `forecast_start_hour`, and region name
- region start/stop indices plus first/last included timestamps
- RMSE, NLL, CRPS, CQM, and interval coverage/width
- normalization/noise fields, training steps/loss/runtime, and stress diagnostics
- prior candidate count, requested count, fallback tier, actual constraints,
  selection rule, and selection-protocol fields where applicable

The stop index is exclusive. The first/last time fields refer to samples that
are actually included in the region; there is no ambiguous boundary-inclusion
flag.

## Resource expectations

The synthetic smoke uses one short target and normally completes on a CPU in
minutes. Validation evaluates 100 targets; the paper preset evaluates 200
targets with up to 1,500 optimization steps and 512 evaluation samples. Running
all neural methods can take many CPU-hours and several gigabytes of artifacts;
a CUDA device substantially reduces training time. The validation-bank run
adds one candidate-selection pass per target and rule. Use `--target-start`,
`--target-stop`, or `--target-ids` to shard the base runner without changing
target identity or seeds, then retain the config and runtime manifests with the
artifacts.
