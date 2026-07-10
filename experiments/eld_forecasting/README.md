# Electricity Load Diagram Forecasting

This experiment evaluates empirical function priors on the UCI
ElectricityLoadDiagrams20112014 (ELD) data. The repository contains one
canonical paper protocol: the original 24-hour-context, 24-hour-forecast
experiment with corrected index-based metrics and explicit reporting fields.

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
the archive hash, and extracts only the expected member. UCI records 96 values
per day. On the 23-hour March clock-change day, values from 01:00 to 02:00 are
zero; on the 25-hour October clock-change day, that interval aggregates both
hours. Preparation preserves this documented convention.

## Canonical paper protocol

The `eld_paper` preset exactly fixes the original experiment:

- Run seeds: `0`, `1`, and `2`.
- Held-out targets: 25 deterministic 2014 windows per seed, frozen in
  `paper_targets.csv`.
- Window: 48 hours at 15-minute resolution (192 points).
- Observed context: first 24 hours (96 points).
- Test forecast: next 24 hours, the half-open index interval `[96, 192)`.
- Historical prior: 2,048 windows from 2011-2013 selected by
  `calendar_prefix_nn`.
- Methods: analog, VIP, FTIP, and empirical GMVIP.
- VIP/FTIP coefficients: 20.
- GMVIP: 192 inducing locations, empirical operator, beta 1.
- Optimization: 500 steps, 8 training Monte Carlo samples, and 256 evaluation
  samples.
- Primary report: median and interquartile range across all 75 seed-target
  windows.

The target manifest is validated before a run starts. A loader change that
would remap a target ID to a different client or date fails instead of silently
changing the experiment. In particular, seed-0 target 18 is client `MT_353`
starting `2014-09-23 00:00:00`.

Metrics use integer half-open intervals. Output rows distinguish `run_seed`
from `target_seed`, record exact region start/stop indices and included times,
and include auditable prior-selection and stress diagnostics. Stress ranks are
computed against the complete 25-target seed set, so sharding cannot change a
target's label.

## Exact reproduction

Run the three canonical seeds from the repository root. The preset already
contains the canonical method list and hyperparameters:

```bash
python -m experiments.eld_forecasting.run \
  --preset eld_paper --seed 0 \
  --output-dir results/eld_forecasting_paper --disable-tqdm
python -m experiments.eld_forecasting.run \
  --preset eld_paper --seed 1 \
  --output-dir results/eld_forecasting_paper --disable-tqdm
python -m experiments.eld_forecasting.run \
  --preset eld_paper --seed 2 \
  --output-dir results/eld_forecasting_paper --disable-tqdm
```

Build the median/IQR report:

```bash
python -m experiments.eld_forecasting.compare \
  --results-root results/eld_forecasting_paper \
  --output results/eld_forecasting_paper/summary_median_iqr.csv
```

Reproduce the VIP/FTIP/GMVIP seed-0 target-18 figure as PNG and PDF:

```bash
python -m experiments.eld_forecasting.plot_predictions \
  --results-root results/eld_forecasting_paper \
  --output-dir results/eld_forecasting_paper/figures_paper_grid \
  --seed 0 --target-ids 18 \
  --methods vip,ftip,gmvip_empirical \
  --layout method_grid --formats png,pdf
```

The generated files are
`seed_0_target_18_vip_ftip_gmvip_empirical_grid.png` and the corresponding PDF.
The checked-in `regeneration.json` records the environment, artifact counts,
and reproducibility hashes from the completed canonical rerun. Large generated
artifacts remain gitignored.

## Sharded execution

Long runs may be split with `--target-start` and `--target-stop`. Each shard
still validates the full frozen target set and uses globally computed stress
ranks. Merge disjoint shards for one method and seed with:

```bash
python -m experiments.eld_forecasting.merge_shards \
  --shard-root results/eld_shards/000_010 \
  --shard-root results/eld_shards/010_020 \
  --shard-root results/eld_shards/020_025 \
  --output-root results/eld_forecasting_paper \
  --method vip --seed 0
```

The merger rejects differing configs, duplicate targets, missing predictions,
and mismatched metric/runtime target sets. `--resume-artifacts` may continue an
interrupted unsharded run only when its stored configuration matches exactly.

## Output schema

Each `<root>/<method>/seed_<seed>/` directory contains `config.yaml`,
`metrics.json`, `runtime.json`, `metrics_per_target_region.csv`, and compressed
per-target prediction files. Metric rows include:

- method, run seed, target seed, target/client identity, and start time;
- `last_observed_hour`, `forecast_start_hour`, and half-open region indices;
- first and last timestamps actually included in each region;
- RMSE, NLL, CRPS, CQM, interval coverage/width, and peak errors;
- training steps/loss/runtime, normalization fields, stress diagnostics, and
  prior-selection diagnostics.

There is no alternate methodology branch or result-version gate in the code.
