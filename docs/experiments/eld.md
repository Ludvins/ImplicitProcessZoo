# Electricity load forecasting

This experiment evaluates empirical function priors on UCI's
ElectricityLoadDiagrams20112014 data. The repository contains one canonical
paper protocol: the original 24-hour-context, 24-hour-forecast experiment with
corrected index-based metrics and explicit reporting fields.

## Data source and preparation

Artur Trindade created the dataset, which the UCI Machine Learning Repository
distributes under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- Dataset: ElectricityLoadDiagrams20112014
- DOI: [`10.24432/C58C86`](https://doi.org/10.24432/C58C86)
- [Official UCI record](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
- Archive SHA-256: `F6C4D0E0DF12ECDB9EA008DD6EEF3518ADB52C559D04A9BAC2E1B81DCFC8D4E1`

```bash
python -m experiments.eld_forecasting.prepare \
  --download --root data/electricity_load_diagrams
```

The downloader uses HTTPS, retries and a timeout, writes atomically, verifies
the archive hash, and extracts only the expected member. UCI records 96 values
per day. On the 23-hour March clock-change day, values from 01:00 to 02:00 are
zero; on the 25-hour October day, that interval aggregates both hours.
Preparation preserves this documented convention.

## Canonical paper protocol

The `eld_paper` preset fixes the experiment:

- run seeds `0`, `1`, and `2`;
- 25 deterministic held-out 2014 windows per seed, frozen in
  `paper_targets.csv`;
- 48-hour windows at 15-minute resolution (192 points);
- the first 24 hours observed (96 points);
- the next 24 hours tested using `[96,192)`;
- 2,048 historical 2011--2013 windows selected by `calendar_prefix_nn`;
- analog, VIP, FTIP, and empirical GMVIP;
- 20 VIP/FTIP coefficients;
- 96 GMVIP inducing locations (one per observed context point), empirical
  operator, beta 1;
- 500 optimization steps, 8 training samples, 256 evaluation samples;
- central 80%, 90%, and 95% interval coverage and width; and
- median and IQR across all 75 seed-target windows as the primary report.

The frozen target manifest is validated before training. Any loader change that
maps a target ID to a different client/date fails. Seed-0 target 18 is client
`MT_353` beginning `2014-09-23 00:00:00`.

The optional training-free `gmvip_empirical_exact` method uses the observed
prefix as the inducing set, $Z=X_{\mathrm{obs}}$, derives the exact
full-covariance Gaussian posterior over its 96 whitened inducing coefficients,
and retains conditional residual paths from the historical trajectory bank.
It performs no optimization. Run it with:

```bash
for seed in 0 1 2; do
  python -m experiments.eld_forecasting.run \
    --preset eld_paper \
    --method gmvip_empirical_exact \
    --seed ${seed} \
    --output-dir results/eld_forecasting_exact_gmvip \
    --disable-tqdm
done

python -m experiments.eld_forecasting.compare \
  --results-root results/eld_forecasting_exact_gmvip \
  --output results/eld_forecasting_exact_gmvip/summary_median_iqr.csv
```

The exact coefficient posterior is

\[
S_a=\left(I+\sigma_y^{-2}L^\top L\right)^{-1},\qquad
m_a=\sigma_y^{-2}S_aL^\top(y_{\mathrm{obs}}-\mu),
\]

where $LL^\top$ is the empirical covariance at the observed locations.
The implementation uses all 2,048 selected paths to estimate these moments,
rather than resampling a second operator bank.

Metrics use integer half-open intervals. Rows distinguish `run_seed` from
`target_seed`, store exact region boundaries/timestamps, and include auditable
prior-selection and stress diagnostics. Stress ranks are computed against all
25 targets before sharding, so shards cannot change labels.

## Exact reproduction

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

python -m experiments.eld_forecasting.compare \
  --results-root results/eld_forecasting_paper \
  --output results/eld_forecasting_paper/summary_median_iqr.csv
```

Reproduce the VIP/FTIP/GMVIP seed-0 target-18 figure:

```bash
python -m experiments.eld_forecasting.plot_predictions \
  --results-root results/eld_forecasting_paper \
  --output-dir results/eld_forecasting_paper/figures_paper_grid \
  --seed 0 --target-ids 18 \
  --methods vip,ftip,gmvip_empirical \
  --layout method_grid --formats png,pdf
```

This writes `seed_0_target_18_vip_ftip_gmvip_empirical_grid.png` and its PDF.
`regeneration.json` records the environment, artifact counts, and hashes from
the completed canonical rerun. Large generated artifacts remain gitignored.

## Sharded execution

Split targets with `--target-start` and `--target-stop`. Every shard still
validates the complete target set and uses global stress ranks. Merge disjoint
shards for one method and seed with:

```bash
python -m experiments.eld_forecasting.merge_shards \
  --shard-root results/eld_shards/000_010 \
  --shard-root results/eld_shards/010_020 \
  --shard-root results/eld_shards/020_025 \
  --output-root results/eld_forecasting_paper \
  --method vip --seed 0
```

The merger rejects differing configs, duplicate targets, missing predictions,
and mismatched metric/runtime target sets. `--resume-artifacts` continues an
interrupted unsharded run only when stored configuration matches exactly.

## Output schema

Each `<root>/<method>/seed_<seed>/` contains `config.yaml`, `metrics.json`,
`runtime.json`, `metrics_per_target_region.csv`, and compressed per-target
predictions. Metric rows include:

- method, run/target seed, target/client identity, and start time;
- `last_observed_hour`, `forecast_start_hour`, half-open region indices, and
  the first/last included timestamp;
- RMSE, NLL, CRPS, CQM, interval coverage/width, and peak errors;
- training steps/loss/runtime and normalization fields; and
- stress and prior-selection diagnostics.

There is no alternate methodology branch or result-version gate in the code.
