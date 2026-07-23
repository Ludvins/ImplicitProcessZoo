# Electricity-load forecasting

The canonical experiment has two entry points:

```bash
python -m experiments.eld_forecasting.benchmark \
  --methods analog,vip,ftip,empirical_gaussian,gmvip_empirical \
  --vip-basis-size 20 --seed 0 --target-ids 0:25

python -m experiments.eld_forecasting.plot
```

Run the benchmark once for each seed in \(\{0,1,2\}\), changing only
`--seed`. Each seed contains 25 frozen 2014 targets, with the first 96
quarter-hours observed and the following 96 held out. Historical prior
windows come from 2011--2013.

The empirical prior contains \(B=2048\) calendar-compatible nearest-neighbor
windows. GMVIP uses \(M=96\) inducing locations. VIP and FTIP use the value
supplied by `--vip-basis-size`; the canonical value is \(S=20\). They use the
same deterministic basis draw, but electricity FTIP is trained directly and
does not use a VIP warm start. All trained methods use 500 Adam steps with
learning rate \(5\times10^{-3}\) and eight Monte Carlo samples per step.
VIP, FTIP, and GMVIP jointly learn scalar observation noise initialized at
\(0.05\) in normalized units. The analog and empirical-Gaussian baselines
retain the fixed \(0.05\) likelihood. There is no validation or checkpoint
selection.

Every target is evaluated with exactly 1,024 posterior function samples. NLL
is the stable equal-weight Gaussian-mixture predictive density, including the
method's learned or fixed observation variance. The report aggregates all 75
seed-target results as mean \(\pm\) sample standard deviation (`ddof=1`).

Canonical results are written to:

```text
results/electricity/seed_<seed>/S_20/<method>/
```

Each method directory contains a manifest, configuration, final checkpoints,
per-target metrics, runtime records, and all 1,024 predictive samples.
`plot.py` validates these artifacts before producing the target-18 PNG/PDF,
the LaTeX table, and the JSON summary under `outputs/electricity/`.

## Reproduced results

The canonical reporter produces the following mean \(\pm\) sample-standard-
deviation results across the 75 seed--target windows. Coverage is in
percentage points.

| Method | RMSE | NLL | CRPS | CQM | Cov. 80% | Cov. 90% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Analog prior | 55.45 ± 137.78 | 5.31 ± 3.80 | 30.73 ± 78.26 | 0.12 ± 0.09 | 74.18 ± 18.53 | 85.96 ± 14.68 |
| VIP | 60.93 ± 161.13 | 99.19 ± 478.25 | 40.60 ± 111.12 | 0.38 ± 0.06 | 20.94 ± 10.04 | 26.26 ± 12.33 |
| FTIP | 68.02 ± 191.35 | 10.03 ± 36.15 | 41.28 ± 121.19 | 0.17 ± 0.10 | 57.25 ± 17.57 | 68.06 ± 17.20 |
| Empirical Gaussian | **48.44 ± 132.24** | 6.63 ± 8.53 | 28.37 ± 78.47 | 0.13 ± 0.10 | **79.32 ± 18.69** | **86.62 ± 15.45** |
| GMVIP | 49.06 ± 130.42 | **4.35 ± 1.19** | **27.48 ± 75.70** | **0.11 ± 0.08** | 76.22 ± 18.26 | 86.53 ± 15.01 |

The learned normalized noise scales average \(0.090\) for VIP, \(0.176\)
for FTIP, and \(0.198\) for GMVIP; their initialization is \(0.05\).
