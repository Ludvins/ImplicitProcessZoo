# Damped-oscillator forecasting

The canonical experiment has two entry points:

```bash
python -m experiments.simulator_forecasting.benchmark \
  --methods vip,ftip,gmvip \
  --vip-basis-size 256 --seed 0 --target-ids 0:20 \
  --regenerate-targets

python -m experiments.simulator_forecasting.plot
```

The deterministic dataset contains 20 target trajectories. Observations with
\(t\le15\) are used for training; \(15<t\le20\) and \(20<t\le30\) are reported
as near and far extrapolation regions. No validation interval or checkpoint
selection is used.

VIP and FTIP share the same \(S=256\) prior basis. FTIP is initialized from
the exact corresponding 3,000-step VIP fit at learning rate
\(5\times10^{-3}\), then receives 3,000 fine-tuning steps at
\(10^{-4}\). Standalone VIP and GMVIP use 3,000 Adam steps at
\(5\times10^{-3}\). Every training stage uses 16 Monte Carlo samples.
GMVIP uses \(B=1024\) prior trajectories and \(M=32\) inducing locations.
All three methods learn scalar observation noise initialized at the simulator
value \(0.05\) and are evaluated with exactly 1,024 posterior function
samples. NLL is the stable equal-weight Gaussian-mixture predictive density
including learned observation variance. The oscillator-period error is not
computed or reported.

Canonical results are written to:

```text
results/oscillator/seed_0/S_256/<method>/
```

Each directory contains a manifest, configuration, final checkpoints,
per-target/region metrics, runtime records, and all 1,024 predictive samples.
`plot.py` validates these artifacts and writes the target-0 PNG/PDF, LaTeX
table, and JSON summary under `outputs/oscillator/`.

## Reproduced results

The canonical reporter produces the following mean \(\pm\) sample-standard-
deviation results across 20 targets on the far-extrapolation interval
\((20,30]\). Coverage is in percentage points.

| Method | RMSE | NLL | CRPS | Cov. 90% |
| --- | ---: | ---: | ---: | ---: |
| VIP | 1.50 ± 0.72 | 2.05 ± 0.80 | 0.94 ± 0.45 | 85.5 ± 20.2 |
| FTIP | 1.38 ± 0.64 | 1.82 ± 0.45 | 0.85 ± 0.38 | **91.0 ± 16.7** |
| GMVIP | **1.20 ± 0.65** | **1.73 ± 0.42** | **0.76 ± 0.37** | 92.0 ± 15.4 |

The learned physical-unit noise scales average \(0.057\) for VIP, \(0.165\)
for FTIP, and \(0.081\) for GMVIP.
