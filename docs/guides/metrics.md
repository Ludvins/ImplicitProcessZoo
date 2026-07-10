# Probabilistic metrics

Experiment runners report the metrics appropriate to their task. Common
regression summaries include:

- root mean squared error (RMSE);
- Gaussian or sample-based negative log likelihood (NLL);
- continuous ranked probability score (CRPS);
- central interval coverage and width; and
- task-specific peak, horizon, residual, or nearest-prior diagnostics.

Classification reports accuracy, NLL, expected calibration error, Brier-style
scores where configured, and optional OOD diagnostics.

Empirical CRPS uses the sorted-sample identity instead of allocating a full
pairwise `[S,S,T]` tensor. It is numerically equivalent to the pairwise
definition while using substantially less memory.

## Aggregation

Do not average already-aggregated values across incompatible regions or data
splits. Prefer the per-target/per-region CSVs, then aggregate with the matching
`compare` module. The ELD paper report uses median and interquartile range over
all seed-target windows, not a mean of per-seed medians.
