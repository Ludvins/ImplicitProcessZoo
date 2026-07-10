# Datasets

The public registry is available through:

```python
from implicit_process_zoo.data import canonical_dataset_name, get_dataset
```

Canonical UCI regression names are `boston`, `concrete`, `energy`, `kin8nm`,
`naval`, `power`, `protein`, `winered`, and `yacht`. The spellings `yatch` and
`heterocedastic` are deprecated aliases for `yacht` and `heteroscedastic`.

The broader loaders also support diagnostic scalar datasets such as `gap`,
`bimodal`, `skewed`, `heteroscedastic`, `snelson`, and `variational_lla`;
large regression (`year`, `airline`, `taxi`); vision/classification datasets;
and trajectory tasks used by the simulator-prior experiments.

## Normalization and validation

Training inputs and targets must be finite, nonempty rank-two arrays with equal
row counts. Target standard deviations are protected by an epsilon to keep
constant-output datasets finite. Normalization statistics used by models are
registered buffers, so they participate in state dictionaries and `.to()`
device/dtype moves.

## Downloads

Data sources live in a registry that records HTTPS URLs, filenames, expected
archive members, hashes, and attribution metadata where applicable. Downloads
use timeouts/retries and atomic temporary files. SHA-256 mismatches fail before
use, and archive extraction rejects unexpected or traversal members.

Some datasets cannot be redistributed. `airline` expects `data/airline.csv`;
`taxi` can use `data/taxi.csv` or build from the configured source when the
additional parquet dependency is available.
