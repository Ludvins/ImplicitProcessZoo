# Reproducibility

## Randomness

Model constructors preserve the caller's global PyTorch RNG state. Stochastic
models own local generators, and seeded prediction temporarily sets and then
restores those generators. Full checkpoints capture Python, NumPy, PyTorch,
CUDA, and model-owned generator states.

## Checkpoints

Schema-version 1 checkpoints contain the model, optimizer, scheduler, global
step, arguments, normalization state, and random states. Resume accepts only a
full versioned checkpoint. Warm start accepts either its model component or a
legacy raw state dictionary. This prevents a model-only file from silently
restarting optimization state.

## Data and results

Downloaded sources are registered with HTTPS URLs and verified hashes. Archive
extraction admits only expected members and rejects traversal paths. Generated
data, checkpoints, plots, and result directories are gitignored.

For scientific reruns, use a named preset, save its generated configuration,
record the Git commit and environment, and write to a new result root rather
than overwriting published artifacts. The ELD runner has one canonical,
corrected methodology described in its [experiment guide](../experiments/eld.md).
