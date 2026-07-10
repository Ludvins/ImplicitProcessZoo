# Checkpoints

Training checkpoints use schema version 1. A bundle contains:

- model, optimizer, and optional scheduler state;
- global optimizer step and experiment arguments;
- normalization buffers;
- Python, NumPy, CPU/CUDA PyTorch RNG states; and
- model-owned generator states.

## Library example

```py
--8<-- "docs/examples/checkpoints.py"
```

## Runner behavior

UCI-style benchmark runners distinguish two operations:

```bash
# Continue model, optimizer, scheduler, step, and RNG state.
python -m experiments.uci.benchmark \
  --model vip --dataset concrete \
  --resume_from_checkpoint results/uci/vip_checkpoint.pt

# Initialize model parameters without resuming optimization.
python -m experiments.uci.benchmark \
  --model ftip --dataset concrete \
  --warm_start_from results/uci/vip_checkpoint.pt
```

Resume rejects legacy model-only files with a migration message. Warm start
accepts either a raw legacy state dictionary or the model component of a
versioned checkpoint. The two flags are mutually exclusive.
