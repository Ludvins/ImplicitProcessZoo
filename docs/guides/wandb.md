# Weights & Biases

W&B tracking is optional and imported lazily. Install the tracking extra and
authenticate using W&B's normal environment or login flow:

```bash
python -m pip install -e ".[tracking]"
```

Enable tracking on supported benchmark runners:

```bash
python -m experiments.uci.benchmark \
  --model vip --dataset concrete \
  --wandb --wandb_project implicit-process-zoo \
  --wandb_tags uci vip
```

`--wandb_entity` defaults to unset, so runs use the authenticated user's or
team's normal default instead of a personal account embedded in the code.
`--wandb_mode offline` records local runs without sending them immediately;
`--wandb_mode disabled` disables the SDK explicitly.

Local JSON/CSV artifacts remain the canonical portable output even when W&B is
enabled. W&B failures should not be treated as a substitute for saving the
runner's result files.
