"""Save and restore a complete versioned training checkpoint."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader, TensorDataset

from implicit_process_zoo.map_baseline import DeterministicMAP
from implicit_process_zoo.utils import (
    build_training_checkpoint,
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)


def make_model() -> DeterministicMAP:
    return DeterministicMAP(
        input_dim=1,
        output_dim=1,
        structure=[4],
        activation=torch.tanh,
        num_data=8,
        seed=3,
        dtype=torch.float64,
    )


x = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
loader = DataLoader(TensorDataset(x, x.square()), batch_size=4, shuffle=False)
model = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
model.fit(loader, optimizer=optimizer, iterations=2)

bundle = build_training_checkpoint(
    model,
    optimizer=optimizer,
    scheduler=None,
    global_step=model._fit_global_step,
    arguments={"model": "map", "seed": 3},
)

with TemporaryDirectory() as directory:
    path = Path(directory) / "checkpoint.pt"
    save_training_checkpoint(path, bundle)
    loaded = load_training_checkpoint(path, map_location="cpu")

    restored_model = make_model()
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-2)
    step = restore_training_checkpoint(loaded, restored_model, restored_optimizer)
    assert step == 2
