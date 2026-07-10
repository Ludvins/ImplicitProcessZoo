"""Train a small model and draw deterministic seeded predictions."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from implicit_process_zoo.map_baseline import DeterministicMAP

dtype = torch.float64
x = torch.linspace(-2.0, 2.0, 24, dtype=dtype).unsqueeze(-1)
y = torch.sin(x)
loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

model = DeterministicMAP(
    input_dim=1,
    output_dim=1,
    structure=[8],
    activation=torch.tanh,
    num_data=len(x),
    seed=5,
    dtype=dtype,
)
losses = model.fit(loader, iterations=4, lr=1e-2, return_loss=True)

f_samples = model.predict_f_samples(x, 5, seed=17)
y_samples = model.predict_y_samples(x, 5, seed=17)
assert len(losses) == 4
assert f_samples.shape == y_samples.shape == (5, 24, 1)
