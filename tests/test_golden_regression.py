import json
from pathlib import Path

import torch

from implicit_process_zoo.map_baseline import DeterministicMAP


def test_map_fixed_seed_matches_publication_golden_output():
    golden_path = Path(__file__).parent / "golden" / "map_seed17.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    model = DeterministicMAP(
        input_dim=2,
        output_dim=1,
        structure=[3],
        activation=torch.tanh,
        num_data=4,
        seed=golden["seed"],
        dtype=torch.float64,
        y_mean=[1.5],
        y_std=[2.0],
    )
    x = torch.tensor([[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5], [0.25, 0.75]], dtype=torch.float64)
    y = torch.tensor([[-0.5], [0.25], [1.0], [0.75]], dtype=torch.float64)

    torch.testing.assert_close(
        model.predict_f(x).detach().flatten(),
        torch.tensor(golden["predict_f"], dtype=torch.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        model.nelbo(x, y).detach(),
        torch.tensor(golden["nelbo"], dtype=torch.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        model.regularizer().detach(),
        torch.tensor(golden["regularizer"], dtype=torch.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
