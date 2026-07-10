"""Human-readable model inspection helpers."""

from __future__ import annotations

import numpy as np
import torch


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """Print a compact hierarchy of trainable parameters."""
    print("\n---- MODEL PARAMETERS ----")
    np.set_printoptions(threshold=3, edgeitems=2)
    sections: list[str] = []
    pad = "  "
    for full_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = full_name.split(".")
        for index in range(len(parts) - 1):
            if parts[index] not in sections:
                print(pad * index, parts[index].upper())
                sections = parts[: index + 1]
        padding = pad * (len(parts) - 1)
        print(padding, f"{parts[-1]}: ({str(list(parameter.shape))[1:-1]})")
        print(
            padding + " " * (len(parts[-1]) + 2),
            parameter.detach().cpu().numpy().flatten(),
        )
    print("\n---------------------------\n\n")
