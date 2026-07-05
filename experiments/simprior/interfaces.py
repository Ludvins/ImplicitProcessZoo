from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class SimulatorPrior(Protocol):
    input_dim: int
    output_dim: int

    def sample_indices(self, n: int, seed: int | None = None) -> torch.LongTensor:
        """Return fixed path identifiers for n simulator prior paths."""

    def evaluate(self, X: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate fixed simulator paths at normalized query inputs."""

    def sample(self, X: torch.Tensor, n: int, seed: int | None = None) -> torch.Tensor:
        ids = self.sample_indices(n, seed)
        return self.evaluate(X, ids)


@dataclass
class SimPriorTask:
    name: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    X_plot: torch.Tensor
    y_plot_true: torch.Tensor
    noise_std: torch.Tensor
    prior: SimulatorPrior
    metadata: dict
