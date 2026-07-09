from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForecastingTask:
    name: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    X_plot: torch.Tensor
    y_plot_true: torch.Tensor
    X_context_observed: torch.Tensor
    X_context_full: torch.Tensor
    noise_std: torch.Tensor
    prior: torch.nn.Module
    metadata: dict
