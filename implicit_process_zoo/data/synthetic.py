"""Synthetic dataset loader exports."""

from ..utils.dataset import (
    Bimodal_Dataset,
    Constant_Dataset,
    Gap_Dataset,
    Heterocedastic_Dataset,
    Linear_Dataset,
    Skewed_Dataset,
    Synthetic_Dataset,
)

Heteroscedastic_Dataset = Heterocedastic_Dataset

__all__ = [
    "Bimodal_Dataset",
    "Constant_Dataset",
    "Gap_Dataset",
    "Heteroscedastic_Dataset",
    "Linear_Dataset",
    "Skewed_Dataset",
    "Synthetic_Dataset",
]
