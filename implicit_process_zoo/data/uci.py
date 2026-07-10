"""UCI and OpenML dataset loader exports."""

from ..utils.dataset import (
    Boston_Dataset,
    Concrete_Dataset,
    Energy_Dataset,
    Kin8nm_Dataset,
    Naval_Dataset,
    Power_Dataset,
    Protein_Bimodal_Dataset,
    Protein_Dataset,
    WineRed_Dataset,
    Yatch_Dataset,
)

Yacht_Dataset = Yatch_Dataset

__all__ = [
    "Boston_Dataset",
    "Concrete_Dataset",
    "Energy_Dataset",
    "Kin8nm_Dataset",
    "Naval_Dataset",
    "Power_Dataset",
    "Protein_Bimodal_Dataset",
    "Protein_Dataset",
    "WineRed_Dataset",
    "Yacht_Dataset",
]
