"""Canonical dataset registry with deprecated spelling aliases."""

from __future__ import annotations

import warnings

from ..utils.dataset import get_dataset as _legacy_get_dataset

ALIASES = {
    "yatch": "yacht",
    "heterocedastic": "heteroscedastic",
}


def canonical_dataset_name(name: str) -> str:
    """Normalize deprecated dataset spellings to their canonical names."""
    if name in ALIASES:
        canonical = ALIASES[name]
        warnings.warn(
            f"Dataset name {name!r} is deprecated; use {canonical!r}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical
    return name


def get_dataset(name: str):
    """Construct a registered dataset after canonicalizing its name."""
    return _legacy_get_dataset(canonical_dataset_name(name))
