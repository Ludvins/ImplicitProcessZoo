"""Registry-backed public dataset interface."""

from __future__ import annotations


def get_dataset(name: str):
    """Construct a dataset by canonical registry name."""
    from .registry import get_dataset as _get_dataset

    return _get_dataset(name)


def canonical_dataset_name(name: str) -> str:
    """Return the canonical name and warn for a deprecated alias."""
    from .registry import canonical_dataset_name as _canonical_dataset_name

    return _canonical_dataset_name(name)


__all__ = ["canonical_dataset_name", "get_dataset"]
