"""Registry-backed public dataset interface."""

from __future__ import annotations


def get_dataset(name: str):
    """Construct a dataset by canonical registry name.

    Parameters
    ----------
    name : str
        Canonical dataset name or a supported deprecated alias.

    Returns
    -------
    object
        Dataset object registered under ``name``.
    """
    from .registry import get_dataset as _get_dataset

    return _get_dataset(name)


def canonical_dataset_name(name: str) -> str:
    """Return the canonical name and warn for a deprecated alias.

    Parameters
    ----------
    name : str
        Dataset name supplied by a caller or command-line interface.

    Returns
    -------
    str
        Canonical spelling used by the dataset registry.

    Warns
    -----
    DeprecationWarning
        If ``name`` is a supported deprecated spelling.
    """
    from .registry import canonical_dataset_name as _canonical_dataset_name

    return _canonical_dataset_name(name)


__all__ = ["canonical_dataset_name", "get_dataset"]
