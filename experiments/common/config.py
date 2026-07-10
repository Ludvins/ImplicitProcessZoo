"""Shared experiment configuration loading and merging."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import yaml


def deep_merge(base: Mapping, override: Mapping) -> dict:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping at the root of {path}, got {type(value).__name__}.")
    return value
