"""Shared normalizing-flow construction."""

from __future__ import annotations

from implicit_process_zoo.flows import CouplingFlow, SplineCoupling1x1Flow, SplineCouplingFlow


def build_flow(
    flow_type: str,
    *,
    depth: int,
    input_dim: int,
    seed: int,
    device,
    dtype,
    num_bins: int = 8,
    domain: float = 3.0,
):
    common = {
        "depth": int(depth),
        "input_dim": int(input_dim),
        "device": device,
        "dtype": dtype,
        "seed": int(seed),
    }
    if flow_type == "affine":
        return CouplingFlow(**common)
    if flow_type == "spline":
        return SplineCouplingFlow(**common, num_bins=int(num_bins), B=float(domain))
    if flow_type == "spline_1x1":
        return SplineCoupling1x1Flow(**common, num_bins=int(num_bins), B=float(domain))
    raise ValueError(f"Unknown flow type {flow_type!r}.")
