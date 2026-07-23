"""Sample-based distances for one-dimensional prior-fidelity experiments."""

from __future__ import annotations

import math

import torch


def as_curves(samples: torch.Tensor) -> torch.Tensor:
    """Normalize function samples to shape ``[num_samples, num_grid_points]``."""
    samples = torch.as_tensor(samples)
    if samples.ndim == 3 and samples.shape[-1] == 1:
        samples = samples[..., 0]
    if samples.ndim != 2:
        raise ValueError(
            "Function samples must have shape [S, N] or [S, N, 1], "
            f"got {tuple(samples.shape)}."
        )
    if samples.shape[0] < 2:
        raise ValueError("At least two function samples are required.")
    return samples


def fit_pointwise_standardizer(
    calibration_samples: torch.Tensor,
    *,
    min_std: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit pointwise mean and standard deviation from an independent prior sample."""
    curves = as_curves(calibration_samples)
    mean = curves.mean(dim=0)
    std = curves.std(dim=0, unbiased=False).clamp_min(float(min_std))
    return mean, std


def standardize_curves(
    samples: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Apply a fixed pointwise affine standardization."""
    curves = as_curves(samples)
    mean = torch.as_tensor(mean, dtype=curves.dtype, device=curves.device)
    std = torch.as_tensor(std, dtype=curves.dtype, device=curves.device)
    if mean.shape != curves.shape[1:] or std.shape != curves.shape[1:]:
        raise ValueError(
            "Standardizer shape must match the sample grid: "
            f"curves={tuple(curves.shape)}, mean={tuple(mean.shape)}, std={tuple(std.shape)}."
        )
    return (curves - mean) / std


def projection_directions(
    dimension: int,
    num_projections: int,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create deterministic random unit directions for sliced distances."""
    dimension = int(dimension)
    num_projections = int(num_projections)
    if dimension <= 0 or num_projections <= 0:
        raise ValueError("dimension and num_projections must be positive.")
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    directions = torch.randn(
        num_projections,
        dimension,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    return directions / directions.norm(dim=1, keepdim=True).clamp_min(
        torch.finfo(dtype).eps
    )


def pointwise_wasserstein1(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Empirical one-dimensional Wasserstein-1 distance at every grid point."""
    reference = as_curves(reference)
    candidate = as_curves(candidate).to(dtype=reference.dtype, device=reference.device)
    if reference.shape != candidate.shape:
        raise ValueError(
            "Pointwise Wasserstein currently requires equal sample and grid sizes, "
            f"got {tuple(reference.shape)} and {tuple(candidate.shape)}."
        )
    reference_sorted = torch.sort(reference, dim=0).values
    candidate_sorted = torch.sort(candidate, dim=0).values
    return (reference_sorted - candidate_sorted).abs().mean(dim=0)


def sliced_wasserstein2(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Sliced Wasserstein-2 distance between complete sampled functions."""
    reference = as_curves(reference)
    candidate = as_curves(candidate).to(dtype=reference.dtype, device=reference.device)
    directions = torch.as_tensor(
        directions, dtype=reference.dtype, device=reference.device
    )
    if reference.shape != candidate.shape:
        raise ValueError(
            "Sliced Wasserstein currently requires equal sample and grid sizes, "
            f"got {tuple(reference.shape)} and {tuple(candidate.shape)}."
        )
    if directions.ndim != 2 or directions.shape[1] != reference.shape[1]:
        raise ValueError(
            "Projection directions must have shape [P, N], "
            f"got {tuple(directions.shape)} for grid size {reference.shape[1]}."
        )
    reference_projection = reference @ directions.T
    candidate_projection = candidate @ directions.T
    reference_projection = torch.sort(reference_projection, dim=0).values
    candidate_projection = torch.sort(candidate_projection, dim=0).values
    projection_w2_sq = (reference_projection - candidate_projection).square().mean(dim=0)
    return projection_w2_sq.mean().sqrt()


def moment_errors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return standardized mean RMSE and relative covariance Frobenius error."""
    reference = as_curves(reference)
    candidate = as_curves(candidate).to(dtype=reference.dtype, device=reference.device)
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError("Reference and candidate must use the same grid.")
    mean_rmse = (reference.mean(dim=0) - candidate.mean(dim=0)).square().mean().sqrt()
    reference_centered = reference - reference.mean(dim=0, keepdim=True)
    candidate_centered = candidate - candidate.mean(dim=0, keepdim=True)
    reference_cov = reference_centered.T @ reference_centered / float(
        reference.shape[0] - 1
    )
    candidate_cov = candidate_centered.T @ candidate_centered / float(
        candidate.shape[0] - 1
    )
    denominator = torch.linalg.matrix_norm(reference_cov).clamp_min(
        torch.finfo(reference.dtype).eps
    )
    covariance_error = torch.linalg.matrix_norm(candidate_cov - reference_cov) / denominator
    return mean_rmse, covariance_error


def _mean_pairwise_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    dimension_scale = math.sqrt(float(left.shape[1]))
    total = left.new_zeros(())
    count = 0
    for left_start in range(0, left.shape[0], int(chunk_size)):
        left_chunk = left[left_start : left_start + int(chunk_size)]
        for right_start in range(0, right.shape[0], int(chunk_size)):
            right_chunk = right[right_start : right_start + int(chunk_size)]
            distances = torch.cdist(left_chunk, right_chunk) / dimension_scale
            total = total + distances.sum()
            count += distances.numel()
    return total / float(count)


def energy_distance(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Biased empirical energy distance using grid-normalized Euclidean distance."""
    reference = as_curves(reference)
    candidate = as_curves(candidate).to(dtype=reference.dtype, device=reference.device)
    cross = _mean_pairwise_distance(reference, candidate, chunk_size=chunk_size)
    within_reference = _mean_pairwise_distance(
        reference, reference, chunk_size=chunk_size
    )
    within_candidate = _mean_pairwise_distance(
        candidate, candidate, chunk_size=chunk_size
    )
    return (2.0 * cross - within_reference - within_candidate).clamp_min(0.0)


def estimate_rbf_bandwidth(
    calibration_samples: torch.Tensor,
    *,
    max_samples: int = 512,
) -> torch.Tensor:
    """Median-heuristic bandwidth on grid-normalized curve distances."""
    curves = as_curves(calibration_samples)
    subset = curves[: min(int(max_samples), curves.shape[0])]
    distances = torch.pdist(subset) / math.sqrt(float(curves.shape[1]))
    positive = distances[distances > 0]
    if positive.numel() == 0:
        return curves.new_tensor(1.0)
    return positive.median().clamp_min(torch.finfo(curves.dtype).eps)


def _mean_rbf_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bandwidth: torch.Tensor,
    scales: tuple[float, ...],
    chunk_size: int,
) -> torch.Tensor:
    dimension = float(left.shape[1])
    total = left.new_zeros(())
    count = 0
    scale_tensor = left.new_tensor(scales)
    bandwidths_sq = (bandwidth * scale_tensor).square().clamp_min(
        torch.finfo(left.dtype).eps
    )
    for left_start in range(0, left.shape[0], int(chunk_size)):
        left_chunk = left[left_start : left_start + int(chunk_size)]
        for right_start in range(0, right.shape[0], int(chunk_size)):
            right_chunk = right[right_start : right_start + int(chunk_size)]
            distance_sq = torch.cdist(left_chunk, right_chunk).square() / dimension
            kernels = torch.exp(
                -0.5 * distance_sq.unsqueeze(-1) / bandwidths_sq
            ).mean(dim=-1)
            total = total + kernels.sum()
            count += kernels.numel()
    return total / float(count)


def rbf_mmd2(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    bandwidth: torch.Tensor | float,
    scales: tuple[float, ...] = (0.5, 1.0, 2.0),
    chunk_size: int = 256,
) -> torch.Tensor:
    """Biased non-negative multi-bandwidth RBF MMD squared."""
    reference = as_curves(reference)
    candidate = as_curves(candidate).to(dtype=reference.dtype, device=reference.device)
    bandwidth = torch.as_tensor(
        bandwidth, dtype=reference.dtype, device=reference.device
    )
    reference_kernel = _mean_rbf_kernel(
        reference,
        reference,
        bandwidth=bandwidth,
        scales=scales,
        chunk_size=chunk_size,
    )
    candidate_kernel = _mean_rbf_kernel(
        candidate,
        candidate,
        bandwidth=bandwidth,
        scales=scales,
        chunk_size=chunk_size,
    )
    cross_kernel = _mean_rbf_kernel(
        reference,
        candidate,
        bandwidth=bandwidth,
        scales=scales,
        chunk_size=chunk_size,
    )
    return (reference_kernel + candidate_kernel - 2.0 * cross_kernel).clamp_min(0.0)


def fidelity_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    directions: torch.Tensor,
    mmd_bandwidth: torch.Tensor | float,
    chunk_size: int = 256,
    robustness_max_samples: int | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
    """Compute the complete metric bundle and return its pointwise W1 profile."""
    reference = as_curves(reference)
    candidate = as_curves(candidate).to(dtype=reference.dtype, device=reference.device)
    pointwise = pointwise_wasserstein1(reference, candidate)
    mean_error, covariance_error = moment_errors(reference, candidate)
    if robustness_max_samples is None:
        robustness_reference = reference
        robustness_candidate = candidate
    else:
        count = min(
            int(robustness_max_samples),
            reference.shape[0],
            candidate.shape[0],
        )
        robustness_reference = reference[:count]
        robustness_candidate = candidate[:count]
    metrics = {
        "joint_sw2": float(
            sliced_wasserstein2(reference, candidate, directions).detach().cpu()
        ),
        "marginal_w1_mean": float(pointwise.mean().detach().cpu()),
        "marginal_w1_max": float(pointwise.max().detach().cpu()),
        "mean_rmse": float(mean_error.detach().cpu()),
        "covariance_rel_fro": float(covariance_error.detach().cpu()),
        "energy_distance": float(
            energy_distance(
                robustness_reference,
                robustness_candidate,
                chunk_size=chunk_size,
            )
            .detach()
            .cpu()
        ),
        "rbf_mmd2": float(
            rbf_mmd2(
                robustness_reference,
                robustness_candidate,
                bandwidth=mmd_bandwidth,
                chunk_size=chunk_size,
            )
            .detach()
            .cpu()
        ),
    }
    return metrics, pointwise.detach()
