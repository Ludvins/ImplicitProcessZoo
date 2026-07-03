import torch


def empirical_mean(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must have shape [S, N].")
    return values.mean(dim=0)


def empirical_cross_cov(
    values_x: torch.Tensor,
    values_z: torch.Tensor,
    mean_x: torch.Tensor | None = None,
    mean_z: torch.Tensor | None = None,
) -> torch.Tensor:
    if values_x.ndim != 2 or values_z.ndim != 2:
        raise ValueError("values_x and values_z must have shape [S, N/M].")
    if values_x.shape[0] != values_z.shape[0]:
        raise ValueError("values_x and values_z must use the same sample count.")
    sample_count = values_x.shape[0]
    if sample_count < 2:
        raise ValueError("At least two samples are required for covariance.")

    if mean_x is None:
        mean_x = empirical_mean(values_x)
    if mean_z is None:
        mean_z = empirical_mean(values_z)

    centered_x = values_x - mean_x.unsqueeze(0)
    centered_z = values_z - mean_z.unsqueeze(0)
    return centered_x.T.matmul(centered_z) / float(sample_count - 1)


def stabilize_covariance(
    K: torch.Tensor,
    jitter: float,
    shrinkage: float = 0.0,
) -> torch.Tensor:
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1].")
    dim = K.shape[0]
    eye = torch.eye(dim, dtype=K.dtype, device=K.device)
    K = 0.5 * (K + K.T)
    if shrinkage > 0.0:
        avg_diag = torch.trace(K) / float(dim)
        K = (1.0 - shrinkage) * K + shrinkage * avg_diag * eye
    return K + float(jitter) * eye


def safe_cholesky(
    K: torch.Tensor,
    initial_jitter: float = 1e-6,
    max_tries: int = 6,
) -> torch.Tensor:
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")
    eye = torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
    jitter = float(initial_jitter)
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(K + jitter * eye)
        if int(info.max().detach().cpu()) == 0:
            return L
        jitter *= 10.0
    return torch.linalg.cholesky(K + jitter * eye)
