import torch


def safe_cholesky(
    K: torch.Tensor,
    initial_jitter: float = 1e-6,
    max_tries: int = 7,
) -> torch.Tensor:
    """Return a lower Cholesky factor of ``K + jitter I``."""
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")

    eye = torch.eye(K.shape[-1], dtype=K.dtype, device=K.device)
    jitter = float(initial_jitter)
    last_info = None
    for _ in range(int(max_tries)):
        L, info = torch.linalg.cholesky_ex(K + jitter * eye)
        last_info = info
        if torch.all(info == 0):
            return L
        jitter *= 10.0

    raise RuntimeError(
        f"Cholesky failed after {max_tries} tries. Last info={last_info.detach().cpu()}."
    )


def right_cholesky_solve(K_xz: torch.Tensor, L_zz: torch.Tensor) -> torch.Tensor:
    """Compute ``K_xz @ inv(K_zz)`` from the Cholesky factor of ``K_zz``."""
    if K_xz.ndim != 2:
        raise ValueError("K_xz must have shape [N, M].")
    if L_zz.ndim != 2 or L_zz.shape[0] != L_zz.shape[1]:
        raise ValueError("L_zz must be a square Cholesky factor.")
    if K_xz.shape[1] != L_zz.shape[0]:
        raise ValueError("K_xz and L_zz dimensions are incompatible.")
    return torch.cholesky_solve(K_xz.T, L_zz).T
