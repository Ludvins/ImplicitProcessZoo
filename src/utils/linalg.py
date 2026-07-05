import torch


def safe_cholesky(
    K: torch.Tensor,
    initial_jitter: float = 1e-6,
    max_tries: int = 7,
) -> torch.Tensor:
    """Return a lower Cholesky factor of ``K + jitter I``."""
    if K.ndim not in (2, 3) or K.shape[-1] != K.shape[-2]:
        raise ValueError("K must be a square matrix or a batch of square matrices.")

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
    if K_xz.ndim not in (2, 3):
        raise ValueError("K_xz must have shape [N, M] or [K, N, M].")
    if L_zz.ndim not in (2, 3) or L_zz.shape[-1] != L_zz.shape[-2]:
        raise ValueError("L_zz must be a square Cholesky factor or a batch of factors.")
    if K_xz.shape[-1] != L_zz.shape[-1]:
        raise ValueError("K_xz and L_zz dimensions are incompatible.")
    if K_xz.ndim == 2:
        if L_zz.ndim != 2:
            raise ValueError("Batched L_zz requires batched K_xz.")
        return torch.cholesky_solve(K_xz.T, L_zz).T
    if L_zz.ndim == 2:
        L_zz = L_zz.expand(K_xz.shape[0], -1, -1)
    if K_xz.shape[0] != L_zz.shape[0]:
        raise ValueError("Batched K_xz and L_zz must have the same batch size.")
    return torch.cholesky_solve(K_xz.transpose(-1, -2), L_zz).transpose(-1, -2)
