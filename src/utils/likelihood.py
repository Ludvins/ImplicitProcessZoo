import math

import torch
from torch.nn.functional import one_hot
from .quadrature import hermgauss, hermgaussquadrature

_LOG2PI = math.log(2 * math.pi)
_SQRT2_INV = 1.0 / math.sqrt(2.0)
_SQRT_PI_INV = 1.0 / math.sqrt(math.pi)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def inv_probit(x):
    """Gaussian CDF with jitter for numerical stability."""
    jitter = 1e-3
    return 0.5 * (1.0 + torch.erf(x * _SQRT2_INV)) * (1 - 2 * jitter) + jitter


# ---------------------------------------------------------------------------
# Log-probability at point samples  p(y | f)
# ---------------------------------------------------------------------------

def gaussian_logp(F, Y, log_variance):
    """log N(Y; F, exp(log_variance))"""
    var = torch.exp(log_variance)
    return -0.5 * (_LOG2PI + log_variance + (F - Y).square() / var)


def bernoulli_logp(F, Y):
    """log Bernoulli(Y; inv_probit(F))"""
    p = inv_probit(F)
    return torch.log(p * Y + (1 - p) * (1 - Y))


def multiclass_logp(F, Y, num_classes, epsilon):
    """Softmax cross-entropy log-likelihood for multiclass classification.

    Parameters
    ----------
    F : (..., N, K) logits
    Y : (N, 1), (N,) class indices, or (N, K) one-hot labels
    num_classes, epsilon : unused, kept for API compatibility
    """
    if Y.dim() >= 2 and Y.shape[-1] == num_classes:
        log_probs = torch.nn.functional.log_softmax(F, dim=-1)
        while Y.dim() < log_probs.dim():
            Y = Y.unsqueeze(0)
        return log_probs * Y.to(device=F.device, dtype=F.dtype)
    y_idx = Y.long().squeeze(-1)                             # (N,)
    log_probs = torch.nn.functional.log_softmax(F, dim=-1)  # (..., N, K)
    # Broadcast index to match log_probs leading dims
    idx = y_idx.unsqueeze(-1)                                # (N, 1)
    while idx.dim() < log_probs.dim():
        idx = idx.unsqueeze(0)
    idx = idx.expand_as(log_probs[..., :1])                  # (..., N, 1)
    return log_probs.gather(-1, idx)                         # (..., N, 1)


# ---------------------------------------------------------------------------
# Predict mean and variance  (integrating over q(f) = N(Fmu, Fvar))
# ---------------------------------------------------------------------------

def predict_mean_and_var_regression(Fmu, Fvar, log_variance):
    """Returns (Fmu, Fvar + exp(log_variance))"""
    return Fmu, Fvar + torch.exp(log_variance)


def predict_mean_and_var_binary(Fmu, Fvar):
    """Returns probit-based mean and Bernoulli variance."""
    p = inv_probit(Fmu / torch.sqrt(1 + Fvar))
    return p, p - torch.square(p)


def predict_mean_and_var_multiclass(Fmu, Fvar, num_classes, epsilon,
                                    num_gh_points, dtype, device):
    """Compute class probabilities via prob_is_largest for each class."""
    K1 = epsilon / (num_classes - 1)
    possible_outputs = [
        torch.full([Fmu.size()[0], 1], i) for i in range(num_classes)
    ]
    ps = []
    for po in possible_outputs:
        p = prob_is_largest(po, Fmu, Fvar, num_classes, num_gh_points,
                            dtype, device)
        ps.append(p * (1 - epsilon) + (1.0 - p) * K1)
    ps = torch.concat(ps, dim=1)
    return ps, ps - torch.square(ps)


# ---------------------------------------------------------------------------
# Variational expectations  E_q(f)[ log p(y | f) ]   (used by VIP)
# ---------------------------------------------------------------------------

def gaussian_variational_expectations(Fmu, Fvar, Y, alpha, log_variance):

    if alpha == 0:
        logpdf = (
            -0.5 * _LOG2PI
            - 0.5 * log_variance
            - 0.5 * ((Y - Fmu).square() + Fvar) / torch.exp(log_variance)
        )
        return torch.mean(logpdf, dim=0)
    # Black-box alpha-energy
    log_S = math.log(Fmu.shape[0])
    _LOG2PI_T = _LOG2PI  # scalar constant

    # log C = 0.5*log(2*pi*var/alpha) - alpha*0.5*log(2*pi*var)
    #       = 0.5*(1-alpha)*(LOG2PI + log_variance) - 0.5*log(alpha)
    log_C = 0.5 * (1 - alpha) * (_LOG2PI_T + log_variance) - 0.5 * math.log(alpha)

    variance = torch.exp(log_variance)
    logpdf = gaussian_logp(Fmu, Y, (Fvar + variance / alpha).log())
    logpdf = torch.logsumexp(logpdf, dim=0) + log_C - log_S
    return logpdf / alpha

def gaussian_variational_expectations_full_cov(Fmu, Fcov, Y, alpha, log_variance):
    """BB-alpha energy under a full-cov posterior q(f) = N(Fmu, Fcov).

    Likelihood is N(y | f, diag(sigma^2)) with sigma^2 = exp(log_variance).

    For alpha = 0 the data-fit term reduces to the diagonal one (off-diagonal
    entries of Fcov vanish under E_q[(y-f)^T Sigma_n^{-1} (y-f)] when Sigma_n
    is diagonal). For alpha > 0 the alpha-energy is the log marginal of an
    MVN whose covariance includes off-diagonal terms.

    Parameters
    ----------
    Fmu : (B, D)
    Fcov: (B, D, D)
    Y   : (B, D)
    alpha : float
    log_variance : (D,)

    Returns
    -------
    Tensor of shape (B, D) for alpha=0 (sum-friendly with the diagonal API)
    or (B,) for alpha>0 (joint contribution per example).
    """
    sigma2 = torch.exp(log_variance)  # (D,)
    D = Y.shape[-1]

    if alpha == 0:
        Fvar_diag = torch.diagonal(Fcov, dim1=-2, dim2=-1)  # (B, D)
        logpdf = (
            -0.5 * _LOG2PI
            - 0.5 * log_variance
            - 0.5 * ((Y - Fmu).square() + Fvar_diag) / sigma2
        )
        return logpdf  # (B, D)

    # alpha > 0: log E_q[N(y|f, diag(sigma^2))^alpha] / alpha
    # = log N(y; Fmu, Fcov + diag(sigma^2/alpha)) / alpha + constants/alpha
    cov_total = Fcov + torch.diag(sigma2 / alpha).unsqueeze(0)  # (B, D, D)
    jitter = 1e-6 * torch.eye(D, dtype=cov_total.dtype, device=cov_total.device)
    mvn = torch.distributions.MultivariateNormal(
        Fmu, covariance_matrix=cov_total + jitter
    )
    log_marg = mvn.log_prob(Y)  # (B,)
    const = ((1 - alpha) * D / 2.0) * _LOG2PI \
        - (D / 2.0) * math.log(alpha) \
        + ((1 - alpha) / 2.0) * log_variance.sum()
    return (const + log_marg) / alpha  # (B,)


def bernoulli_variational_expectations(Fmu, Fvar, Y, num_gh_points,
                                       dtype, device):
    """Gauss-Hermite quadrature of bernoulli_logp."""
    return hermgaussquadrature(
        bernoulli_logp, num_gh_points, Fmu, Fvar, Y, dtype, device,
    )


def multiclass_variational_expectations(Fmu, Fvar, Y, num_classes, epsilon,
                                        num_gh_points, dtype, device):
    """prob_is_largest then p*log(1-eps) + (1-p)*log(K1)."""
    K1 = epsilon / (num_classes - 1)
    p = prob_is_largest(Y, Fmu, Fvar, num_classes, num_gh_points, dtype, device)
    ve = p * torch.log(torch.tensor(1.0 - epsilon)) + (1.0 - p) * torch.log(
        torch.tensor(K1)
    )
    return torch.sum(ve, dim=-1)


# ---------------------------------------------------------------------------
# Multiclass helper
# ---------------------------------------------------------------------------

def prob_is_largest(Y, Fmu, Fvar, num_classes, num_gh_points, dtype, device):
    """Gauss-Hermite quadrature to compute P(f_Y > f_k for all k != Y).

    Reference: http://mlg.eng.cam.ac.uk/matthews/thesis.pdf
    """
    gh_x, gh_w = hermgauss(num_gh_points, dtype, device)

    oh_on = one_hot(Y.long().flatten(), num_classes).to(dtype=dtype, device=device)

    mu_selected = torch.sum(oh_on * Fmu, 1).reshape(-1, 1)
    var_selected = torch.sum(oh_on * Fvar, 1).reshape(-1, 1)

    sqrt_selected = torch.sqrt(torch.clip(2.0 * var_selected, min=1e-10))

    X = mu_selected + gh_x * sqrt_selected

    dist = (X.unsqueeze(1) - Fmu.unsqueeze(2)) / Fvar.unsqueeze(2)
    cdfs = 0.5 * (1.0 + torch.erf(dist * _SQRT2_INV))
    cdfs = cdfs * (1 - 2e-6) + 1e-6

    oh_off = torch.abs(oh_on - 1)
    cdfs = cdfs * oh_off.unsqueeze(2) + oh_on.unsqueeze(2)

    gh_w = gh_w.unsqueeze(-1) * _SQRT_PI_INV
    return torch.prod(cdfs, dim=1) @ gh_w


# ---------------------------------------------------------------------------
# GP log marginal likelihood  (Appendix C.5, Ma et al. 2019)
# ---------------------------------------------------------------------------

def compute_gp_log_marginal(m_star, phi, y, sigma2):
    """
    Log marginal likelihood of the moment-matched GP prior.

    Uses cached intermediates from predict_f to avoid recomputing
    the generative function forward pass.

    Parameters
    ----------
    m_star : [N, D]
        Prior mean (mean of fantasy samples, already computed in predict_f).
    phi : [S, N, D]
        Normalized centered features (already computed in predict_f),
        where phi = (f - m) / sqrt(S - 1). K_star = phi^T @ phi.
    y : [N, D]
        Observed targets (normalized).
    sigma2 : scalar tensor
        Observation noise variance, exp(log_variance).

    Returns
    -------
    Scalar tensor with gradient.
    """
    if phi.ndim == 2:
        phi = phi.unsqueeze(-1)
    if m_star.ndim == 1:
        m_star = m_star.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    S, M, D = phi.shape
    dtype, device = phi.dtype, phi.device

    log_q = torch.tensor(0.0, dtype=dtype, device=device)

    # Support both scalar sigma2 and per-output-dim sigma2 (shape (D,)).
    sigma2_t = sigma2 if torch.is_tensor(sigma2) else torch.as_tensor(
        sigma2, dtype=dtype, device=device)
    if sigma2_t.ndim == 0 or sigma2_t.numel() == 1:
        sigma2_t = sigma2_t.reshape(()).expand(D)
    else:
        sigma2_t = sigma2_t.reshape(D)

    # Vectorize the whole D-wise loop with a batched Cholesky.
    phi_dms = phi.permute(2, 0, 1)                               # (D, S, M)
    K_star = torch.einsum("dsm, dsn -> dmn", phi_dms, phi_dms)   # (D, M, M)

    jitter = 1e-6
    eye = torch.eye(M, dtype=dtype, device=device).unsqueeze(0)   # (1, M, M)
    K_y = K_star + (sigma2_t + jitter).view(D, 1, 1) * eye        # (D, M, M)
    L = torch.linalg.cholesky(K_y)                                # (D, M, M)

    residual = (y - m_star).T.unsqueeze(-1)                       # (D, M, 1)
    alpha = torch.cholesky_solve(residual, L)                      # (D, M, 1)

    log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # (D,)
    quad = (residual * alpha).sum(dim=(-2, -1))                    # (D,)

    log_q = -0.5 * (quad + log_det + M * _LOG2PI).sum()
    return log_q
