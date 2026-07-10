"""Rational-quadratic spline utilities for flow layers."""

import torch
import torch.nn.functional as F

_MIN_BIN_WIDTH = 1e-3
_MIN_BIN_HEIGHT = 1e-3
_MIN_DERIVATIVE = 1e-3


def rq_spline_forward(y, widths, heights, derivatives, B, inverse=False):
    """Apply a batched 1D rational-quadratic spline.

    Parameters
    ----------
    y : Tensor of shape (N,)
        Input values. If ``inverse`` is False these are data points mapped to
        the base; if True, base points mapped back to data.
    widths, heights : Tensor of shape (N, K)
        Normalized widths/heights (each row sums to 1). Multiplied by ``2B``
        internally.
    derivatives : Tensor of shape (N, K + 1)
        Positive derivative values at knots. Boundary entries should already
        be 1 (identity tails).
    B : float
        Half-width of the spline domain.
    inverse : bool
        If True, solve the quadratic to invert the spline.

    Returns
    -------
    out : Tensor of shape (N,)
    log_det : Tensor of shape (N,)
        ``log|d out / d y|``. For the inverse, this is ``log|d y / d u|`` —
        i.e. the Jacobian of the inverse map. We return the signed log-det in
        the direction of the map being applied.
    """
    K = widths.shape[-1]

    widths = widths * (2 * B)
    heights = heights * (2 * B)

    # Cumulative knot positions in [-B, B].
    x_knots = torch.cumsum(widths, dim=-1)
    x_knots = F.pad(x_knots, (1, 0), value=0.0) - B  # (N, K + 1)
    y_knots = torch.cumsum(heights, dim=-1)
    y_knots = F.pad(y_knots, (1, 0), value=0.0) - B  # (N, K + 1)

    # Snap the upper endpoint exactly to B to avoid cumulative-sum drift.
    x_knots = x_knots.clone()
    y_knots = y_knots.clone()
    x_knots[..., -1] = B
    y_knots[..., -1] = B

    # Inside-domain mask.
    if inverse:
        in_domain = (y >= -B) & (y <= B)
    else:
        in_domain = (y >= -B) & (y <= B)

    out = y.clone()
    log_det = torch.zeros_like(y)

    if not in_domain.any():
        return out, log_det

    y_in = y[in_domain]
    w_in = widths[in_domain]
    h_in = heights[in_domain]
    d_in = derivatives[in_domain]
    xk_in = x_knots[in_domain]
    yk_in = y_knots[in_domain]

    # Bin lookup.
    if inverse:
        # y_in lies in [y_knots[k], y_knots[k+1])
        bin_idx = torch.searchsorted(yk_in, y_in.unsqueeze(-1)).squeeze(-1) - 1
    else:
        bin_idx = torch.searchsorted(xk_in, y_in.unsqueeze(-1)).squeeze(-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)

    def gather(t):
        return t.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    def gather_next(t):
        return t.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

    w_k = gather(w_in)
    h_k = gather(h_in)
    x_k = gather(xk_in)
    y_k = gather(yk_in)
    d_k = gather(d_in)
    d_kp1 = gather_next(d_in)
    s_k = h_k / w_k  # slope of bin

    if inverse:
        # Solve for xi via quadratic formula.
        delta_u = y_in - y_k
        a = h_k * (s_k - d_k) + delta_u * (d_k + d_kp1 - 2 * s_k)
        b = h_k * d_k - delta_u * (d_k + d_kp1 - 2 * s_k)
        c = -s_k * delta_u

        discriminant = b * b - 4 * a * c
        discriminant = discriminant.clamp_min(0.0)
        xi = 2 * c / (-b - torch.sqrt(discriminant))
        xi = xi.clamp(0.0, 1.0)

        out_in = xi * w_k + x_k
        denom = s_k + (d_k + d_kp1 - 2 * s_k) * xi * (1 - xi)
        num_deriv = s_k**2 * (d_kp1 * xi**2 + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi) ** 2)
        # This is log|du/dy| evaluated at the forward spline; the inverse has
        # log|dy/du| = -log|du/dy|.
        log_det_in = 2 * torch.log(denom) - torch.log(num_deriv)
    else:
        xi = (y_in - x_k) / w_k
        xi = xi.clamp(0.0, 1.0)

        numerator = h_k * (s_k * xi**2 + d_k * xi * (1 - xi))
        denom = s_k + (d_k + d_kp1 - 2 * s_k) * xi * (1 - xi)
        out_in = y_k + numerator / denom

        num_deriv = s_k**2 * (d_kp1 * xi**2 + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi) ** 2)
        log_det_in = torch.log(num_deriv) - 2 * torch.log(denom)

    out[in_domain] = out_in
    log_det[in_domain] = log_det_in
    return out, log_det
