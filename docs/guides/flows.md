# Normalizing flows

FTIP and flow-based GMVIP coefficient posteriors transform a standard Gaussian
latent through invertible maps. The repository provides:

- `AffineLayer` for a learned diagonal affine transform;
- affine coupling through `CouplingLayer` and `CouplingFlow`;
- rational-quadratic spline coupling through `SplineCouplingLayer` and
  `SplineCouplingFlow`; and
- spline coupling interleaved with Glow-style invertible $1\times1$ mixing
  through `SplineCoupling1x1Flow`.

Forward methods return transformed coefficients and a log-Jacobian term used
in the density change of variables. Experiment flow factories expose depth,
initial scale, spline bins, and mixing choices as runner arguments.

FTIP can initialize an affine component from VIP's Gaussian coefficient
posterior. This warm start changes the total training procedure and must be
reported explicitly in equal-budget comparisons.

See the [flow API](../api/flows.md) for source-backed signatures.
