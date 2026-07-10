# Models

The suite spans deterministic, weight-space, sampled-feature, and
inducing-variable approximations. Every model is a PyTorch module and the
supported evaluation interface is the common sample-first prediction contract.

| Model | Variational object | Regularizer |
| --- | --- | --- |
| MAP | Network parameters | $L_2$ penalty |
| MFVI | Diagonal Gaussian weights | Analytic weight-space KL |
| FBNN | Bayesian neural-network functions | Score-based functional KL |
| TFSVI | Linearized Gaussian functions | Analytic functional Gaussian KL |
| VIP | Gaussian basis coefficients | Analytic coefficient KL |
| FTIP | Flow-transformed coefficients | Monte Carlo coefficient KL |
| GMVIP | Inducing coefficients | Gaussian or flow coefficient KL |
| SIP | Implicit inducing values | Critic-estimated density-ratio KL |

The mathematical pages describe the implementations, not a promise that every
model supports every likelihood or experiment family. Consult runner `--help`
output for compatible combinations.
