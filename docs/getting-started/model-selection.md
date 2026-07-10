# Choose a model

No method dominates every task. Use the simplest method that represents the
uncertainty and prior structure required by the experiment.

| Goal | Suggested starting point | Why |
| --- | --- | --- |
| Establish a deterministic baseline | MAP | Fast and easy to interpret |
| Standard Bayesian neural-network baseline | MFVI | Familiar diagonal weight posterior |
| Enforce a functional prior with a BNN posterior | FBNN | Direct score-based functional regularization |
| Use a tractable Gaussian function approximation | TFSVI | Analytic linearized function-space KL |
| Reuse samples from an implicit prior | VIP | Compact finite coefficient posterior |
| Model a non-Gaussian coefficient posterior | FTIP | Normalizing flows on VIP coefficients |
| Condition prior samples through inducing points | GMVIP | Matheron update with empirical or RBF operator |
| Keep the inducing posterior implicit | SIP | Neural sampler and critic-estimated KL |

## Practical trade-offs

- **MAP and MFVI** are useful calibration baselines and have relatively small
  memory footprints.
- **FBNN and TFSVI** work directly with a function-space regularizer, but their
  measurement/context sets add computation.
- **VIP and FTIP** depend on the number of sampled prior features. FTIP adds
  expressiveness and flow-optimization cost.
- **GMVIP and SIP** control complexity through inducing inputs. GMVIP keeps an
  explicit latent coefficient density; SIP uses adversarial density-ratio
  estimation.

Use the same data splits, optimization budgets, and evaluation sample counts
when comparing methods. The experiment presets encode the intended scientific
protocols and should be preferred over ad hoc flag combinations for reported
results.
