# Prior construction

The inference methods accept priors that can generate coherent function
samples. Coherence means that one sampled latent state represents the same
function at every input set, rather than resampling independent values for
each minibatch.

## Bayesian neural-network priors

`BayesianNN` composes stochastic `BayesLinear` layers. For a bank size $S$, its
output has shape `[S, N, D]`: prior functions, observations, and outputs.
Fixing layer noise reuses the same sampled weights across evaluations.

## Coherent sampling adapter

`CoherentPriorFunctionSampler` supports three prior styles:

1. an explicit `sample_latents` / `evaluate_latents` interface;
2. a collection of callable sampled functions; or
3. the repository's Bayesian-network modules, whose stochastic layer noise is
   captured and replayed.

`PriorFunctionBank` freezes a sampled latent bank and evaluates it at arbitrary
inputs. VIP/FTIP use sampled basis functions, while empirical GMVIP uses a bank
to estimate prior means and covariance operators.

## Gradient policy

Freezing a bank is different from freezing prior parameters. A fixed bank can
still carry gradients into a learnable prior unless detached. Experiment flags
make this distinction explicit; record prior-learning settings in reported
comparisons.
