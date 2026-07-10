# Objectives

Most variational models minimize a minibatch-scaled negative objective

$$
\mathcal L = -\frac{N}{|B|}\mathcal A_B + \mathcal R,
$$

where $\mathcal A_B$ measures data fit and $\mathcal R$ is the method-specific
KL or regularizer.

## Expected log likelihood

With `--bb_alpha 0`, the data term is the expected log likelihood:

$$
\mathcal A_B = \sum_{i\in B}\mathbb E_q[\log p(y_i\mid f_i)].
$$

## BB-alpha energy

For nonzero $\alpha$, the implementations use the Monte Carlo BB-alpha energy

$$
\mathcal A_B = \sum_{i\in B}\frac{1}{\alpha}
\log\left[\frac{1}{S}\sum_{s=1}^{S}
\exp\{\alpha\log p(y_i\mid f_i^{(s)})\}\right].
$$

Its $\alpha\to0$ limit is the expected log likelihood. Classification follows
the same structure with Bernoulli or multiclass likelihood terms where the
method supports them.

## Comparison discipline

An equal optimizer-step count does not always imply equal total work. Context
construction, prior banks, flow warm starts, critic updates, and posterior
sample counts all affect cost. Report the complete preset/configuration and
runtime metadata with metrics rather than comparing only the final objective.
