# Concepts

An implicit process is a distribution over functions that is easy to sample
but may not have a tractable finite-dimensional density. A Bayesian neural
network is the central example: sample its weights and evaluate the network to
obtain a coherent function draw.

This repository studies variational approximations that operate in weight,
function, coefficient, or inducing-variable space while sharing a common
likelihood and prediction interface.

For scalar regression, let

$$
\mathcal D = \{(x_i,y_i)\}_{i=1}^{N}, \qquad
p(y_i\mid f_i)=\mathcal N(y_i;f_i,\sigma_y^2).
$$

The model-specific pages explain how each method represents $q(f)$ and its
regularizer. The remaining concept pages cover the shared
[objectives](objectives.md), [prior construction](priors.md),
[prediction contract](prediction.md), and
[reproducibility rules](reproducibility.md).
