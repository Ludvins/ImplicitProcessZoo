# Generalized Matheron VIP

GMVIP applies an inducing-point Matheron update to a coherent prior draw
$g(\cdot)$:

$$
f(X)=g(X)+\Psi_Z(X)[\mu_Z+D_Za-g(Z)].
$$

At the inducing inputs,

$$
u=f(Z)=\mu_Z+D_Za,
$$

and the latent prior is $p(a)=\mathcal N(0,I)$. The Gaussian posterior uses
$q(a)=\mathcal N(m_a,L_aL_a^\top)$; the RealNVP option transforms a standard
normal and uses the same latent-variable KL form as FTIP.

## Empirical operator

The empirical cardinal operator estimates moments from a coherent function
bank evaluated jointly at $[X;Z]$:

$$
\Psi_Z(X)=K_{XZ}^{\mathrm{emp}}
(K_{ZZ}^{\mathrm{emp}})^{-1}.
$$

Its inducing mean is the empirical prior mean and $D_Z$ is a stabilized
empirical Cholesky factor.

## RBF operator

The RBF cardinal operator uses

$$
\Psi_Z(X)=K_{\mathrm{RBF}}(X,Z)
K_{\mathrm{RBF}}(Z,Z)^{-1}.
$$

Lengthscales and output scale may be learned or fixed. The canonical RBF
setting uses $\mu_Z=0$ and
$D_Z=\operatorname{chol}(K_{\mathrm{RBF}}(Z,Z))$.

The maximized objective is

$$
\operatorname{ELBO}
=\mathbb E_{q(a),g}[\log p(y_B\mid f_B)]
-\beta\operatorname{KL}(q(a)\Vert\mathcal N(0,I)),
$$

with minibatch likelihood scaling and an optional BB-alpha term. Only the
latent coefficient KL is used.

[API reference](../api/models.md#implicit_process_zoo.gmvip.gmvip.GeneralizedMatheronVIP)
