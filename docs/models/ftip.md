# Flow-Transformed Implicit Processes

FTIP retains VIP's finite sampled-function basis and replaces its Gaussian
coefficient posterior with an invertible normalizing flow. Let

$$
z\sim\mathcal N(0,I),\qquad a=T_\phi(z).
$$

The transformed density is

$$
\log q_\phi(a)=\log p(z)-
\log\left|\det\frac{\partial T_\phi(z)}{\partial z}\right|.
$$

The posterior process remains

$$
f(X)=m(X)+\Phi(X)^\top a,
$$

but $q(a)$ can be non-Gaussian. The implementation supports affine coupling,
rational-quadratic spline coupling, and spline coupling with Glow-style
invertible $1\times1$ mixing.

Its Monte Carlo objective is

$$
\mathcal L_{\mathrm{FTIP}}
=-\frac{N}{|B|}\mathcal A_B
+\mathbb E_{a\sim q_\phi}
[\log q_\phi(a)-\log\mathcal N(a;0,I)].
$$

Antithetic coefficient sampling returns exactly the requested positive sample
count for both odd and even values. FTIP can warm-start from a trained VIP by
initializing an affine flow from VIP's Gaussian coefficient posterior. Disable
automatic warm start when the reported budget must count only FTIP optimizer
steps.

[API reference](../api/models.md#implicit_process_zoo.ftip.ftip.FTIP)
