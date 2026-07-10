# Variational Implicit Processes

VIP follows Ma et al. (2019). Given coherent prior-function samples $g_s$, it
constructs a finite basis

$$
m(X)=\frac{1}{S}\sum_{s=1}^{S}g_s(X),
\qquad
\Phi_s(X)=\frac{g_s(X)-m(X)}{\sqrt{S-1}}.
$$

The coefficient posterior is a full-covariance Gaussian

$$
q(a)=\mathcal N(q_\mu,L_qL_q^\top),
\qquad p(a)=\mathcal N(0,I),
$$

and the induced process is

$$
f(X)=m(X)+\Phi(X)^\top a.
$$

For regression its finite-dimensional moments are analytic:

$$
\mathbb E_q[f(X)]=m(X)+\Phi(X)^\top q_\mu,
$$

$$
\operatorname{Cov}_q[f(X)]
=\Phi(X)^\top L_qL_q^\top\Phi(X).
$$

The negative objective is

$$
\mathcal L_{\mathrm{VIP}}
=-\frac{N}{|B|}\mathbb E_{q(f_B)}[\log p(y_B\mid f_B)]
+\operatorname{KL}(q(a)\Vert\mathcal N(0,I)).
$$

VIP is finite-dimensional in coefficient space but function-valued after
composition with the prior basis. Experiments explicitly control whether the
prior BNN is frozen or learned.

[API reference](../api/models.md#implicit_process_zoo.vip.vip.VIP)
