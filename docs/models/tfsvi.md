# Tractable Function-Space VI

TFSVI follows Rudner et al. (2022). It places a mean-field Gaussian over
parameters,

$$
q(\theta)=\mathcal N(\mu,\operatorname{diag}(\sigma^2)),
\qquad p(\theta)=\mathcal N(0,\sigma_p^2I),
$$

and linearizes the network at the variational mean:

$$
f(X_C;\theta)\approx f(X_C;\mu)+J_C(\theta-\mu).
$$

This yields Gaussian function approximations

$$
\widetilde q(f_C)=\mathcal N\!\left(
f(X_C;\mu),J_C\operatorname{diag}(\sigma^2)J_C^\top\right),
$$

$$
\widetilde p(f_C)=\mathcal N\!\left(
f(X_C;\mu)-J_C\mu,\sigma_p^2J_CJ_C^\top\right).
$$

The objective uses the largest context-set KL:

$$
\mathcal L_{\mathrm{TFSVI}}
=-\frac{N}{|B|}\mathcal A_B
+\max_{c=1,\ldots,S_{\mathrm{ctx}}}
\operatorname{KL}(\widetilde q(f_{C_c})\Vert\widetilde p(f_{C_c})).
$$

Scalar-output Gaussian KLs are summed across output dimensions. The base
network is an architecture template; trainable state lives in the flat
variational mean and log standard deviation.

[API reference](../api/models.md#implicit_process_zoo.tfsvi.tfsvi.TFSVI)
