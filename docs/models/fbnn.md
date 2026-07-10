# Functional Bayesian Neural Network

FBNN follows Sun et al. (2019). It keeps a trainable Bayesian neural-network
posterior but regularizes it in function space against a GP or BNN prior.

At each step it forms a measurement set

$$
X_M=X_{\mathrm{context}}\cup X_{\mathrm{train\ subset}}
$$

and obtains posterior samples $f_M^{(s)}=f^{(s)}(X_M)$. The functional KL is
optimized through a score-based surrogate:

$$
\operatorname{KL}(q(f_M)\Vert p_0(f_M))
\Longrightarrow
\mathbb E_{q(f_M)}\!\left[
f_M^\top\bigl(\nabla_{f_M}\log q(f_M)-
\nabla_{f_M}\log p_0(f_M)\bigr)\right].
$$

The posterior score is estimated with a spectral Stein gradient estimator
(SSGE). For a Gaussian-process prior, the prior score is analytic:

$$
\nabla_f\log p_0(f_M)
=-(K_{MM}+\gamma^2I)^{-1}(f_M-m_M).
$$

The training objective is

$$
\mathcal L_{\mathrm{FBNN}}
=-\frac{N}{|B|}\mathcal A_B
+\lambda_{\mathrm{KL}}\widehat{\operatorname{KL}}_{\mathrm{func}}.
$$

Context points constrain the posterior toward the prior away from observed
training inputs. The shared fit loop calls FBNN's preparation hook once to
construct the required context state.

[API reference](../api/models.md#implicit_process_zoo.fbnn.fbnn.FBNN)
