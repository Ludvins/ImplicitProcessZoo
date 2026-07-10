# Sparse Implicit Process

SIP follows Rodriguez Santana et al. (2022). It introduces inducing variables
$u=f(Z)$ and the sparse factorization

$$
q_\phi(f_X,u)=p_\theta(f_X\mid u)q_\phi(u).
$$

The inducing posterior is implicit:

$$
\xi\sim\mathcal N(0,I),\qquad
\epsilon=m_\epsilon+exp(\tfrac12\ell_\epsilon)\odot\xi,
\qquad u=h_\phi(\epsilon).
$$

Coherent prior samples at $[X;Z]$ estimate $m_X,m_Z,K_{ZZ},K_{XZ}$ and
$K_{XX}$. These moments define

$$
\mu_{X\mid u}=m_X+K_{XZ}K_{ZZ}^{-1}(u-m_Z),
$$

$$
\Sigma_{X\mid u}=K_{XX}-K_{XZ}K_{ZZ}^{-1}K_{ZX}.
$$

Training uses the diagonal conditional for efficient minibatch likelihood
terms; prediction uses the full conditional covariance. Both paths use
jittered Cholesky solves rather than explicit matrix inverses.

Because both $q_\phi(u)$ and $p_\theta(u)$ may be implicit, a critic estimates
their log density ratio. It minimizes

$$
-\frac12\left[
\mathbb E_{q_\phi(u)}\log\sigma(T_\omega(u))
+\mathbb E_{p_\theta(u)}\log(1-\sigma(T_\omega(u)))
\right].
$$

At the optimum, $T_{\omega^*}(u)=\log q_\phi(u)-\log p_\theta(u)$. The primal
update uses the symmetrized estimate

$$
\mathcal R_{\mathrm{KL}}\approx\frac12\left(
\mathbb E_q[T_{\omega^*}(u)]-
\mathbb E_p[T_{\omega^*}(u)]\right).
$$

The critic is optimized separately before each primal update. Training logs
its loss, accuracy, saturation, forward/reverse KL estimates, and active beta.
The adversarial estimation cost is controlled by the inducing dimension;
predictions at arbitrary inputs are recovered through the sparse conditional.

[API reference](../api/models.md#implicit_process_zoo.sip.sip.SIP)
