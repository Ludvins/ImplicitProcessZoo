# Mean-Field Variational Inference

MFVI performs weight-space variational inference with independent Gaussian
marginals:

$$
q_\lambda(\theta)=\mathcal N(\theta;\mu,\operatorname{diag}(\sigma^2)),
\qquad
\theta^{(s)}=\mu+\sigma\odot\epsilon^{(s)},
\quad\epsilon^{(s)}\sim\mathcal N(0,I).
$$

The prior is a standard normal over the same flat weight vector. Its ELBO-style
negative objective is

$$
\mathcal L_{\mathrm{MFVI}}
=-\frac{N}{|B|}\sum_{i\in B}
\mathbb E_{q(\theta)}[\log p(y_i\mid f_\theta(x_i))]
+\operatorname{KL}(q_\lambda(\theta)\Vert p(\theta)).
$$

Fresh Monte Carlo weight noise is used during training. All posterior
uncertainty therefore originates in the diagonal weight distribution and is
propagated through the network.

[API reference](../api/models.md#implicit_process_zoo.mfvi.mfvi.MFVI)
