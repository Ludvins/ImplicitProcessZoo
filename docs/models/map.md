# MAP

MAP is the deterministic calibration baseline. It uses an MLP $f_\theta(x)$
and learns an isotropic Gaussian observation variance $\sigma_y^2$. There is
no posterior over functions or weights.

The optimized objective is

$$
\mathcal L_{\mathrm{MAP}}(\theta,\sigma_y^2)
=-\frac{N}{|B|}\sum_{i\in B}
\log\mathcal N(y_i;f_\theta(x_i),\sigma_y^2)
+\frac{\lambda}{2}\lVert\theta\rVert_2^2.
$$

The predictive distribution is

$$
p(y_*\mid x_*,\mathcal D)
\approx\mathcal N(f_\theta(x_*),\sigma_y^2).
$$

`predict_f_samples` repeats the deterministic latent prediction for the
requested sample count; `predict_y_samples` draws independent likelihood
noise. The method is useful for measuring whether posterior uncertainty
improves calibration or predictive scores beyond a learned-noise point
estimate.

[API reference](../api/models.md#implicit_process_zoo.map_baseline.DeterministicMAP)
