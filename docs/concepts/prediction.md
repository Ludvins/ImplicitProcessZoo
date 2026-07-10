# Prediction contract

Every supported inference model exposes sample-first tensor methods:

```python
predict_f_samples(x, num_samples, *, seed=None)  # [S, N, D]
predict_y_samples(x, num_samples, *, seed=None)  # [S, N, D]
```

- `S` is the requested posterior sample count.
- `N` is the number of observations in `x`.
- `D` is the output dimension.

Function samples represent latent $f(x)$. Observation samples additionally
include likelihood noise for regression. Classification models return their
latent/sample representation according to the implemented likelihood.

The optional seed makes repeated calls reproducible without permanently
changing the model-owned generator. The shared batched utility concatenates
along `N`, including when the final minibatch is smaller, and passes the same
seed to every batch so pathwise methods reuse one function draw.

Legacy `forward()` return shapes vary for historical and optimization reasons;
external evaluation code should use the two explicit prediction methods.
