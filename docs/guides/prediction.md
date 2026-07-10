# Batched prediction

Use explicit sample methods rather than model-specific `forward()` outputs:

```python
f = model.predict_f_samples(x, 256, seed=0)
y = model.predict_y_samples(x, 256, seed=0)
```

Both tensors have shape `[samples, observations, outputs]`.

For a loader or iterable of batches, use the shared utility:

```python
from implicit_process_zoo.utils import batched_predict_samples

y = batched_predict_samples(
    model,
    test_loader,
    256,
    kind="y",
    device="cuda",
    seed=0,
)
```

Results are moved to CPU and concatenated along the observation axis. Uneven
final minibatches are supported. A shared seed is passed to each batch so
pathwise models evaluate the same posterior functions over all observations.

Use `kind="f"` to exclude observation noise. A nonpositive sample count, empty
batch iterable, missing prediction method, or invalid model return shape raises
an explicit error.
