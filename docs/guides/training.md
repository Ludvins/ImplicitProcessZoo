# Training

Models expose a `fit` method backed by the shared `fit_loop`. Specify exactly
one duration:

```python
losses = model.fit(loader, iterations=1000, return_loss=True)
# or
losses = model.fit(loader, epochs=50, return_loss=True)
```

Providing both `epochs` and `iterations`, neither, or a nonpositive value raises
`ValueError`. The cosine scheduler retains epoch-based semantics: in iteration
mode it steps after each loader-length block, and its effective period is
clamped to at least one.

The shared loop moves minibatches to the model device and calls the model's
`_train_step` hook. FBNN uses a preparation hook for its measurement context;
SIP uses a before-step hook for critic updates. These hooks are implementation
details, not external extension points.

## Example

```py
--8<-- "docs/examples/train_and_predict.py"
```

Experiment runners are preferred for reported results because they also
capture arguments, data splits, metrics, checkpoints, and comparison files.
