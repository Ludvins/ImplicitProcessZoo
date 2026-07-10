"""Deterministic MLP that can be evaluated from a flat parameter vector.

Shared by methods that parameterize a neural network by a single flat weight
vector (e.g. TFSVI): an architecture template is built as a normal
``torch.nn.Module`` so PyTorch's init logic applies, then its parameters are
collected into (name, shape) lists and their actual values are supplied at
forward time via :func:`forward_with_flat_params`.
"""

import torch
import torch.nn as nn


class FlatMLP(nn.Module):
    """Deterministic MLP used as an architecture template.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    structure : list of int
        Hidden layer widths, e.g. ``[50, 50]``.
    activation : callable
        Activation module applied between inner layers.
    dtype : torch dtype
    device : torch.device or None
    """

    def __init__(
        self, input_dim, output_dim, structure, activation, dtype=torch.float64, device=None
    ):
        super().__init__()
        self.activation = activation
        dims = [input_dim] + list(structure) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(_in, _out, dtype=dtype, device=device) for _in, _out in zip(dims, dims[1:])]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


def collect_param_spec(base_net):
    """Return ``(names, shapes, total)`` for a network's trainable parameters.

    The parameters themselves are left untouched — the caller chooses whether
    to freeze them. The returned ``names`` / ``shapes`` can be fed to
    :func:`unflatten_params` to materialise a dict compatible with the net's
    forward pass.
    """
    names, shapes = [], []
    for name, p in base_net.named_parameters():
        names.append(name)
        shapes.append(p.shape)
    total = sum(s.numel() for s in shapes)
    return names, shapes, total


def unflatten_params(flat, names, shapes):
    """Reshape a flat parameter vector back into a ``{name: tensor}`` dict."""
    params = {}
    offset = 0
    for name, shape in zip(names, shapes):
        numel = shape.numel()
        params[name] = flat[offset : offset + numel].reshape(shape)
        offset += numel
    return params


def forward_with_flat_params(base_net, flat_params, x):
    """Forward pass through ``base_net`` using ``flat_params`` instead of its own weights.

    Assumes ``base_net`` is a :class:`FlatMLP` (or structurally identical): a
    ``ModuleList`` of ``nn.Linear`` layers under ``layers``, with an
    ``activation`` attribute applied between all but the last layer.

    Uses ``nn.Linear`` convention: ``output = x @ weight.T + bias``. Avoids
    ``torch.func.functional_call`` for broad PyTorch compatibility.
    """
    names, shapes, _ = collect_param_spec(base_net)
    params = unflatten_params(flat_params, names, shapes)
    num_layers = len(base_net.layers)
    activation = base_net.activation
    for i in range(num_layers - 1):
        w = params[f"layers.{i}.weight"]
        b = params[f"layers.{i}.bias"]
        x = activation(x @ w.T + b)
    i = num_layers - 1
    w = params[f"layers.{i}.weight"]
    b = params[f"layers.{i}.bias"]
    return x @ w.T + b
