# Curated API reference

This reference documents the supported library-level interfaces used by the
experiment runners. It intentionally omits internal optimization hooks, legacy
dataset classes, and experiment-only helpers.

- [Models](models.md): the eight inference implementations.
- [Priors](priors.md): coherent prior generators and function banks.
- [GMVIP components](gmvip.md): kernels, operators, likelihoods, and coefficient posteriors.
- [Flows](flows.md): coefficient transformations.
- [Data registry](data.md): canonical dataset lookup.
- [Training](training.md), [prediction](prediction.md), and
  [checkpoints](checkpoints.md): shared utilities.

Signatures and source links are generated from the installed package by
mkdocstrings. Documentation pages describe stable usage; helpers not listed in
this section should be treated as implementation details.
