# Synthetic Plot Runner

This runner trains models on the fixed synthetic dataset and creates
publication-style predictive plots.

See the [synthetic experiment guide](https://ludvins.github.io/ImplicitProcessZoo/experiments/synthetic/)
for commands and artifact behavior.

```bash
python -m experiments.synthetic.plot --help
```

HMC is an explicit optional reference and is not included by plain
`--models all`:

```bash
python -m pip install -e ".[experiments,hmc]"
python -m experiments.synthetic.plot --models gmvip hmc
```

The standard HMC protocol copies the transition and prediction settings from
BayesiPy's `Synthetic_1D_regression.ipynb`: one float64 Hamiltorch chain on
CUDA, 1,000 draws, no burn-in, step size `0.0005`, 500 leapfrog steps, and
fixed diagonal inverse mass `0.1`. It retains this repository's existing
1-10-10-1 BNN posterior and MAP-based initialization, samples normalized
observation noise jointly with the weights, and renders the result with the
same sample-band panel style as the other synthetic methods. The one-chain
reference is not convergence-certified.
