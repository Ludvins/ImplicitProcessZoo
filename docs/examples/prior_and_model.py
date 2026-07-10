"""Construct a Bayesian neural-network prior and a VIP posterior."""

import torch

from implicit_process_zoo.priors.generative_functions import BayesianNN, BayesLinear
from implicit_process_zoo.vip import VIP

device = torch.device("cpu")
dtype = torch.float64

prior = BayesianNN(
    structure=[8],
    activation=torch.tanh,
    num_samples=8,
    input_dim=1,
    output_dim=1,
    layer_model=BayesLinear,
    fix_random_noise=True,
    zero_mean_prior=True,
    seed=7,
    device=device,
    dtype=dtype,
)

model = VIP(
    generative_function=prior,
    num_regression_coeffs=8,
    output_dim=1,
    likelihood="regression",
    num_data=16,
    seed=11,
    device=device,
    dtype=dtype,
)

x = torch.linspace(-1.0, 1.0, 16, dtype=dtype).unsqueeze(-1)
function_samples = model.predict_f_samples(x, 4, seed=23)
assert function_samples.shape == (4, 16, 1)
