# Priors

## Bayesian layers and networks

::: implicit_process_zoo.priors.generative_functions.BayesLinear
    options:
      members:
        - forward

::: implicit_process_zoo.priors.generative_functions.BayesianNN
    options:
      members:
        - forward
        - freeze_parameters
        - defreeze_parameters

::: implicit_process_zoo.priors.generative_functions.GP
    options:
      members:
        - forward

## Coherent sampling

::: implicit_process_zoo.priors.function_bank.CoherentPriorFunctionSampler
    options:
      members:
        - sample_latents
        - evaluate_latents
        - sample_values

::: implicit_process_zoo.priors.function_bank.PriorFunctionBank
    options:
      members:
        - evaluate
