# GMVIP components

## Kernel and inducing initialization

::: implicit_process_zoo.gmvip.kernels.RBFKernel

::: implicit_process_zoo.gmvip.inducing.initialize_inducing_points

## Matheron operators

::: implicit_process_zoo.gmvip.operators.BaseMatheronOperator
    options:
      members:
        - psi
        - inducing_mean
        - inducing_scale_matrix
        - apply

::: implicit_process_zoo.gmvip.operators.EmpiricalCovarianceMatheronOperator
    options:
      members:
        - psi
        - inducing_mean
        - inducing_scale_matrix
        - mean_at

::: implicit_process_zoo.gmvip.operators.RBFCardinalMatheronOperator
    options:
      members:
        - psi
        - inducing_mean
        - inducing_scale_matrix
        - mean_at

## Coefficient posteriors

::: implicit_process_zoo.gmvip.posteriors.CholeskyGaussianCoefficientPosterior

::: implicit_process_zoo.gmvip.posteriors.RealNVPCoefficientPosterior

## Likelihood

::: implicit_process_zoo.gmvip.likelihoods.GaussianRegressionLikelihood
