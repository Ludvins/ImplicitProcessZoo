param(
  [string]$Python = "python",
  [string]$Device = "cuda",
  [int[]]$Seeds = @(0, 1, 2, 3, 4),
  [int]$Iterations = 30000
)

$ErrorActionPreference = "Stop"
$operators = @("rbf", "empirical")
$posteriors = @("gaussian", "realnvp")

foreach ($seed in $Seeds) {
  foreach ($operator in $operators) {
    foreach ($posterior in $posteriors) {
      & $Python scripts/uci_benchmark.py `
        --model gmvip `
        --dataset concrete `
        --seed $seed `
        --iterations $Iterations `
        --device $Device `
        --gmvip_operator_type $operator `
        --gmvip_posterior_type $posterior `
        --gmvip_learn_Z `
        --gmvip_learn_prior
    }
  }
}
