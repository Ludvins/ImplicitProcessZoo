param(
  [string]$Python = "python",
  [string]$Device = "cuda",
  [int[]]$Seeds = @(0, 1, 2, 3, 4),
  [int]$Iterations = 30000
)

$ErrorActionPreference = "Stop"
$baselines = @("vip", "ftip", "mfvi", "fbnn", "tfsvi", "map")
$operators = @("rbf", "empirical")
$posteriors = @("gaussian", "realnvp")

foreach ($seed in $Seeds) {
  foreach ($model in $baselines) {
    & $Python scripts/uci_benchmark.py `
      --model $model `
      --dataset concrete `
      --seed $seed `
      --iterations $Iterations `
      --device $Device
  }
  foreach ($operator in $operators) {
    foreach ($posterior in $posteriors) {
      & $Python scripts/uci_benchmark.py `
        --model gmvip `
        --dataset concrete `
        --seed $seed `
        --iterations $Iterations `
        --device $Device `
        --gmvip_operator_type $operator `
        --gmvip_posterior_type $posterior
    }
  }
}
