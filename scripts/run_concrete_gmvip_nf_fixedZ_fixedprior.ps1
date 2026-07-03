param(
  [string]$Python = "python",
  [string]$Device = "cuda",
  [int[]]$Seeds = @(0, 1, 2, 3, 4),
  [int]$Iterations = 30000
)

$ErrorActionPreference = "Stop"

foreach ($seed in $Seeds) {
  & $Python scripts/uci_benchmark.py `
    --model gmvip `
    --dataset concrete `
    --seed $seed `
    --iterations $Iterations `
    --device $Device `
    --gmvip_operator_type rbf `
    --gmvip_posterior_type realnvp `
    --no-gmvip_learn_Z `
    --no-gmvip_learn_prior
}
