param(
  [string]$Python = "python",
  [string]$Device = "cuda",
  [int[]]$Seeds = @(0, 1, 2, 3, 4),
  [int]$Iterations = 30000
)

$ErrorActionPreference = "Stop"
$states = @(
  @{ LearnZ = $true; LearnPrior = $false },
  @{ LearnZ = $false; LearnPrior = $true },
  @{ LearnZ = $true; LearnPrior = $true }
)

foreach ($seed in $Seeds) {
  foreach ($state in $states) {
    $zFlag = if ($state.LearnZ) { "--gmvip_learn_Z" } else { "--no-gmvip_learn_Z" }
    $priorFlag = if ($state.LearnPrior) { "--gmvip_learn_prior" } else { "--no-gmvip_learn_prior" }
    & $Python scripts/uci_benchmark.py `
      --model gmvip `
      --dataset concrete `
      --seed $seed `
      --iterations $Iterations `
      --device $Device `
      --gmvip_operator_type rbf `
      --gmvip_posterior_type realnvp `
      $zFlag `
      $priorFlag
  }
}
