param(
  [string]$Python = "python",
  [string]$Device = "cuda",
  [int]$Seed = 0,
  [int]$Iterations = 30000
)

$ErrorActionPreference = "Stop"
$datasets = @("boston", "energy")
$models = @("gmvip", "vip", "ftip", "mfvi", "fbnn", "tfsvi", "map")

foreach ($dataset in $datasets) {
  foreach ($model in $models) {
    & $Python scripts/uci_benchmark.py `
      --model $model `
      --dataset $dataset `
      --seed $Seed `
      --iterations $Iterations `
      --device $Device
  }
}
