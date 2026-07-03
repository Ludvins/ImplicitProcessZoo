param(
  [string]$Python = "python",
  [string[]]$Datasets = @("boston", "energy", "concrete"),
  [string]$Device = "cuda",
  [int]$Seed = 0,
  [int]$Iterations = 30000
)

$ErrorActionPreference = "Stop"

foreach ($dataset in $Datasets) {
  & $Python scripts/uci_benchmark.py `
    --model gmvip `
    --dataset $dataset `
    --seed $Seed `
    --iterations $Iterations `
    --device $Device `
    --gmvip_operator_type rbf `
    --gmvip_posterior_type gaussian `
    --gmvip_num_inducing 100 `
    --gmvip_num_operator_bank_samples 512 `
    --gmvip_num_train_samples 512 `
    --gmvip_num_eval_samples 512
}
