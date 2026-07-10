# UCI regression

`experiments.uci.benchmark` is the main scalar-regression benchmark. It builds
models from the library, loads and normalizes a dataset, trains one or more
methods, and writes metrics and optional checkpoints.

## Datasets and methods

The benchmark datasets are:

```text
boston, concrete, energy, kin8nm, naval, power, protein, winered, yacht
```

Supported methods are `map`, `mfvi`, `fbnn`, `tfsvi`, `vip`, `ftip`, `gmvip`,
and `sip`. Use `--model all` or `--dataset all` to expand a comparison. The
historical spelling `yatch` remains a deprecated CLI alias.

## Commands

```bash
python -m experiments.uci.benchmark --model gmvip --dataset concrete
python -m experiments.uci.benchmark --model all --dataset boston
```

A representative explicit configuration is:

```bash
python -m experiments.uci.benchmark \
  --model vip \
  --dataset boston \
  --seed 0 \
  --iterations 30000 \
  --bb_alpha 0 \
  --batch_size 100 \
  --lr 0.001 \
  --hidden_dims 10 10 \
  --activation tanh \
  --layer_model BayesLinear \
  --device cuda
```

Prior learning is explicit for the sampled-prior methods:

```text
--vip_learn_prior / --no-vip_learn_prior
--ftip_learn_prior / --no-ftip_learn_prior
--gmvip_learn_prior / --no-gmvip_learn_prior
--sip_learn_prior / --no-sip_learn_prior
```

Direct FTIP runs enable VIP warm-starting by default. Pass
`--no_auto_warm_start` for comparisons where FTIP must receive exactly the
requested number of its own optimizer steps.

## GMVIP configuration

The Matheron form is

$$
f(X)=g(X)+\Psi_Z(X)[\mu_Z+D_Za-g(Z)].
$$

The runner supports empirical and RBF operators, Gaussian and RealNVP
coefficient posteriors, learnable inducing locations, antithetic sampling, and
tunable prior BNNs. A representative empirical run is:

```bash
python -m experiments.uci.benchmark \
  --model gmvip --dataset concrete \
  --iterations 30000 --bb_alpha 0 \
  --gmvip_operator_type empirical \
  --gmvip_posterior_type gaussian \
  --gmvip_num_inducing 100 \
  --gmvip_inducing_method kmeans \
  --gmvip_learn_Z \
  --gmvip_num_operator_bank_samples 512 \
  --gmvip_num_train_samples 512 \
  --gmvip_num_eval_samples 512 \
  --gmvip_mean_mode prior_sample \
  --gmvip_inducing_scale prior_cholesky \
  --gmvip_jitter 0.001 \
  --gmvip_max_log_noise none \
  --gmvip_learn_prior
```

Use `--help` for the complete current flag set. Results default to
`results/uci`; individual jobs write JSON results and multi-job expansions add
JSON/CSV comparisons. Checkpoints are saved unless `--no_save_checkpoint` is
passed.
