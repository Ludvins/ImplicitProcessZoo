"""Launch the comparable 8-method UCI sweep.

The sweep is intentionally explicit: every command receives its W&B name and
group so the online project groups contain only the five seeds for a single
dataset/method variant.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("concrete", "boston", "energy")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
OUTPUT_DIR = "results/uci_comparable_8_methods_30k"
LOG_DIR = REPO / "outputs" / "wandb_logs" / "uci_comparable_8_methods_30k"


@dataclass(frozen=True)
class Variant:
    label: str
    model: str
    extra_args: tuple[str, ...]


VARIANTS: tuple[Variant, ...] = (
    Variant("MAP", "map", ()),
    Variant("MFVI", "mfvi", ("--regression_coeffs", "512", "--mfvi_num_eval_samples", "512")),
    Variant(
        "FBNN",
        "fbnn",
        (
            "--fbnn_prior",
            "bnn",
            "--fbnn_freeze_prior",
            "--regression_coeffs",
            "512",
            "--fbnn_num_eval_samples",
            "512",
        ),
    ),
    Variant(
        "TFSVI",
        "tfsvi",
        ("--tfsvi_num_train_samples", "512", "--tfsvi_num_eval_samples", "512"),
    ),
    Variant("VIP Tunable Prior", "vip", ("--regression_coeffs", "20", "--vip_learn_prior")),
    Variant("VIP Fixed Prior", "vip", ("--regression_coeffs", "20", "--no-vip_learn_prior")),
    Variant(
        "FTIP Tunable Prior",
        "ftip",
        (
            "--regression_coeffs",
            "20",
            "--num_samples",
            "512",
            "--eval_samples",
            "512",
            "--flow_type",
            "spline_1x1",
            "--no_auto_warm_start",
            "--ftip_learn_prior",
        ),
    ),
    Variant(
        "FTIP Fixed Prior",
        "ftip",
        (
            "--regression_coeffs",
            "20",
            "--num_samples",
            "512",
            "--eval_samples",
            "512",
            "--flow_type",
            "spline_1x1",
            "--no_auto_warm_start",
            "--no-ftip_learn_prior",
        ),
    ),
    Variant(
        "GMVIP Tunable Prior",
        "gmvip",
        (
            "--gmvip_operator_type",
            "empirical",
            "--gmvip_posterior_type",
            "gaussian",
            "--gmvip_num_inducing",
            "100",
            "--gmvip_inducing_method",
            "kmeans",
            "--gmvip_learn_Z",
            "--gmvip_num_train_samples",
            "512",
            "--gmvip_num_eval_samples",
            "512",
            "--gmvip_num_operator_bank_samples",
            "512",
            "--gmvip_beta",
            "1",
            "--gmvip_beta_warmup_steps",
            "0",
            "--gmvip_data_alpha",
            "0",
            "--gmvip_max_log_noise",
            "none",
            "--gmvip_jitter",
            "0.001",
            "--gmvip_mean_mode",
            "prior_sample",
            "--gmvip_inducing_scale",
            "prior_cholesky",
            "--gmvip_learn_prior",
        ),
    ),
    Variant(
        "GMVIP Fixed Prior",
        "gmvip",
        (
            "--gmvip_operator_type",
            "empirical",
            "--gmvip_posterior_type",
            "gaussian",
            "--gmvip_num_inducing",
            "100",
            "--gmvip_inducing_method",
            "kmeans",
            "--gmvip_learn_Z",
            "--gmvip_num_train_samples",
            "512",
            "--gmvip_num_eval_samples",
            "512",
            "--gmvip_num_operator_bank_samples",
            "512",
            "--gmvip_beta",
            "1",
            "--gmvip_beta_warmup_steps",
            "0",
            "--gmvip_data_alpha",
            "0",
            "--gmvip_max_log_noise",
            "none",
            "--gmvip_jitter",
            "0.001",
            "--gmvip_mean_mode",
            "prior_sample",
            "--gmvip_inducing_scale",
            "prior_cholesky",
            "--no-gmvip_learn_prior",
        ),
    ),
    Variant(
        "SIP Tunable Prior",
        "sip",
        (
            "--sip_num_inducing",
            "100",
            "--sip_inducing_method",
            "kmeans",
            "--sip_learn_inducing",
            "--sip_num_prior_samples",
            "512",
            "--sip_num_train_samples",
            "512",
            "--sip_num_eval_samples",
            "512",
            "--no-sip_fix_random_noise",
            "--sip_jitter",
            "1e-5",
            "--sip_beta",
            "1",
            "--sip_beta_warmup_steps",
            "0",
            "--sip_learn_prior",
        ),
    ),
    Variant(
        "SIP Fixed Prior",
        "sip",
        (
            "--sip_num_inducing",
            "100",
            "--sip_inducing_method",
            "kmeans",
            "--sip_learn_inducing",
            "--sip_num_prior_samples",
            "512",
            "--sip_num_train_samples",
            "512",
            "--sip_num_eval_samples",
            "512",
            "--no-sip_fix_random_noise",
            "--sip_jitter",
            "1e-5",
            "--sip_beta",
            "1",
            "--sip_beta_warmup_steps",
            "0",
            "--no-sip_learn_prior",
        ),
    ),
)


def pretty_dataset(dataset: str) -> str:
    return {"boston": "Boston", "concrete": "Concrete", "energy": "Energy"}.get(
        dataset,
        dataset.title(),
    )


def parse_ints(values: list[str] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return default
    return tuple(int(value) for value in values)


def select_variants(labels: list[str] | None) -> tuple[Variant, ...]:
    if not labels:
        return VARIANTS
    variants_by_label = {variant.label.lower(): variant for variant in VARIANTS}
    selected: list[Variant] = []
    unknown: list[str] = []
    for label in labels:
        variant = variants_by_label.get(label.lower())
        if variant is None:
            unknown.append(label)
        else:
            selected.append(variant)
    if unknown:
        valid = ", ".join(variant.label for variant in VARIANTS)
        raise ValueError(f"Unknown variant label(s): {unknown}. Valid labels: {valid}")
    return tuple(selected)


def build_command(python: str, dataset: str, seed: int, variant: Variant) -> list[str]:
    dataset_label = pretty_dataset(dataset)
    group = f"{dataset_label} | {variant.label}"
    name = f"{group} | seed {seed}"
    return [
        python,
        "-m",
        "scripts.uci_benchmark",
        "--model",
        variant.model,
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--iterations",
        "30000",
        "--bb_alpha",
        "0",
        "--batch_size",
        "100",
        "--lr",
        "0.001",
        "--hidden_dims",
        "10",
        "10",
        "--activation",
        "tanh",
        "--layer_model",
        "BayesLinear",
        "--device",
        "cuda",
        "--output_dir",
        OUTPUT_DIR,
        "--wandb",
        "--wandb_mode",
        "online",
        "--wandb_project",
        "apfsvi",
        "--wandb_entity",
        "ludvins",
        "--wandb_group",
        group,
        "--wandb_name",
        name,
        *variant.extra_args,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="*", default=[str(seed) for seed in DEFAULT_SEEDS])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="*",
        help="Optional exact variant labels to run, e.g. 'SIP Tunable Prior'.",
    )
    args = parser.parse_args()

    seeds = parse_ints(args.seeds, DEFAULT_SEEDS)
    variants = select_variants(args.variants)
    commands: list[tuple[str, list[str]]] = []
    for dataset in args.datasets:
        for seed in seeds:
            for variant in variants:
                label = f"{pretty_dataset(dataset)}__{variant.label.replace(' ', '_')}__seed{seed}"
                commands.append((label, build_command(args.python, dataset, seed, variant)))

    selected = commands[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    runtime = REPO / "outputs" / "wandb_runtime"
    for child in ("dir", "data", "cache"):
        (runtime / child).mkdir(parents=True, exist_ok=True)
    env["APFSVI_DISABLE_TQDM"] = "1"
    env["WANDB_DIR"] = str(runtime / "dir")
    env["WANDB_DATA_DIR"] = str(runtime / "data")
    env["WANDB_CACHE_DIR"] = str(runtime / "cache")
    env["WANDB_INIT_TIMEOUT"] = "120"
    env["WANDB_X_DISABLE_STATS"] = "true"

    for absolute_index, (label, cmd) in enumerate(selected, start=args.start_index):
        print(f"[{absolute_index + 1}/{len(commands)}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        safe_label = label.replace("|", "_").replace("/", "_").replace("\\", "_")
        out_log = LOG_DIR / f"{absolute_index:03d}_{safe_label}.out.log"
        err_log = LOG_DIR / f"{absolute_index:03d}_{safe_label}.err.log"
        started = time.time()
        with out_log.open("wb") as stdout, err_log.open("wb") as stderr:
            proc = subprocess.run(
                cmd,
                cwd=REPO,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        elapsed = time.time() - started
        print(f"  returncode={proc.returncode} elapsed_s={elapsed:.1f}", flush=True)
        if proc.returncode != 0:
            print(f"  failed stdout={out_log}", flush=True)
            print(f"  failed stderr={err_log}", flush=True)
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
