"""Frozen-CLIP linear-head benchmark on CIFAR-10.

Examples
--------
python -m experiments.classification.clip_benchmark --stage smoke
python -m experiments.classification.clip_benchmark --stage all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from experiments.common import build_flow
from implicit_process_zoo.fbnn import FBNN
from implicit_process_zoo.ftip import FTIP
from implicit_process_zoo.gmvip import GeneralizedMatheronVIP, initialize_inducing_points
from implicit_process_zoo.mfvi import MFVI
from implicit_process_zoo.priors.generative_functions import BayesianNN, BayesLinear
from implicit_process_zoo.sip import SIP
from implicit_process_zoo.tfsvi import TFSVI
from implicit_process_zoo.utils.training import prepare_model_for_fit
from implicit_process_zoo.vip import VIP

try:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This experiment requires torchvision and transformers. Install the vision "
        "dependencies before running it."
    ) from exc


METHODS = ("map", "mfvi", "fbnn", "tfsvi", "vip", "ftip", "gmvip", "sip")
PRIOR_METHODS = ("fbnn", "vip", "ftip", "gmvip", "sip")
SIZES = (500, 1_000, 5_000, 10_000, 45_000)
NUM_CLASSES = 10
FEATURE_DIM = 512
CALIBRATION_SIZE = 5_000
TEST_SIZE = 10_000


@dataclass(frozen=True)
class Split:
    features: Tensor
    targets: Tensor


@dataclass(frozen=True)
class SplitBundle:
    tune_train: Split
    selection: Split
    calibration: Split
    final_train: Split
    test: Split
    indices: dict[str, Tensor]
    hashes: dict[str, str]


class ClipCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, targets = zip(*batch)
        pixels = self.processor(images=list(images), return_tensors="pt").pixel_values
        return pixels, torch.tensor(targets, dtype=torch.long)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Frozen-CLIP CIFAR-10 benchmark with linear inference heads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=("embeddings", "smoke", "tune", "final", "all"),
        default="all",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS + ("all",), default=["all"])
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/classification/clip_cifar10"),
    )
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--temperature-iterations", type=int, default=100)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args(argv)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if "all" in args.methods:
        if len(args.methods) != 1:
            parser.error("Use --methods all by itself, or list individual methods.")
        args.methods = list(METHODS)
    else:
        args.methods = list(dict.fromkeys(args.methods))
    args.sizes = sorted(dict.fromkeys(args.sizes))
    if not args.sizes or any(size not in SIZES for size in args.sizes):
        parser.error(f"--sizes must be drawn from {SIZES}.")
    if not args.seeds:
        parser.error("At least one seed is required.")
    args.seeds = list(dict.fromkeys(args.seeds))
    for name in (
        "embedding_batch_size",
        "batch_size",
        "eval_batch_size",
        "eval_samples",
        "epochs",
        "patience",
        "temperature_iterations",
        "ece_bins",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.min_delta < 0:
        parser.error("--min-delta must be non-negative.")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.embedding_cache is None:
        args.embedding_cache = args.output_dir / "clip_embeddings.pt"
    else:
        args.embedding_cache = args.embedding_cache.expanduser().resolve()
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_value),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def config_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, default=json_value).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_hash(tensor):
    array = tensor.detach().cpu().to(torch.int64).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def extract_split(encoder, loader, device, name, no_tqdm):
    features = []
    targets = []
    encoder.eval()
    with torch.inference_mode():
        for pixels, labels in tqdm(
            loader,
            desc=f"Embedding {name}",
            unit="batch",
            disable=no_tqdm,
        ):
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                image_features = encoder(pixel_values=pixels).image_embeds
            features.append(F.normalize(image_features.float(), dim=-1).cpu())
            targets.append(labels)
    return Split(torch.cat(features), torch.cat(targets))


def build_embeddings(args, device):
    train_data = CIFAR10(args.data_dir, train=True, download=args.download)
    test_data = CIFAR10(args.data_dir, train=False, download=args.download)
    processor = CLIPImageProcessor.from_pretrained(
        args.clip_model,
        local_files_only=not args.download,
    )
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.clip_model,
        local_files_only=not args.download,
    ).to(device)
    collator = ClipCollator(processor)
    loader_args = {
        "batch_size": args.embedding_batch_size,
        "shuffle": False,
        "num_workers": 0,
        "collate_fn": collator,
        "pin_memory": device.type == "cuda",
    }
    train = extract_split(
        encoder,
        DataLoader(train_data, **loader_args),
        device,
        "train",
        args.no_tqdm,
    )
    test = extract_split(
        encoder,
        DataLoader(test_data, **loader_args),
        device,
        "test",
        args.no_tqdm,
    )
    payload = {
        "model_name": args.clip_model,
        "normalized": True,
        "train_features": train.features,
        "train_targets": train.targets,
        "test_features": test.features,
        "test_targets": test.targets,
    }
    args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(args.embedding_cache, payload)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return train, test


def load_embeddings(args, device):
    if not args.embedding_cache.exists():
        return build_embeddings(args, device)
    payload = torch.load(args.embedding_cache, map_location="cpu", weights_only=True)
    if payload.get("model_name") != args.clip_model or not payload.get("normalized"):
        raise ValueError("The embedding cache does not match the requested CLIP model.")
    train = Split(payload["train_features"].float(), payload["train_targets"].long())
    test = Split(payload["test_features"].float(), payload["test_targets"].long())
    if train.features.shape != (50_000, FEATURE_DIM):
        raise ValueError(f"Expected train embeddings [50000,{FEATURE_DIM}].")
    if test.features.shape != (TEST_SIZE, FEATURE_DIM):
        raise ValueError(f"Expected test embeddings [{TEST_SIZE},{FEATURE_DIM}].")
    return train, test


def take(split, indices):
    return Split(split.features[indices], split.targets[indices])


def balanced_prefix(pool, targets, total):
    per_class = total // NUM_CLASSES
    return torch.cat(
        [pool[targets[pool].eq(class_id)][:per_class] for class_id in range(NUM_CLASSES)]
    )


def shuffled(indices, seed):
    generator = torch.Generator().manual_seed(seed)
    return indices[torch.randperm(len(indices), generator=generator)]


def base_indices(targets, split_seed):
    generator = torch.Generator().manual_seed(split_seed + 73_001)
    tune_parts = []
    selection_parts = []
    calibration_parts = []
    for class_id in range(NUM_CLASSES):
        indices = torch.where(targets.eq(class_id))[0]
        if len(indices) != 5_000:
            raise ValueError("CIFAR-10 must contain 5,000 training images per class.")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        calibration_parts.append(indices[:500])
        selection_parts.append(indices[500:1_000])
        tune_parts.append(indices[1_000:])

    def join(parts):
        values = torch.cat(parts)
        return values[torch.randperm(len(values), generator=generator)]

    tune = join(tune_parts)
    selection = join(selection_parts)
    calibration = join(calibration_parts)
    return {"tune": tune, "selection": selection, "calibration": calibration}


def split_sizes(size):
    selection = min(5_000, size // 5)
    return size - selection, selection


def make_splits(train, test, sizes, split_seed):
    pools = base_indices(train.targets, split_seed)
    bundles = {}
    previous = None
    for size in sorted(sizes):
        tune_size, selection_size = split_sizes(size)
        if size == 45_000:
            tune_indices = pools["tune"]
            selection_indices = pools["selection"]
        else:
            tune_indices = balanced_prefix(pools["tune"], train.targets, tune_size)
            selection_indices = balanced_prefix(pools["selection"], train.targets, selection_size)
            tune_indices = shuffled(tune_indices, split_seed + 110_000 + size)
            selection_indices = shuffled(selection_indices, split_seed + 210_000 + size)
        final_indices = shuffled(
            torch.cat([tune_indices, selection_indices]),
            split_seed + 310_000 + size,
        )
        indices = {
            "tune_train": tune_indices,
            "selection": selection_indices,
            "calibration": pools["calibration"],
            "final_train": final_indices,
        }
        hashes = {name: tensor_hash(values) for name, values in indices.items()}
        bundle = SplitBundle(
            tune_train=take(train, tune_indices),
            selection=take(train, selection_indices),
            calibration=take(train, pools["calibration"]),
            final_train=take(train, final_indices),
            test=test,
            indices=indices,
            hashes=hashes,
        )
        validate_split(bundle, size, previous)
        bundles[size] = bundle
        previous = bundle
    return bundles


def validate_split(bundle, size, previous):
    tune_size, selection_size = split_sizes(size)
    expected = {
        "tune_train": tune_size,
        "selection": selection_size,
        "calibration": CALIBRATION_SIZE,
        "final_train": size,
    }
    for name, count in expected.items():
        split = getattr(bundle, name)
        if len(split.features) != count or split.features.requires_grad:
            raise AssertionError(f"Invalid {name} split for N={size}.")
        class_counts = [int(split.targets.eq(label).sum()) for label in range(NUM_CLASSES)]
        if len(set(class_counts)) != 1:
            raise AssertionError(f"{name} is not class-balanced for N={size}.")
    tune = set(bundle.indices["tune_train"].tolist())
    selection = set(bundle.indices["selection"].tolist())
    calibration = set(bundle.indices["calibration"].tolist())
    final = set(bundle.indices["final_train"].tolist())
    if tune & selection or final & calibration or final != tune | selection:
        raise AssertionError(f"Split overlap for N={size}.")
    if previous is not None:
        for name in ("tune_train", "selection", "final_train"):
            old = set(previous.indices[name].tolist())
            if not old.issubset(set(bundle.indices[name].tolist())):
                raise AssertionError(f"{name} is not nested at N={size}.")
        if not torch.equal(previous.indices["calibration"], bundle.indices["calibration"]):
            raise AssertionError("Calibration split changed across sizes.")


def candidate_grids():
    grids = {
        "map": [{"lr": 1e-3, "weight_decay": weight_decay} for weight_decay in (0.0, 1e-4)],
        "mfvi": [
            {"lr": 1e-3, "weight_log_std": log_std, "train_samples": samples}
            for log_std in (-3.0, -2.0, -1.0)
            for samples in (10, 20)
        ],
        "vip": [
            {"lr": 1e-3, "basis_samples": samples, "prior_log_std": log_std}
            for samples in (32, 64, 128)
            for log_std in (-1.0, 0.0)
        ],
        "ftip": [
            {
                "lr": 1e-3,
                "basis_samples": samples,
                "prior_log_std": 0.0,
                "flow_type": flow_type,
            }
            for samples in (20, 64, 128)
            for flow_type in ("affine", "spline_1x1")
        ],
        "fbnn": [
            {
                "lr": 1e-3,
                "posterior_samples": 20,
                "prior_samples": prior_samples,
                "measurement": measurement,
                "context": context,
                "lambda_kl": lambda_kl,
            }
            for prior_samples, measurement, context, lambda_kl in (
                (64, 16, 16, 1.0),
                (64, 32, 32, 1.0),
                (64, 64, 64, 1.0),
                (64, 32, 32, 0.1),
                (64, 32, 32, 10.0),
                (128, 32, 32, 1.0),
            )
        ],
        "tfsvi": [
            {
                "lr": 1e-3,
                "sigma_prior": sigma,
                "context_sets": sets,
                "context_points": 16,
                "train_samples": 20,
            }
            for sigma in (0.1, 1.0, 10.0)
            for sets in (1, 3)
        ],
        "gmvip": [
            {
                "lr": 1e-3,
                "num_inducing": inducing,
                "prior_log_std": log_std,
                "prior_samples": 64,
                "train_samples": 20,
            }
            for inducing in (32, 64, 128)
            for log_std in (-1.0, 0.0)
        ],
        "sip": [
            {
                "lr": 1e-3,
                "num_inducing": inducing,
                "beta": beta,
                "prior_samples": 64,
                "train_samples": 20,
            }
            for inducing in (16, 32, 64)
            for beta in (0.1, 1.0)
        ],
    }
    for method, candidates in grids.items():
        expected = 2 if method == "map" else 6
        if len(candidates) != expected:
            raise AssertionError(f"Unexpected candidate count for {method}.")
        for index, candidate in enumerate(candidates):
            candidate["candidate_index"] = index
            candidate["candidate_id"] = f"c{index:02d}_{config_hash(candidate)[:10]}"
    return grids


def make_bayesian_linear(
    num_samples: int,
    seed: int,
    device: torch.device,
    *,
    fix_random_noise: bool,
    weight_log_std: float,
) -> BayesianNN:
    return BayesianNN(
        structure=[],
        activation=nn.Identity(),
        num_samples=int(num_samples),
        input_dim=FEATURE_DIM,
        output_dim=NUM_CLASSES,
        layer_model=BayesLinear,
        dropout=0.0,
        seed=int(seed),
        fix_random_noise=bool(fix_random_noise),
        zero_mean_prior=False,
        weight_log_sigma_init=float(weight_log_std),
        device=device,
        dtype=torch.float32,
    )


def build_model(
    method: str,
    candidate: dict[str, Any],
    train: Split,
    seed: int,
    device: torch.device,
) -> nn.Module:
    seed_everything(seed)
    num_data = len(train.features)

    if method == "map":
        return nn.Linear(
            FEATURE_DIM,
            NUM_CLASSES,
            device=device,
            dtype=torch.float32,
        )

    if method == "mfvi":
        posterior = make_bayesian_linear(
            candidate["train_samples"],
            seed + 101,
            device,
            fix_random_noise=False,
            weight_log_std=candidate["weight_log_std"],
        )
        return MFVI(
            generative_function=posterior,
            output_dim=NUM_CLASSES,
            likelihood="multiclass",
            num_data=num_data,
            num_samples=candidate["train_samples"],
            bb_alpha=0.0,
            num_classes=NUM_CLASSES,
            device=device,
            dtype=torch.float32,
        )

    if method == "vip":
        prior = make_bayesian_linear(
            candidate["basis_samples"],
            seed + 201,
            device,
            fix_random_noise=True,
            weight_log_std=candidate["prior_log_std"],
        )
        return VIP(
            generative_function=prior,
            num_regression_coeffs=candidate["basis_samples"],
            output_dim=NUM_CLASSES,
            likelihood="multiclass",
            num_data=num_data,
            bb_alpha=0.0,
            num_classes=NUM_CLASSES,
            num_mc_samples=20,
            use_prior_regularizer=False,
            device=device,
            dtype=torch.float32,
            seed=seed + 202,
        )

    if method == "ftip":
        basis_samples = candidate["basis_samples"]
        prior = make_bayesian_linear(
            basis_samples,
            seed + 301,
            device,
            fix_random_noise=True,
            weight_log_std=candidate["prior_log_std"],
        )
        flow = build_flow(
            candidate["flow_type"],
            depth=2,
            input_dim=basis_samples * NUM_CLASSES,
            seed=seed + 302,
            device=device,
            dtype=torch.float32,
            num_bins=8,
            domain=3.0,
        )
        return FTIP(
            generative_function=prior,
            num_regression_coeffs=basis_samples,
            output_dim=NUM_CLASSES,
            flow=flow,
            likelihood="multiclass",
            num_data=num_data,
            num_samples=20,
            bb_alpha=0.0,
            num_classes=NUM_CLASSES,
            use_prior_regularizer=False,
            max_grad_norm=None,
            device=device,
            dtype=torch.float32,
            seed=seed + 303,
        )

    if method == "fbnn":
        posterior = make_bayesian_linear(
            candidate["posterior_samples"],
            seed + 401,
            device,
            fix_random_noise=True,
            weight_log_std=-3.0,
        )
        prior = make_bayesian_linear(
            candidate["prior_samples"],
            seed + 402,
            device,
            fix_random_noise=True,
            weight_log_std=0.0,
        )
        return FBNN(
            generative_function=posterior,
            prior_function=prior,
            output_dim=NUM_CLASSES,
            likelihood="multiclass",
            num_data=num_data,
            num_samples=candidate["posterior_samples"],
            num_measurement=candidate["measurement"],
            num_context=candidate["context"],
            context_std=1.5,
            bb_alpha=0.0,
            lambda_kl=candidate["lambda_kl"],
            num_eigs=None,
            nugget=1e-4,
            reservoir_size=min(5_000, num_data),
            num_classes=NUM_CLASSES,
            freeze_prior=False,
            device=device,
            dtype=torch.float32,
        )

    if method == "tfsvi":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 501)
            template = nn.Linear(
                FEATURE_DIM,
                NUM_CLASSES,
                device=device,
                dtype=torch.float32,
            )
        return TFSVI(
            input_dim=FEATURE_DIM,
            output_dim=NUM_CLASSES,
            structure=[],
            activation=nn.Identity(),
            likelihood="multiclass",
            num_data=num_data,
            sigma_prior=candidate["sigma_prior"],
            num_samples=candidate["train_samples"],
            bb_alpha=0.0,
            S_ctx=candidate["context_sets"],
            K_ctx=candidate["context_points"],
            num_classes=NUM_CLASSES,
            generative_function=template,
            device=device,
            dtype=torch.float32,
        )

    if method == "gmvip":
        inducing = initialize_inducing_points(
            train.features,
            num_inducing=candidate["num_inducing"],
            method="random_subset",
            seed=seed + 31,
        ).to(device)
        prior = make_bayesian_linear(
            max(candidate["prior_samples"], candidate["train_samples"], 2),
            seed + 601,
            device,
            fix_random_noise=True,
            weight_log_std=candidate["prior_log_std"],
        )
        return GeneralizedMatheronVIP(
            base_prior=prior,
            inducing_points=inducing,
            operator_type="empirical",
            posterior_type="gaussian",
            likelihood="multiclass",
            output_dim=NUM_CLASSES,
            num_classes=NUM_CLASSES,
            num_operator_bank_samples=candidate["prior_samples"],
            learn_noise=False,
            freeze_base_prior=False,
            detach_prior_samples=False,
            jitter=1e-4,
            shrinkage=0.02,
            learn_Z=True,
            learn_kernel=False,
            ard=True,
            init_lengthscale="median",
            init_outputscale="prior_marginal",
            inducing_scale="prior_cholesky",
            mean_mode="prior_sample",
            posterior_max_log_std=None,
            antithetic_samples=True,
            num_data=num_data,
            num_train_samples=candidate["train_samples"],
            beta=1.0,
            beta_warmup_steps=0,
            data_alpha=0.0,
            max_grad_norm=None,
            operator_bank_seed=seed + 1601,
        )

    if method == "sip":
        inducing = initialize_inducing_points(
            train.features,
            num_inducing=candidate["num_inducing"],
            method="random_subset",
            seed=seed + 41,
        ).to(device)
        prior = make_bayesian_linear(
            max(candidate["prior_samples"], candidate["train_samples"], 2),
            seed + 701,
            device,
            fix_random_noise=False,
            weight_log_std=0.0,
        )
        return SIP(
            generative_function=prior,
            inducing_inputs=inducing,
            output_dim=NUM_CLASSES,
            likelihood="multiclass",
            num_data=num_data,
            num_prior_samples=candidate["prior_samples"],
            num_train_samples=candidate["train_samples"],
            num_eval_samples=100,
            bb_alpha=0.0,
            beta=candidate["beta"],
            beta_warmup_steps=0,
            learn_inducing=True,
            detach_covariances=False,
            critic_hidden_dim=50,
            critic_lr=1e-3,
            critic_steps=1,
            posterior_noise_dim=100,
            posterior_hidden_dim=50,
            posterior_depth=2,
            fresh_prior_samples=True,
            num_classes=NUM_CLASSES,
            jitter=1e-4,
            device=device,
            dtype=torch.float32,
            seed=seed + 702,
        )

    raise ValueError(f"Unknown method: {method}")


def prior_module(model: nn.Module, method: str) -> nn.Module | None:
    if method in ("vip", "ftip"):
        return model.generative_function
    if method == "fbnn":
        return model.prior_function
    if method == "gmvip":
        return model.base_prior
    if method == "sip":
        return model.generative_function
    return None


def trainable_prior(model: nn.Module, method: str) -> bool:
    prior = prior_module(model, method)
    return prior is not None and all(parameter.requires_grad for parameter in prior.parameters())


def trainable_inducing_locations(model: nn.Module, method: str) -> bool:
    if method == "gmvip":
        return bool(model.operator.learn_Z and model.operator.Z.requires_grad)
    if method == "sip":
        return bool(model.Z.requires_grad)
    return False


def check_model(method: str, model: nn.Module) -> None:
    if method in PRIOR_METHODS:
        prior = prior_module(model, method)
        if prior is None or not list(prior.parameters()):
            raise AssertionError(f"{method} has no Bayesian linear prior.")
        if not trainable_prior(model, method):
            raise AssertionError(f"{method} prior is not fully trainable.")
        if not isinstance(prior, BayesianNN) or prior.structure != []:
            raise AssertionError(f"{method} prior is not one linear layer.")
    if method in ("gmvip", "sip") and not trainable_inducing_locations(model, method):
        raise AssertionError(f"{method} does not have trainable inducing locations.")
    if method == "map" and not isinstance(model, nn.Linear):
        raise AssertionError("MAP must be exactly one linear layer.")
    if method == "tfsvi" and not isinstance(model.base_net, nn.Linear):
        raise AssertionError("TFSVI must use a linear architecture template.")


def make_loader(
    split: Split,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(split.features, split.targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def optimizer_for(
    method: str,
    model: nn.Module,
    candidate: dict[str, Any],
) -> torch.optim.Optimizer:
    if method == "sip":
        parameters = list(model.vi_parameters())
    else:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"{method} has no optimizer parameters.")
    if method == "map":
        return torch.optim.AdamW(
            parameters,
            lr=candidate["lr"],
            weight_decay=candidate["weight_decay"],
        )
    return torch.optim.Adam(parameters, lr=candidate["lr"])


def training_step(
    method: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: Tensor,
    targets: Tensor,
) -> float:
    if method == "map":
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features), targets.long())
        loss.backward()
        optimizer.step()
    else:
        loss = model._train_step(optimizer, features, targets.long())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite training loss for {method}.")
    return float(loss.detach())


@torch.no_grad()
def predictive_log_probabilities(
    method: str,
    model: nn.Module,
    split: Split,
    device: torch.device,
    batch_size: int,
    num_samples: int,
) -> tuple[Tensor, tuple[int, int, int]]:
    model.eval()
    chunks: list[Tensor] = []
    prediction_shape: tuple[int, int, int] | None = None
    loader = make_loader(split, batch_size, shuffle=False, seed=0)
    for batch_index, (features, _) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        if method == "map":
            logits = model(features).unsqueeze(0).expand(num_samples, -1, -1)
        else:
            logits = model.predict_f_samples(
                features,
                num_samples=num_samples,
                seed=100_000 + batch_index,
            )
        if logits.ndim != 3 or logits.shape != (
            num_samples,
            len(features),
            NUM_CLASSES,
        ):
            raise AssertionError(
                f"Expected [{num_samples}, batch, {NUM_CLASSES}], got {tuple(logits.shape)}."
            )
        prediction_shape = tuple(logits.shape)
        log_probabilities = F.log_softmax(logits, dim=-1)
        chunks.append(torch.logsumexp(log_probabilities, dim=0).cpu())
        chunks[-1] -= math.log(num_samples)
    if prediction_shape is None:
        raise AssertionError("Cannot evaluate an empty split.")
    result = torch.cat(chunks)
    if not torch.isfinite(result).all():
        raise FloatingPointError(f"{method} produced non-finite predictions.")
    return result, prediction_shape


def expected_calibration_error(
    probabilities: Tensor,
    targets: Tensor,
    num_bins: int,
) -> float:
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(targets)
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    ece = torch.tensor(0.0)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            fraction = in_bin.float().mean()
            accuracy = correct[in_bin].float().mean()
            average_confidence = confidence[in_bin].mean()
            ece += fraction * (accuracy - average_confidence).abs()
    return float(ece)


def classification_metrics(
    log_probabilities: Tensor,
    targets: Tensor,
    bins: int,
) -> dict[str, float]:
    probabilities = log_probabilities.exp()
    return {
        "accuracy": float(probabilities.argmax(dim=1).eq(targets).float().mean()),
        "nll": float(F.nll_loss(log_probabilities, targets)),
        "ece": expected_calibration_error(probabilities, targets, bins),
    }


def fit_temperature(
    log_probabilities: Tensor,
    targets: Tensor,
    iterations: int,
) -> float:
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=iterations,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-6.0, 6.0).exp()
        scaled = F.log_softmax(log_probabilities / temperature, dim=1)
        loss = F.nll_loss(scaled, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-6.0, 6.0).exp())


def apply_temperature(log_probabilities: Tensor, temperature: float) -> Tensor:
    return F.log_softmax(log_probabilities / temperature, dim=1)


def balanced_folds(targets: Tensor, num_folds: int, seed: int) -> list[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    folds: list[list[Tensor]] = [[] for _ in range(num_folds)]
    for class_index in range(NUM_CLASSES):
        indices = torch.where(targets == class_index)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for fold_index, chunk in enumerate(torch.tensor_split(indices, num_folds)):
            folds[fold_index].append(chunk)
    return [torch.cat(parts).sort().values for parts in folds]


def cross_fitted_metrics(
    log_probabilities: Tensor,
    targets: Tensor,
    *,
    seed: int,
    iterations: int,
    bins: int,
) -> dict[str, Any]:
    scaled = torch.empty_like(log_probabilities)
    temperatures = []
    folds = balanced_folds(targets, 5, seed)
    all_indices = torch.arange(len(targets))
    for validation_indices in folds:
        fit_mask = torch.ones(len(targets), dtype=torch.bool)
        fit_mask[validation_indices] = False
        fit_indices = all_indices[fit_mask]
        temperature = fit_temperature(
            log_probabilities[fit_indices],
            targets[fit_indices],
            iterations,
        )
        temperatures.append(temperature)
        scaled[validation_indices] = apply_temperature(
            log_probabilities[validation_indices],
            temperature,
        )
    metrics: dict[str, Any] = classification_metrics(scaled, targets, bins)
    metrics["temperatures"] = temperatures
    metrics["temperature_mean"] = float(np.mean(temperatures))
    return metrics


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_nll: float,
    stale_epochs: int,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_nll": best_nll,
        "stale_epochs": stale_epochs,
    }
    critic_optimizer = getattr(model, "critic_optimizer", None)
    if critic_optimizer is not None:
        payload["critic_optimizer"] = critic_optimizer.state_dict()
    atomic_torch_save(path, payload)


def restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, float, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    critic_optimizer = getattr(model, "critic_optimizer", None)
    if critic_optimizer is not None and "critic_optimizer" in payload:
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
    return (
        int(payload["epoch"]),
        float(payload["best_nll"]),
        int(payload["stale_epochs"]),
    )


def fit_model(
    method: str,
    candidate: dict[str, Any],
    train: Split,
    validation: Split | None,
    *,
    epochs: int,
    seed: int,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    seed_everything(seed)
    model = build_model(method, candidate, train, seed, args.device)
    preparation_loader = make_loader(
        train,
        args.batch_size,
        shuffle=False,
        seed=seed,
    )
    prepare_model_for_fit(model, preparation_loader)
    check_model(method, model)

    optimizer = optimizer_for(method, model, candidate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=candidate["lr"] / 100,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    start_epoch = 1
    best_nll = float("inf")
    stale_epochs = 0
    if args.resume and not args.overwrite and latest_path.exists():
        completed_epoch, best_nll, stale_epochs = restore_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
        )
        start_epoch = completed_epoch + 1

    reset_peak_memory(args.device)
    training_seconds = 0.0
    history: list[dict[str, float]] = []
    best_epoch = max(0, start_epoch - 1)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loader = make_loader(
            train,
            args.batch_size,
            shuffle=True,
            seed=seed * 10_000 + epoch,
        )
        losses = []
        started = time.perf_counter()
        for features, targets in loader:
            features = features.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            losses.append(training_step(method, model, optimizer, features, targets))
        training_seconds += time.perf_counter() - started
        scheduler.step()

        if validation is None:
            validation_nll = float("nan")
            improved = True
        else:
            log_probabilities, _ = predictive_log_probabilities(
                method,
                model,
                validation,
                args.device,
                args.eval_batch_size,
                args.eval_samples,
            )
            validation_nll = float(F.nll_loss(log_probabilities, validation.targets))
            improved = validation_nll < best_nll - args.min_delta

        if improved:
            if validation is not None:
                best_nll = validation_nll
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_nll,
                stale_epochs,
            )
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "selection_nll": validation_nll,
            }
        )
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_nll,
            stale_epochs,
        )
        if validation is not None and stale_epochs >= args.patience:
            break

    if best_path.exists():
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        best_epoch = int(payload["epoch"])
        best_nll = float(payload["best_nll"])

    return model, {
        "best_epoch": best_epoch,
        "best_selection_nll": None if validation is None else best_nll,
        "epochs_ran": history[-1]["epoch"] if history else start_epoch - 1,
        "history": history,
        "training_seconds": training_seconds,
        "peak_gpu_memory_mb": peak_memory_mb(args.device),
        "trainable_parameters": trainable_parameter_count(model),
    }


def state_template(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "smoke": {},
        "tuning": {},
        "winners": {},
        "final": {},
        "failures": [],
    }


def load_state(args: argparse.Namespace, metadata: dict[str, Any]) -> tuple[dict, Path]:
    path = args.output_dir / "benchmark_state.json"
    if args.overwrite or not path.exists():
        state = state_template(metadata)
        atomic_write_json(path, state)
        return state, path
    if not args.resume:
        raise RuntimeError(f"{path} already exists. Use --resume or --overwrite.")
    state = json.loads(path.read_text(encoding="utf-8"))
    old_signature = state.get("metadata", {}).get("run_signature")
    if old_signature != metadata["run_signature"]:
        raise RuntimeError(
            "The saved run signature does not match this command. "
            "Use a different output directory or --overwrite."
        )
    return state, path


def save_state(state: dict[str, Any], path: Path) -> None:
    atomic_write_json(path, state)


def record_failure(
    state: dict[str, Any],
    state_path: Path,
    *,
    stage: str,
    size: int,
    method: str,
    candidate_id: str | None,
    seed: int,
    error: BaseException,
) -> dict[str, Any]:
    failure = {
        "stage": stage,
        "size": size,
        "method": method,
        "candidate_id": candidate_id,
        "seed": seed,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    state["failures"].append(failure)
    save_state(state, state_path)
    return failure


def completed(record: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    if not record:
        return False
    if record.get("status") == "complete":
        return True
    return record.get("status") == "failed" and not args.retry_failures


def choose_winner(records: dict[str, Any]) -> dict[str, Any]:
    successful = [record for record in records.values() if record.get("status") == "complete"]
    if not successful:
        raise RuntimeError("No candidate completed successfully.")
    best_nll = min(record["selection"]["nll"] for record in successful)
    tied = [record for record in successful if record["selection"]["nll"] <= best_nll + 1e-4]
    return min(
        tied,
        key=lambda record: (
            -record["selection"]["accuracy"],
            record["trainable_parameters"],
            record["training_seconds"],
            record["candidate"]["candidate_index"],
        ),
    )


def run_smoke(
    args: argparse.Namespace,
    grids: dict[str, list[dict[str, Any]]],
    splits: dict[int, SplitBundle],
    state: dict[str, Any],
    state_path: Path,
) -> None:
    bundle = splits[500]
    smoke_train = bundle.tune_train
    smoke_validation = bundle.selection
    for method in args.methods:
        old_record = state["smoke"].get(method)
        if completed(old_record, args):
            continue
        print(f"smoke {method}")
        try:
            candidate = grids[method][0]
            model, fit = fit_model(
                method,
                candidate,
                smoke_train,
                None,
                epochs=1,
                seed=0,
                args=args,
                checkpoint_dir=args.output_dir / "checkpoints" / "smoke" / method,
            )
            log_probabilities, prediction_shape = predictive_log_probabilities(
                method,
                model,
                smoke_validation,
                args.device,
                args.eval_batch_size,
                min(4, args.eval_samples),
            )
            if not torch.isfinite(log_probabilities).all():
                raise FloatingPointError("Smoke predictions are not finite.")
            state["smoke"][method] = {
                "status": "complete",
                "loss": fit["history"][-1]["training_loss"],
                "prediction_shape": prediction_shape,
                "trainable_parameters": fit["trainable_parameters"],
                "trainable_prior": trainable_prior(model, method),
                "trainable_inducing_locations": trainable_inducing_locations(model, method),
            }
        except (RuntimeError, FloatingPointError, ValueError, AssertionError) as error:
            state["smoke"][method] = {
                "status": "failed",
                "error": str(error),
            }
            record_failure(
                state,
                state_path,
                stage="smoke",
                size=500,
                method=method,
                candidate_id=None,
                seed=0,
                error=error,
            )
        save_state(state, state_path)


def run_tuning(
    args: argparse.Namespace,
    grids: dict[str, list[dict[str, Any]]],
    splits: dict[int, SplitBundle],
    state: dict[str, Any],
    state_path: Path,
) -> None:
    for size in args.sizes:
        size_key = str(size)
        state["tuning"].setdefault(size_key, {})
        state["winners"].setdefault(size_key, {})
        bundle = splits[size]
        for method in args.methods:
            method_records = state["tuning"][size_key].setdefault(method, {})
            for candidate in grids[method]:
                candidate_id = candidate["candidate_id"]
                if completed(method_records.get(candidate_id), args):
                    continue
                print(f"tune size={size} method={method} candidate={candidate_id}")
                checkpoint_dir = (
                    args.output_dir / "checkpoints" / "tune" / size_key / method / candidate_id
                )
                try:
                    model, fit = fit_model(
                        method,
                        candidate,
                        bundle.tune_train,
                        bundle.selection,
                        epochs=args.epochs,
                        seed=0,
                        args=args,
                        checkpoint_dir=checkpoint_dir,
                    )
                    log_probabilities, _ = predictive_log_probabilities(
                        method,
                        model,
                        bundle.selection,
                        args.device,
                        args.eval_batch_size,
                        args.eval_samples,
                    )
                    raw = classification_metrics(
                        log_probabilities,
                        bundle.selection.targets,
                        args.ece_bins,
                    )
                    selection = cross_fitted_metrics(
                        log_probabilities,
                        bundle.selection.targets,
                        seed=args.split_seed,
                        iterations=args.temperature_iterations,
                        bins=args.ece_bins,
                    )
                    method_records[candidate_id] = {
                        "status": "complete",
                        "candidate": candidate,
                        "best_epoch": fit["best_epoch"],
                        "raw_selection": raw,
                        "selection": selection,
                        "trainable_parameters": fit["trainable_parameters"],
                        "training_seconds": fit["training_seconds"],
                        "peak_gpu_memory_mb": fit["peak_gpu_memory_mb"],
                        "history": fit["history"],
                    }
                    del model
                    if args.device.type == "cuda":
                        torch.cuda.empty_cache()
                except (
                    RuntimeError,
                    FloatingPointError,
                    ValueError,
                    AssertionError,
                    np.linalg.LinAlgError,
                ) as error:
                    method_records[candidate_id] = {
                        "status": "failed",
                        "candidate": candidate,
                        "error": str(error),
                    }
                    record_failure(
                        state,
                        state_path,
                        stage="tune",
                        size=size,
                        method=method,
                        candidate_id=candidate_id,
                        seed=0,
                        error=error,
                    )
                save_state(state, state_path)

            try:
                winner = choose_winner(method_records)
                state["winners"][size_key][method] = {
                    "candidate_id": winner["candidate"]["candidate_id"],
                    "candidate": winner["candidate"],
                    "selected_epoch": winner["best_epoch"],
                    "selection": winner["selection"],
                }
            except RuntimeError as error:
                record_failure(
                    state,
                    state_path,
                    stage="winner",
                    size=size,
                    method=method,
                    candidate_id=None,
                    seed=0,
                    error=error,
                )
            save_state(state, state_path)


def run_final(
    args: argparse.Namespace,
    splits: dict[int, SplitBundle],
    state: dict[str, Any],
    state_path: Path,
) -> None:
    for size in args.sizes:
        size_key = str(size)
        state["final"].setdefault(size_key, {})
        bundle = splits[size]
        for method in args.methods:
            method_records = state["final"][size_key].setdefault(method, {})
            winner = state.get("winners", {}).get(size_key, {}).get(method)
            if winner is None:
                error = RuntimeError(f"No tuned winner for size={size}, method={method}.")
                record_failure(
                    state,
                    state_path,
                    stage="final",
                    size=size,
                    method=method,
                    candidate_id=None,
                    seed=-1,
                    error=error,
                )
                continue
            candidate = winner["candidate"]
            epochs = int(winner["selected_epoch"])
            for seed in args.seeds:
                seed_key = str(seed)
                if completed(method_records.get(seed_key), args):
                    continue
                print(f"final size={size} method={method} seed={seed} epochs={epochs}")
                checkpoint_dir = (
                    args.output_dir / "checkpoints" / "final" / size_key / method / seed_key
                )
                try:
                    model, fit = fit_model(
                        method,
                        candidate,
                        bundle.final_train,
                        None,
                        epochs=epochs,
                        seed=seed,
                        args=args,
                        checkpoint_dir=checkpoint_dir,
                    )
                    calibration_logp, _ = predictive_log_probabilities(
                        method,
                        model,
                        bundle.calibration,
                        args.device,
                        args.eval_batch_size,
                        args.eval_samples,
                    )
                    temperature = fit_temperature(
                        calibration_logp,
                        bundle.calibration.targets,
                        args.temperature_iterations,
                    )
                    calibrated_calibration = apply_temperature(
                        calibration_logp,
                        temperature,
                    )
                    if not torch.equal(
                        calibration_logp.argmax(dim=1),
                        calibrated_calibration.argmax(dim=1),
                    ):
                        raise AssertionError("Temperature scaling changed predicted classes.")

                    test_logp, _ = predictive_log_probabilities(
                        method,
                        model,
                        bundle.test,
                        args.device,
                        args.eval_batch_size,
                        args.eval_samples,
                    )
                    calibrated_test = apply_temperature(test_logp, temperature)
                    if not torch.equal(
                        test_logp.argmax(dim=1),
                        calibrated_test.argmax(dim=1),
                    ):
                        raise AssertionError("Temperature scaling changed predicted classes.")
                    method_records[seed_key] = {
                        "status": "complete",
                        "candidate": candidate,
                        "epochs": epochs,
                        "temperature": temperature,
                        "calibration": classification_metrics(
                            calibrated_calibration,
                            bundle.calibration.targets,
                            args.ece_bins,
                        ),
                        "raw_test": classification_metrics(
                            test_logp,
                            bundle.test.targets,
                            args.ece_bins,
                        ),
                        "test": classification_metrics(
                            calibrated_test,
                            bundle.test.targets,
                            args.ece_bins,
                        ),
                        "trainable_parameters": fit["trainable_parameters"],
                        "training_seconds": fit["training_seconds"],
                        "peak_gpu_memory_mb": fit["peak_gpu_memory_mb"],
                    }
                    del model
                    if args.device.type == "cuda":
                        torch.cuda.empty_cache()
                except (
                    RuntimeError,
                    FloatingPointError,
                    ValueError,
                    AssertionError,
                    np.linalg.LinAlgError,
                ) as error:
                    method_records[seed_key] = {
                        "status": "failed",
                        "candidate": candidate,
                        "epochs": epochs,
                        "error": str(error),
                    }
                    record_failure(
                        state,
                        state_path,
                        stage="final",
                        size=size,
                        method=method,
                        candidate_id=candidate["candidate_id"],
                        seed=seed,
                        error=error,
                    )
                save_state(state, state_path)


def mean_std(values: list[float], digits: int = 4) -> str:
    if not values:
        return ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def final_rows(args: argparse.Namespace, state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for size in args.sizes:
        size_key = str(size)
        for method in args.methods:
            records = state.get("final", {}).get(size_key, {}).get(method, {})
            runs = [record for record in records.values() if record.get("status") == "complete"]
            if not runs:
                continue
            rows.append(
                {
                    "training_size": size,
                    "method": method.upper(),
                    "accuracy": mean_std([record["test"]["accuracy"] for record in runs]),
                    "calibrated_nll": mean_std([record["test"]["nll"] for record in runs]),
                    "calibrated_ece": mean_std([record["test"]["ece"] for record in runs]),
                    "temperature": mean_std([record["temperature"] for record in runs]),
                    "trainable_parameters": mean_std(
                        [float(record["trainable_parameters"]) for record in runs],
                        digits=0,
                    ),
                    "training_seconds": mean_std(
                        [record["training_seconds"] for record in runs],
                        digits=1,
                    ),
                    "peak_gpu_memory_mb": mean_std(
                        [record["peak_gpu_memory_mb"] for record in runs],
                        digits=1,
                    ),
                    "completed_seeds": len(runs),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("No completed runs.\n", encoding="utf-8")
        return
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def winner_rows(args: argparse.Namespace, state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for size in args.sizes:
        for method in args.methods:
            winner = state.get("winners", {}).get(str(size), {}).get(method)
            if winner is None:
                continue
            rows.append(
                {
                    "training_size": size,
                    "method": method.upper(),
                    "candidate_id": winner["candidate_id"],
                    "selected_epoch": winner["selected_epoch"],
                    "selection_nll": winner["selection"]["nll"],
                    "selection_accuracy": winner["selection"]["accuracy"],
                    "configuration": json.dumps(
                        winner["candidate"],
                        sort_keys=True,
                    ),
                }
            )
    return rows


def map_gmvip_rows(args: argparse.Namespace, state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for size in args.sizes:
        final = state.get("final", {}).get(str(size), {})
        map_records = final.get("map", {})
        gmvip_records = final.get("gmvip", {})
        common_seeds = [
            str(seed)
            for seed in args.seeds
            if map_records.get(str(seed), {}).get("status") == "complete"
            and gmvip_records.get(str(seed), {}).get("status") == "complete"
        ]
        if not common_seeds:
            continue
        accuracy_differences = [
            gmvip_records[seed]["test"]["accuracy"] - map_records[seed]["test"]["accuracy"]
            for seed in common_seeds
        ]
        nll_differences = [
            map_records[seed]["test"]["nll"] - gmvip_records[seed]["test"]["nll"]
            for seed in common_seeds
        ]
        rows.append(
            {
                "training_size": size,
                "gmvip_minus_map_accuracy": mean_std(accuracy_differences),
                "map_minus_gmvip_calibrated_nll": mean_std(nll_differences),
                "paired_seeds": len(common_seeds),
            }
        )
    return rows


def write_tables(args: argparse.Namespace, state: dict[str, Any]) -> None:
    headline = final_rows(args, state)
    winners = winner_rows(args, state)
    comparison = map_gmvip_rows(args, state)
    write_csv(args.output_dir / "headline.csv", headline)
    write_markdown(args.output_dir / "headline.md", headline)
    write_csv(args.output_dir / "winners.csv", winners)
    write_markdown(args.output_dir / "winners.md", winners)
    write_csv(args.output_dir / "map_vs_gmvip.csv", comparison)
    write_markdown(args.output_dir / "map_vs_gmvip.md", comparison)


def build_metadata(
    args: argparse.Namespace,
    grids: dict[str, list[dict[str, Any]]],
    splits: dict[int, SplitBundle],
) -> dict[str, Any]:
    split_hashes = {str(size): bundle.hashes for size, bundle in splits.items()}
    settings = {
        "clip_model": args.clip_model,
        "normalized_embeddings": True,
        "feature_dim": FEATURE_DIM,
        "num_classes": NUM_CLASSES,
        "methods": args.methods,
        "sizes": args.sizes,
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "eval_samples": args.eval_samples,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "temperature_iterations": args.temperature_iterations,
        "ece_bins": args.ece_bins,
        "candidate_grids": {method: grids[method] for method in args.methods},
        "split_hashes": split_hashes,
    }
    return {
        **settings,
        "run_signature": config_hash(settings),
        "embedding_cache": str(args.embedding_cache),
        "device": str(args.device),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def verify_reproduction(
    train: Split,
    test: Split,
    sizes: list[int],
    split_seed: int,
    splits: dict[int, SplitBundle],
) -> None:
    repeated = make_splits(train, test, sizes, split_seed)
    for size in sizes:
        if splits[size].hashes != repeated[size].hashes:
            raise AssertionError(f"Split reproduction failed for N={size}.")


def main(argv=None):
    args = parse_args(argv)
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_embeddings(args, args.device)
    print(f"embeddings: {args.embedding_cache}")
    if args.stage == "embeddings":
        return

    splits = make_splits(train, test, args.sizes, args.split_seed)
    verify_reproduction(train, test, args.sizes, args.split_seed, splits)
    grids = candidate_grids()
    metadata = build_metadata(args, grids, splits)
    state, state_path = load_state(args, metadata)

    if args.stage in ("smoke", "all"):
        if 500 not in splits:
            smoke_splits = make_splits(train, test, [500], args.split_seed)
        else:
            smoke_splits = splits
        run_smoke(args, grids, smoke_splits, state, state_path)
    if args.stage in ("tune", "all"):
        run_tuning(args, grids, splits, state, state_path)
    if args.stage in ("final", "all"):
        run_final(args, splits, state, state_path)

    write_tables(args, state)
    save_state(state, state_path)
    print(f"state: {state_path}")
    print(f"headline: {args.output_dir / 'headline.md'}")


if __name__ == "__main__":
    main()
