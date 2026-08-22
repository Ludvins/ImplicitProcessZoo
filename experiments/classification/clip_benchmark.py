"""Frozen-CLIP linear-head benchmark on CIFAR-10.

Examples
--------
python -m experiments.classification.clip_benchmark --stage smoke
python -m experiments.classification.clip_benchmark --stage all
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from experiments.common import frozen_feature_classification as frozen

try:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This experiment requires torchvision and transformers. Install the vision "
        "dependencies before running it."
    ) from exc


METHODS = frozen.METHODS
SIZES = (500, 1_000, 5_000, 10_000, 45_000)
NUM_CLASSES = 10
FEATURE_DIM = 512
CALIBRATION_SIZE = 5_000
TEST_SIZE = 10_000
SPEC = frozen.FrozenFeatureSpec(FEATURE_DIM, NUM_CLASSES, train_fbnn_prior=True)


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
    return frozen.Split(torch.cat(features), torch.cat(targets))


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
    frozen.atomic_torch_save(args.embedding_cache, payload)
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
    train = frozen.Split(payload["train_features"].float(), payload["train_targets"].long())
    test = frozen.Split(payload["test_features"].float(), payload["test_targets"].long())
    if train.features.shape != (50_000, FEATURE_DIM):
        raise ValueError(f"Expected train embeddings [50000,{FEATURE_DIM}].")
    if test.features.shape != (TEST_SIZE, FEATURE_DIM):
        raise ValueError(f"Expected test embeddings [{TEST_SIZE},{FEATURE_DIM}].")
    return train, test


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

    return {
        "tune": join(tune_parts),
        "selection": join(selection_parts),
        "calibration": join(calibration_parts),
    }


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
        hashes = {name: frozen.tensor_hash(values) for name, values in indices.items()}
        bundle = frozen.SplitBundle(
            tune_train=frozen.take(train, tune_indices),
            selection=frozen.take(train, selection_indices),
            calibration=frozen.take(train, pools["calibration"]),
            final_train=frozen.take(train, final_indices),
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


def run_smoke(args, grids, splits, state, state_path):
    bundle = splits[500]
    for method in args.methods:
        if frozen.completed(state["smoke"].get(method), args):
            continue
        print(f"smoke {method}")
        try:
            model, fit = frozen.fit_model(
                SPEC,
                method,
                grids[method][0],
                bundle.tune_train,
                None,
                epochs=1,
                seed=0,
                args=args,
                checkpoint_dir=args.output_dir / "checkpoints" / "smoke" / method,
            )
            log_probabilities, prediction_shape = frozen.predictive_log_probabilities(
                SPEC,
                method,
                model,
                bundle.selection,
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
                "trainable_prior": frozen.trainable_prior(model, method),
                "trainable_inducing_locations": frozen.trainable_inducing_locations(model, method),
            }
        except (RuntimeError, FloatingPointError, ValueError, AssertionError) as error:
            state["smoke"][method] = {"status": "failed", "error": str(error)}
            frozen.record_failure(
                state,
                state_path,
                stage="smoke",
                size=500,
                method=method,
                candidate_id=None,
                seed=0,
                error=error,
            )
        frozen.save_state(state, state_path)


def run_final(args, splits, state, state_path):
    for size in args.sizes:
        size_key = str(size)
        state["final"].setdefault(size_key, {})
        bundle = splits[size]
        for method in args.methods:
            method_records = state["final"][size_key].setdefault(method, {})
            winner = state.get("winners", {}).get(size_key, {}).get(method)
            if winner is None:
                error = RuntimeError(f"No tuned winner for size={size}, method={method}.")
                frozen.record_failure(
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
                if frozen.completed(method_records.get(seed_key), args):
                    continue
                print(f"final size={size} method={method} seed={seed} epochs={epochs}")
                checkpoint_dir = (
                    args.output_dir / "checkpoints" / "final" / size_key / method / seed_key
                )
                try:
                    model, fit = frozen.fit_model(
                        SPEC,
                        method,
                        candidate,
                        bundle.final_train,
                        None,
                        epochs=epochs,
                        seed=seed,
                        args=args,
                        checkpoint_dir=checkpoint_dir,
                    )
                    calibration_logp, _ = frozen.predictive_log_probabilities(
                        SPEC,
                        method,
                        model,
                        bundle.calibration,
                        args.device,
                        args.eval_batch_size,
                        args.eval_samples,
                    )
                    temperature = frozen.fit_temperature(
                        calibration_logp,
                        bundle.calibration.targets,
                        args.temperature_iterations,
                    )
                    calibrated_calibration = frozen.apply_temperature(
                        calibration_logp,
                        temperature,
                    )
                    if not torch.equal(
                        calibration_logp.argmax(dim=1),
                        calibrated_calibration.argmax(dim=1),
                    ):
                        raise AssertionError("Temperature scaling changed predicted classes.")

                    test_logp, _ = frozen.predictive_log_probabilities(
                        SPEC,
                        method,
                        model,
                        bundle.test,
                        args.device,
                        args.eval_batch_size,
                        args.eval_samples,
                    )
                    calibrated_test = frozen.apply_temperature(test_logp, temperature)
                    if not torch.equal(test_logp.argmax(1), calibrated_test.argmax(1)):
                        raise AssertionError("Temperature scaling changed predicted classes.")
                    method_records[seed_key] = {
                        "status": "complete",
                        "candidate": candidate,
                        "epochs": epochs,
                        "temperature": temperature,
                        "calibration": frozen.classification_metrics(
                            calibrated_calibration,
                            bundle.calibration.targets,
                            args.ece_bins,
                        ),
                        "raw_test": frozen.classification_metrics(
                            test_logp,
                            bundle.test.targets,
                            args.ece_bins,
                        ),
                        "test": frozen.classification_metrics(
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
                    frozen.record_failure(
                        state,
                        state_path,
                        stage="final",
                        size=size,
                        method=method,
                        candidate_id=candidate["candidate_id"],
                        seed=seed,
                        error=error,
                    )
                frozen.save_state(state, state_path)


def mean_std(values: list[float], digits: int = 4) -> str:
    if not values:
        return ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def final_rows(args, state):
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
                        [float(record["trainable_parameters"]) for record in runs], 0
                    ),
                    "training_seconds": mean_std(
                        [record["training_seconds"] for record in runs], 1
                    ),
                    "peak_gpu_memory_mb": mean_std(
                        [record["peak_gpu_memory_mb"] for record in runs], 1
                    ),
                    "completed_seeds": len(runs),
                }
            )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    if not rows:
        path.write_text("No completed runs.\n", encoding="utf-8")
        return
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def winner_rows(args, state):
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
                    "configuration": json.dumps(winner["candidate"], sort_keys=True),
                }
            )
    return rows


def map_gmvip_rows(args, state):
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


def write_tables(args, state):
    for stem, rows in (
        ("headline", final_rows(args, state)),
        ("winners", winner_rows(args, state)),
        ("map_vs_gmvip", map_gmvip_rows(args, state)),
    ):
        write_csv(args.output_dir / f"{stem}.csv", rows)
        write_markdown(args.output_dir / f"{stem}.md", rows)


def build_metadata(args, grids, splits):
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
        "run_signature": frozen.config_hash(settings),
        "embedding_cache": str(args.embedding_cache),
        "device": str(args.device),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def verify_reproduction(train, test, sizes, split_seed, splits):
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
    grids = frozen.candidate_grids()
    metadata = build_metadata(args, grids, splits)
    state, state_path = frozen.load_state(args, metadata)

    if args.stage in ("smoke", "all"):
        smoke_splits = splits if 500 in splits else make_splits(train, test, [500], args.split_seed)
        run_smoke(args, grids, smoke_splits, state, state_path)
    if args.stage in ("tune", "all"):
        frozen.run_tuning(SPEC, args, grids, splits, state, state_path)
    if args.stage in ("final", "all"):
        run_final(args, splits, state, state_path)

    write_tables(args, state)
    frozen.save_state(state, state_path)
    print(f"state: {state_path}")
    print(f"headline: {args.output_dir / 'headline.md'}")


if __name__ == "__main__":
    main()
