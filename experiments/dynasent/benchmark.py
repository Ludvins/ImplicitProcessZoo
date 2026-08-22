"""Frozen-CLIP linear-head benchmark on three-class DynaSent.

Round 1 supplies the fitting, selection, calibration, and in-distribution test
data. Round 2 is used only as an adversarial-shift test set.

Examples
--------
python -m experiments.dynasent.benchmark --stage smoke
python -m experiments.dynasent.benchmark --stage all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from experiments.common import frozen_feature_classification as frozen

try:
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This experiment requires transformers.") from exc


METHODS = frozen.METHODS
LABELS = {"negative": 0, "neutral": 1, "positive": 2}
NUM_CLASSES = 3
FEATURE_DIM = 512
SPEC = frozen.FrozenFeatureSpec(FEATURE_DIM, NUM_CLASSES, train_fbnn_prior=False)
DATA_URL = "https://raw.githubusercontent.com/cgpotts/dynasent/main/dynasent-v1.1.zip"
FILES = {
    "round1_train": "dynasent-v1.1-round01-yelp-train.jsonl",
    "round1_dev": "dynasent-v1.1-round01-yelp-dev.jsonl",
    "round1_test": "dynasent-v1.1-round01-yelp-test.jsonl",
    "round2_test": "dynasent-v1.1-round02-dynabench-test.jsonl",
}


@dataclass(frozen=True)
class TextSplit:
    sentences: list[str]
    targets: Tensor
    text_ids: list[str]


@dataclass(frozen=True)
class ExperimentData:
    bundle: frozen.SplitBundle
    round2_test: frozen.Split
    full_size: int
    hashes: dict[str, str]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Frozen-CLIP DynaSent multiclass benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=("embeddings", "smoke", "tune", "final", "all"),
        default="all",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS + ("all",), default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dynasent"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dynasent/clip_multiclass"),
    )
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recompute-embeddings", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
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
            parser.error("Use --methods all alone, or list individual methods.")
        args.methods = list(METHODS)
    else:
        args.methods = list(dict.fromkeys(args.methods))
    args.seeds = list(dict.fromkeys(args.seeds))
    if not args.seeds:
        parser.error("At least one final seed is required.")
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

    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.archive is None:
        args.archive = args.data_dir / "dynasent-v1.1.zip"
    else:
        args.archive = args.archive.expanduser().resolve()
    if args.embedding_cache is None:
        args.embedding_cache = args.output_dir / "clip_text_embeddings.pt"
    else:
        args.embedding_cache = args.embedding_cache.expanduser().resolve()
    return args


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records_hash(splits: dict[str, TextSplit]) -> str:
    digest = hashlib.sha256()
    for name in sorted(splits):
        split = splits[name]
        for text_id, sentence, target in zip(split.text_ids, split.sentences, split.targets):
            record = [name, text_id, sentence, int(target)]
            digest.update(json.dumps(record, ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def download_archive(args) -> None:
    if args.archive.exists():
        return
    if not args.download:
        raise FileNotFoundError(f"DynaSent archive not found: {args.archive}")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.archive.with_suffix(".zip.part")
    print(f"downloading {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, temporary)
    temporary.replace(args.archive)


def extract_archive(args) -> dict[str, Path]:
    download_archive(args)
    found = {
        name: next(iter(args.data_dir.rglob(filename)), None) for name, filename in FILES.items()
    }
    if all(path is not None for path in found.values()):
        return {name: path for name, path in found.items() if path is not None}

    root = args.data_dir.resolve()
    with zipfile.ZipFile(args.archive) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if root != destination and root not in destination.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(root)

    result = {name: next(iter(root.rglob(filename)), None) for name, filename in FILES.items()}
    missing = [FILES[name] for name, path in result.items() if path is None]
    if missing:
        raise FileNotFoundError(f"Archive is missing: {', '.join(missing)}")
    return {name: path for name, path in result.items() if path is not None}


def load_jsonl(path: Path) -> TextSplit:
    sentences = []
    targets = []
    text_ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            label = record.get("gold_label")
            if label not in LABELS:
                continue
            sentence = " ".join(str(record["sentence"]).split())
            text_id = str(record["text_id"])
            if not sentence or not text_id:
                raise ValueError(f"Empty sentence or id in {path}")
            sentences.append(sentence)
            targets.append(LABELS[label])
            text_ids.append(text_id)
    if not sentences:
        raise ValueError(f"No ternary examples in {path}")
    if len(text_ids) != len(set(text_ids)):
        raise ValueError(f"Duplicate text ids in {path}")
    tensor = torch.tensor(targets, dtype=torch.long)
    if set(tensor.tolist()) != set(range(NUM_CLASSES)):
        raise ValueError(f"Not all three labels occur in {path}")
    return TextSplit(sentences, tensor, text_ids)


def load_text_data(args):
    paths = extract_archive(args)
    splits = {name: load_jsonl(path) for name, path in paths.items()}
    all_ids = [text_id for split in splits.values() for text_id in split.text_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("DynaSent text ids overlap across official splits.")
    metadata = {
        "archive": str(args.archive),
        "archive_sha256": file_hash(args.archive),
        "records_sha256": records_hash(splits),
        "label_mapping": LABELS,
        "splits": {
            name: {
                "examples": len(split.sentences),
                "class_counts": {
                    label: int(split.targets.eq(index).sum()) for label, index in LABELS.items()
                },
                "file": str(paths[name]),
            }
            for name, split in splits.items()
        },
    }
    return splits, metadata


def embedding_signature(args, metadata):
    return {
        "clip_model": args.clip_model,
        "max_length": 77,
        "pooling": "clip_text_projection",
        "l2_normalized": True,
        "records_sha256": metadata["records_sha256"],
        "label_mapping": LABELS,
    }


def embed_texts(model, tokenizer, split, args, name):
    features = []
    truncated = 0
    model.eval()
    with torch.inference_mode():
        batches = range(0, len(split.sentences), args.embedding_batch_size)
        for start in tqdm(batches, desc=f"embedding {name}", disable=args.no_tqdm):
            texts = split.sentences[start : start + args.embedding_batch_size]
            lengths = tokenizer(
                texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_length=True,
            )["length"]
            truncated += sum(int(length) > 77 for length in lengths)
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens = {key: value.to(args.device) for key, value in tokens.items()}
            with torch.autocast(
                device_type=args.device.type,
                dtype=torch.float16,
                enabled=args.device.type == "cuda",
            ):
                encoded = model(**tokens).text_embeds
            features.append(F.normalize(encoded.float(), dim=-1).cpu())
    print(f"{name}: {len(split.sentences)} examples, {truncated} truncated at 77 tokens")
    result = torch.cat(features)
    if result.shape != (len(split.sentences), FEATURE_DIM) or not torch.isfinite(result).all():
        raise ValueError(f"Invalid embedding tensor for {name}: {tuple(result.shape)}")
    return frozen.Split(result, split.targets.clone())


def load_or_build_embeddings(args, text_splits, metadata):
    signature = embedding_signature(args, metadata)
    if args.embedding_cache.exists() and not args.recompute_embeddings:
        payload = torch.load(args.embedding_cache, map_location="cpu", weights_only=True)
        if payload.get("signature") != signature:
            raise RuntimeError(
                "The embedding cache signature differs. Use --recompute-embeddings or "
                "a different --output-dir."
            )
        result = {
            name: frozen.Split(values["features"].float(), values["targets"].long())
            for name, values in payload["splits"].items()
        }
    else:
        tokenizer = CLIPTokenizerFast.from_pretrained(
            args.clip_model,
            local_files_only=not args.download,
        )
        model = CLIPTextModelWithProjection.from_pretrained(
            args.clip_model,
            local_files_only=not args.download,
        ).to(args.device)
        model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError("CLIP must remain frozen.")
        result = {
            name: embed_texts(model, tokenizer, split, args, name)
            for name, split in text_splits.items()
        }
        payload = {
            "signature": signature,
            "splits": {
                name: {"features": split.features, "targets": split.targets}
                for name, split in result.items()
            },
        }
        frozen.atomic_torch_save(args.embedding_cache, payload)
        frozen.atomic_write_json(
            args.embedding_cache.with_suffix(".json"),
            {"signature": signature, "dataset": metadata},
        )
        del model
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    if set(result) != set(FILES):
        raise ValueError(f"Embedding cache splits must be {sorted(FILES)}")
    for name, split in result.items():
        if split.features.shape != (len(text_splits[name].sentences), FEATURE_DIM):
            raise ValueError(f"Unexpected cached shape for {name}: {tuple(split.features.shape)}")
        if not torch.equal(split.targets, text_splits[name].targets):
            raise ValueError(f"Cached labels differ for {name}")
    return result


def stratified_dev_split(targets: Tensor, seed: int):
    generator = torch.Generator().manual_seed(seed + 82_019)
    selection = []
    calibration = []
    for class_index in range(NUM_CLASSES):
        indices = torch.where(targets.eq(class_index))[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        midpoint = len(indices) // 2
        selection.append(indices[:midpoint])
        calibration.append(indices[midpoint:])

    def shuffled(parts):
        values = torch.cat(parts)
        return values[torch.randperm(len(values), generator=generator)]

    return shuffled(selection), shuffled(calibration)


def split_hash(indices: Tensor) -> str:
    return frozen.tensor_hash(indices)


def make_experiment_data(embeddings, split_seed):
    train = embeddings["round1_train"]
    dev = embeddings["round1_dev"]
    selection_indices, calibration_indices = stratified_dev_split(dev.targets, split_seed)
    selection = frozen.take(dev, selection_indices)
    calibration = frozen.take(dev, calibration_indices)
    final_train = frozen.Split(
        torch.cat([train.features, selection.features]),
        torch.cat([train.targets, selection.targets]),
    )
    final_indices = torch.arange(len(final_train.features), dtype=torch.long)
    train_indices = torch.arange(len(train.features), dtype=torch.long)
    hashes = {
        "round1_train": split_hash(train_indices),
        "selection_from_round1_dev": split_hash(selection_indices),
        "calibration_from_round1_dev": split_hash(calibration_indices),
        "final_train": split_hash(final_indices),
        "round1_test_targets": frozen.tensor_hash(embeddings["round1_test"].targets),
        "round2_test_targets": frozen.tensor_hash(embeddings["round2_test"].targets),
    }
    if set(selection_indices.tolist()) & set(calibration_indices.tolist()):
        raise AssertionError("Selection and calibration splits overlap.")
    if len(selection_indices) + len(calibration_indices) != len(dev.features):
        raise AssertionError("Round 1 dev was not partitioned exactly.")
    repeated = stratified_dev_split(dev.targets, split_seed)
    if not torch.equal(selection_indices, repeated[0]) or not torch.equal(
        calibration_indices, repeated[1]
    ):
        raise AssertionError("Dev split is not deterministic.")

    bundle = frozen.SplitBundle(
        tune_train=train,
        selection=selection,
        calibration=calibration,
        final_train=final_train,
        test=embeddings["round1_test"],
        indices={
            "tune_train": train_indices,
            "selection": selection_indices,
            "calibration": calibration_indices,
            "final_train": final_indices,
        },
        hashes=hashes,
    )
    return ExperimentData(bundle, embeddings["round2_test"], len(final_train.features), hashes)


def balanced_subset(split, per_class, seed):
    generator = torch.Generator().manual_seed(seed)
    parts = []
    for class_index in range(NUM_CLASSES):
        indices = torch.where(split.targets.eq(class_index))[0]
        indices = indices[torch.randperm(len(indices), generator=generator)][:per_class]
        parts.append(indices)
    joined = torch.cat(parts)
    joined = joined[torch.randperm(len(joined), generator=generator)]
    return frozen.take(split, joined)


def run_smoke(args, grids, data, state, state_path):
    train = balanced_subset(data.bundle.tune_train, 128, args.split_seed + 1)
    validation = balanced_subset(data.bundle.selection, 32, args.split_seed + 2)
    for method in args.methods:
        if frozen.completed(state["smoke"].get(method), args):
            continue
        print(f"smoke method={method}")
        try:
            model, fit = frozen.fit_model(
                SPEC,
                method,
                grids[method][0],
                train,
                None,
                epochs=1,
                seed=0,
                args=args,
                checkpoint_dir=args.output_dir / "checkpoints" / "smoke" / method,
            )
            logp, shape = frozen.predictive_log_probabilities(
                SPEC,
                method,
                model,
                validation,
                args.device,
                args.eval_batch_size,
                min(4, args.eval_samples),
            )
            if shape[2] != NUM_CLASSES or not torch.isfinite(logp).all():
                raise AssertionError("Invalid smoke predictions.")
            prior_trainable = frozen.trainable_prior(model, method)
            if method in ("vip", "ftip", "gmvip", "sip") and not prior_trainable:
                raise AssertionError(f"{method} prior is not trainable.")
            if method == "fbnn" and prior_trainable:
                raise AssertionError("FBNN prior must be fixed.")
            inducing_trainable = frozen.trainable_inducing_locations(model, method)
            if method in ("gmvip", "sip") and not inducing_trainable:
                raise AssertionError(f"{method} inducing locations are not trainable.")
            state["smoke"][method] = {
                "status": "complete",
                "loss": fit["history"][-1]["training_loss"],
                "prediction_shape": shape,
                "trainable_parameters": fit["trainable_parameters"],
                "trainable_prior": prior_trainable,
                "trainable_inducing_locations": inducing_trainable,
            }
            del model
        except (RuntimeError, FloatingPointError, ValueError, AssertionError) as error:
            state["smoke"][method] = {"status": "failed", "error": str(error)}
            frozen.record_failure(
                state,
                state_path,
                stage="smoke",
                size=data.full_size,
                method=method,
                candidate_id=None,
                seed=0,
                error=error,
            )
        frozen.save_state(state, state_path)


def macro_f1(probabilities: Tensor, targets: Tensor) -> float:
    predictions = probabilities.argmax(dim=1)
    scores = []
    for class_index in range(NUM_CLASSES):
        predicted = predictions.eq(class_index)
        actual = targets.eq(class_index)
        tp = (predicted & actual).sum().float()
        fp = (predicted & ~actual).sum().float()
        fn = (~predicted & actual).sum().float()
        denominator = 2 * tp + fp + fn
        scores.append(torch.where(denominator > 0, 2 * tp / denominator, 0.0))
    return float(torch.stack(scores).mean())


def extra_metrics(logp: Tensor, targets: Tensor):
    probabilities = logp.exp()
    one_hot = F.one_hot(targets, NUM_CLASSES).float()
    confidence, prediction = probabilities.max(dim=1)
    errors = prediction.ne(targets).float()
    order = confidence.argsort(descending=True)
    risk_curve = errors[order].cumsum(0) / torch.arange(1, len(errors) + 1)
    entropy = -(probabilities * logp).sum(dim=1)
    return {
        "macro_f1": macro_f1(probabilities, targets),
        "brier": float((probabilities - one_hot).square().sum(dim=1).mean()),
        "predictive_entropy": float(entropy.mean()),
        "aurc": float(risk_curve.mean()),
    }


def all_metrics(logp, targets, bins):
    return {**frozen.classification_metrics(logp, targets, bins), **extra_metrics(logp, targets)}


def entropy_scores(logp):
    probabilities = logp.exp()
    return -(probabilities * logp).sum(dim=1)


def binary_auroc(negative_scores, positive_scores):
    scores = torch.cat([negative_scores, positive_scores])
    labels = torch.cat([torch.zeros(len(negative_scores)), torch.ones(len(positive_scores))])
    order = scores.argsort(descending=True)
    labels = labels[order]
    tpr = labels.cumsum(0) / max(1, len(positive_scores))
    fpr = (1.0 - labels).cumsum(0) / max(1, len(negative_scores))
    tpr = torch.cat([torch.zeros(1), tpr])
    fpr = torch.cat([torch.zeros(1), fpr])
    return float(torch.trapezoid(tpr, fpr))


def run_final(args, data, state, state_path):
    size_key = str(data.full_size)
    state["final"].setdefault(size_key, {})
    tests = {"round1_id": data.bundle.test, "round2_shift": data.round2_test}
    for method in args.methods:
        records = state["final"][size_key].setdefault(method, {})
        winner = state.get("winners", {}).get(size_key, {}).get(method)
        if winner is None:
            error = RuntimeError(f"No tuned winner for {method}.")
            frozen.record_failure(
                state,
                state_path,
                stage="final",
                size=data.full_size,
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
            if frozen.completed(records.get(seed_key), args):
                continue
            print(f"final method={method} seed={seed} epochs={epochs}")
            try:
                model, fit = frozen.fit_model(
                    SPEC,
                    method,
                    candidate,
                    data.bundle.final_train,
                    None,
                    epochs=epochs,
                    seed=seed,
                    args=args,
                    checkpoint_dir=args.output_dir / "checkpoints" / "final" / method / seed_key,
                )
                calibration_logp, _ = frozen.predictive_log_probabilities(
                    SPEC,
                    method,
                    model,
                    data.bundle.calibration,
                    args.device,
                    args.eval_batch_size,
                    args.eval_samples,
                )
                temperature = frozen.fit_temperature(
                    calibration_logp,
                    data.bundle.calibration.targets,
                    args.temperature_iterations,
                )
                calibrated = frozen.apply_temperature(calibration_logp, temperature)
                if not torch.equal(calibration_logp.argmax(1), calibrated.argmax(1)):
                    raise AssertionError("Temperature scaling changed calibration classes.")

                test_results = {}
                calibrated_logps = {}
                for name, split in tests.items():
                    raw_logp, _ = frozen.predictive_log_probabilities(
                        SPEC,
                        method,
                        model,
                        split,
                        args.device,
                        args.eval_batch_size,
                        args.eval_samples,
                    )
                    scaled_logp = frozen.apply_temperature(raw_logp, temperature)
                    if not torch.equal(raw_logp.argmax(1), scaled_logp.argmax(1)):
                        raise AssertionError(f"Temperature scaling changed classes on {name}.")
                    test_results[name] = {
                        "raw": all_metrics(raw_logp, split.targets, args.ece_bins),
                        "calibrated": all_metrics(scaled_logp, split.targets, args.ece_bins),
                    }
                    calibrated_logps[name] = scaled_logp

                records[seed_key] = {
                    "status": "complete",
                    "candidate": candidate,
                    "epochs": epochs,
                    "temperature": temperature,
                    "calibration": all_metrics(
                        calibrated,
                        data.bundle.calibration.targets,
                        args.ece_bins,
                    ),
                    "tests": test_results,
                    "round2_entropy_ood_auroc": binary_auroc(
                        entropy_scores(calibrated_logps["round1_id"]),
                        entropy_scores(calibrated_logps["round2_shift"]),
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
                records[seed_key] = {
                    "status": "failed",
                    "candidate": candidate,
                    "epochs": epochs,
                    "error": str(error),
                }
                frozen.record_failure(
                    state,
                    state_path,
                    stage="final",
                    size=data.full_size,
                    method=method,
                    candidate_id=candidate["candidate_id"],
                    seed=seed,
                    error=error,
                )
            frozen.save_state(state, state_path)


def mean_std(values, digits=4):
    if not values:
        return ""
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f} +/- {deviation:.{digits}f}"


def write_csv(path, rows):
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


def write_reports(args, data, state):
    size_key = str(data.full_size)
    headline = []
    raw = []
    for method in args.methods:
        records = state.get("final", {}).get(size_key, {}).get(method, {})
        runs = [record for record in records.values() if record.get("status") == "complete"]
        for evaluation in ("round1_id", "round2_shift"):
            if not runs:
                continue
            metrics = [record["tests"][evaluation]["calibrated"] for record in runs]
            headline.append(
                {
                    "method": method.upper(),
                    "evaluation": evaluation,
                    "accuracy": mean_std([item["accuracy"] for item in metrics]),
                    "nll": mean_std([item["nll"] for item in metrics]),
                    "ece": mean_std([item["ece"] for item in metrics]),
                    "macro_f1": mean_std([item["macro_f1"] for item in metrics]),
                    "brier": mean_std([item["brier"] for item in metrics]),
                    "entropy": mean_std([item["predictive_entropy"] for item in metrics]),
                    "aurc": mean_std([item["aurc"] for item in metrics]),
                    "temperature": mean_std([record["temperature"] for record in runs]),
                    "ood_entropy_auroc": mean_std(
                        [record["round2_entropy_ood_auroc"] for record in runs]
                    ),
                    "trainable_parameters": mean_std(
                        [float(record["trainable_parameters"]) for record in runs], 0
                    ),
                    "training_seconds": mean_std(
                        [record["training_seconds"] for record in runs], 1
                    ),
                    "peak_gpu_memory_mb": mean_std(
                        [record["peak_gpu_memory_mb"] for record in runs], 1
                    ),
                    "seeds": len(runs),
                }
            )
        for seed, record in records.items():
            if record.get("status") != "complete":
                continue
            for evaluation, values in record["tests"].items():
                raw.append(
                    {
                        "method": method,
                        "seed": seed,
                        "evaluation": evaluation,
                        **values["raw"],
                        **{
                            f"calibrated_{key}": value
                            for key, value in values["calibrated"].items()
                        },
                        "temperature": record["temperature"],
                        "ood_entropy_auroc": record["round2_entropy_ood_auroc"],
                        "trainable_parameters": record["trainable_parameters"],
                        "training_seconds": record["training_seconds"],
                        "peak_gpu_memory_mb": record["peak_gpu_memory_mb"],
                    }
                )

    winners = []
    for method in args.methods:
        winner = state.get("winners", {}).get(size_key, {}).get(method)
        if winner is not None:
            winners.append(
                {
                    "method": method.upper(),
                    "selected_epoch": winner["selected_epoch"],
                    "selection_nll": winner["selection"]["nll"],
                    "selection_accuracy": winner["selection"]["accuracy"],
                    "configuration": json.dumps(winner["candidate"], sort_keys=True),
                }
            )

    for stem, rows in (("headline", headline), ("winners", winners)):
        write_csv(args.output_dir / f"{stem}.csv", rows)
        write_markdown(args.output_dir / f"{stem}.md", rows)
    write_csv(args.output_dir / "raw_metrics.csv", raw)
    frozen.atomic_write_json(args.output_dir / "failures.json", state["failures"])


def build_metadata(args, data, grids, dataset_metadata):
    settings = {
        "dataset": "DynaSent v1.1",
        "protocol": "round1_fit_round1_id_round2_shift",
        "dataset_records_sha256": dataset_metadata["records_sha256"],
        "clip_model": args.clip_model,
        "feature_dim": FEATURE_DIM,
        "num_classes": NUM_CLASSES,
        "label_mapping": LABELS,
        "full_training_size": data.full_size,
        "selection_size": len(data.bundle.selection.features),
        "calibration_size": len(data.bundle.calibration.features),
        "round1_test_size": len(data.bundle.test.features),
        "round2_test_size": len(data.round2_test.features),
        "split_hashes": data.hashes,
        "methods": args.methods,
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
        "fbnn_prior_trainable": False,
        "candidate_grids": {method: grids[method] for method in args.methods},
    }
    return {
        **settings,
        "run_signature": frozen.config_hash(settings),
        "embedding_cache": str(args.embedding_cache),
        "device": str(args.device),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main(argv=None):
    args = parse_args(argv)
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_splits, dataset_metadata = load_text_data(args)
    embeddings = load_or_build_embeddings(args, text_splits, dataset_metadata)
    print(f"embeddings: {args.embedding_cache}")
    if args.stage == "embeddings":
        return

    data = make_experiment_data(embeddings, args.split_seed)
    args.sizes = [data.full_size]
    grids = frozen.candidate_grids()
    metadata = build_metadata(args, data, grids, dataset_metadata)
    state, state_path = frozen.load_state(args, metadata)

    if args.stage in ("smoke", "all"):
        run_smoke(args, grids, data, state, state_path)
    if args.stage in ("tune", "all"):
        frozen.run_tuning(
            SPEC,
            args,
            grids,
            {data.full_size: data.bundle},
            state,
            state_path,
        )
    if args.stage in ("final", "all"):
        run_final(args, data, state, state_path)

    write_reports(args, data, state)
    frozen.save_state(state, state_path)
    print(f"state: {state_path}")
    print(f"headline: {args.output_dir / 'headline.md'}")


if __name__ == "__main__":
    main()
