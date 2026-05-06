"""Multiclass classification benchmark.

Runs MNIST, FashionMNIST, and CIFAR10 with VIP, FTIP, MFVI, FBNN, or TFSVI. Each run writes a JSON result file and, by default, a checkpoint.

Example:
    python -m scripts.classification_benchmark --model ftip --dataset MNIST
"""

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.fbnn import FBNN
from src.flows import CouplingFlow, SplineCoupling1x1Flow, SplineCouplingFlow
from src.ftip import FTIP
from src.mfvi import MFVI
from src.priors.generative_functions import (
    BayesianCNN,
    BayesianCNNFull,
    BayesianNN,
    BayesianResNet,
    BayesLinear,
    GP,
    SimplerBayesLinear,
)
from src.tfsvi import TFSVI
from src.utils.dataset import get_dataset
from src.utils.metrics import MetricsClassification
from src.utils.utils import infinite_loader
from src.vip import VIP


CLASSIFICATION_DATASETS = ["MNIST", "FashionMNIST", "CIFAR10"]
ACTIVATIONS = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}
LAYER_MODELS = {
    "BayesLinear": BayesLinear,
    "SimplerBayesLinear": SimplerBayesLinear,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Multiclass classification benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", required=True,
                   choices=["vip", "ftip", "mfvi", "fbnn", "tfsvi"])
    p.add_argument("--dataset", required=True,
                   help="Dataset name or 'all'.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--device", default=None)
    p.add_argument("--output_dir", default="results")

    p.add_argument("--gen_model", choices=["bnn", "cnn", "bayes_cnn", "resnet"],
                   default="cnn")
    p.add_argument("--resnet_backbone", default="resnet18")
    p.add_argument("--resnet_cifar_stem", action="store_true", default=True)
    p.add_argument("--no_resnet_cifar_stem", action="store_true")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[120, 84])
    p.add_argument("--activation", choices=list(ACTIVATIONS.keys()), default="relu")
    p.add_argument("--layer_model", choices=list(LAYER_MODELS.keys()),
                   default="BayesLinear")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight_log_sigma_init", type=float, default=None)

    p.add_argument("--regression_coeffs", type=int, default=20)
    p.add_argument("--bb_alpha", type=float, default=0.5)
    p.add_argument("--use_prior_regularizer", action="store_true", default=False)
    p.add_argument("--regularizer_mode", choices=["evidence", "KL"],
                   default="evidence")
    p.add_argument("--prior_regularizer_scaler", type=float, default=1.0)

    p.add_argument("--flow_type", choices=["affine", "spline", "spline_1x1"],
                   default="spline_1x1")
    p.add_argument("--flow_num_bins", type=int, default=8)
    p.add_argument("--flow_domain", type=float, default=3.0)
    p.add_argument("--flow_depth", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--eval_samples", type=int, default=1000)
    p.add_argument("--warm_start_from", default=None)
    p.add_argument("--learnable_affine", action="store_true", default=True)
    p.add_argument("--no_learnable_affine", action="store_true")

    p.add_argument("--auto_warm_start", action="store_true", default=True)
    p.add_argument("--no_auto_warm_start", action="store_true")
    p.add_argument("--vip_epochs", type=int, default=None)
    p.add_argument("--vip_iterations", type=int, default=None)
    p.add_argument("--vip_lr", type=float, default=1e-3)
    p.add_argument("--ftip_lr", type=float, default=1e-4)

    p.add_argument("--fbnn_prior", choices=["gp", "bnn"], default="gp")
    p.add_argument("--fbnn_freeze_prior", action="store_true", default=False)
    p.add_argument("--fbnn_gp_inner_dim", type=int, default=10)
    p.add_argument("--fbnn_num_measurement", type=int, default=20)
    p.add_argument("--fbnn_num_context", type=int, default=20)
    p.add_argument("--fbnn_context_std", type=float, default=2.0)
    p.add_argument("--fbnn_lambda_kl", type=float, default=1.0)
    p.add_argument("--fbnn_num_eval_samples", type=int, default=200)

    p.add_argument("--tfsvi_sigma_prior", type=float, default=1.0)
    p.add_argument("--tfsvi_S_ctx", type=int, default=5)
    p.add_argument("--tfsvi_K_ctx", type=int, default=20)
    p.add_argument("--tfsvi_num_train_samples", type=int, default=20)
    p.add_argument("--tfsvi_num_eval_samples", type=int, default=200)
    p.add_argument("--mfvi_num_eval_samples", type=int, default=200)

    p.add_argument("--save_checkpoint", action="store_true", default=True)
    p.add_argument("--no_save_checkpoint", action="store_true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--cosine_annealing", action="store_true", default=True)
    p.add_argument("--no_cosine_annealing", action="store_true")
    args = p.parse_args()

    if args.no_cosine_annealing:
        args.cosine_annealing = False
    if args.no_save_checkpoint:
        args.save_checkpoint = False
    if args.no_auto_warm_start:
        args.auto_warm_start = False
    if args.no_learnable_affine:
        args.learnable_affine = False
    if args.no_resnet_cifar_stem:
        args.resnet_cifar_stem = False
    if args.warm_start_from:
        args.auto_warm_start = False
    if args.iterations is None and args.epochs is None:
        args.epochs = 100
    if args.vip_epochs is None and args.vip_iterations is None:
        args.vip_epochs = args.epochs
        args.vip_iterations = args.iterations
    if args.weight_log_sigma_init is None:
        args.weight_log_sigma_init = -1.0 if args.model in ("mfvi", "fbnn") else 0.0
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _ensure_layers_alias(gen_fn):
    if hasattr(gen_fn, "layers"):
        return
    pieces = []
    for name in ("conv1", "conv2"):
        layer = getattr(gen_fn, name, None)
        if layer is not None and hasattr(layer, "fix_random_noise"):
            pieces.append(layer)
    if hasattr(gen_fn, "head"):
        pieces.extend(list(gen_fn.head))
    if pieces:
        object.__setattr__(gen_fn, "layers", pieces)


def _set_bnn_num_samples(gen_fn, num_samples):
    old = gen_fn.num_samples
    gen_fn.num_samples = num_samples
    if not hasattr(gen_fn, "layers"):
        return
    for layer in gen_fn.layers:
        if hasattr(layer, "num_samples"):
            layer.num_samples = num_samples
        if (hasattr(layer, "fix_random_noise") and layer.fix_random_noise
                and num_samples != old):
            layer.noise = layer.get_noise(first_call=True)


def _set_bnn_fix_random_noise(gen_fn, fix_random_noise):
    if hasattr(gen_fn, "fix_random_noise"):
        gen_fn.fix_random_noise = fix_random_noise
    if not hasattr(gen_fn, "layers"):
        return
    for layer in gen_fn.layers:
        if hasattr(layer, "fix_random_noise"):
            layer.fix_random_noise = fix_random_noise


def _build_gen_fn(args, train_dataset, num_classes, num_samples):
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    common = dict(
        input_dim=train_dataset.input_dim,
        num_samples=num_samples,
        output_dim=num_classes,
        layer_model=LAYER_MODELS[args.layer_model],
        dropout=args.dropout,
        weight_log_sigma_init=args.weight_log_sigma_init,
        device=device,
        seed=args.seed,
        dtype=dtype,
    )
    if args.gen_model == "cnn":
        return BayesianCNN(head_dims=args.hidden_dims, **common)
    if args.gen_model == "bayes_cnn":
        return BayesianCNNFull(head_dims=args.hidden_dims, **common)
    if args.gen_model == "resnet":
        head_dims = args.hidden_dims if args.hidden_dims != [120, 84] else []
        return BayesianResNet(
            head_dims=head_dims,
            backbone=args.resnet_backbone,
            cifar_stem=args.resnet_cifar_stem,
            **common,
        )
    return BayesianNN(
        structure=args.hidden_dims,
        activation=ACTIVATIONS[args.activation],
        **common,
    )


def _build_flow(args, input_dim, device, dtype):
    flow_kwargs = dict(
        depth=args.flow_depth,
        input_dim=input_dim,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    if args.flow_type == "affine":
        return CouplingFlow(**flow_kwargs)
    if args.flow_type == "spline":
        return SplineCouplingFlow(
            **flow_kwargs, num_bins=args.flow_num_bins, B=args.flow_domain,
        )
    return SplineCoupling1x1Flow(
        **flow_kwargs, num_bins=args.flow_num_bins, B=args.flow_domain,
    )


def build_model(args, train_dataset, num_classes, model_type=None):
    if model_type is None:
        model_type = args.model
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    gen_fn = _build_gen_fn(args, train_dataset, num_classes, args.regression_coeffs)
    common = dict(
        generative_function=gen_fn,
        num_regression_coeffs=args.regression_coeffs,
        output_dim=num_classes,
        likelihood="multiclass",
        num_data=len(train_dataset),
        num_classes=num_classes,
        bb_alpha=args.bb_alpha,
        use_prior_regularizer=args.use_prior_regularizer,
        regularizer_mode=args.regularizer_mode,
        prior_regularizer_scaler=args.prior_regularizer_scaler,
        y_mean=0.0,
        y_std=1.0,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    if model_type == "vip":
        return VIP(**common)
    if model_type == "ftip":
        flow = _build_flow(
            args, args.regression_coeffs * num_classes, device, dtype,
        )
        return FTIP(**common, flow=flow, num_samples=args.num_samples)
    if model_type == "mfvi":
        _ensure_layers_alias(gen_fn)
        _set_bnn_fix_random_noise(gen_fn, False)
        return MFVI(
            generative_function=gen_fn,
            output_dim=num_classes,
            likelihood="multiclass",
            num_data=len(train_dataset),
            num_samples=args.regression_coeffs,
            num_classes=num_classes,
            bb_alpha=args.bb_alpha,
            y_mean=0.0,
            y_std=1.0,
            device=device,
            dtype=dtype,
        )
    if model_type == "tfsvi":
        _ensure_layers_alias(gen_fn)
        _set_bnn_num_samples(gen_fn, 1)
        return TFSVI(
            input_dim=train_dataset.input_dim,
            output_dim=num_classes,
            structure=args.hidden_dims,
            activation=ACTIVATIONS[args.activation],
            likelihood="multiclass",
            num_data=len(train_dataset),
            sigma_prior=args.tfsvi_sigma_prior,
            num_samples=args.tfsvi_num_train_samples,
            num_classes=num_classes,
            bb_alpha=args.bb_alpha,
            S_ctx=args.tfsvi_S_ctx,
            K_ctx=args.tfsvi_K_ctx,
            y_mean=0.0,
            y_std=1.0,
            generative_function=gen_fn,
            device=device,
            dtype=dtype,
        )
    if model_type == "fbnn":
        _ensure_layers_alias(gen_fn)
        if args.fbnn_prior == "gp":
            prior = GP(
                input_dim=train_dataset.input_dim,
                output_dim=num_classes,
                inner_layer_dim=args.fbnn_gp_inner_dim,
                seed=args.seed,
                device=device,
                dtype=dtype,
            )
        else:
            prior = _build_gen_fn(
                args, train_dataset, num_classes, args.regression_coeffs,
            )
        return FBNN(
            generative_function=gen_fn,
            prior_function=prior,
            output_dim=num_classes,
            likelihood="multiclass",
            num_data=len(train_dataset),
            num_samples=args.regression_coeffs,
            num_measurement=args.fbnn_num_measurement,
            num_context=args.fbnn_num_context,
            context_std=args.fbnn_context_std,
            bb_alpha=args.bb_alpha,
            lambda_kl=args.fbnn_lambda_kl,
            num_classes=num_classes,
            y_mean=0.0,
            y_std=1.0,
            freeze_prior=args.fbnn_freeze_prior,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _eval_samples_for(args, model_type):
    if model_type == "fbnn":
        return args.fbnn_num_eval_samples
    if model_type == "tfsvi":
        return args.tfsvi_num_eval_samples
    if model_type == "mfvi":
        return args.mfvi_num_eval_samples
    return args.eval_samples


def evaluate(model, dataset, args, model_type=None, batch_size=1024, light=False):
    if model_type is None:
        model_type = args.model
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    x = torch.tensor(dataset.inputs, dtype=dtype, device=device)
    y = torch.tensor(dataset.targets, dtype=torch.long, device=device)

    metrics = MetricsClassification(num_data=len(dataset), device=device)
    was_training = model.training
    model.eval()

    old_mc = getattr(model, "num_mc_samples", None)
    if model_type == "vip":
        model.num_mc_samples = args.eval_samples
    eval_samples = _eval_samples_for(args, model_type)
    fbnn_old_samples = None
    if model_type == "fbnn":
        fbnn_old_samples = model.num_samples
        model._set_num_samples(eval_samples)
        model.num_samples = eval_samples

    with torch.no_grad():
        coeffs = None
        if model_type == "ftip":
            coeffs = model.sample_flow_coefficients(args.eval_samples)
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            yb = y[i:i + batch_size]
            if model_type == "ftip":
                samples, _ = model.forward_with_coefficients(xb, coeffs)
            elif model_type == "vip":
                samples, _ = model(xb)
            else:
                samples = model.predict_y_samples(xb, eval_samples)
            metrics.update(yb, loss=torch.tensor(0.0), mean_pred=samples)

    if old_mc is not None:
        model.num_mc_samples = old_mc
    if fbnn_old_samples is not None:
        model._set_num_samples(fbnn_old_samples)
        model.num_samples = fbnn_old_samples
    if was_training:
        model.train()
    d = metrics.get_dict()
    if light:
        return {"Error": d["Error"], "NLL": d["NLL"]}
    return d


def train_with_metrics(model, train_loader, train_test_dataset,
                       validation_dataset, args, lr=None, epochs=None,
                       iterations=None, model_type=None, desc="Training"):
    if lr is None:
        lr = args.lr
    if model_type is None:
        model_type = args.model
    if epochs is None and iterations is None:
        epochs = args.epochs
        iterations = args.iterations

    device = torch.device(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if args.cosine_annealing:
        if iterations is not None:
            t_max = max(1, math.ceil(iterations / len(train_loader)))
        else:
            t_max = max(1, epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=lr / 100,
        )

    losses = []
    metrics_history = {"iterations": [], "train": [], "validation": []}
    if model_type == "tfsvi":
        all_inputs = [inputs for inputs, _ in train_loader]
        model._train_inputs = torch.cat(all_inputs, dim=0).to(device)

    model.train()
    if iterations is not None:
        data_stream = infinite_loader(train_loader)
        iters_per_epoch = len(train_loader)
        loop = tqdm(range(iterations), unit=" iter", desc=desc)
        for i in loop:
            inputs, target = next(data_stream)
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = model._train_step(optimizer, inputs, target)
            losses.append(loss.item())
            if scheduler is not None and (i + 1) % iters_per_epoch == 0:
                scheduler.step()
            if (i + 1) % args.eval_every == 0:
                metrics_history["iterations"].append(i + 1)
                metrics_history["train"].append(evaluate(
                    model, train_test_dataset, args,
                    model_type=model_type, light=True,
                ))
                metrics_history["validation"].append(evaluate(
                    model, validation_dataset, args,
                    model_type=model_type, light=True,
                ))
            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    else:
        loop = tqdm(range(epochs), unit=" epoch", desc=desc)
        step = 0
        for _ in loop:
            for inputs, target in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                loss = model._train_step(optimizer, inputs, target)
                losses.append(loss.item())
                step += 1
                if step % args.eval_every == 0:
                    metrics_history["iterations"].append(step)
                    metrics_history["train"].append(evaluate(
                        model, train_test_dataset, args,
                        model_type=model_type, light=True,
                    ))
                    metrics_history["validation"].append(evaluate(
                        model, validation_dataset, args,
                        model_type=model_type, light=True,
                    ))
            if scheduler is not None:
                scheduler.step()
            loop.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    diagnostics = {
        "KLs": [float(v) for v in getattr(model, "KLs", [])],
        "bb_alphas": [float(v) for v in getattr(model, "bb_alphas", [])],
    }
    if hasattr(model, "prior_regularizers"):
        diagnostics["prior_regularizers"] = [
            float(v) for v in model.prior_regularizers
        ]
    if model_type == "ftip":
        diagnostics["base_KLs"] = [float(v) for v in model.base_KLs]
        diagnostics["flow_ldj"] = [float(v) for v in model.flow_ldj]
    return losses, metrics_history, diagnostics


def _ckpt_path(args, dataset_name, model_type):
    alpha_tag = f"_alpha{args.bb_alpha}"
    layer_tag = "_simpler" if args.layer_model == "SimplerBayesLinear" else "_bayes"
    gen_tag = f"_{args.gen_model}" if args.gen_model != "bnn" else ""
    return os.path.join(
        args.output_dir,
        f"{model_type}_{dataset_name}{alpha_tag}{layer_tag}{gen_tag}_seed{args.seed}.pt",
    )


def _build_result(dataset_name, model_type, model, args, train_loader,
                  train_test_dataset, test_dataset, lr=None,
                  epochs=None, iterations=None, desc="Training"):
    if lr is None:
        lr = args.lr
    t0 = time.time()
    losses, metrics_history, diagnostics = train_with_metrics(
        model, train_loader, train_test_dataset, test_dataset, args,
        lr=lr, epochs=epochs, iterations=iterations,
        model_type=model_type, desc=desc,
    )
    train_time = time.time() - t0
    train_metrics = evaluate(
        model, train_test_dataset, args, model_type=model_type,
    )
    test_metrics = evaluate(model, test_dataset, args, model_type=model_type)
    result = {
        "dataset": dataset_name,
        "model": model_type,
        "task": "multiclass",
        "hyperparameters": {
            "lr": lr,
            "batch_size": args.batch_size,
            "iterations": iterations if iterations is not None else args.iterations,
            "epochs": epochs if epochs is not None else args.epochs,
            "cosine_annealing": args.cosine_annealing,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": args.device,
            "gen_model": args.gen_model,
            "hidden_dims": args.hidden_dims,
            "activation": args.activation,
            "layer_model": args.layer_model,
            "dropout": args.dropout,
            "weight_log_sigma_init": args.weight_log_sigma_init,
            "regression_coeffs": args.regression_coeffs,
            "bb_alpha": args.bb_alpha,
        },
        "train_time_s": round(train_time, 2),
        "train": train_metrics,
        "test": test_metrics,
        "losses": losses,
        "metrics_history": metrics_history,
        "diagnostics": diagnostics,
    }
    if args.save_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), _ckpt_path(args, dataset_name, model_type))
    return result, model


def run_single(dataset_name, args):
    dataset = get_dataset(dataset_name)
    num_classes = dataset.classes
    train_dataset, train_test_dataset, test_dataset = dataset.get_split()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=bool(args.device and "cuda" in args.device),
        num_workers=0,
    )
    results = []
    use_warm_start = args.model == "ftip" and args.auto_warm_start

    if use_warm_start:
        vip_model = build_model(args, train_dataset, num_classes, model_type="vip")
        vip_ws_result, vip_model = _build_result(
            dataset_name, "vip", vip_model, args, train_loader,
            train_test_dataset, test_dataset, lr=args.vip_lr,
            epochs=args.vip_epochs, iterations=args.vip_iterations,
            desc="VIP pre-training",
        )
        vip_state = {name: param.data.clone()
                     for name, param in vip_model.named_parameters()}
        vip_baseline_result, _ = _build_result(
            dataset_name, "vip", vip_model, args, train_loader,
            train_test_dataset, test_dataset, lr=args.vip_lr,
            epochs=args.epochs, iterations=args.iterations,
            desc="VIP baseline",
        )
        vip_baseline_result["train_time_s"] = round(
            vip_ws_result["train_time_s"] + vip_baseline_result["train_time_s"], 2,
        )
        results.append(vip_baseline_result)
        for name, param in vip_model.named_parameters():
            param.data.copy_(vip_state[name])

        ftip_model = build_model(
            args, train_dataset, num_classes, model_type="ftip",
        )
        ftip_model.warm_start_from_vip(
            vip_model, learnable_affine=args.learnable_affine,
        )
        ftip_result, _ = _build_result(
            dataset_name, "ftip", ftip_model, args, train_loader,
            train_test_dataset, test_dataset, lr=args.ftip_lr,
            epochs=args.epochs, iterations=args.iterations,
            desc="FTIP fine-tuning",
        )
        ftip_result["warm_start"] = {
            "enabled": True,
            "vip_epochs": args.vip_epochs,
            "vip_iterations": args.vip_iterations,
            "vip_lr": args.vip_lr,
            "learnable_affine": args.learnable_affine,
        }
        ftip_result["total_time_s"] = round(
            vip_ws_result["train_time_s"] + ftip_result["train_time_s"], 2,
        )
        results.append(ftip_result)
        return results

    model = build_model(args, train_dataset, num_classes)
    if args.warm_start_from and args.model == "ftip":
        vip_model = build_model(args, train_dataset, num_classes, model_type="vip")
        vip_model.load_state_dict(
            torch.load(
                args.warm_start_from,
                map_location=torch.device(args.device),
                weights_only=True,
            )
        )
        model.warm_start_from_vip(
            vip_model, learnable_affine=args.learnable_affine,
        )
    lr = args.lr
    result, _ = _build_result(
            dataset_name, args.model, model, args, train_loader,
            train_test_dataset, test_dataset, lr=lr,
        )
    results.append(result)
    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    datasets = CLASSIFICATION_DATASETS if args.dataset == "all" else [args.dataset]
    all_results = []
    for dataset_name in datasets:
        alpha_tag = f"_alpha{args.bb_alpha}"
        layer_tag = "_simpler" if args.layer_model == "SimplerBayesLinear" else "_bayes"
        gen_tag = f"_{args.gen_model}" if args.gen_model != "bnn" else ""
        out_path = os.path.join(
            args.output_dir,
            f"{args.model}_{dataset_name}{alpha_tag}{layer_tag}{gen_tag}_seed{args.seed}.json",
        )
        if os.path.exists(out_path):
            with open(out_path) as f:
                all_results.extend(json.load(f))
            continue
        results = run_single(dataset_name, args)
        all_results.extend(results)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
