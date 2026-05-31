"""Image classification benchmark for FashionMNIST and CIFAR10.

Runs MAP, VIP, FBNN, and AP-FSVI with CNN generative functions. The Bayesian
layers are always the full ``BayesLinear`` implementation, matching the current
UCI benchmark policy.

Examples
--------
python -m scripts.classification_benchmark --dataset FashionMNIST --model ap_fsvi
python -m scripts.classification_benchmark --dataset CIFAR10 --model all --ap_variant all
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ap_fsvi import APFSVI
from src.fbnn import FBNN
from src.priors.generative_functions import (
    BayesLinear,
    BayesianCNN,
    BayesianCNNFull,
    BayesianResNet,
)
from src.utils.dataset import get_dataset
from src.utils.metrics import MetricsClassification
from src.utils.utils import infinite_loader
from src.vip import VIP

from scripts.benchmark_utils import (
    add_wandb_args,
    finish_wandb_run,
    init_wandb_run,
    pretty_model_name,
    wandb_log_eval,
    wandb_log_result,
    wandb_log_train_step,
)


CLASSIFICATION_DATASETS = ["FashionMNIST", "CIFAR10"]
CLASSIFICATION_MODELS = ["map", "vip", "fbnn", "ap_fsvi"]

AP_VARIANTS = {
    "mmd": {
        "display": "MMD",
        "discrepancy": "mmd",
        "sample_projection_mode": "random",
    },
    "energy": {
        "display": "Energy",
        "discrepancy": "energy",
        "sample_projection_mode": "random",
    },
    "sample_sliced_random": {
        "display": "Sample Sliced KL",
        "discrepancy": "sample_sliced_kl",
        "sample_projection_mode": "random",
    },
    "sample_sliced_fixed_random": {
        "display": "Sample Sliced KL Fixed",
        "discrepancy": "sample_sliced_kl",
        "sample_projection_mode": "fixed_random",
    },
    "sample_sliced_gaussian_random": {
        "display": "Sample Sliced Gaussian KL",
        "discrepancy": "sample_sliced_gaussian_kl",
        "sample_projection_mode": "random",
    },
    "sample_sliced_quantile_transport_random": {
        "display": "Sliced Quantile-Transport KL",
        "discrepancy": "sample_sliced_quantile_transport_kl",
        "sample_projection_mode": "random",
    },
    "spectral_projected": {
        "display": "Spectral Projected KL",
        "discrepancy": "spectral_projected_kl",
        "sample_projection_mode": "random",
    },
}

AP_VARIANT_ALIASES = {
    "sample_sliced": "sample_sliced_random",
    "sample_sliced_kl": "sample_sliced_random",
    "sample_sliced_gaussian": "sample_sliced_gaussian_random",
    "sample_sliced_gaussian_kl": "sample_sliced_gaussian_random",
    "sample_sliced_quantile_transport": "sample_sliced_quantile_transport_random",
    "sample_sliced_quantile_transport_kl": "sample_sliced_quantile_transport_random",
    "sliced_quantile_transport_kl": "sample_sliced_quantile_transport_random",
    "sqtkl": "sample_sliced_quantile_transport_random",
    "spectral_projected_kl": "spectral_projected",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="FashionMNIST/CIFAR10 classification benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--dataset",
        required=True,
        choices=CLASSIFICATION_DATASETS + ["all"],
        help="Dataset to run.",
    )
    p.add_argument(
        "--model",
        required=True,
        choices=CLASSIFICATION_MODELS + ["all"],
        help="Model to train.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Tensor dtype.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda when available.",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join("results", "classification"),
        help="Directory for JSON results and checkpoints.",
    )
    p.add_argument(
        "--limit_train",
        type=int,
        default=None,
        help="Optional train subset size for smoke tests.",
    )
    p.add_argument(
        "--limit_test",
        type=int,
        default=None,
        help="Optional test subset size for smoke tests.",
    )

    # CNN generator.
    p.add_argument(
        "--backbone",
        choices=["lenet", "resnet18"],
        default="lenet",
        help="CNN backbone. resnet18 is CIFAR10-only.",
    )
    p.add_argument(
        "--full_bayes_cnn",
        action="store_true",
        default=False,
        help="Use Bayesian conv layers as well as BayesLinear head layers.",
    )
    p.add_argument(
        "--head_dims",
        type=int,
        nargs="*",
        default=None,
        help="Bayesian classifier head widths. Defaults are architecture-specific.",
    )
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--weight_log_sigma_init",
        type=float,
        default=-3.0,
        help="Initial posterior log std for Bayesian layers.",
    )
    p.add_argument(
        "--prior_weight_log_sigma_init",
        type=float,
        default=0.0,
        help="Initial log std for BNN prior layers.",
    )

    # Shared VI settings.
    p.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Training samples / VIP basis size.",
    )
    p.add_argument(
        "--eval_samples",
        type=int,
        default=100,
        help="Posterior samples used at evaluation.",
    )
    p.add_argument(
        "--bb_alpha",
        type=float,
        default=0.0,
        help="BB-alpha parameter. 0 gives the ELBO-style objective.",
    )
    p.add_argument(
        "--use_prior_regularizer",
        action="store_true",
        default=False,
        help="Enable optional method-specific prior regularizers when available.",
    )
    p.add_argument(
        "--prior_regularizer_scaler",
        type=float,
        default=1.0,
        help="Scale for optional prior regularizers.",
    )

    # MAP.
    p.add_argument("--map_l2", type=float, default=1e-4)

    # FBNN.
    p.add_argument(
        "--fbnn_num_samples",
        type=int,
        default=None,
        help="FBNN posterior samples per step. Defaults to --num_samples.",
    )
    p.add_argument(
        "--fbnn_num_prior_samples",
        type=int,
        default=64,
        help="Frozen BNN prior samples used for SSGE prior score estimation.",
    )
    p.add_argument("--fbnn_num_measurement", type=int, default=32)
    p.add_argument("--fbnn_num_context", type=int, default=32)
    p.add_argument("--fbnn_context_std", type=float, default=1.5)
    p.add_argument("--fbnn_lambda_kl", type=float, default=1.0)
    p.add_argument("--fbnn_num_eigs", type=int, default=None)
    p.add_argument("--fbnn_nugget", type=float, default=1e-4)
    p.add_argument("--fbnn_reservoir_size", type=int, default=5000)
    p.add_argument(
        "--fbnn_learn_prior",
        action="store_true",
        default=False,
        help="Let FBNN prior parameters train. Default is a fixed BNN prior.",
    )

    # AP-FSVI.
    p.add_argument(
        "--ap_variant",
        default="sample_sliced_random",
        choices=list(AP_VARIANTS.keys()) + list(AP_VARIANT_ALIASES.keys()) + ["all"],
        help="AP-FSVI discrepancy preset.",
    )
    p.add_argument(
        "--ap_fsvi_prior",
        choices=["bnn", "gp"],
        default="bnn",
        help="AP-FSVI prior. BNN is the image-classification default.",
    )
    p.add_argument(
        "--ap_fsvi_num_samples",
        type=int,
        default=None,
        help="AP-FSVI posterior samples per step. Defaults to --num_samples.",
    )
    p.add_argument("--ap_fsvi_num_prior_samples", type=int, default=64)
    p.add_argument("--ap_fsvi_num_measurement", type=int, default=64)
    p.add_argument("--ap_fsvi_beta", type=float, default=1.0)
    p.add_argument("--ap_fsvi_beta_start", type=float, default=0.0)
    p.add_argument("--ap_fsvi_beta_warmup_steps", type=int, default=1000)
    p.add_argument("--ap_fsvi_data_pretrain_steps", type=int, default=0)
    p.add_argument(
        "--ap_fsvi_data_loss",
        choices=["expected_nll", "predictive_nll"],
        default="expected_nll",
    )
    p.add_argument(
        "--ap_fsvi_discrepancy_projections",
        type=int,
        default=64,
        help="Projection/mode count for sliced or spectral AP-FSVI variants.",
    )
    p.add_argument(
        "--ap_fsvi_spectral_estimator",
        choices=["full_gaussian", "gaussian", "knn_entropy"],
        default="full_gaussian",
    )
    p.add_argument("--ap_fsvi_spectral_cov_shrinkage", type=float, default=0.05)
    p.add_argument("--ap_fsvi_sample_gaussian_shrinkage", type=float, default=0.05)
    p.add_argument(
        "--ap_fsvi_quantile_transport_k",
        type=int,
        default=3,
        help="Local spacing window for AP-FSVI sliced quantile-transport KL.",
    )
    p.add_argument(
        "--ap_fsvi_measurement_weights",
        type=float,
        nargs=3,
        default=[0.4, 0.4, 0.2],
        metavar=("DATA", "NEAR", "DOMAIN"),
    )
    p.add_argument("--ap_fsvi_near_data_noise", type=float, default=0.05)
    p.add_argument("--ap_fsvi_domain_std", type=float, default=1.0)
    p.add_argument(
        "--no_ap_fsvi_unit_domain_bounds",
        action="store_true",
        help="Do not constrain AP-FSVI domain measurement points to [0, 1].",
    )
    p.add_argument(
        "--ap_fsvi_fixed_measure_points",
        action="store_true",
        default=False,
        help="Use a fixed AP-FSVI measurement set sampled before training.",
    )
    p.add_argument(
        "--ap_fsvi_adaptive_measure_points",
        action="store_true",
        default=False,
        help="Enable AP-FSVI adaptive measurement-point optimization.",
    )
    p.add_argument(
        "--ap_fsvi_adaptive_measure_mode",
        choices=["gradient", "candidate", "candidate_then_one_step"],
        default="gradient",
    )
    p.add_argument("--ap_fsvi_adaptive_measure_steps", type=int, default=1)
    p.add_argument("--ap_fsvi_adaptive_measure_every", type=int, default=1)
    p.add_argument("--ap_fsvi_adaptive_measure_lr", type=float, default=0.02)
    p.add_argument("--ap_fsvi_reservoir_size", type=int, default=5000)
    p.add_argument("--ap_fsvi_max_grad_norm", type=float, default=None)

    # Training.
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument(
        "--eval_train_examples",
        type=int,
        default=5000,
        help="Training examples for periodic eval. 0 disables train eval.",
    )
    p.add_argument(
        "--final_train_examples",
        type=int,
        default=None,
        help="Training examples for final train metrics. Default evaluates all.",
    )
    p.add_argument("--cosine_annealing", action="store_true", default=True)
    p.add_argument("--no_cosine_annealing", action="store_true")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--no_tqdm", action="store_true", default=False)

    # Checkpoints.
    p.add_argument("--save_checkpoint", action="store_true", default=True)
    p.add_argument("--no_save_checkpoint", action="store_true")

    add_wandb_args(p)

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.no_cosine_annealing:
        args.cosine_annealing = False
    if args.no_save_checkpoint:
        args.save_checkpoint = False
    if args.iterations is not None and args.epochs is not None:
        args.epochs = None
    if args.fbnn_num_samples is None:
        args.fbnn_num_samples = args.num_samples
    if args.ap_fsvi_num_samples is None:
        args.ap_fsvi_num_samples = args.num_samples
    if args.ap_variant in AP_VARIANT_ALIASES:
        args.ap_variant = AP_VARIANT_ALIASES[args.ap_variant]
    return args


class AttributeSubset(Dataset):
    """Subset wrapper that preserves dataset attributes used by builders."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)
        for name in (
            "input_dim",
            "output_dim",
            "targets_mean",
            "targets_std",
            "inputs_mean",
            "inputs_std",
            "n_samples",
        ):
            if hasattr(dataset, name):
                setattr(self, name, getattr(dataset, name))
        if hasattr(dataset, "inputs"):
            self.inputs = dataset.inputs[self.indices]
        if hasattr(dataset, "targets"):
            self.targets = dataset.targets[self.indices]
        self.n_samples = len(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


class DeterministicCNNMAP(torch.nn.Module):
    """Deterministic CNN classifier trained with CE plus L2."""

    _SHAPES = {
        784: (1, 28, 28),
        3072: (3, 32, 32),
    }

    def __init__(
        self,
        input_dim,
        output_dim,
        num_data,
        backbone="lenet",
        head_dims=None,
        dropout=0.0,
        l2=1e-4,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        if input_dim not in self._SHAPES:
            raise ValueError(f"Unsupported image input_dim={input_dim}.")
        if backbone == "resnet18" and input_dim != 3072:
            raise ValueError("resnet18 is only supported for CIFAR10/input_dim=3072.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_data = num_data
        self.backbone = backbone
        self.image_shape = self._SHAPES[input_dim]
        self.l2 = l2
        self.device = device
        self.dtype = dtype
        self.data_terms = []
        self.KLs = []
        self.l2_terms = []

        if backbone == "resnet18":
            self.features, feat_dim = self._build_resnet_features(dtype=dtype)
            if head_dims is None:
                head_dims = []
        else:
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(self.image_shape[0], 6, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(6, 16, kernel_size=5),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(kernel_size=2, stride=2),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *self.image_shape, dtype=dtype)
                feat_dim = self.features(dummy).reshape(1, -1).shape[1]
            if head_dims is None:
                head_dims = [120, 84]

        layers = []
        last = feat_dim
        for width in head_dims:
            layers.append(torch.nn.Linear(last, width))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            last = width
        layers.append(torch.nn.Linear(last, output_dim))
        self.classifier = torch.nn.Sequential(*layers)
        self.to(device=device, dtype=dtype)

    def _build_resnet_features(self, dtype):
        import torchvision.models as tvm

        net = tvm.resnet18(weights=None)
        net.conv1 = torch.nn.Conv2d(
            3, net.conv1.out_channels, 3, stride=1, padding=1, bias=False
        )
        net.maxpool = torch.nn.Identity()
        feat_dim = net.fc.in_features
        net.fc = torch.nn.Identity()
        return net.to(dtype=dtype), feat_dim

    def predict_logits(self, X):
        if X.dtype != self.dtype:
            X = X.to(self.dtype)
        X = X.to(self.device)
        x_img = X.reshape(X.shape[0], *self.image_shape)
        feat = self.features(x_img)
        feat = feat.reshape(feat.shape[0], -1)
        return self.classifier(feat)

    def predict_f_samples(self, X, S=1):
        logits = self.predict_logits(X)
        return logits.unsqueeze(0).expand(S, *logits.shape)

    def forward(self, X):
        return self.predict_f_samples(X, S=1)

    def nelbo(self, X, y):
        X = X.to(dtype=self.dtype, device=self.device)
        y = y.to(device=self.device).long().view(-1)
        logits = self.predict_logits(X)
        data_term = (
            self.num_data
            / X.shape[0]
            * torch.nn.functional.cross_entropy(logits, y, reduction="sum")
        )
        l2_term = 0.5 * self.l2 * sum(
            param.square().sum() for param in self.parameters()
        )
        self.data_terms.append(data_term.detach())
        self.KLs.append(l2_term.detach())
        self.l2_terms.append(l2_term.detach())
        return data_term + l2_term

    def _train_step(self, optimizer, X, y):
        optimizer.zero_grad(set_to_none=True)
        loss = self.nelbo(X, y)
        loss.backward()
        optimizer.step()
        return loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_args(args):
    return torch.float64 if args.dtype == "float64" else torch.float32


def maybe_limit_dataset(dataset, limit, seed):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=limit, replace=False)
    return AttributeSubset(dataset, np.sort(indices))


def annotate_classification_split(split, source_dataset):
    """Carry class-count metadata from the dataset wrapper to split objects."""
    num_classes = getattr(
        source_dataset,
        "classes",
        getattr(source_dataset, "output_dim", getattr(split, "output_dim", None)),
    )
    if num_classes is None:
        raise ValueError("Classification dataset does not expose a class count.")
    split.output_dim = int(num_classes)
    split.classes = int(num_classes)
    return split


def build_bayesian_classifier(
    args,
    train_dataset,
    *,
    num_samples,
    seed,
    fix_random_noise=True,
    weight_log_sigma_init=None,
):
    device = torch.device(args.device)
    dtype = dtype_from_args(args)
    input_dim = train_dataset.input_dim
    output_dim = train_dataset.output_dim
    weight_log_sigma_init = (
        args.weight_log_sigma_init
        if weight_log_sigma_init is None
        else weight_log_sigma_init
    )

    if args.backbone == "resnet18":
        if input_dim != 3072:
            raise ValueError("resnet18 backbone is only valid for CIFAR10.")
        if args.full_bayes_cnn:
            raise ValueError("full_bayes_cnn is not implemented for resnet18.")
        return BayesianResNet(
            num_samples=num_samples,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_model=BayesLinear,
            head_dims=args.head_dims,
            dropout=args.dropout,
            backbone="resnet18",
            cifar_stem=True,
            device=device,
            fix_random_noise=fix_random_noise,
            weight_log_sigma_init=weight_log_sigma_init,
            seed=seed,
            dtype=dtype,
        )

    cls = BayesianCNNFull if args.full_bayes_cnn else BayesianCNN
    return cls(
        num_samples=num_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        layer_model=BayesLinear,
        head_dims=args.head_dims,
        dropout=args.dropout,
        device=device,
        fix_random_noise=fix_random_noise,
        weight_log_sigma_init=weight_log_sigma_init,
        seed=seed,
        dtype=dtype,
    )


def build_model(args, train_dataset, model_type, ap_variant=None):
    device = torch.device(args.device)
    dtype = dtype_from_args(args)
    input_dim = train_dataset.input_dim
    output_dim = train_dataset.output_dim
    num_classes = output_dim

    if model_type == "map":
        return DeterministicCNNMAP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_data=len(train_dataset),
            backbone=args.backbone,
            head_dims=args.head_dims,
            dropout=args.dropout,
            l2=args.map_l2,
            device=device,
            dtype=dtype,
        )

    if model_type == "vip":
        gen_fn = build_bayesian_classifier(
            args,
            train_dataset,
            num_samples=args.num_samples,
            seed=args.seed,
            fix_random_noise=True,
            weight_log_sigma_init=args.weight_log_sigma_init,
        )
        return VIP(
            generative_function=gen_fn,
            num_regression_coeffs=args.num_samples,
            output_dim=output_dim,
            likelihood="multiclass",
            num_data=len(train_dataset),
            bb_alpha=args.bb_alpha,
            num_classes=num_classes,
            num_mc_samples=args.eval_samples,
            use_prior_regularizer=args.use_prior_regularizer,
            prior_regularizer_scaler=args.prior_regularizer_scaler,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    if model_type == "fbnn":
        gen_fn = build_bayesian_classifier(
            args,
            train_dataset,
            num_samples=args.fbnn_num_samples,
            seed=args.seed,
            fix_random_noise=True,
            weight_log_sigma_init=args.weight_log_sigma_init,
        )
        prior_fn = build_bayesian_classifier(
            args,
            train_dataset,
            num_samples=args.fbnn_num_prior_samples,
            seed=args.seed + 1,
            fix_random_noise=True,
            weight_log_sigma_init=args.prior_weight_log_sigma_init,
        )
        return FBNN(
            generative_function=gen_fn,
            prior_function=prior_fn,
            output_dim=output_dim,
            likelihood="multiclass",
            num_data=len(train_dataset),
            num_samples=args.fbnn_num_samples,
            num_measurement=args.fbnn_num_measurement,
            num_context=args.fbnn_num_context,
            context_std=args.fbnn_context_std,
            bb_alpha=args.bb_alpha,
            lambda_kl=args.fbnn_lambda_kl,
            num_eigs=args.fbnn_num_eigs,
            nugget=args.fbnn_nugget,
            reservoir_size=args.fbnn_reservoir_size,
            num_classes=num_classes,
            freeze_prior=not args.fbnn_learn_prior,
            device=device,
            dtype=dtype,
        )

    if model_type == "ap_fsvi":
        if ap_variant is None:
            ap_variant = args.ap_variant
        spec = AP_VARIANTS[ap_variant]
        gen_fn = build_bayesian_classifier(
            args,
            train_dataset,
            num_samples=args.ap_fsvi_num_samples,
            seed=args.seed,
            fix_random_noise=False,
            weight_log_sigma_init=args.weight_log_sigma_init,
        )
        prior_fn = None
        if args.ap_fsvi_prior == "bnn":
            prior_fn = build_bayesian_classifier(
                args,
                train_dataset,
                num_samples=args.ap_fsvi_num_prior_samples,
                seed=args.seed + 1,
                fix_random_noise=False,
                weight_log_sigma_init=args.prior_weight_log_sigma_init,
            )
        domain_bounds = None if args.no_ap_fsvi_unit_domain_bounds else [0.0, 1.0]
        return APFSVI(
            generative_function=gen_fn,
            prior_function=prior_fn,
            input_dim=input_dim,
            output_dim=output_dim,
            likelihood="multiclass",
            num_classes=num_classes,
            num_data=len(train_dataset),
            num_samples=args.ap_fsvi_num_samples,
            num_prior_samples=args.ap_fsvi_num_prior_samples,
            num_measurement=args.ap_fsvi_num_measurement,
            beta=args.ap_fsvi_beta,
            beta_start=args.ap_fsvi_beta_start,
            beta_warmup_steps=args.ap_fsvi_beta_warmup_steps,
            data_pretrain_steps=args.ap_fsvi_data_pretrain_steps,
            data_loss=args.ap_fsvi_data_loss,
            measurement_weights=args.ap_fsvi_measurement_weights,
            near_data_noise=args.ap_fsvi_near_data_noise,
            domain_bounds=domain_bounds,
            domain_std=args.ap_fsvi_domain_std,
            adaptive_measure_points=args.ap_fsvi_adaptive_measure_points,
            adaptive_measure_mode=args.ap_fsvi_adaptive_measure_mode,
            adaptive_measure_steps=args.ap_fsvi_adaptive_measure_steps,
            adaptive_measure_every=args.ap_fsvi_adaptive_measure_every,
            adaptive_measure_lr=args.ap_fsvi_adaptive_measure_lr,
            fixed_measure_points=args.ap_fsvi_fixed_measure_points,
            function_discrepancy=spec["discrepancy"],
            discrepancy_num_projections=args.ap_fsvi_discrepancy_projections,
            sample_projection_mode=spec["sample_projection_mode"],
            quantile_transport_k=args.ap_fsvi_quantile_transport_k,
            spectral_estimator=args.ap_fsvi_spectral_estimator,
            spectral_cov_shrinkage=args.ap_fsvi_spectral_cov_shrinkage,
            sample_gaussian_shrinkage=args.ap_fsvi_sample_gaussian_shrinkage,
            reservoir_size=args.ap_fsvi_reservoir_size,
            max_grad_norm=args.ap_fsvi_max_grad_norm,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def predict_logits_samples(model, xb, args, model_type):
    if model_type == "map":
        return model.predict_f_samples(xb, S=1)
    if model_type == "vip":
        old = model.num_mc_samples
        model.num_mc_samples = args.eval_samples
        samples, _ = model(xb)
        model.num_mc_samples = old
        return samples
    if model_type == "fbnn":
        return model.predict(xb, S=args.eval_samples)
    if model_type == "ap_fsvi":
        return model.predict(xb, S=args.eval_samples)
    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_classification(
    model,
    dataset,
    args,
    model_type,
    *,
    max_examples=None,
    batch_size=None,
):
    device = torch.device(args.device)
    dtype = dtype_from_args(args)
    batch_size = batch_size or args.eval_batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics = MetricsClassification(num_data=len(dataset), device=device)
    model.eval()
    seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            if max_examples is not None and max_examples > 0:
                remaining = max_examples - seen
                if remaining <= 0:
                    break
                if xb.shape[0] > remaining:
                    xb = xb[:remaining]
                    yb = yb[:remaining]
            xb = xb.to(device=device, dtype=dtype, non_blocking=True)
            yb = yb.to(device=device, non_blocking=True)
            samples = predict_logits_samples(model, xb, args, model_type)
            metrics.update(
                yb,
                loss=torch.tensor(0.0, dtype=dtype, device=device),
                mean_pred=samples,
                light=False,
            )
            seen += xb.shape[0]
    model.train()
    return metrics.get_dict()


def initialize_function_context(model, model_type, train_loader):
    if model_type == "fbnn" and hasattr(model, "_fill_reservoir"):
        model._fill_reservoir(train_loader)
    if model_type == "ap_fsvi":
        if hasattr(model, "_fill_reservoir"):
            model._fill_reservoir(train_loader)
        if hasattr(model, "_initialize_fixed_measurement_set"):
            model._initialize_fixed_measurement_set(train_loader)


def train_with_metrics(
    model,
    train_loader,
    train_eval_dataset,
    test_dataset,
    args,
    model_type,
    *,
    desc,
):
    device = torch.device(args.device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    scheduler = None
    if args.cosine_annealing:
        t_max = args.iterations if args.iterations is not None else max(1, args.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, t_max), eta_min=args.lr / 100
        )

    if args.compile:
        try:
            model.nelbo = torch.compile(model.nelbo)
        except Exception:
            print("  [warn] torch.compile unavailable for this model.")

    initialize_function_context(model, model_type, train_loader)
    model.train()
    losses = []
    metrics_history = {"iterations": [], "train": [], "validation": []}

    if args.iterations is not None:
        data_stream = infinite_loader(train_loader)
        iterator = tqdm(
            range(args.iterations),
            unit=" iter",
            desc=desc,
            disable=args.no_tqdm,
        )
        for step_idx in iterator:
            xb, yb = next(data_stream)
            xb = xb.to(device=device, non_blocking=True)
            yb = yb.to(device=device, non_blocking=True)
            loss = model._train_step(optimizer, xb, yb)
            losses.append(float(loss.detach().cpu()))
            step = step_idx + 1
            if scheduler is not None:
                scheduler.step()
            if not args.no_tqdm:
                iterator.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            wandb_log_train_step(args, step, loss, optimizer, model, model_type)
            maybe_evaluate_during_training(
                model,
                train_eval_dataset,
                test_dataset,
                args,
                model_type,
                step,
                metrics_history,
            )
    else:
        step = 0
        epoch_iterator = tqdm(
            range(args.epochs),
            unit=" epoch",
            desc=desc,
            disable=args.no_tqdm,
        )
        for _ in epoch_iterator:
            for xb, yb in train_loader:
                step += 1
                xb = xb.to(device=device, non_blocking=True)
                yb = yb.to(device=device, non_blocking=True)
                loss = model._train_step(optimizer, xb, yb)
                losses.append(float(loss.detach().cpu()))
                wandb_log_train_step(args, step, loss, optimizer, model, model_type)
                maybe_evaluate_during_training(
                    model,
                    train_eval_dataset,
                    test_dataset,
                    args,
                    model_type,
                    step,
                    metrics_history,
                )
            if scheduler is not None:
                scheduler.step()
            if not args.no_tqdm:
                epoch_iterator.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    diagnostics = extract_diagnostics(model)
    return losses, metrics_history, diagnostics


def maybe_evaluate_during_training(
    model,
    train_eval_dataset,
    test_dataset,
    args,
    model_type,
    step,
    metrics_history,
):
    if args.eval_every <= 0 or step % args.eval_every != 0:
        return
    train_metrics = None
    if args.eval_train_examples is None or args.eval_train_examples != 0:
        train_metrics = evaluate_classification(
            model,
            train_eval_dataset,
            args,
            model_type,
            max_examples=args.eval_train_examples,
        )
    test_metrics = evaluate_classification(model, test_dataset, args, model_type)
    metrics_history["iterations"].append(step)
    metrics_history["train"].append(train_metrics or {})
    metrics_history["validation"].append(test_metrics)
    wandb_log_eval(step, train_metrics, test_metrics)


def extract_diagnostics(model):
    diagnostics = {}
    for attr in (
        "KLs",
        "bb_alphas",
        "prior_regularizers",
        "data_terms",
        "function_terms",
        "betas",
        "l2_terms",
    ):
        if hasattr(model, attr):
            values = getattr(model, attr)
            diagnostics[attr] = [to_float(v) for v in values]
    return diagnostics


def to_float(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    return float(value)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total)}


def result_file_name(dataset_name, model_type, args, ap_variant):
    parts = [dataset_name, model_type]
    if model_type == "ap_fsvi":
        parts.append(ap_variant)
        parts.append(args.ap_fsvi_prior)
    parts.append(args.backbone)
    if args.full_bayes_cnn:
        parts.append("fullbayescnn")
    parts.append(f"seed{args.seed}")
    return "_".join(parts).replace(os.sep, "_") + ".json"


def checkpoint_file_name(dataset_name, model_type, args, ap_variant):
    return result_file_name(dataset_name, model_type, args, ap_variant).replace(
        ".json", ".pt"
    )


def run_single(dataset_name, model_type, args, ap_variant=None):
    set_seed(args.seed)
    dataset = get_dataset(dataset_name)
    train_dataset, train_eval_dataset, test_dataset = dataset.get_split(0.1, args.seed)
    train_dataset = annotate_classification_split(train_dataset, dataset)
    train_eval_dataset = annotate_classification_split(train_eval_dataset, dataset)
    test_dataset = annotate_classification_split(test_dataset, dataset)
    train_dataset = maybe_limit_dataset(train_dataset, args.limit_train, args.seed)
    train_eval_dataset = maybe_limit_dataset(
        train_eval_dataset, args.limit_train, args.seed
    )
    test_dataset = maybe_limit_dataset(test_dataset, args.limit_test, args.seed + 1)

    if args.backbone == "resnet18" and dataset_name != "CIFAR10":
        raise ValueError("resnet18 backbone is only supported for CIFAR10.")

    variant_label = None
    if model_type == "ap_fsvi":
        variant_label = ap_variant or args.ap_variant
    model = build_model(args, train_dataset, model_type, variant_label)
    params = count_parameters(model)

    use_cuda = "cuda" in str(args.device).lower()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )

    display_model = pretty_model_name(model_type)
    display_suffix = None
    if model_type == "ap_fsvi":
        spec = AP_VARIANTS[variant_label]
        display_suffix = f"{spec['display']} | {args.ap_fsvi_prior.upper()} prior"
    run_name = classification_run_name(
        dataset_name,
        model_type,
        args.seed,
        display_suffix,
    )
    group = classification_group(dataset_name, model_type, args, variant_label)
    tags = [
        "classification",
        dataset_name,
        model_type,
        args.backbone,
        "BayesLinear",
    ]
    if model_type == "ap_fsvi":
        tags.extend([variant_label, f"prior:{args.ap_fsvi_prior}"])
    run = init_wandb_run(
        args,
        name=run_name,
        group=group,
        tags=tags,
        config={
            "dataset_name": dataset_name,
            "model_type": model_type,
            "ap_variant": variant_label,
            "parameter_count": params,
        },
    )

    print(f"\n{'=' * 72}")
    print(f"Dataset: {dataset_name} | Model: {display_model}")
    if display_suffix:
        print(f"Variant: {display_suffix}")
    print(f"Backbone: {args.backbone} | full_bayes_cnn={args.full_bayes_cnn}")
    print(f"Parameters: trainable={params['trainable']:,} total={params['total']:,}")
    print(f"{'=' * 72}")

    try:
        t0 = time.time()
        losses, metrics_history, diagnostics = train_with_metrics(
            model,
            train_loader,
            train_eval_dataset,
            test_dataset,
            args,
            model_type,
            desc=f"{dataset_name} {display_model}",
        )
        train_time = time.time() - t0

        train_metrics = evaluate_classification(
            model,
            train_eval_dataset,
            args,
            model_type,
            max_examples=args.final_train_examples,
        )
        test_metrics = evaluate_classification(model, test_dataset, args, model_type)

        result = {
            "dataset": dataset_name,
            "model": model_type if model_type != "ap_fsvi" else f"ap_fsvi_{variant_label}",
            "model_type": model_type,
            "ap_variant": variant_label,
            "train_time_s": round(train_time, 2),
            "train": train_metrics,
            "test": test_metrics,
            "losses": losses,
            "metrics_history": metrics_history,
            "diagnostics": diagnostics,
            "parameter_count": params,
            "hyperparameters": result_hyperparameters(args, model_type, variant_label),
        }

        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(
            args.output_dir,
            result_file_name(dataset_name, model_type, args, variant_label),
        )
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        result["result_path"] = result_path

        if args.save_checkpoint:
            ckpt_path = os.path.join(
                args.output_dir,
                checkpoint_file_name(dataset_name, model_type, args, variant_label),
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "result": result,
                    "args": vars(args),
                },
                ckpt_path,
            )
            result["checkpoint_path"] = ckpt_path

        print_metrics("Train", train_metrics)
        print_metrics("Test", test_metrics)
        print(f"Time: {train_time:.1f}s")
        print(f"Result: {result_path}")
        wandb_log_result(result)
        return result
    finally:
        finish_wandb_run(run)


def classification_run_name(dataset_name, model_type, seed, suffix=None):
    parts = ["Classification", dataset_name, pretty_model_name(model_type)]
    if suffix:
        parts.append(suffix)
    parts.append(f"seed {seed}")
    return " | ".join(parts)


def classification_group(dataset_name, model_type, args, ap_variant=None):
    parts = ["classification", dataset_name.lower(), model_type]
    if model_type == "ap_fsvi":
        parts.extend([args.ap_fsvi_prior, ap_variant])
    parts.append(args.backbone)
    if args.full_bayes_cnn:
        parts.append("full_bayes_cnn")
    return "_".join(str(p) for p in parts if p)


def result_hyperparameters(args, model_type, ap_variant):
    keys = [
        "seed",
        "dtype",
        "device",
        "backbone",
        "full_bayes_cnn",
        "head_dims",
        "dropout",
        "weight_log_sigma_init",
        "prior_weight_log_sigma_init",
        "num_samples",
        "eval_samples",
        "bb_alpha",
        "batch_size",
        "eval_batch_size",
        "lr",
        "iterations",
        "epochs",
        "cosine_annealing",
        "use_prior_regularizer",
        "prior_regularizer_scaler",
    ]
    h = {key: getattr(args, key) for key in keys}
    if model_type == "map":
        h["map_l2"] = args.map_l2
    if model_type == "fbnn":
        h.update(
            {
                "fbnn_num_samples": args.fbnn_num_samples,
                "fbnn_num_prior_samples": args.fbnn_num_prior_samples,
                "fbnn_num_measurement": args.fbnn_num_measurement,
                "fbnn_num_context": args.fbnn_num_context,
                "fbnn_context_std": args.fbnn_context_std,
                "fbnn_lambda_kl": args.fbnn_lambda_kl,
                "fbnn_num_eigs": args.fbnn_num_eigs,
                "fbnn_nugget": args.fbnn_nugget,
                "fbnn_reservoir_size": args.fbnn_reservoir_size,
                "fbnn_learn_prior": args.fbnn_learn_prior,
            }
        )
    if model_type == "ap_fsvi":
        spec = AP_VARIANTS[ap_variant]
        h.update(
            {
                "ap_variant": ap_variant,
                "ap_fsvi_prior": args.ap_fsvi_prior,
                "ap_fsvi_discrepancy": spec["discrepancy"],
                "ap_fsvi_sample_projection_mode": spec["sample_projection_mode"],
                "ap_fsvi_num_samples": args.ap_fsvi_num_samples,
                "ap_fsvi_num_prior_samples": args.ap_fsvi_num_prior_samples,
                "ap_fsvi_num_measurement": args.ap_fsvi_num_measurement,
                "ap_fsvi_beta": args.ap_fsvi_beta,
                "ap_fsvi_beta_start": args.ap_fsvi_beta_start,
                "ap_fsvi_beta_warmup_steps": args.ap_fsvi_beta_warmup_steps,
                "ap_fsvi_data_pretrain_steps": args.ap_fsvi_data_pretrain_steps,
                "ap_fsvi_data_loss": args.ap_fsvi_data_loss,
                "ap_fsvi_discrepancy_projections": args.ap_fsvi_discrepancy_projections,
                "ap_fsvi_spectral_estimator": args.ap_fsvi_spectral_estimator,
                "ap_fsvi_spectral_cov_shrinkage": args.ap_fsvi_spectral_cov_shrinkage,
                "ap_fsvi_sample_gaussian_shrinkage": args.ap_fsvi_sample_gaussian_shrinkage,
                "ap_fsvi_quantile_transport_k": args.ap_fsvi_quantile_transport_k,
                "ap_fsvi_measurement_weights": args.ap_fsvi_measurement_weights,
                "ap_fsvi_near_data_noise": args.ap_fsvi_near_data_noise,
                "ap_fsvi_domain_std": args.ap_fsvi_domain_std,
                "ap_fsvi_unit_domain_bounds": not args.no_ap_fsvi_unit_domain_bounds,
                "ap_fsvi_fixed_measure_points": args.ap_fsvi_fixed_measure_points,
                "ap_fsvi_adaptive_measure_points": args.ap_fsvi_adaptive_measure_points,
                "ap_fsvi_adaptive_measure_mode": args.ap_fsvi_adaptive_measure_mode,
                "ap_fsvi_adaptive_measure_steps": args.ap_fsvi_adaptive_measure_steps,
                "ap_fsvi_adaptive_measure_every": args.ap_fsvi_adaptive_measure_every,
                "ap_fsvi_adaptive_measure_lr": args.ap_fsvi_adaptive_measure_lr,
                "ap_fsvi_reservoir_size": args.ap_fsvi_reservoir_size,
                "ap_fsvi_max_grad_norm": args.ap_fsvi_max_grad_norm,
            }
        )
    return h


def print_metrics(label, metrics):
    print(
        f"{label}: "
        f"NLL={metrics['NLL']:.4f}  "
        f"Error={metrics['Error']:.4f}  "
        f"ECE={metrics['ECE']:.4f}  "
        f"Brier={metrics['Brier']:.4f}"
    )


def expand_jobs(args):
    datasets = CLASSIFICATION_DATASETS if args.dataset == "all" else [args.dataset]
    models = CLASSIFICATION_MODELS if args.model == "all" else [args.model]
    jobs = []
    for dataset_name in datasets:
        for model_type in models:
            if model_type == "ap_fsvi":
                variants = (
                    list(AP_VARIANTS.keys())
                    if args.ap_variant == "all"
                    else [args.ap_variant]
                )
                for variant in variants:
                    jobs.append((dataset_name, model_type, variant))
            else:
                jobs.append((dataset_name, model_type, None))
    return jobs


def print_comparison(results):
    if not results:
        return
    by_dataset = {}
    for result in results:
        by_dataset.setdefault(result["dataset"], []).append(result)

    for dataset_name, rows in by_dataset.items():
        print(f"\nTest comparison: {dataset_name} (sorted by Error)")
        rows = sorted(rows, key=lambda r: r["test"]["Error"])
        header = f"{'model':38s}  {'NLL':>8s}  {'Error':>8s}  {'ECE':>8s}  {'Brier':>8s}"
        print(header)
        print("-" * len(header))
        for result in rows:
            metrics = result["test"]
            model_name = result["model"]
            print(
                f"{model_name:38s}  "
                f"{metrics['NLL']:8.4f}  "
                f"{metrics['Error']:8.4f}  "
                f"{metrics['ECE']:8.4f}  "
                f"{metrics['Brier']:8.4f}"
            )


def save_comparison(results, output_dir):
    if not results:
        return
    rows = []
    for result in results:
        row = {
            "dataset": result["dataset"],
            "model": result["model"],
            "train_time_s": result["train_time_s"],
        }
        for split in ("train", "test"):
            for key, value in result[split].items():
                row[f"{split}_{key}"] = value
        rows.append(row)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "classification_comparison.json")
    csv_path = os.path.join(output_dir, "classification_comparison.csv")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    columns = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "dataset",
        "model",
        "test_NLL",
        "test_Error",
        "test_ECE",
        "test_Brier",
        "train_NLL",
        "train_Error",
        "train_ECE",
        "train_Brier",
        "train_time_s",
    ]
    columns = [c for c in preferred if c in columns] + [
        c for c in columns if c not in preferred
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nComparison JSON: {json_path}")
    print(f"Comparison CSV:  {csv_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available() and "cuda" in str(args.device).lower():
        torch.set_float32_matmul_precision("high")

    jobs = expand_jobs(args)
    results = []
    for dataset_name, model_type, ap_variant in jobs:
        run_args = copy.deepcopy(args)
        result = run_single(dataset_name, model_type, run_args, ap_variant)
        results.append(result)

    print_comparison(results)
    save_comparison(results, args.output_dir)


if __name__ == "__main__":
    main()
