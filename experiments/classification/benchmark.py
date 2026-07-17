"""Image classification benchmark for FashionMNIST and CIFAR10.

Runs the repository's classification-capable methods with CNN generative
functions. The Bayesian layers are always the full ``BayesLinear``
implementation, matching the current UCI benchmark policy. GMVIP uses its
native vector-valued multiclass likelihood.

Examples
--------
python -m experiments.classification.benchmark --dataset FashionMNIST --model vip
python -m experiments.classification.benchmark --dataset CIFAR10 --model all
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiments.benchmark_utils import (
    add_wandb_args,
    finish_wandb_run,
    init_wandb_run,
    pretty_model_name,
    wandb_log_eval,
    wandb_log_result,
    wandb_log_train_step,
)
from experiments.common import build_flow as build_common_flow
from experiments.common import write_csv_rows, write_json
from implicit_process_zoo.data import get_dataset
from implicit_process_zoo.fbnn import FBNN
from implicit_process_zoo.ftip import FTIP
from implicit_process_zoo.gmvip import GeneralizedMatheronVIP, initialize_inducing_points
from implicit_process_zoo.mfvi import MFVI
from implicit_process_zoo.priors.generative_functions import (
    BayesianCNN,
    BayesianCNNFull,
    BayesianResNet,
    BayesLinear,
)
from implicit_process_zoo.sip import SIP
from implicit_process_zoo.tfsvi import TFSVI
from implicit_process_zoo.utils import build_training_checkpoint, save_training_checkpoint
from implicit_process_zoo.utils.metrics import MetricsClassification
from implicit_process_zoo.utils.utils import infinite_loader
from implicit_process_zoo.vip import VIP

CLASSIFICATION_DATASETS = ["FashionMNIST", "CIFAR10"]
CLASSIFICATION_MODELS = [
    "map",
    "mfvi",
    "fbnn",
    "tfsvi",
    "vip",
    "ftip",
    "gmvip",
    "sip",
]


def parse_args(argv=None):
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
    p.add_argument(
        "--regularizer_mode",
        type=str,
        default="evidence",
        choices=["evidence", "KL"],
        help="Optional prior regularizer mode for VIP/FTIP.",
    )

    # MAP.
    p.add_argument("--map_l2", type=float, default=1e-4)

    # VIP / FTIP.
    p.add_argument(
        "--vip_learn_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the VIP Bayesian CNN generator/prior parameters.",
    )
    p.add_argument(
        "--ftip_learn_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the FTIP Bayesian CNN generator/prior parameters.",
    )
    p.add_argument(
        "--ftip_num_samples",
        type=int,
        default=None,
        help="FTIP flow posterior samples per training step. Defaults to --num_samples.",
    )
    p.add_argument(
        "--ftip_eval_samples",
        type=int,
        default=None,
        help="FTIP posterior samples at evaluation. Defaults to --eval_samples.",
    )
    p.add_argument(
        "--flow_type",
        type=str,
        default="spline_1x1",
        choices=["affine", "spline", "spline_1x1"],
        help="FTIP coefficient normalizing flow.",
    )
    p.add_argument("--flow_depth", type=int, default=2)
    p.add_argument("--flow_num_bins", type=int, default=8)
    p.add_argument("--flow_domain", type=float, default=3.0)
    p.add_argument("--ftip_max_grad_norm", type=float, default=None)

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

    # MFVI.
    p.add_argument(
        "--mfvi_num_eval_samples",
        type=int,
        default=None,
        help="MFVI posterior samples at evaluation. Defaults to --eval_samples.",
    )

    # TFSVI.
    p.add_argument("--tfsvi_sigma_prior", type=float, default=1.0)
    p.add_argument("--tfsvi_S_ctx", type=int, default=3)
    p.add_argument("--tfsvi_K_ctx", type=int, default=16)
    p.add_argument(
        "--tfsvi_num_train_samples",
        type=int,
        default=None,
        help="TFSVI likelihood samples per step. Defaults to --num_samples.",
    )
    p.add_argument(
        "--tfsvi_num_eval_samples",
        type=int,
        default=None,
        help="TFSVI posterior samples at evaluation. Defaults to --eval_samples.",
    )

    # SIP.
    p.add_argument("--sip_num_inducing", type=int, default=32)
    p.add_argument(
        "--sip_inducing_method",
        type=str,
        default="random_subset",
        choices=["grid", "random_subset", "kmeans", "grid_1d", "train_quantiles"],
    )
    p.add_argument("--sip_num_prior_samples", type=int, default=64)
    p.add_argument("--sip_num_train_samples", type=int, default=None)
    p.add_argument("--sip_num_eval_samples", type=int, default=None)
    p.add_argument(
        "--sip_learn_inducing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--sip_learn_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--sip_detach_covariances",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--sip_jitter", type=float, default=1e-4)
    p.add_argument(
        "--sip_fix_random_noise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--sip_beta", type=float, default=1.0)
    p.add_argument("--sip_beta_warmup_steps", type=int, default=0)
    p.add_argument("--sip_critic_hidden_dim", type=int, default=50)
    p.add_argument("--sip_critic_lr", type=float, default=1e-3)
    p.add_argument("--sip_critic_steps", type=int, default=1)
    p.add_argument("--sip_posterior_noise_dim", type=int, default=100)
    p.add_argument("--sip_posterior_hidden_dim", type=int, default=50)
    p.add_argument("--sip_posterior_depth", type=int, default=2)

    # GMVIP.
    p.add_argument("--gmvip_operator_type", choices=["empirical", "rbf"], default="empirical")
    p.add_argument(
        "--gmvip_posterior_type",
        choices=["gaussian", "realnvp"],
        default="gaussian",
    )
    p.add_argument("--gmvip_num_inducing", type=int, default=16)
    p.add_argument(
        "--gmvip_inducing_method",
        type=str,
        default="random_subset",
        choices=["grid", "random_subset", "kmeans", "grid_1d", "train_quantiles"],
    )
    p.add_argument("--gmvip_num_operator_bank_samples", type=int, default=64)
    p.add_argument("--gmvip_num_train_samples", type=int, default=None)
    p.add_argument("--gmvip_num_eval_samples", type=int, default=None)
    p.add_argument(
        "--gmvip_learn_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--gmvip_learn_Z",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--gmvip_learn_kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--gmvip_beta", type=float, default=1.0)
    p.add_argument("--gmvip_beta_warmup_steps", type=int, default=0)
    p.add_argument("--gmvip_data_alpha", type=float, default=0.0)
    p.add_argument("--gmvip_jitter", type=float, default=1e-4)
    p.add_argument("--gmvip_shrinkage", type=float, default=0.02)
    p.add_argument("--gmvip_flow_depth", type=int, default=2)
    p.add_argument("--gmvip_flow_hidden_dim", type=int, default=64)
    p.add_argument("--gmvip_flow_num_layers", type=int, default=2)
    p.add_argument("--gmvip_flow_dropout", type=float, default=0.0)
    p.add_argument("--gmvip_flow_scale_bound", type=float, default=2.0)
    p.add_argument("--gmvip_max_grad_norm", type=float, default=None)

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

    args = p.parse_args(argv)
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
    if args.ftip_num_samples is None:
        args.ftip_num_samples = args.num_samples
    if args.ftip_eval_samples is None:
        args.ftip_eval_samples = args.eval_samples
    if args.mfvi_num_eval_samples is None:
        args.mfvi_num_eval_samples = args.eval_samples
    if args.tfsvi_num_train_samples is None:
        args.tfsvi_num_train_samples = args.num_samples
    if args.tfsvi_num_eval_samples is None:
        args.tfsvi_num_eval_samples = args.eval_samples
    if args.sip_num_train_samples is None:
        args.sip_num_train_samples = args.sip_num_prior_samples
    if args.sip_num_eval_samples is None:
        args.sip_num_eval_samples = args.eval_samples
    if args.gmvip_num_train_samples is None:
        args.gmvip_num_train_samples = args.num_samples
    if args.gmvip_num_eval_samples is None:
        args.gmvip_num_eval_samples = args.eval_samples
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
            ).to(device=device, dtype=dtype)
            with torch.no_grad():
                dummy = torch.zeros(1, *self.image_shape, dtype=dtype, device=device)
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
        net.conv1 = torch.nn.Conv2d(3, net.conv1.out_channels, 3, stride=1, padding=1, bias=False)
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

    def predict_f_samples(self, X, num_samples, *, seed=None):
        logits = self.predict_logits(X)
        return logits.unsqueeze(0).expand(num_samples, *logits.shape)

    def forward(self, X):
        return self.predict_f_samples(X, num_samples=1)

    def nelbo(self, X, y):
        X = X.to(dtype=self.dtype, device=self.device)
        y = y.to(device=self.device).long().view(-1)
        logits = self.predict_logits(X)
        data_term = (
            self.num_data
            / X.shape[0]
            * torch.nn.functional.cross_entropy(logits, y, reduction="sum")
        )
        l2_term = 0.5 * self.l2 * sum(param.square().sum() for param in self.parameters())
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
    output_dim_override=None,
):
    device = torch.device(args.device)
    dtype = dtype_from_args(args)
    input_dim = train_dataset.input_dim
    output_dim = (
        train_dataset.output_dim if output_dim_override is None else int(output_dim_override)
    )
    weight_log_sigma_init = (
        args.weight_log_sigma_init if weight_log_sigma_init is None else weight_log_sigma_init
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


def freeze_if_requested(module, learn_prior):
    if learn_prior:
        return
    if hasattr(module, "freeze_parameters"):
        module.freeze_parameters()
        return
    for param in module.parameters():
        param.requires_grad_(False)


def build_flow(args, input_dim, device, dtype):
    return build_common_flow(
        args.flow_type,
        depth=args.flow_depth,
        input_dim=int(input_dim),
        device=device,
        dtype=dtype,
        seed=args.seed,
        num_bins=args.flow_num_bins,
        domain=args.flow_domain,
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

    if model_type == "gmvip":
        train_inputs = torch.as_tensor(
            train_dataset.inputs,
            dtype=dtype,
            device=device,
        )
        inducing_inputs = initialize_inducing_points(
            train_inputs,
            num_inducing=args.gmvip_num_inducing,
            method=args.gmvip_inducing_method,
            seed=args.seed + 31,
        )
        prior_samples = max(
            int(args.gmvip_num_operator_bank_samples),
            int(args.gmvip_num_train_samples),
            2,
        )
        learn_prior = bool(args.gmvip_learn_prior)
        if args.gmvip_operator_type == "empirical":
            mean_mode = "prior_sample"
            inducing_scale = "prior_cholesky"
        else:
            mean_mode = "zero"
            inducing_scale = "rbf_cholesky"
        prior = build_bayesian_classifier(
            args,
            train_dataset,
            num_samples=prior_samples,
            seed=args.seed + 101,
            fix_random_noise=True,
            weight_log_sigma_init=args.prior_weight_log_sigma_init,
        )
        freeze_if_requested(prior, learn_prior)
        return GeneralizedMatheronVIP(
            base_prior=prior,
            inducing_points=inducing_inputs,
            operator_type=args.gmvip_operator_type,
            posterior_type=args.gmvip_posterior_type,
            likelihood="multiclass",
            output_dim=output_dim,
            num_classes=num_classes,
            num_operator_bank_samples=args.gmvip_num_operator_bank_samples,
            learn_noise=False,
            init_log_noise=-10.0,
            min_log_noise=None,
            max_log_noise=None,
            freeze_base_prior=not learn_prior,
            detach_prior_samples=not learn_prior,
            jitter=args.gmvip_jitter,
            shrinkage=args.gmvip_shrinkage,
            learn_Z=args.gmvip_learn_Z,
            learn_kernel=bool(args.gmvip_learn_kernel and args.gmvip_operator_type == "rbf"),
            ard=True,
            init_lengthscale="median",
            init_outputscale="prior_marginal",
            inducing_scale=inducing_scale,
            mean_mode=mean_mode,
            posterior_max_log_std=None,
            flow_depth=args.gmvip_flow_depth,
            flow_hidden_dim=args.gmvip_flow_hidden_dim,
            flow_num_layers=args.gmvip_flow_num_layers,
            flow_dropout=args.gmvip_flow_dropout,
            flow_scale_bound=args.gmvip_flow_scale_bound,
            antithetic_samples=True,
            num_data=len(train_dataset),
            num_train_samples=args.gmvip_num_train_samples,
            beta=args.gmvip_beta,
            beta_warmup_steps=args.gmvip_beta_warmup_steps,
            data_alpha=args.gmvip_data_alpha,
            max_grad_norm=args.gmvip_max_grad_norm,
            operator_bank_seed=args.seed + 1009,
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
        freeze_if_requested(gen_fn, args.vip_learn_prior)
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
            regularizer_mode=args.regularizer_mode,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    if model_type == "ftip":
        gen_fn = build_bayesian_classifier(
            args,
            train_dataset,
            num_samples=args.num_samples,
            seed=args.seed,
            fix_random_noise=True,
            weight_log_sigma_init=args.weight_log_sigma_init,
        )
        freeze_if_requested(gen_fn, args.ftip_learn_prior)
        flow = build_flow(
            args,
            input_dim=args.num_samples * output_dim,
            device=device,
            dtype=dtype,
        )
        return FTIP(
            generative_function=gen_fn,
            num_regression_coeffs=args.num_samples,
            output_dim=output_dim,
            flow=flow,
            likelihood="multiclass",
            num_data=len(train_dataset),
            num_samples=args.ftip_num_samples,
            bb_alpha=args.bb_alpha,
            num_classes=num_classes,
            use_prior_regularizer=args.use_prior_regularizer,
            prior_regularizer_scaler=args.prior_regularizer_scaler,
            regularizer_mode=args.regularizer_mode,
            max_grad_norm=args.ftip_max_grad_norm,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    if model_type == "mfvi":
        gen_fn = build_bayesian_classifier(
            args,
            train_dataset,
            num_samples=args.num_samples,
            seed=args.seed,
            fix_random_noise=False,
            weight_log_sigma_init=args.weight_log_sigma_init,
        )
        return MFVI(
            generative_function=gen_fn,
            output_dim=output_dim,
            likelihood="multiclass",
            num_data=len(train_dataset),
            num_samples=args.num_samples,
            bb_alpha=args.bb_alpha,
            num_classes=num_classes,
            device=device,
            dtype=dtype,
        )

    if model_type == "tfsvi":
        gen_fn = build_bayesian_classifier(
            args,
            train_dataset,
            num_samples=1,
            seed=args.seed,
            fix_random_noise=True,
            weight_log_sigma_init=args.weight_log_sigma_init,
        )
        return TFSVI(
            input_dim=input_dim,
            output_dim=output_dim,
            structure=[],
            activation=torch.nn.ReLU(),
            likelihood="multiclass",
            num_data=len(train_dataset),
            sigma_prior=args.tfsvi_sigma_prior,
            num_samples=args.tfsvi_num_train_samples,
            bb_alpha=args.bb_alpha,
            S_ctx=args.tfsvi_S_ctx,
            K_ctx=args.tfsvi_K_ctx,
            num_classes=num_classes,
            generative_function=gen_fn,
            device=device,
            dtype=dtype,
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

    if model_type == "sip":
        train_inputs = torch.as_tensor(
            train_dataset.inputs,
            dtype=dtype,
            device=device,
        )
        inducing_inputs = initialize_inducing_points(
            train_inputs,
            num_inducing=args.sip_num_inducing,
            method=args.sip_inducing_method,
            seed=args.seed + 17,
        )
        prior_samples = max(
            int(args.sip_num_prior_samples),
            int(args.sip_num_train_samples),
            2,
        )
        prior_fn = build_bayesian_classifier(
            args,
            train_dataset,
            num_samples=prior_samples,
            seed=args.seed + 1,
            fix_random_noise=args.sip_fix_random_noise,
            weight_log_sigma_init=args.prior_weight_log_sigma_init,
        )
        freeze_if_requested(prior_fn, args.sip_learn_prior)
        return SIP(
            generative_function=prior_fn,
            inducing_inputs=inducing_inputs,
            output_dim=output_dim,
            likelihood="multiclass",
            num_data=len(train_dataset),
            num_prior_samples=args.sip_num_prior_samples,
            num_train_samples=args.sip_num_train_samples,
            num_eval_samples=args.sip_num_eval_samples,
            bb_alpha=args.bb_alpha,
            beta=args.sip_beta,
            beta_warmup_steps=args.sip_beta_warmup_steps,
            learn_inducing=args.sip_learn_inducing,
            detach_covariances=args.sip_detach_covariances,
            critic_hidden_dim=args.sip_critic_hidden_dim,
            critic_lr=args.sip_critic_lr,
            critic_steps=args.sip_critic_steps,
            posterior_noise_dim=args.sip_posterior_noise_dim,
            posterior_hidden_dim=args.sip_posterior_hidden_dim,
            posterior_depth=args.sip_posterior_depth,
            fresh_prior_samples=not args.sip_fix_random_noise,
            num_classes=num_classes,
            jitter=args.sip_jitter,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def predict_logits_samples(model, xb, args, model_type):
    if model_type == "map":
        return model.predict_f_samples(xb, num_samples=1)
    if model_type == "vip":
        old = model.num_mc_samples
        model.num_mc_samples = args.eval_samples
        samples, _ = model(xb)
        model.num_mc_samples = old
        return samples
    if model_type == "ftip":
        return model.predict_y(xb, S=args.ftip_eval_samples)
    if model_type == "mfvi":
        return model.predict(xb, num_samples=args.mfvi_num_eval_samples)
    if model_type == "fbnn":
        return model.predict(xb, num_samples=args.eval_samples)
    if model_type == "tfsvi":
        return model.predict(xb, num_samples=args.tfsvi_num_eval_samples)
    if model_type == "sip":
        return model.predict_f_samples(xb, num_samples=args.sip_num_eval_samples)
    if model_type == "gmvip":
        return model.predict_f_samples(xb, num_samples=args.gmvip_num_eval_samples)
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
    if model_type == "tfsvi" and hasattr(model, "_train_inputs"):
        device = torch.device(model.device)
        chunks = []
        for inputs, _ in train_loader:
            chunks.append(inputs.to(device=device, dtype=model.dtype))
        model._train_inputs = torch.cat(chunks, dim=0)


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
    return losses, metrics_history, diagnostics, optimizer, scheduler


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
    variant = classification_variant_slug(args, model_type)
    if variant:
        parts.append(variant)
    parts.append(args.backbone)
    if args.full_bayes_cnn:
        parts.append("fullbayescnn")
    parts.append(f"seed{args.seed}")
    return "_".join(parts).replace(os.sep, "_") + ".json"


def checkpoint_file_name(dataset_name, model_type, args, ap_variant):
    return result_file_name(dataset_name, model_type, args, ap_variant).replace(".json", ".pt")


def run_single(dataset_name, model_type, args, ap_variant=None):
    set_seed(args.seed)
    dataset = get_dataset(dataset_name)
    train_dataset, train_eval_dataset, test_dataset = dataset.get_split(0.1, args.seed)
    train_dataset = annotate_classification_split(train_dataset, dataset)
    train_eval_dataset = annotate_classification_split(train_eval_dataset, dataset)
    test_dataset = annotate_classification_split(test_dataset, dataset)
    train_dataset = maybe_limit_dataset(train_dataset, args.limit_train, args.seed)
    train_eval_dataset = maybe_limit_dataset(train_eval_dataset, args.limit_train, args.seed)
    test_dataset = maybe_limit_dataset(test_dataset, args.limit_test, args.seed + 1)

    if args.backbone == "resnet18" and dataset_name != "CIFAR10":
        raise ValueError("resnet18 backbone is only supported for CIFAR10.")

    model = build_model(args, train_dataset, model_type, ap_variant)
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
    display_suffix = classification_variant_label(args, model_type)
    run_name = classification_run_name(
        dataset_name,
        model_type,
        args.seed,
        display_suffix,
    )
    group = classification_group(dataset_name, model_type, args, ap_variant)
    tags = [
        "classification",
        dataset_name,
        model_type,
        args.backbone,
        "BayesLinear",
    ]
    run = init_wandb_run(
        args,
        name=run_name,
        group=group,
        tags=tags,
        config={
            "dataset_name": dataset_name,
            "model_type": model_type,
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
        losses, metrics_history, diagnostics, optimizer, scheduler = train_with_metrics(
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
            "model": model_type,
            "model_type": model_type,
            "train_time_s": round(train_time, 2),
            "train": train_metrics,
            "test": test_metrics,
            "losses": losses,
            "metrics_history": metrics_history,
            "diagnostics": diagnostics,
            "parameter_count": params,
            "hyperparameters": result_hyperparameters(args, model_type, ap_variant),
        }

        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(
            args.output_dir,
            result_file_name(dataset_name, model_type, args, ap_variant),
        )
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        result["result_path"] = result_path

        if args.save_checkpoint:
            ckpt_path = os.path.join(
                args.output_dir,
                checkpoint_file_name(dataset_name, model_type, args, ap_variant),
            )
            checkpoint = build_training_checkpoint(
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=len(losses),
                arguments=vars(args),
            )
            save_training_checkpoint(ckpt_path, checkpoint)
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
    variant = classification_variant_slug(args, model_type)
    if variant:
        parts.append(variant)
    parts.append(args.backbone)
    if args.full_bayes_cnn:
        parts.append("full_bayes_cnn")
    return "_".join(str(p) for p in parts if p)


def classification_variant_label(args, model_type):
    if model_type == "vip":
        return "Tunable Prior" if args.vip_learn_prior else "Fixed Prior"
    if model_type == "ftip":
        return "Tunable Prior" if args.ftip_learn_prior else "Fixed Prior"
    if model_type == "fbnn":
        return "Tunable Prior" if args.fbnn_learn_prior else "Fixed Prior"
    if model_type == "sip":
        prior = "Tunable Prior" if args.sip_learn_prior else "Fixed Prior"
        z = "Learn Z" if args.sip_learn_inducing else "Fixed Z"
        return f"{prior}, {z}"
    if model_type == "gmvip":
        prior = "Tunable Prior" if args.gmvip_learn_prior else "Fixed Prior"
        z = "Learn Z" if args.gmvip_learn_Z else "Fixed Z"
        return f"{prior}, {z}, {args.gmvip_operator_type.upper()}"
    return None


def classification_variant_slug(args, model_type):
    label = classification_variant_label(args, model_type)
    if not label:
        return None
    return label.lower().replace(",", "").replace(" ", "_").replace("/", "_")


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
    if model_type == "vip":
        h["vip_learn_prior"] = args.vip_learn_prior
    if model_type == "ftip":
        h.update(
            {
                "ftip_learn_prior": args.ftip_learn_prior,
                "ftip_num_samples": args.ftip_num_samples,
                "ftip_eval_samples": args.ftip_eval_samples,
                "flow_type": args.flow_type,
                "flow_depth": args.flow_depth,
                "flow_num_bins": args.flow_num_bins,
                "flow_domain": args.flow_domain,
                "ftip_max_grad_norm": args.ftip_max_grad_norm,
            }
        )
    if model_type == "mfvi":
        h["mfvi_num_eval_samples"] = args.mfvi_num_eval_samples
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
    if model_type == "tfsvi":
        h.update(
            {
                "tfsvi_sigma_prior": args.tfsvi_sigma_prior,
                "tfsvi_S_ctx": args.tfsvi_S_ctx,
                "tfsvi_K_ctx": args.tfsvi_K_ctx,
                "tfsvi_num_train_samples": args.tfsvi_num_train_samples,
                "tfsvi_num_eval_samples": args.tfsvi_num_eval_samples,
            }
        )
    if model_type == "gmvip":
        h.update(
            {
                "gmvip_operator_type": args.gmvip_operator_type,
                "gmvip_posterior_type": args.gmvip_posterior_type,
                "gmvip_num_inducing": args.gmvip_num_inducing,
                "gmvip_inducing_method": args.gmvip_inducing_method,
                "gmvip_num_operator_bank_samples": args.gmvip_num_operator_bank_samples,
                "gmvip_num_train_samples": args.gmvip_num_train_samples,
                "gmvip_num_eval_samples": args.gmvip_num_eval_samples,
                "gmvip_learn_prior": args.gmvip_learn_prior,
                "gmvip_learn_Z": args.gmvip_learn_Z,
                "gmvip_learn_kernel": args.gmvip_learn_kernel,
                "gmvip_beta": args.gmvip_beta,
                "gmvip_beta_warmup_steps": args.gmvip_beta_warmup_steps,
                "gmvip_data_alpha": args.gmvip_data_alpha,
                "gmvip_jitter": args.gmvip_jitter,
                "gmvip_shrinkage": args.gmvip_shrinkage,
                "gmvip_flow_depth": args.gmvip_flow_depth,
                "gmvip_flow_hidden_dim": args.gmvip_flow_hidden_dim,
                "gmvip_flow_num_layers": args.gmvip_flow_num_layers,
                "gmvip_flow_dropout": args.gmvip_flow_dropout,
                "gmvip_flow_scale_bound": args.gmvip_flow_scale_bound,
                "gmvip_max_grad_norm": args.gmvip_max_grad_norm,
            }
        )
    if model_type == "sip":
        h.update(
            {
                "sip_num_inducing": args.sip_num_inducing,
                "sip_inducing_method": args.sip_inducing_method,
                "sip_num_prior_samples": args.sip_num_prior_samples,
                "sip_num_train_samples": args.sip_num_train_samples,
                "sip_num_eval_samples": args.sip_num_eval_samples,
                "sip_learn_inducing": args.sip_learn_inducing,
                "sip_learn_prior": args.sip_learn_prior,
                "sip_detach_covariances": args.sip_detach_covariances,
                "sip_jitter": args.sip_jitter,
                "sip_fix_random_noise": args.sip_fix_random_noise,
                "sip_beta": args.sip_beta,
                "sip_beta_warmup_steps": args.sip_beta_warmup_steps,
                "sip_critic_hidden_dim": args.sip_critic_hidden_dim,
                "sip_critic_lr": args.sip_critic_lr,
                "sip_critic_steps": args.sip_critic_steps,
                "sip_posterior_noise_dim": args.sip_posterior_noise_dim,
                "sip_posterior_hidden_dim": args.sip_posterior_hidden_dim,
                "sip_posterior_depth": args.sip_posterior_depth,
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
    write_json(json_path, rows)

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
    columns = [c for c in preferred if c in columns] + [c for c in columns if c not in preferred]
    write_csv_rows(csv_path, rows, fieldnames=columns)
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
        try:
            result = run_single(dataset_name, model_type, run_args, ap_variant)
        except NotImplementedError as exc:
            if args.model != "all":
                raise
            print(f"\nSkipping {dataset_name} | {model_type}: {exc}")
            continue
        results.append(result)

    print_comparison(results)
    save_comparison(results, args.output_dir)


if __name__ == "__main__":
    main()
