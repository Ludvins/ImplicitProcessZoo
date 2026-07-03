"""Train GMVIP on the synthetic Gap dataset with a BNN prior."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gmvip import GeneralizedMatheronVIP, initialize_inducing_points  # noqa: E402
from src.priors.generative_functions import BayesLinear, BayesianNN  # noqa: E402
from src.utils.dataset import Test_Dataset, Training_Dataset, get_dataset  # noqa: E402
from src.utils.utils import infinite_loader  # noqa: E402


def optional_float(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return None
    return float(value)


def float_or_median(value):
    text = str(value).strip().lower()
    if text == "median":
        return "median"
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GMVIP on the Gap dataset with a BNN prior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dataset_name", choices=["gap", "bimodal"], default="gap")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--data_alpha", type=float, default=0.0)
    parser.add_argument("--beta_warmup_steps", type=int, default=0)
    parser.add_argument("--operator_type", choices=["empirical", "rbf"], default="rbf")
    parser.add_argument("--posterior_type", choices=["gaussian", "realnvp"], default="gaussian")
    parser.add_argument("--num_inducing", type=int, default=48)
    parser.add_argument(
        "--inducing_method",
        choices=["train_quantiles", "random_subset", "grid_1d", "kmeans"],
        default="train_quantiles",
    )
    parser.add_argument("--num_operator_bank_samples", type=int, default=384)
    parser.add_argument("--num_mc_samples", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument(
        "--antithetic_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use antithetic base Gaussian pairs for Gaussian/RealNVP coefficient samples.",
    )
    parser.add_argument("--bnn_hidden_dim", type=int, default=48)
    parser.add_argument("--bnn_hidden_layers", type=int, default=2)
    parser.add_argument("--bnn_weight_log_sigma_init", type=float, default=0.0)
    parser.add_argument(
        "--learn_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the BNN prior/basis parameters while keeping bank noise identities fixed.",
    )
    parser.add_argument("--jitter", type=float, default=1e-5)
    parser.add_argument("--shrinkage", type=float, default=0.02)
    parser.add_argument("--learn_Z", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learn_kernel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init_lengthscale", type=float_or_median, default="median")
    parser.add_argument("--init_outputscale", type=optional_float, default=None)
    parser.add_argument("--mean_mode", choices=["prior_sample", "zero", "prior_api"], default="prior_sample")
    parser.add_argument(
        "--inducing_scale",
        choices=["prior_cholesky", "rbf_cholesky", "prior_diag", "identity"],
        default="prior_cholesky",
        help="Map from whitened coefficients a to inducing values u: D_Z in u=mu_Z+D_Za.",
    )
    parser.add_argument("--flow_depth", type=int, default=4)
    parser.add_argument("--flow_hidden_dim", type=int, default=128)
    parser.add_argument("--flow_num_layers", type=int, default=2)
    parser.add_argument("--flow_dropout", type=float, default=0.0)
    parser.add_argument("--flow_scale_bound", type=float, default=2.0)
    parser.add_argument("--posterior_init_mean", type=float, default=0.0)
    parser.add_argument("--posterior_init_log_std", type=float, default=0.0)
    parser.add_argument("--posterior_min_log_std", type=optional_float, default=-8.0)
    parser.add_argument("--posterior_max_log_std", type=optional_float, default=4.0)
    parser.add_argument("--learn_noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init_log_noise", type=float, default=-2.5)
    parser.add_argument("--min_log_noise", type=optional_float, default=-5.0)
    parser.add_argument(
        "--max_log_noise",
        type=optional_float,
        default=None,
        help="Upper clamp for log noise; use none or omit for no maximum.",
    )
    parser.add_argument("--max_grad_norm", type=optional_float, default=10.0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--normalize_inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--converge", action="store_true")
    parser.add_argument("--min_iterations", type=int, default=None)
    parser.add_argument("--convergence_eval_every", type=int, default=300)
    parser.add_argument("--convergence_window", type=int, default=3)
    parser.add_argument("--convergence_patience", type=int, default=2)
    parser.add_argument("--convergence_rel_tol", type=float, default=0.01)
    parser.add_argument("--convergence_abs_tol", type=float, default=0.0)
    parser.add_argument("--convergence_num_mc_samples", type=int, default=16)
    parser.add_argument("--convergence_eval_batch_size", type=int, default=None)
    parser.add_argument("--plot_samples", type=int, default=24)
    parser.add_argument("--output_dir", default=os.path.join("outputs", "gmvip_gap_bnn"))
    return parser.parse_args()


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def tensor_to_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(()))
    return float(value)


def metric_to_float(metrics, key):
    return tensor_to_float(metrics[key]) if key in metrics else None


def build_bnn_prior(args, input_dim, output_dim, device, dtype):
    prior_samples = args.num_operator_bank_samples
    prior = BayesianNN(
        structure=[args.bnn_hidden_dim] * args.bnn_hidden_layers,
        activation=torch.nn.Tanh(),
        num_samples=prior_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        layer_model=BayesLinear,
        seed=args.seed + 17,
        fix_random_noise=True,
        zero_mean_prior=not args.learn_prior,
        weight_log_sigma_init=args.bnn_weight_log_sigma_init,
        device=device,
        dtype=dtype,
    )
    if not args.learn_prior:
        prior.freeze_parameters()
    return prior


def make_run_name(args):
    return (
        f"gmvip_{args.dataset_name}_{args.operator_type}_{args.posterior_type}"
        f"_{args.mean_mode}_{args.inducing_scale}"
    )


def make_regression_split(dataset, test_size, seed, normalize_inputs):
    if normalize_inputs:
        return dataset.get_split(test_size=test_size, seed=seed)
    train_indexes, test_indexes = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        random_state=seed,
    )
    train_dataset = Training_Dataset(
        dataset.inputs[train_indexes],
        dataset.targets[train_indexes],
        normalize_targets=dataset.type == "regression",
        normalize_inputs=False,
    )
    train_eval_dataset = Test_Dataset(
        dataset.inputs[train_indexes],
        dataset.targets[train_indexes],
        train_dataset.inputs_mean,
        train_dataset.inputs_std,
    )
    test_dataset = Test_Dataset(
        dataset.inputs[test_indexes],
        dataset.targets[test_indexes],
        train_dataset.inputs_mean,
        train_dataset.inputs_std,
    )
    return train_dataset, train_eval_dataset, test_dataset


def dataset_plot_domain(dataset):
    if hasattr(dataset, "plot_domain"):
        return dataset.plot_domain
    x = np.asarray(dataset.inputs).reshape(-1)
    pad = 0.06 * max(float(x.max() - x.min()), 1e-8)
    return float(x.min() - pad), float(x.max() + pad)


def bimodal_branches(x):
    branch_cos = 20.0 * np.cos(x - 0.5)
    branch_sin = 20.0 * np.sin(x - 0.5)
    return branch_cos, branch_sin


def dataset_reference_curves(dataset_name, dataset, x_orig):
    if dataset_name == "gap":
        true = dataset.true_function(x_orig)
        return {
            "primary": true,
            "curves": [("true function", true, "black", 1.2, 1.0)],
            "conditional_mean": true,
        }
    if dataset_name == "bimodal":
        branch_cos, branch_sin = bimodal_branches(x_orig)
        conditional_mean = 0.5 * (branch_cos + branch_sin)
        return {
            "primary": conditional_mean,
            "curves": [
                ("cos branch", branch_cos, "black", 1.1, 0.95),
                ("sin branch", branch_sin, "#7f7f7f", 1.1, 0.95),
                ("conditional mean", conditional_mean, "#444444", 1.0, 0.55),
            ],
            "conditional_mean": conditional_mean,
            "branches": [branch_cos, branch_sin],
        }
    raise ValueError(f"Unsupported dataset_name={dataset_name!r}.")


def region_masks(dataset_name, dataset, x_orig, plot_domain):
    if dataset_name == "gap":
        gap_mask = (x_orig >= dataset.gap_bounds[0]) & (x_orig <= dataset.gap_bounds[1])
        observed_mask = (
            ((x_orig >= -4.0) & (x_orig <= dataset.gap_bounds[0]))
            | ((x_orig >= dataset.gap_bounds[1]) & (x_orig <= 4.0))
        )
        return {
            "heldout": gap_mask,
            "observed": observed_mask,
            "heldout_name": "gap",
            "observed_name": "observed_region",
        }
    center_width = 0.25 * (plot_domain[1] - plot_domain[0])
    center = 0.5 * (plot_domain[0] + plot_domain[1])
    central_mask = np.abs(x_orig - center) <= 0.5 * center_width
    outer_mask = ~central_mask
    return {
        "heldout": central_mask,
        "observed": outer_mask,
        "heldout_name": "central_region",
        "observed_name": "outer_region",
    }


def full_train_eval(model, X, y, args, step):
    model_was_training = model.training
    model.eval()
    y_eval = y[..., 0] if y.ndim == 2 and y.shape[-1] == 1 else y
    eval_batch_size = args.convergence_eval_batch_size or args.batch_size or X.shape[0]
    eval_batch_size = max(1, min(int(eval_batch_size), int(X.shape[0])))
    seed = args.seed + 1_000_003
    with torch.no_grad():
        expected_log_lik = torch.zeros((), dtype=X.dtype, device=X.device)
        for start in range(0, X.shape[0], eval_batch_size):
            stop = min(start + eval_batch_size, X.shape[0])
            f_samples, _, _ = model.sample_posterior_values_with_kl(
                X[start:stop],
                args.convergence_num_mc_samples,
                seed=seed,
            )
            log_prob = model.likelihood.log_prob(y_eval[start:stop].unsqueeze(0), f_samples)
            expected_log_lik = expected_log_lik + model._alpha_sample_log_likelihood(
                log_prob,
                data_alpha=args.data_alpha,
            )
        if model.posterior_type == "realnvp":
            kl = model.posterior.kl_to_standard_normal(
                num_samples=args.convergence_num_mc_samples,
                generator=model._make_generator(seed + 17),
                antithetic=model.antithetic_samples,
            )
        else:
            kl = model.posterior.kl_to_standard_normal()
        _, _, diagnostics = model.posterior.rsample_with_kl(
            args.convergence_num_mc_samples,
            generator=model._make_generator(seed + 23),
            antithetic=model.antithetic_samples,
        )
        q_std_mean = diagnostics["q_std_mean"]
        coefficient_displacement = diagnostics["coefficient_displacement"]
        elbo = expected_log_lik - float(args.beta) * kl
        loss = -elbo
    model.train(model_was_training)
    return {
        "step": int(step),
        "eval_loss": tensor_to_float(loss),
        "eval_elbo": tensor_to_float(elbo),
        "eval_data_nll": tensor_to_float(-expected_log_lik),
        "eval_kl": tensor_to_float(kl),
        "eval_noise": tensor_to_float(model.noise_std),
        "eval_q_std_mean": tensor_to_float(q_std_mean),
        "eval_coefficient_displacement": tensor_to_float(coefficient_displacement),
    }


def convergence_progress(eval_history, window):
    losses = [row["eval_loss"] for row in eval_history]
    if len(losses) < 2 * window:
        return None
    previous = np.asarray(losses[-2 * window : -window], dtype=np.float64)
    current = np.asarray(losses[-window:], dtype=np.float64)
    previous_mean = float(previous.mean())
    current_mean = float(current.mean())
    abs_improvement = previous_mean - current_mean
    rel_improvement = abs_improvement / max(abs(previous_mean), 1.0)
    return {
        "previous_mean": previous_mean,
        "current_mean": current_mean,
        "abs_improvement": float(abs_improvement),
        "rel_improvement": float(rel_improvement),
    }


def save_svg_fallback(
    path,
    x_orig,
    reference_curves,
    f_mean,
    f_std,
    y_std,
    f_samples,
    noise_std,
    prior_samples,
    train_x,
    train_y,
    inducing_x,
    gap_bounds,
    plot_domain,
    title,
    max_plot_samples,
):
    width, height = 1000, 560
    left, right, top, bottom = 70, 30, 40, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    visible = np.concatenate(
        [
            *[curve[1].reshape(-1) for curve in reference_curves],
            f_mean.reshape(-1),
            (f_samples[: min(16, f_samples.shape[0])] - 2.0 * noise_std).reshape(-1),
            (f_samples[: min(16, f_samples.shape[0])] + 2.0 * noise_std).reshape(-1),
            f_samples[: min(16, f_samples.shape[0])].reshape(-1),
            train_y.reshape(-1),
        ]
    )
    y_min = float(np.nanmin(visible))
    y_max = float(np.nanmax(visible))
    y_pad = 0.08 * max(1e-8, y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def sx(x):
        return left + (float(x) - plot_domain[0]) / max(1e-12, plot_domain[1] - plot_domain[0]) * plot_w

    def sy(y):
        return top + (y_max - float(y)) / max(1e-12, y_max - y_min) * plot_h

    def polyline(xs, ys):
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))

    def band_polygon(lower, upper):
        pts = [(sx(x), sy(y)) for x, y in zip(x_orig, lower)]
        pts += [(sx(x), sy(y)) for x, y in reversed(list(zip(x_orig, upper)))]
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560">',
        '<rect width="1000" height="560" fill="white"/>',
        f'<text x="70" y="26" font-family="Arial" font-size="20" fill="#111">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
    ]
    if gap_bounds is not None:
        gap_x0, gap_x1 = sx(gap_bounds[0]), sx(gap_bounds[1])
        lines.append(
            f'<rect x="{gap_x0:.2f}" y="{top}" width="{gap_x1 - gap_x0:.2f}" height="{plot_h}" fill="#f3d36b" fill-opacity="0.35"/>'
        )
    z_marker_y = top + plot_h - 10
    for z in inducing_x:
        x_screen = sx(z)
        lines.append(
            f'<line x1="{x_screen:.2f}" y1="{top}" x2="{x_screen:.2f}" y2="{top + plot_h}" '
            'stroke="#2ca02c" stroke-opacity="0.45" stroke-width="1.2" stroke-dasharray="4,4"/>'
        )
        lines.append(
            f'<polygon points="{x_screen - 5:.2f},{z_marker_y + 8:.2f} '
            f'{x_screen + 5:.2f},{z_marker_y + 8:.2f} {x_screen:.2f},{z_marker_y:.2f}" '
            'fill="#2ca02c" fill-opacity="0.95"/>'
        )
    for sample in prior_samples[: min(8, prior_samples.shape[0])]:
        lines.append(
            f'<polyline points="{polyline(x_orig, sample)}" fill="none" stroke="#888888" stroke-opacity="0.18" stroke-width="0.8"/>'
        )
    for sample in f_samples[: min(max_plot_samples, f_samples.shape[0])]:
        lines.append(
            f'<polygon points="{band_polygon(sample - 2.0 * noise_std, sample + 2.0 * noise_std)}" '
            'fill="#1f77b4" fill-opacity="0.055"/>'
        )
        lines.append(
            f'<polyline points="{polyline(x_orig, sample)}" fill="none" stroke="#1f77b4" stroke-opacity="0.42" stroke-width="0.8"/>'
        )
    for label, values, color, width_px, opacity in reference_curves:
        lines.append(
            f'<polyline points="{polyline(x_orig, values)}" fill="none" stroke="{color}" '
            f'stroke-width="{width_px}" stroke-opacity="{opacity}"/>'
        )
    lines.append(f'<polyline points="{polyline(x_orig, f_mean)}" fill="none" stroke="#1f77b4" stroke-width="2.2"/>')
    for x, y in zip(train_x.reshape(-1), train_y.reshape(-1)):
        lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.2" fill="#333333" fill-opacity="0.25"/>')
    legend_x = left + 12
    legend_y = top + 16
    lines.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="22" height="9" fill="#1f77b4" fill-opacity="0.18"/>',
            f'<text x="{legend_x + 30}" y="{legend_y + 9}" font-family="Arial" font-size="12" fill="#222">posterior sample +/- 2 noise std</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 24}" x2="{legend_x + 22}" y2="{legend_y + 24}" stroke="#1f77b4" stroke-width="2"/>',
            f'<text x="{legend_x + 30}" y="{legend_y + 28}" font-family="Arial" font-size="12" fill="#222">posterior mean</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 44}" x2="{legend_x + 22}" y2="{legend_y + 44}" stroke="#2ca02c" stroke-width="1.4" stroke-dasharray="4,4"/>',
            f'<polygon points="{legend_x + 7},{legend_y + 54} {legend_x + 15},{legend_y + 54} {legend_x + 11},{legend_y + 46}" fill="#2ca02c"/>',
            f'<text x="{legend_x + 30}" y="{legend_y + 50}" font-family="Arial" font-size="12" fill="#222">inducing Z</text>',
        ]
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = make_run_name(args)

    dataset = get_dataset(args.dataset_name)
    train_dataset, train_eval_dataset, test_dataset = make_regression_split(
        dataset,
        test_size=args.test_size,
        seed=args.seed,
        normalize_inputs=args.normalize_inputs,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed + 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    train_inputs = torch.tensor(train_dataset.inputs, dtype=dtype, device=device)
    train_targets = torch.tensor(train_dataset.targets, dtype=dtype, device=device)
    Z = initialize_inducing_points(
        train_inputs,
        num_inducing=args.num_inducing,
        method=args.inducing_method,
        seed=args.seed + 31,
    )
    initial_Z = Z.detach().clone()
    prior = build_bnn_prior(
        args,
        input_dim=train_dataset.input_dim,
        output_dim=train_dataset.output_dim,
        device=device,
        dtype=dtype,
    )
    model = GeneralizedMatheronVIP(
        base_prior=prior,
        inducing_points=Z,
        operator_type=args.operator_type,
        posterior_type=args.posterior_type,
        num_operator_bank_samples=args.num_operator_bank_samples,
        learn_noise=args.learn_noise,
        jitter=args.jitter,
        shrinkage=args.shrinkage,
        init_log_noise=args.init_log_noise,
        min_log_noise=args.min_log_noise,
        max_log_noise=args.max_log_noise,
        freeze_base_prior=not args.learn_prior,
        detach_prior_samples=not args.learn_prior,
        learn_Z=args.learn_Z,
        learn_kernel=args.learn_kernel,
        init_lengthscale=args.init_lengthscale,
        init_outputscale=(
            "prior_marginal" if args.init_outputscale is None else args.init_outputscale
        ),
        mean_mode=args.mean_mode,
        inducing_scale=args.inducing_scale,
        flow_depth=args.flow_depth,
        flow_hidden_dim=args.flow_hidden_dim,
        flow_num_layers=args.flow_num_layers,
        flow_dropout=args.flow_dropout,
        flow_scale_bound=args.flow_scale_bound,
        antithetic_samples=args.antithetic_samples,
        posterior_init_mean=args.posterior_init_mean,
        posterior_init_log_std=args.posterior_init_log_std,
        posterior_min_log_std=args.posterior_min_log_std,
        posterior_max_log_std=args.posterior_max_log_std,
        num_data=len(train_dataset),
        num_train_samples=args.num_mc_samples,
        beta=args.beta,
        beta_warmup_steps=args.beta_warmup_steps,
        data_alpha=args.data_alpha,
        max_grad_norm=args.max_grad_norm,
        operator_bank_seed=args.seed + 101,
    )
    model.prepare_for_training(train_loader)

    optimizer = torch.optim.Adam(model.vi_parameters(), lr=args.lr)
    stream = infinite_loader(train_loader)
    history = []
    eval_history = []
    convergence_checks_passed = 0
    convergence = {
        "enabled": bool(args.converge),
        "converged": False,
        "reason": "max_iterations_reached" if args.converge else "disabled",
        "checks_passed": 0,
        "last_check": None,
        "min_iterations": args.min_iterations,
        "max_iterations": args.iterations,
        "eval_every": args.convergence_eval_every,
        "window": args.convergence_window,
        "patience": args.convergence_patience,
        "rel_tol": args.convergence_rel_tol,
        "abs_tol": args.convergence_abs_tol,
        "num_mc_samples": args.convergence_num_mc_samples,
        "eval_batch_size": args.convergence_eval_batch_size or args.batch_size,
    }
    min_iterations = args.min_iterations
    if min_iterations is None:
        min_iterations = max(
            args.beta_warmup_steps,
            2 * args.convergence_window * args.convergence_eval_every,
        )
    convergence["min_iterations"] = int(min_iterations)

    model.train()
    actual_iterations = 0
    for step in range(1, args.iterations + 1):
        actual_iterations = int(step)
        X_batch, y_batch = next(stream)
        X_batch = X_batch.to(dtype=dtype, device=device)
        y_batch = y_batch.to(dtype=dtype, device=device)
        loss = model._train_step(optimizer, X_batch, y_batch)
        diagnostics = model.last_train_metrics
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.detach().cpu()}")
        beta = tensor_to_float(diagnostics["beta"])

        if step == 1 or step % args.log_every == 0 or step == args.iterations:
            row = {
                "step": step,
                "loss": tensor_to_float(diagnostics["loss"]),
                "elbo": tensor_to_float(diagnostics["elbo"]),
                "data_nll": tensor_to_float(diagnostics["data_nll"]),
                "kl": tensor_to_float(diagnostics["kl"]),
                "noise": tensor_to_float(diagnostics["noise"]),
                "q_std_mean": metric_to_float(diagnostics, "q_std_mean"),
                "coefficient_displacement": metric_to_float(diagnostics, "coefficient_displacement"),
                "beta": beta,
            }
            history.append(row)
            print(
                "step={step} loss={loss:.3f} data_nll={data_nll:.3f} "
                "kl={kl:.3f} noise={noise:.4f} q_std={q_std_mean} beta={beta:.3f}".format(
                    **row
                )
            )

        should_eval = (
            step == 1
            or step % args.convergence_eval_every == 0
            or step == args.iterations
        )
        if should_eval:
            eval_row = full_train_eval(model, train_inputs, train_targets, args, step)
            eval_history.append(eval_row)
            print(
                "eval step={step} loss={eval_loss:.3f} data_nll={eval_data_nll:.3f} "
                "kl={eval_kl:.3f} noise={eval_noise:.4f} q_std={eval_q_std_mean}".format(
                    **eval_row
                )
            )
            check = convergence_progress(eval_history, max(1, args.convergence_window))
            convergence["last_check"] = check
            if args.converge and step >= min_iterations and check is not None:
                abs_improvement = check["abs_improvement"]
                rel_improvement = check["rel_improvement"]
                abs_ok = 0.0 <= abs_improvement <= args.convergence_abs_tol
                rel_ok = 0.0 <= rel_improvement <= args.convergence_rel_tol
                small_worsening = abs_improvement < 0.0 and abs(abs_improvement) <= args.convergence_abs_tol
                if rel_ok or abs_ok:
                    convergence_checks_passed += 1
                elif small_worsening:
                    convergence_checks_passed += 1
                else:
                    convergence_checks_passed = 0
                convergence["checks_passed"] = int(convergence_checks_passed)
                if convergence_checks_passed >= args.convergence_patience:
                    convergence["converged"] = True
                    convergence["reason"] = "full_train_eval_loss_plateau"
                    print(
                        "converged step={step} rel_improvement={rel:.6f} "
                        "abs_improvement={abs_imp:.3f}".format(
                            step=step,
                            rel=check["rel_improvement"],
                            abs_imp=check["abs_improvement"],
                        )
                    )
                    break

    model.eval()
    with torch.no_grad():
        x_train = torch.tensor(train_eval_dataset.inputs, dtype=dtype, device=device)
        y_train_original = torch.tensor(train_eval_dataset.targets, dtype=dtype, device=device)
        train_pred = model.predict(x_train, num_samples=args.eval_samples)
        train_mean_original = (
            train_pred["y_mean"].unsqueeze(-1) * float(np.asarray(train_dataset.targets_std).reshape(-1)[0])
            + float(np.asarray(train_dataset.targets_mean).reshape(-1)[0])
        )
        train_rmse = torch.sqrt((train_mean_original - y_train_original).square().mean())

        x_test = torch.tensor(test_dataset.inputs, dtype=dtype, device=device)
        y_test_original = torch.tensor(test_dataset.targets, dtype=dtype, device=device)
        test_pred = model.predict(x_test, num_samples=args.eval_samples)
        test_mean_original = (
            test_pred["y_mean"].unsqueeze(-1) * float(np.asarray(train_dataset.targets_std).reshape(-1)[0])
            + float(np.asarray(train_dataset.targets_mean).reshape(-1)[0])
        )
        test_rmse = torch.sqrt((test_mean_original - y_test_original).square().mean())

        plot_domain = dataset_plot_domain(dataset)
        grid_orig = np.linspace(plot_domain[0], plot_domain[1], 401)[:, None]
        grid_norm = (grid_orig - train_dataset.inputs_mean) / train_dataset.inputs_std
        x_grid = torch.tensor(grid_norm, dtype=dtype, device=device)
        pred = model.predict(x_grid, num_samples=args.eval_samples, include_noise=True)
        prior_samples = model.sample_prior_values(x_grid, min(args.eval_samples, 128))

    target_mean = float(np.asarray(train_dataset.targets_mean).reshape(-1)[0])
    target_std = float(np.asarray(train_dataset.targets_std).reshape(-1)[0])
    f_samples = pred["f_samples"].detach().cpu() * target_std + target_mean
    f_mean = pred["f_mean"].detach().cpu() * target_std + target_mean
    f_std = pred["f_var"].detach().cpu().sqrt() * target_std
    y_std = pred["y_var"].detach().cpu().sqrt() * target_std
    noise_std_original = float(model.noise_std.detach().cpu()) * target_std
    prior_samples = prior_samples.detach().cpu() * target_std + target_mean
    train_x_orig = train_eval_dataset.inputs * train_dataset.inputs_std + train_dataset.inputs_mean
    train_y_orig = train_eval_dataset.targets
    x_orig = grid_orig[:, 0]
    references = dataset_reference_curves(args.dataset_name, dataset, x_orig)
    reference_curves = references["curves"]
    reference_mean = references["conditional_mean"]
    masks = region_masks(args.dataset_name, dataset, x_orig, plot_domain)
    heldout_mask = masks["heldout"]
    observed_mask = masks["observed"]

    train_rmse = float(train_rmse.detach().cpu())
    test_rmse = float(test_rmse.detach().cpu())
    heldout_function_std = float(f_std[heldout_mask].mean())
    observed_function_std = float(f_std[observed_mask].mean())
    heldout_predictive_std = float(y_std[heldout_mask].mean())
    observed_predictive_std = float(y_std[observed_mask].mean())
    f_mean_np = f_mean.numpy()
    final_minibatch = history[-1] if history else None
    final_eval = eval_history[-1] if eval_history else None
    best_eval = min(eval_history, key=lambda row: row["eval_loss"]) if eval_history else None
    final_metrics_source = "eval" if final_eval is not None else "minibatch"

    def final_metric(eval_key, minibatch_key):
        if final_eval is not None:
            return final_eval.get(eval_key)
        if final_minibatch is not None:
            return final_minibatch.get(minibatch_key)
        return None

    summary = {
        "model": "gmvip",
        "prior": "bnn",
        "dataset_name": args.dataset_name,
        "device": str(device),
        "seed": args.seed,
        "iterations": args.iterations,
        "actual_iterations": actual_iterations,
        "convergence": convergence,
        "operator_type": args.operator_type,
        "posterior_type": args.posterior_type,
        "num_inducing": args.num_inducing,
        "inducing_method": args.inducing_method,
        "normalize_inputs": args.normalize_inputs,
        "learn_prior": args.learn_prior,
        "learn_Z": args.learn_Z,
        "num_operator_bank_samples": args.num_operator_bank_samples,
        "num_mc_samples": args.num_mc_samples,
        "eval_samples": args.eval_samples,
        "antithetic_samples": args.antithetic_samples,
        "convergence_eval_batch_size": args.convergence_eval_batch_size or args.batch_size,
        "mean_mode": args.mean_mode,
        "inducing_scale": args.inducing_scale,
        "learn_kernel": args.learn_kernel,
        "init_lengthscale": args.init_lengthscale,
        "init_outputscale": args.init_outputscale,
        "posterior_init_mean": args.posterior_init_mean,
        "posterior_init_log_std": args.posterior_init_log_std,
        "posterior_min_log_std": args.posterior_min_log_std,
        "posterior_max_log_std": args.posterior_max_log_std,
        "flow_depth": args.flow_depth,
        "flow_hidden_dim": args.flow_hidden_dim,
        "flow_num_layers": args.flow_num_layers,
        "flow_dropout": args.flow_dropout,
        "flow_scale_bound": args.flow_scale_bound,
        "learn_noise": args.learn_noise,
        "beta": args.beta,
        "data_alpha": args.data_alpha,
        "beta_warmup_steps": args.beta_warmup_steps,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "reference_mean_grid_rmse": float(np.sqrt(np.mean((f_mean_np - reference_mean) ** 2))),
        f"{masks['observed_name']}_reference_mean_rmse": float(
            np.sqrt(np.mean((f_mean_np[observed_mask] - reference_mean[observed_mask]) ** 2))
        ),
        f"{masks['heldout_name']}_reference_mean_rmse": float(
            np.sqrt(np.mean((f_mean_np[heldout_mask] - reference_mean[heldout_mask]) ** 2))
        ),
        f"{masks['heldout_name']}_function_std_mean": heldout_function_std,
        f"{masks['observed_name']}_function_std_mean": observed_function_std,
        f"{masks['heldout_name']}_to_{masks['observed_name']}_std_ratio": heldout_function_std / max(observed_function_std, 1e-12),
        f"{masks['heldout_name']}_predictive_std_mean": heldout_predictive_std,
        f"{masks['observed_name']}_predictive_std_mean": observed_predictive_std,
        f"{masks['heldout_name']}_to_{masks['observed_name']}_predictive_std_ratio": heldout_predictive_std / max(observed_predictive_std, 1e-12),
        "final_metrics_source": final_metrics_source,
        "final_loss": final_metric("eval_loss", "loss"),
        "final_data_nll": final_metric("eval_data_nll", "data_nll"),
        "final_kl": final_metric("eval_kl", "kl"),
        "final_noise": final_metric("eval_noise", "noise"),
        "final_noise_original_units": noise_std_original,
        "final_q_std_mean": final_metric("eval_q_std_mean", "q_std_mean"),
        "final_coefficient_displacement": final_metric("eval_coefficient_displacement", "coefficient_displacement"),
        "final_eval_loss": final_eval["eval_loss"] if final_eval else None,
        "final_eval_data_nll": final_eval["eval_data_nll"] if final_eval else None,
        "final_eval_kl": final_eval["eval_kl"] if final_eval else None,
        "final_eval_noise": final_eval["eval_noise"] if final_eval else None,
        "best_eval_step": best_eval["step"] if best_eval else None,
        "best_eval_loss": best_eval["eval_loss"] if best_eval else None,
        "best_eval_data_nll": best_eval["eval_data_nll"] if best_eval else None,
        "best_eval_kl": best_eval["eval_kl"] if best_eval else None,
        "best_eval_noise": best_eval["eval_noise"] if best_eval else None,
        "final_minibatch_loss": final_minibatch["loss"] if final_minibatch else None,
        "final_minibatch_data_nll": final_minibatch["data_nll"] if final_minibatch else None,
        "final_minibatch_kl": final_minibatch["kl"] if final_minibatch else None,
        "final_minibatch_noise": final_minibatch["noise"] if final_minibatch else None,
    }
    if "branches" in references:
        branches = np.stack(references["branches"], axis=0)
        nearest_branch_error = np.min(np.abs(f_mean_np[None, :] - branches), axis=0)
        sample_branch_dist = np.min(
            np.abs(f_samples.numpy()[:, None, :] - branches[None, :, :]),
            axis=1,
        )
        summary["nearest_branch_mae"] = float(nearest_branch_error.mean())
        summary["sample_nearest_branch_mae"] = float(sample_branch_dist.mean())
    kernel = getattr(model.operator, "kernel", None)
    if kernel is not None:
        lengthscale = kernel.lengthscale.detach().cpu()
        if lengthscale.ndim == 0:
            summary["final_kernel_lengthscale"] = float(lengthscale)
        else:
            summary["final_kernel_lengthscale"] = [float(v) for v in lengthscale.reshape(-1)]
        summary["final_kernel_outputscale"] = float(kernel.outputscale.detach().cpu().reshape(()))

    initial_z_orig = initial_Z.detach().cpu().numpy() * train_dataset.inputs_std + train_dataset.inputs_mean
    z_orig = model.Z.detach().cpu().numpy() * train_dataset.inputs_std + train_dataset.inputs_mean
    summary["initial_inducing_points_original_units"] = [
        float(value) for value in initial_z_orig.reshape(-1)
    ]
    summary["final_inducing_points_original_units"] = [
        float(value) for value in z_orig.reshape(-1)
    ]
    title = f"GMVIP {args.operator_type}/{args.posterior_type} with BNN prior on {args.dataset_name}"
    if plt is None:
        figure_path = output_dir / f"{run_name}.svg"
        save_svg_fallback(
            figure_path,
            x_orig=x_orig,
            reference_curves=reference_curves,
            f_mean=f_mean.numpy(),
            f_std=f_std.numpy(),
            y_std=y_std.numpy(),
            f_samples=f_samples.numpy(),
            noise_std=noise_std_original,
            prior_samples=prior_samples.numpy(),
            train_x=train_x_orig,
            train_y=train_y_orig,
            inducing_x=z_orig[:, 0],
            gap_bounds=getattr(dataset, "gap_bounds", None),
            plot_domain=plot_domain,
            title=title,
            max_plot_samples=args.plot_samples,
        )
    else:
        figure_path = output_dir / f"{run_name}.png"
        fig, ax = plt.subplots(figsize=(10, 5.5))
        gap_bounds = getattr(dataset, "gap_bounds", None)
        if gap_bounds is not None:
            ax.axvspan(gap_bounds[0], gap_bounds[1], color="#f3d36b", alpha=0.35, label="held-out gap")
        for sample in prior_samples[: min(8, prior_samples.shape[0])]:
            ax.plot(x_orig, sample.numpy(), color="#888888", linewidth=0.7, alpha=0.18)
        plotted_samples = f_samples[: min(args.plot_samples, f_samples.shape[0])]
        for index, sample in enumerate(plotted_samples):
            sample_np = sample.numpy()
            ax.fill_between(
                x_orig,
                sample_np - 2.0 * noise_std_original,
                sample_np + 2.0 * noise_std_original,
                color="tab:blue",
                alpha=0.045,
                linewidth=0.0,
                label="sample +/-2 noise std" if index == 0 else None,
            )
            ax.plot(
                x_orig,
                sample_np,
                color="tab:blue",
                linewidth=0.8,
                alpha=0.42,
                label="posterior function sample" if index == 0 else None,
            )
        for label, values, color, width_px, alpha in reference_curves:
            ax.plot(x_orig, values, color=color, linewidth=width_px, alpha=alpha, label=label)
        ax.plot(x_orig, f_mean.numpy(), color="tab:blue", linewidth=2.0, label="posterior mean")
        ax.scatter(train_x_orig[:, 0], train_y_orig[:, 0], s=5, color="#333333", alpha=0.25, label="train")
        for index, z in enumerate(z_orig[:, 0]):
            ax.axvline(
                float(z),
                color="tab:green",
                linestyle="--",
                linewidth=0.9,
                alpha=0.45,
                label="inducing Z" if index == 0 else None,
            )
        ax.scatter(
            z_orig[:, 0],
            np.full_like(z_orig[:, 0], 0.02),
            marker="^",
            s=52,
            color="tab:green",
            edgecolor="white",
            linewidth=0.5,
            transform=ax.get_xaxis_transform(),
            zorder=5,
            clip_on=False,
        )
        ax.set_xlim(plot_domain)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=160)
        plt.close(fig)

    summary["figure_path"] = str(figure_path)
    summary_path = output_dir / f"{run_name}_summary.json"
    history_path = output_dir / f"{run_name}_history.json"
    eval_history_path = output_dir / f"{run_name}_eval_history.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with eval_history_path.open("w", encoding="utf-8") as handle:
        json.dump(eval_history, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"summary_path={summary_path}")
    print(f"history_path={history_path}")
    print(f"eval_history_path={eval_history_path}")
    print(f"figure_path={figure_path}")


if __name__ == "__main__":
    main()
