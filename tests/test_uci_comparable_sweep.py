import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.uci.benchmark import (
    _variant_tag,
    build_model,
    parse_args,
    train_with_metrics,
)


class TinyTrainDataset:
    def __init__(self):
        rng = np.random.RandomState(0)
        self.inputs = rng.randn(12, 2).astype(np.float64)
        self.targets = rng.randn(12, 1).astype(np.float64)
        self.input_dim = 2
        self.output_dim = 1
        self.targets_mean = 0.0
        self.targets_std = 1.0

    def __len__(self):
        return self.inputs.shape[0]


def _args(extra):
    return parse_args(
        [
            "--model",
            "map",
            "--dataset",
            "boston",
            "--iterations",
            "30000",
            "--device",
            "cpu",
            "--dtype",
            "float64",
            *extra,
        ]
    )


def test_wandb_names_and_groups_are_launcher_controlled():
    args = _args(
        [
            "--model",
            "gmvip",
            "--wandb_name",
            "Boston | GMVIP Tunable Prior | seed 0",
            "--wandb_group",
            "Boston | GMVIP Tunable Prior",
        ]
    )

    assert args.wandb_name == "Boston | GMVIP Tunable Prior | seed 0"
    assert args.wandb_group == "Boston | GMVIP Tunable Prior"


def test_wandb_stats_are_disabled_by_default_but_configurable():
    default_args = _args(["--model", "map"])
    enabled_args = _args(["--model", "map", "--no-wandb_disable_stats"])

    assert default_args.wandb_disable_stats is True
    assert enabled_args.wandb_disable_stats is False


def test_mfvi_defaults_to_alpha_zero():
    args = _args(["--model", "mfvi"])
    explicit_args = _args(["--model", "mfvi", "--bb_alpha", "0.5"])

    assert args.bb_alpha == 0.0
    assert explicit_args.bb_alpha == 0.5


def test_variant_tags_distinguish_prior_and_inducing_states():
    ftip_learn = _args(["--model", "ftip", "--ftip_learn_prior"])
    ftip_fixed = _args(["--model", "ftip", "--no-ftip_learn_prior"])
    assert _variant_tag(ftip_learn, "ftip") == "_learnprior"
    assert _variant_tag(ftip_fixed, "ftip") == "_fixedprior"

    sip_learn = _args(
        [
            "--model",
            "sip",
            "--sip_num_inducing",
            "100",
            "--sip_inducing_method",
            "kmeans",
            "--sip_learn_inducing",
            "--sip_learn_prior",
        ]
    )
    sip_fixed = _args(
        [
            "--model",
            "sip",
            "--sip_num_inducing",
            "100",
            "--sip_inducing_method",
            "kmeans",
            "--sip_learn_inducing",
            "--no-sip_learn_prior",
        ]
    )
    assert _variant_tag(sip_learn, "sip") != _variant_tag(sip_fixed, "sip")
    assert "learnZ" in _variant_tag(sip_learn, "sip")
    assert "learnprior" in _variant_tag(sip_learn, "sip")
    assert "fixedprior" in _variant_tag(sip_fixed, "sip")

    gmvip_full = _args(["--model", "gmvip"])
    gmvip_inducing_only = _args(["--model", "gmvip", "--gmvip_path_mode", "inducing_only"])
    assert gmvip_full.gmvip_path_mode == "full"
    assert _variant_tag(gmvip_full, "gmvip") != _variant_tag(gmvip_inducing_only, "gmvip")
    assert "_full_" in _variant_tag(gmvip_full, "gmvip")
    assert "_inducing_only_" in _variant_tag(gmvip_inducing_only, "gmvip")


def test_ftip_prior_flag_controls_generator_trainability():
    dataset = TinyTrainDataset()
    learn_args = _args(["--model", "ftip", "--regression_coeffs", "6", "--ftip_learn_prior"])
    fixed_args = _args(["--model", "ftip", "--regression_coeffs", "6", "--no-ftip_learn_prior"])

    learn_model = build_model(learn_args, dataset)
    fixed_model = build_model(fixed_args, dataset)

    assert any(param.requires_grad for param in learn_model.generative_function.parameters())
    assert not any(param.requires_grad for param in fixed_model.generative_function.parameters())


def test_gmvip_and_sip_build_with_learn_z_and_both_prior_states():
    dataset = TinyTrainDataset()
    base = [
        "--device",
        "cpu",
        "--dtype",
        "float64",
        "--hidden_dims",
        "4",
        "--iterations",
        "1",
    ]
    gmvip_common = [
        "--model",
        "gmvip",
        "--dataset",
        "boston",
        *base,
        "--gmvip_operator_type",
        "empirical",
        "--gmvip_posterior_type",
        "gaussian",
        "--gmvip_num_inducing",
        "4",
        "--gmvip_inducing_method",
        "kmeans",
        "--gmvip_learn_Z",
        "--gmvip_num_train_samples",
        "8",
        "--gmvip_num_eval_samples",
        "8",
        "--gmvip_num_operator_bank_samples",
        "8",
    ]
    sip_common = [
        "--model",
        "sip",
        "--dataset",
        "boston",
        *base,
        "--sip_num_inducing",
        "4",
        "--sip_inducing_method",
        "kmeans",
        "--sip_learn_inducing",
        "--sip_num_prior_samples",
        "8",
        "--sip_num_eval_samples",
        "8",
    ]

    for prior_flag, should_train in [
        ("--gmvip_learn_prior", True),
        ("--no-gmvip_learn_prior", False),
    ]:
        model = build_model(parse_args([*gmvip_common, prior_flag]), dataset)
        assert model.Z.requires_grad
        assert any(p.requires_grad for p in model.base_prior.parameters()) is should_train

    inducing_only_model = build_model(
        parse_args([*gmvip_common, "--gmvip_path_mode", "inducing_only"]), dataset
    )
    assert inducing_only_model.path_mode == "inducing_only"

    for prior_flag, should_train in [("--sip_learn_prior", True), ("--no-sip_learn_prior", False)]:
        model = build_model(parse_args([*sip_common, prior_flag]), dataset)
        assert isinstance(model.Z, torch.nn.Parameter)
        assert any(p.requires_grad for p in model.generative_function.parameters()) is should_train


def test_gmvip_grid_inducing_initializer_builds_for_multidimensional_uci_inputs():
    dataset = TinyTrainDataset()
    args = _args(
        [
            "--model",
            "gmvip",
            "--hidden_dims",
            "4",
            "--gmvip_operator_type",
            "empirical",
            "--gmvip_posterior_type",
            "gaussian",
            "--gmvip_num_inducing",
            "4",
            "--gmvip_inducing_method",
            "grid",
            "--gmvip_num_train_samples",
            "8",
            "--gmvip_num_eval_samples",
            "8",
            "--gmvip_num_operator_bank_samples",
            "8",
        ]
    )

    model = build_model(args, dataset)

    assert model.Z.shape == (4, 2)


def test_custom_uci_loop_runs_prepare_fit_hook(monkeypatch):
    class PreparingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
            self.prepared_loader = None

        def _prepare_fit(self, train_loader):
            self.prepared_loader = train_loader

        def _train_step(self, optimizer, inputs, target):
            optimizer.zero_grad()
            loss = ((self.weight * inputs) - target).square().mean()
            loss.backward()
            optimizer.step()
            return loss

    inputs = torch.ones(4, 1, dtype=torch.float64)
    targets = torch.zeros(4, 1, dtype=torch.float64)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = PreparingModel()
    args = _args(["--model", "fbnn"])
    monkeypatch.setenv("IPZOO_DISABLE_TQDM", "1")

    train_with_metrics(
        model,
        loader,
        train_test_dataset=None,
        validation_dataset=None,
        args=args,
        iterations=1,
        model_type="fbnn",
    )

    assert model.prepared_loader is loader
