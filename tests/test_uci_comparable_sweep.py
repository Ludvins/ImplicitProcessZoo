import numpy as np
import torch

from scripts.uci_benchmark import (
    _variant_tag,
    _wandb_run_metadata,
    build_model,
    parse_args,
)
from scripts.run_uci_comparable_8_methods_30k import VARIANTS, build_command, select_variants


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


def test_comparable_wandb_names_and_groups_are_exact():
    cases = [
        ("map", [], "Boston | MAP | seed 0", "Boston | MAP"),
        ("mfvi", [], "Boston | MFVI | seed 0", "Boston | MFVI"),
        ("fbnn", [], "Boston | FBNN | seed 0", "Boston | FBNN"),
        ("tfsvi", [], "Boston | TFSVI | seed 0", "Boston | TFSVI"),
        ("vip", ["--vip_learn_prior"], "Boston | VIP Tunable Prior | seed 0", "Boston | VIP Tunable Prior"),
        ("vip", ["--no-vip_learn_prior"], "Boston | VIP Fixed Prior | seed 0", "Boston | VIP Fixed Prior"),
        ("ftip", ["--ftip_learn_prior"], "Boston | FTIP Tunable Prior | seed 0", "Boston | FTIP Tunable Prior"),
        ("ftip", ["--no-ftip_learn_prior"], "Boston | FTIP Fixed Prior | seed 0", "Boston | FTIP Fixed Prior"),
        ("gmvip", ["--gmvip_learn_prior"], "Boston | GMVIP Tunable Prior | seed 0", "Boston | GMVIP Tunable Prior"),
        ("gmvip", ["--no-gmvip_learn_prior"], "Boston | GMVIP Fixed Prior | seed 0", "Boston | GMVIP Fixed Prior"),
        ("sip", ["--sip_learn_prior"], "Boston | SIP Tunable Prior | seed 0", "Boston | SIP Tunable Prior"),
        ("sip", ["--no-sip_learn_prior"], "Boston | SIP Fixed Prior | seed 0", "Boston | SIP Fixed Prior"),
    ]
    for model, extra, expected_name, expected_group in cases:
        args = _args(["--model", model, "--seed", "0", *extra])
        name, group, _ = _wandb_run_metadata(args, "boston")
        assert name == expected_name
        assert group == expected_group


def test_wandb_stats_are_disabled_by_default_but_configurable():
    default_args = _args(["--model", "map"])
    enabled_args = _args(["--model", "map", "--no-wandb_disable_stats"])

    assert default_args.wandb_disable_stats is True
    assert enabled_args.wandb_disable_stats is False


def test_comparable_ftip_runs_disable_auto_warm_start():
    ftip_variants = [variant for variant in VARIANTS if variant.model == "ftip"]

    assert len(ftip_variants) == 2
    for variant in ftip_variants:
        command = build_command("python", "concrete", 0, variant)
        assert "--no_auto_warm_start" in command
        assert "--auto_warm_start" not in command


def test_comparable_launcher_can_filter_exact_variants():
    selected = select_variants(["SIP Tunable Prior", "SIP Fixed Prior"])

    assert [variant.label for variant in selected] == ["SIP Tunable Prior", "SIP Fixed Prior"]
    assert all(variant.model == "sip" for variant in selected)


def test_variant_tags_distinguish_prior_and_inducing_states():
    ftip_learn = _args(["--model", "ftip", "--ftip_learn_prior"])
    ftip_fixed = _args(["--model", "ftip", "--no-ftip_learn_prior"])
    assert _variant_tag(ftip_learn, "ftip") == "_learnprior"
    assert _variant_tag(ftip_fixed, "ftip") == "_fixedprior"

    sip_learn = _args([
        "--model",
        "sip",
        "--sip_num_inducing",
        "100",
        "--sip_inducing_method",
        "kmeans",
        "--sip_learn_inducing",
        "--sip_learn_prior",
    ])
    sip_fixed = _args([
        "--model",
        "sip",
        "--sip_num_inducing",
        "100",
        "--sip_inducing_method",
        "kmeans",
        "--sip_learn_inducing",
        "--no-sip_learn_prior",
    ])
    assert _variant_tag(sip_learn, "sip") != _variant_tag(sip_fixed, "sip")
    assert "learnZ" in _variant_tag(sip_learn, "sip")
    assert "learnprior" in _variant_tag(sip_learn, "sip")
    assert "fixedprior" in _variant_tag(sip_fixed, "sip")


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

    for prior_flag, should_train in [("--gmvip_learn_prior", True), ("--no-gmvip_learn_prior", False)]:
        model = build_model(parse_args([*gmvip_common, prior_flag]), dataset)
        assert model.Z.requires_grad
        assert any(p.requires_grad for p in model.base_prior.parameters()) is should_train

    for prior_flag, should_train in [("--sip_learn_prior", True), ("--no-sip_learn_prior", False)]:
        model = build_model(parse_args([*sip_common, prior_flag]), dataset)
        assert isinstance(model.Z, torch.nn.Parameter)
        assert any(p.requires_grad for p in model.generative_function.parameters()) is should_train
