import json
import math

import numpy as np
import pytest
import torch

from experiments.synthetic import hmc
from experiments.synthetic.hmc import (
    HMC_PARAMETER_COUNT,
    NOISE_SITE_NAME,
    WEIGHT_SITE_SHAPES,
    HMCConfig,
    _flatten_params,
    _split_rhat_and_ess,
    _unflatten_params,
    bnn_forward,
    hmc_log_joint,
    posterior_predictive,
    run_hmc_reference,
    save_hmc_artifacts,
    select_draw_indices,
)
from experiments.synthetic.plot import (
    DEFAULT_MODELS,
    missing_model_reason,
    parse_synthetic_args,
    resolve_models,
)
from implicit_process_zoo.utils import dataset as dataset_module


def _zero_samples(chains=2, draws=6):
    samples = {
        name: torch.zeros((chains, draws, *shape), dtype=torch.float64)
        for name, shape in WEIGHT_SITE_SHAPES.items()
    }
    samples[NOISE_SITE_NAME] = torch.full(
        (chains, draws),
        math.log(0.2),
        dtype=torch.float64,
    )
    return samples


def test_synthetic_dataset_uses_canonical_name_and_path(monkeypatch):
    loaded = {}
    synthetic_data = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 4, dtype=np.float64),
            np.linspace(0.0, 0.3, 4, dtype=np.float64),
        ]
    )

    monkeypatch.setattr(dataset_module.os.path, "exists", lambda _path: True)

    def fake_load(path):
        loaded["path"] = str(path).replace("\\", "/")
        return synthetic_data

    monkeypatch.setattr(dataset_module.np, "load", fake_load)

    dataset = dataset_module.get_dataset("synthetic")
    assert isinstance(dataset, dataset_module.Synthetic_Dataset)
    assert loaded["path"].endswith("/data/synthetic/data.npy")
    assert not hasattr(dataset_module, "LegacySynthetic_Dataset")
    for removed_name in ("legacy_synthetic", "variational_lla", "valla"):
        with pytest.raises(KeyError):
            dataset_module.get_dataset(removed_name)


def test_hmc_is_explicit_only_in_synthetic_model_resolution():
    assert resolve_models(["all"]) == DEFAULT_MODELS
    assert resolve_models(["all", "hmc"]) == [*DEFAULT_MODELS, "hmc"]
    assert resolve_models(["gmvip", "hmc", "gmvip"]) == ["gmvip", "hmc"]

    synthetic_args, base_args = parse_synthetic_args(["--models", "hmc"])
    assert base_args.dataset == "synthetic"
    assert synthetic_args.hmc_chains == 1
    assert synthetic_args.hmc_warmup_steps == 0
    assert synthetic_args.hmc_num_samples == 1000
    assert synthetic_args.hmc_num_predictive_samples == 1000
    assert synthetic_args.hmc_step_size == 0.0005
    assert synthetic_args.hmc_num_steps == 500
    assert synthetic_args.hmc_inverse_mass == 0.1
    assert synthetic_args.hmc_map_warmstart_steps == 1000
    assert synthetic_args.hmc_map_warmstart_lr == 0.003
    assert synthetic_args.hmc_initialization_jitter == 0.01
    assert synthetic_args.hmc_device == "cuda"
    assert synthetic_args.figure_name == "synthetic_predictive_distributions.png"
    assert synthetic_args.pdf_name == "synthetic_predictive_distributions.pdf"
    assert synthetic_args.results_name == "synthetic_predictive_distributions.json"


def test_hmc_cli_rejects_impossible_predictive_draw_count():
    with pytest.raises(SystemExit):
        parse_synthetic_args(
            [
                "--models",
                "hmc",
                "--hmc_chains",
                "2",
                "--hmc_num_samples",
                "4",
                "--hmc_num_predictive_samples",
                "9",
            ]
        )


def test_missing_hamiltorch_message_is_actionable(monkeypatch):
    monkeypatch.setattr(hmc, "hamiltorch_available", lambda: False)
    reason = missing_model_reason("hmc")
    assert "optional pinned Hamiltorch dependency" in reason
    assert ".[experiments,hmc]" in reason
    assert missing_model_reason("gmvip") is None


def test_bnn_forward_matches_direct_tanh_network():
    assert HMC_PARAMETER_COUNT == 141
    params = {
        "w1": torch.arange(10, dtype=torch.float64).reshape(1, 10) / 10,
        "b1": torch.linspace(-0.2, 0.2, 10, dtype=torch.float64),
        "w2": torch.eye(10, dtype=torch.float64) * 0.5,
        "b2": torch.linspace(0.1, -0.1, 10, dtype=torch.float64),
        "w3": torch.linspace(-0.5, 0.5, 10, dtype=torch.float64).reshape(10, 1),
        "b3": torch.tensor([0.3], dtype=torch.float64),
    }
    x = torch.tensor([[-1.0], [0.5]], dtype=torch.float64)
    expected = torch.tanh(torch.tanh(x @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"])
    expected = expected @ params["w3"] + params["b3"]
    torch.testing.assert_close(bnn_forward(params, x), expected)


def test_hmc_log_joint_matches_explicit_gaussian_terms():
    params = {
        name: torch.full(shape, 0.1, dtype=torch.float64)
        for name, shape in WEIGHT_SITE_SHAPES.items()
    }
    params[NOISE_SITE_NAME] = torch.tensor(-1.7, dtype=torch.float64)
    x = torch.tensor([[-0.5], [0.25], [0.75]], dtype=torch.float64)
    y = torch.tensor([[-0.1], [0.2], [0.4]], dtype=torch.float64)

    standard = torch.distributions.Normal(0.0, 1.0)
    expected = sum(standard.log_prob(params[name]).sum() for name in WEIGHT_SITE_SHAPES)
    expected += torch.distributions.Normal(-2.5, 1.0).log_prob(params[NOISE_SITE_NAME])
    expected += (
        torch.distributions.Normal(
            bnn_forward(params, x),
            torch.exp(params[NOISE_SITE_NAME]),
        )
        .log_prob(y)
        .sum()
    )
    torch.testing.assert_close(hmc_log_joint(params, x, y), expected)


def test_hamiltorch_flat_parameter_mapping_round_trips():
    params = {
        name: torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
        for name, shape in WEIGHT_SITE_SHAPES.items()
    }
    params[NOISE_SITE_NAME] = torch.tensor(-2.0, dtype=torch.float64)
    restored = _unflatten_params(_flatten_params(params))
    for name, expected in params.items():
        torch.testing.assert_close(restored[name], expected)


def test_torch_diagnostics_recognize_well_mixed_chains():
    generator = torch.Generator().manual_seed(17)
    values = torch.randn((4, 1000, 3), generator=generator, dtype=torch.float64)
    rhat, ess = _split_rhat_and_ess(values)
    assert torch.max(torch.abs(rhat - 1.0)) < 0.02
    assert torch.min(ess) > 800


def test_posterior_predictive_matches_gmvip_component_convention():
    samples = _zero_samples()
    offsets = torch.arange(12, dtype=torch.float64).reshape(2, 6) / 10
    samples["b3"][..., 0] = offsets
    x_grid = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)

    predictive, selected = posterior_predictive(
        samples,
        x_grid,
        y_mean=2.0,
        y_std=3.0,
        num_predictive_samples=6,
    )
    selected_offsets = offsets[selected[:, 0], selected[:, 1]].numpy()
    expected_means = selected_offsets[:, None] * 3.0 + 2.0
    expected_means = np.broadcast_to(expected_means, (6, 3))
    expected_stds = np.full((6, 3), 0.6)
    expected_mixture_mean = expected_means.mean(axis=0)
    expected_mixture_std = np.sqrt(
        np.mean(expected_stds**2 + expected_means**2, axis=0) - expected_mixture_mean**2
    )

    np.testing.assert_allclose(predictive.means, expected_means)
    np.testing.assert_allclose(predictive.stds, expected_stds)
    np.testing.assert_allclose(predictive.mixture_mean, expected_mixture_mean)
    np.testing.assert_allclose(predictive.mixture_std, expected_mixture_std)


def test_draw_selection_is_deterministic_and_chain_balanced():
    first = select_draw_indices(4, 1000, 1024)
    second = select_draw_indices(4, 1000, 1024)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1024, 2)
    counts = np.bincount(first[:, 0], minlength=4)
    assert counts.max() - counts.min() <= 1
    assert len(np.unique(first, axis=0)) == 1024


def test_hmc_artifact_contains_raw_and_predictive_arrays(tmp_path):
    samples = _zero_samples(draws=4)
    x_grid = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    predictive, selected = posterior_predictive(
        samples,
        x_grid,
        y_mean=0.0,
        y_std=1.0,
        num_predictive_samples=4,
    )
    result = {"model": "hmc", "inference": {"chains": 2}, "diagnostics": {}}
    artifacts = save_hmc_artifacts(
        tmp_path,
        samples=samples,
        selected_indices=selected,
        x_grid=x_grid.numpy(),
        predictive=predictive,
        result=result,
    )

    arrays = np.load(artifacts["posterior_predictive"])
    assert arrays["posterior_w2"].shape == (2, 4, 10, 10)
    assert arrays["posterior_log_sigma_y"].shape == (2, 4)
    assert arrays["predictive_component_means"].shape == (4, 3)
    assert arrays["predictive_component_stds"].shape == (4, 3)
    assert arrays["selected_chain_draw_indices"].shape == (4, 2)
    assert json.loads(arrays["configuration_json"].item()) == {"chains": 2}
    assert json.loads((tmp_path / "hmc_summary.json").read_text())["model"] == "hmc"


def test_hmc_config_validates_diagnostic_requirements():
    with pytest.raises(ValueError, match="at least one chain"):
        HMCConfig(chains=0).validate()
    with pytest.raises(ValueError, match="at least one retained"):
        HMCConfig(num_samples=0).validate()
    with pytest.raises(ValueError, match="non-negative"):
        HMCConfig(warmup_steps=-1).validate()
    with pytest.raises(ValueError, match="noise-prior log scale"):
        HMCConfig(noise_log_scale=0.0).validate()
    with pytest.raises(ValueError, match="step size"):
        HMCConfig(step_size=0.0).validate()
    with pytest.raises(ValueError, match="leapfrog"):
        HMCConfig(num_steps=0).validate()
    with pytest.raises(ValueError, match="inverse mass"):
        HMCConfig(inverse_mass=0.0).validate()
    with pytest.raises(ValueError, match="MAP warm-start"):
        HMCConfig(map_warmstart_steps=-1).validate()


def test_hamiltorch_small_integration_smoke(tmp_path):
    pytest.importorskip("hamiltorch")
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float64)[:, None]
    y = (0.2 * np.sin(x)).astype(np.float64)
    grid = np.linspace(-1.2, 1.2, 5, dtype=np.float64)[:, None]
    config = HMCConfig(
        chains=2,
        warmup_steps=4,
        num_samples=4,
        num_predictive_samples=4,
        step_size=1e-4,
        num_steps=2,
        device="cpu",
        map_warmstart_steps=0,
        seed=7,
        diagnostic_grid_points=3,
        disable_progress=True,
    )

    result, predictive = run_hmc_reference(
        train_x=x,
        train_y=y,
        x_grid=grid,
        y_mean=1.5,
        y_std=2.0,
        output_dir=tmp_path,
        config=config,
    )
    assert result["model"] == "hmc"
    assert result["inference"]["chain_execution"] == "sequential"
    assert result["inference"]["sampler"] == "hamiltorch_hmc"
    assert result["inference"]["burn"] == -1
    assert result["inference"]["joint_parameter_count"] == 142
    assert predictive.means.shape == (4, 5)
    assert predictive.stds.shape == (4, 5)
    assert (tmp_path / "hmc_posterior_samples.npz").is_file()
    assert (tmp_path / "hmc_summary.json").is_file()


def test_hamiltorch_notebook_single_chain_saves_uncertified_artifacts(tmp_path):
    pytest.importorskip("hamiltorch")
    x = np.linspace(-1.0, 1.0, 6, dtype=np.float64)[:, None]
    y = np.zeros_like(x)
    output_dir = tmp_path / "failed"
    config = HMCConfig(
        chains=1,
        warmup_steps=0,
        num_samples=4,
        num_predictive_samples=4,
        step_size=1e-4,
        num_steps=2,
        device="cpu",
        map_warmstart_steps=0,
        seed=11,
        diagnostic_grid_points=3,
        disable_progress=True,
    )

    result, predictive = run_hmc_reference(
        train_x=x,
        train_y=y,
        x_grid=x,
        y_mean=0.0,
        y_std=1.0,
        output_dir=output_dir,
        config=config,
    )

    assert result["converged"] is None
    assert result["diagnostics"]["assessed"] is False
    assert predictive.means.shape == (4, 6)
    assert (output_dir / "hmc_posterior_samples.npz").is_file()
    assert (output_dir / "hmc_summary.json").is_file()
