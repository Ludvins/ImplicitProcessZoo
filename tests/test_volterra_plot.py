import csv
import json

import numpy as np

from experiments.volterra.plot import METRICS, main


def _write_result(
    root,
    method,
    basis_size,
    *,
    with_prediction,
    nll_definition=None,
):
    if nll_definition is None:
        learned_methods = {
            "map",
            "mfvi",
            "vip",
            "ftip",
            "sip",
            "gmvip_empirical",
            "gmvip_rbf",
        }
        mode = "learned" if method in learned_methods else "fixed"
        nll_definition = (
            f"equal_weight_gaussian_mixture_with_{mode}_observation_variance"
        )
    method_dir = root / "seed_0" / f"S_{basis_size}" / method
    method_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "experiment": "lotka_volterra",
        "method": method,
        "seed": 0,
        "vip_basis_size": basis_size,
        "nll": nll_definition,
        "checkpoint_selection": "none_final_step_only",
        "data_usage": {
            "training": "t<=15",
            "unused_gap": "15<t<=20",
            "test": "20<t<=30",
        },
        "status": "complete",
        "protocol_hash": f"{method}-{basis_size}",
        "dataset": {"sha256": "same-dataset"},
    }
    (method_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    row = {
        "experiment": "lotka_volterra",
        "method": method,
        "metric_partition": "test_(20,30]",
        "target_id": 9,
        **{metric: 0.1 + 0.01 * basis_size for metric, _ in METRICS},
    }
    if "learned" in nll_definition:
        row["observation_noise_std_prey"] = 0.1
        row["observation_noise_std_predator"] = 0.2
    with (method_dir / "metrics_per_target.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    if with_prediction:
        prediction_dir = method_dir / "predictions"
        prediction_dir.mkdir()
        t = np.linspace(0.0, 30.0, 7)
        truth = np.stack((1.0 + 0.1 * t, 2.0 + 0.05 * t), axis=-1)
        samples = np.broadcast_to(truth, (1024, *truth.shape)).copy()
        np.savez_compressed(
            prediction_dir / "target_9.npz",
            t_plot=t,
            y_true=truth,
            y_train_x=t[:4],
            y_train=truth[:4],
            samples=samples,
            mean=samples.mean(axis=0),
            std=samples.std(axis=0),
            evaluation_seed=np.asarray(501),
            evaluation_samples=np.asarray(1024),
        )


def test_plot_generates_validated_figure_and_tables(tmp_path):
    result_root = tmp_path / "results"
    methods = (
        "analog_prior",
        "gmvip_surrogate_prior",
        "vip",
        "ftip",
        "gmvip_empirical",
    )
    for method in methods:
        _write_result(result_root, method, 20, with_prediction=True)
    for basis_size in (64, 128, 256):
        for method in ("vip", "ftip"):
            _write_result(result_root, method, basis_size, with_prediction=False)
    fixed_root = tmp_path / "fixed_results"
    for method in methods:
        _write_result(
            fixed_root,
            method,
            20,
            with_prediction=False,
            nll_definition=(
                "equal_weight_gaussian_mixture_with_fixed_observation_variance"
            ),
        )

    out_dir = tmp_path / "artifacts"
    result = main(
        [
            "--results-root",
            str(result_root),
            "--methods",
            ",".join(methods),
            "--target-ids",
            "9",
            "--aggregate-target-ids",
            "9",
            "--fixed-noise-results-root",
            str(fixed_root),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert (out_dir / "volterra_target_9.png").exists()
    assert (out_dir / "volterra_target_9.pdf").exists()
    assert (out_dir / "volterra_main_table.tex").exists()
    assert (out_dir / "volterra_basis_table.tex").exists()
    assert (out_dir / "volterra_noise_comparison_table.tex").exists()
    assert (out_dir / "volterra_noise_comparison.json").exists()
    assert "Period error" not in (out_dir / "volterra_main_table.tex").read_text(
        encoding="utf-8"
    )
    assert "oscillation_period_error" not in (
        out_dir / "volterra_noise_comparison.json"
    ).read_text(encoding="utf-8")
    assert result["figures"][0]["target_id"] == 9
    assert result["aggregate_target_ids"] == (9,)
