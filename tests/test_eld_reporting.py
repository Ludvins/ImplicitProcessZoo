import hashlib
import json

import numpy as np
import pandas as pd

from experiments.eld_forecasting.plot import plot_target_method_grid, write_main_table

METHODS = ("vip", "ftip", "gmvip_empirical")


def _method_dir(root, method, seed=0):
    return root / f"seed_{seed}" / "S_20" / method


def _write_manifest(root, method, seed=0):
    method_dir = _method_dir(root, method, seed)
    method_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "experiment": "electricity_forecasting",
        "method": method,
        "seed": seed,
        "vip_basis_size": 20,
        "evaluation_samples": 1024,
        "checkpoint_selection": "none_final_step_only",
        "status": "complete",
        "data_usage": {"validation": "none"},
        "observation_noise": {"mode": "learned_scalar"},
        "dataset": {"sha256": "test-dataset"},
        "config": {"gmvip": {"num_inducing": 96}},
    }
    (method_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_metric_rows(root, method, seed, values):
    _write_manifest(root, method, seed)
    path = _method_dir(root, method, seed) / "metrics_per_target_region.csv"
    pd.DataFrame(
        [
            {
                "method": method,
                "run_seed": seed,
                "target_id": target_id,
                "region": "test_forecast",
                "rmse": value,
                "nll": value + 1.0,
                "crps": value / 2.0,
                "cqm": value / 3.0,
                "cov80": 0.80,
                "cov90": 0.90,
            }
            for target_id, value in enumerate(values)
        ]
    ).to_csv(path, index=False)


def test_electricity_table_reports_mean_plus_sample_std(tmp_path):
    for method, offset in zip(METHODS, (0.0, 1.0, 2.0)):
        _write_metric_rows(tmp_path, method, 0, np.arange(25, dtype=float) + offset)

    report = write_main_table(
        tmp_path / "table.tex",
        tmp_path / "summary.json",
        root=tmp_path,
        methods=list(METHODS),
        seeds=[0],
        basis_size=20,
    )
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert report["table"].endswith("table.tex")
    assert summary["aggregation"] == "mean_plus_sample_standard_deviation_ddof_1"
    expected = float(np.arange(25, dtype=float).std(ddof=1))
    assert np.isclose(summary["rows"]["vip"]["rmse"]["std"], expected)


def _write_prediction(root, method):
    path = _method_dir(root, method) / "predictions" / "target_18.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(8, dtype=np.float64) * 0.25
    truth = (10.0 + np.arange(8, dtype=np.float64)).reshape(-1, 1)
    offsets = np.linspace(-1.0, 1.0, 1024, dtype=np.float64).reshape(-1, 1, 1)
    samples = truth[None, :, :] + offsets
    np.savez_compressed(
        path,
        target_id=np.asarray(18),
        client_id=np.asarray("MT_353"),
        start_time=np.asarray("2014-09-23 00:00:00"),
        run_seed=np.asarray(0),
        target_seed=np.asarray(18000),
        evaluation_samples=np.asarray(1024),
        forecast_start_hour=np.asarray(1.0),
        t=t,
        truth=truth,
        context_t=t[:4],
        context_y=truth[:4],
        train_t=t[:4],
        train_y=truth[:4],
        samples=samples,
    )


def test_target_18_grid_is_byte_reproducible(tmp_path):
    results = tmp_path / "results"
    for method in METHODS:
        _write_prediction(results, method)

    first = plot_target_method_grid(
        results,
        18,
        tmp_path / "first",
        list(METHODS),
        formats=["png", "pdf"],
    )
    second = plot_target_method_grid(
        results,
        18,
        tmp_path / "second",
        list(METHODS),
        formats=["png", "pdf"],
    )

    assert [hashlib.sha256(path.read_bytes()).digest() for path in first] == [
        hashlib.sha256(path.read_bytes()).digest() for path in second
    ]
