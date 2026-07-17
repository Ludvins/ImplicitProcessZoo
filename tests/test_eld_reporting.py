import hashlib

import numpy as np
import pandas as pd

from experiments.eld_forecasting.compare import aggregate
from experiments.eld_forecasting.plot_predictions import plot_target_method_grid


def _write_metric_rows(root, method, run_seed, values):
    path = root / method / f"seed_{run_seed}" / "metrics_per_target_region.csv"
    path.parent.mkdir(parents=True)
    (path.parent / "config.yaml").write_text("experiment: eld_forecasting\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "method": method,
                "run_seed": run_seed,
                "target_id": target_id,
                "client_id": f"client_{target_id}",
                "start_time": f"time_{target_id}",
                "region": "test_forecast",
                "rmse": value,
                "crps": value / 2.0,
            }
            for target_id, value in enumerate(values)
        ]
    ).to_csv(path, index=False)


def test_eld_aggregate_uses_original_median_iqr_summary(tmp_path):
    _write_metric_rows(tmp_path, "vip", 0, [1.0, 3.0])
    _write_metric_rows(tmp_path, "vip", 1, [5.0, 7.0])

    summary = aggregate(tmp_path)
    overall = summary[
        (summary["run_seed"].astype(str) == "all") & (summary["method"] == "vip")
    ].iloc[0]

    assert overall["n"] == 4
    assert overall["rmse_median"] == 4.0
    assert overall["rmse_q25"] == 2.5
    assert overall["rmse_q75"] == 5.5
    assert overall["rmse_mean"] == 4.0


def test_eld_aggregate_backfills_80_percent_intervals_from_predictions(tmp_path):
    metric_path = tmp_path / "vip" / "seed_0" / "metrics_per_target_region.csv"
    metric_path.parent.mkdir(parents=True)
    (metric_path.parent / "config.yaml").write_text(
        "experiment: eld_forecasting\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "method": "vip",
                "run_seed": 0,
                "target_id": 18,
                "client_id": "MT_353",
                "start_time": "2014-09-23 00:00:00",
                "region": "test_forecast",
                "region_start_idx": 4,
                "region_stop_idx": 8,
                "rmse": 0.0,
            }
        ]
    ).to_csv(metric_path, index=False)
    _write_prediction(tmp_path, "vip")

    summary = aggregate(tmp_path)
    overall = summary.iloc[0]

    assert overall["cov80_median"] == 1.0
    assert np.isclose(overall["width80_median"], 1.6)


def _write_prediction(root, method):
    path = root / method / "seed_0" / "predictions" / "target_18.npz"
    path.parent.mkdir(parents=True)
    t = np.arange(8, dtype=np.float64) * 0.25
    truth = (10.0 + np.arange(8, dtype=np.float64)).reshape(-1, 1)
    offsets = np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(-1, 1, 1)
    samples = truth[None, :, :] + offsets
    np.savez_compressed(
        path,
        target_id=np.asarray(18),
        client_id=np.asarray("MT_353"),
        start_time=np.asarray("2014-09-23 00:00:00"),
        run_seed=np.asarray(0),
        target_seed=np.asarray(18000),
        forecast_start_hour=np.asarray(1.0),
        t=t,
        truth=truth,
        context_t=t[:4],
        context_y=truth[:4],
        train_t=t[:4],
        train_y=truth[:4],
        samples=samples,
    )


def test_target_18_paper_grid_is_byte_reproducible(tmp_path):
    results = tmp_path / "results"
    for method in ("vip", "ftip", "gmvip_empirical"):
        _write_prediction(results, method)

    first = plot_target_method_grid(
        results,
        18,
        tmp_path / "first",
        ["vip", "ftip", "gmvip_empirical"],
        formats=["png", "pdf"],
    )
    second = plot_target_method_grid(
        results,
        18,
        tmp_path / "second",
        ["vip", "ftip", "gmvip_empirical"],
        formats=["png", "pdf"],
    )

    assert [hashlib.sha256(path.read_bytes()).digest() for path in first] == [
        hashlib.sha256(path.read_bytes()).digest() for path in second
    ]
