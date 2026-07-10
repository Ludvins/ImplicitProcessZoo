import json

import numpy as np
import pytest

from experiments.common import write_csv_rows, write_json
from experiments.eld_forecasting.merge_shards import merge_method_shards


def _write_shard(root, target_id, *, config="methodology_version: 2\n"):
    method_dir = root / "vip" / "seed_0"
    method_dir.mkdir(parents=True)
    (method_dir / "config.yaml").write_text(config, encoding="utf-8")
    write_csv_rows(
        method_dir / "metrics_per_target_region.csv",
        [
            {
                "methodology_version": 2,
                "method": "vip",
                "target_id": target_id,
                "region": "full_forecast",
                "rmse": float(target_id + 1),
            }
        ],
    )
    write_json(method_dir / "runtime.json", [{"target_id": target_id, "steps": 2}])
    prediction_dir = method_dir / "predictions"
    prediction_dir.mkdir()
    np.savez_compressed(prediction_dir / f"target_{target_id}.npz", target_id=target_id)


def test_merge_method_shards_rebuilds_summary_and_artifacts(tmp_path):
    shard_a = tmp_path / "a"
    shard_b = tmp_path / "b"
    _write_shard(shard_a, 0)
    _write_shard(shard_b, 1)

    destination = merge_method_shards([shard_a, shard_b], tmp_path / "merged", method="vip", seed=0)

    metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["methodology_version"] == 2
    assert metrics["targets"] == [0, 1]
    assert metrics["summary"]["vip|full_forecast"]["rmse"]["mean"] == pytest.approx(1.5)
    assert (destination / "predictions" / "target_0.npz").is_file()
    assert (destination / "predictions" / "target_1.npz").is_file()


def test_merge_method_shards_rejects_duplicate_targets(tmp_path):
    shard_a = tmp_path / "a"
    shard_b = tmp_path / "b"
    _write_shard(shard_a, 0)
    _write_shard(shard_b, 0)

    with pytest.raises(ValueError, match="more than one shard"):
        merge_method_shards([shard_a, shard_b], tmp_path / "merged", method="vip")
