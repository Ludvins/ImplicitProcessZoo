import pytest

from scripts import regression_benchmark


def test_large_regression_parser_defaults():
    args = regression_benchmark.parse_args([
        "--model",
        "map",
        "--dataset",
        "year",
        "--iterations",
        "1",
        "--device",
        "cpu",
    ])

    assert args.dataset == "year"
    assert args.output_dir == "results/regression"
    assert args.bb_alpha == 0.0


def test_large_regression_parser_restricts_datasets():
    with pytest.raises(SystemExit):
        regression_benchmark.parse_args([
            "--model",
            "map",
            "--dataset",
            "boston",
            "--iterations",
            "1",
        ])


def test_large_regression_all_dataset_order(monkeypatch):
    calls = []

    def fake_run_from_args(args, *, dataset_names, default_iters):
        calls.append((args, dataset_names, default_iters))
        return ["ok"]

    monkeypatch.setattr(regression_benchmark, "run_from_args", fake_run_from_args)

    result = regression_benchmark.main([
        "--model",
        "map",
        "--dataset",
        "all",
        "--iterations",
        "1",
        "--device",
        "cpu",
    ])

    assert result == ["ok"]
    assert calls[0][1] == ["year", "airline", "taxi"]
    assert calls[0][2]["year"] == 30_000
