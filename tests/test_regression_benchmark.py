from io import BytesIO

import pytest

from experiments.regression import benchmark as regression_benchmark
from implicit_process_zoo.utils import dataset as dataset_module


def test_large_regression_parser_defaults():
    args = regression_benchmark.parse_args(
        [
            "--model",
            "map",
            "--dataset",
            "year",
            "--iterations",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert args.dataset == "year"
    assert args.output_dir == "results/regression"
    assert args.bb_alpha == 0.0
    assert args.hidden_dims == [50, 50]


def test_large_regression_parser_restricts_datasets():
    with pytest.raises(SystemExit):
        regression_benchmark.parse_args(
            [
                "--model",
                "map",
                "--dataset",
                "boston",
                "--iterations",
                "1",
            ]
        )


def test_large_regression_all_dataset_order(monkeypatch):
    calls = []

    def fake_run_from_args(args, *, dataset_names, default_iters):
        calls.append((args, dataset_names, default_iters))
        return ["ok"]

    monkeypatch.setattr(regression_benchmark, "run_from_args", fake_run_from_args)

    result = regression_benchmark.main(
        [
            "--model",
            "map",
            "--dataset",
            "all",
            "--iterations",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert result == ["ok"]
    assert calls[0][1] == ["year", "airline", "taxi"]
    assert calls[0][2]["year"] == 60_000
    assert calls[0][2]["airline"] == 60_000
    assert calls[0][2]["taxi"] == 120_000


def test_large_regression_parser_preserves_explicit_hidden_dims():
    args = regression_benchmark.parse_args(
        [
            "--model",
            "map",
            "--dataset",
            "year",
            "--iterations",
            "1",
            "--hidden_dims",
            "10",
            "10",
            "--device",
            "cpu",
        ]
    )

    assert args.hidden_dims == [10, 10]
    assert args._hidden_dims_user_supplied is True


def test_large_regression_run_applies_per_dataset_hidden_defaults(monkeypatch):
    seen = []

    def fake_uci_run_from_args(args, *, dataset_names, default_iters):
        seen.append((args.dataset, list(args.hidden_dims), default_iters[args.dataset]))
        return [args.dataset]

    monkeypatch.setattr(
        regression_benchmark,
        "run_uci_from_args",
        fake_uci_run_from_args,
    )

    args = regression_benchmark.parse_args(
        [
            "--model",
            "map",
            "--dataset",
            "all",
            "--device",
            "cpu",
        ]
    )
    result = regression_benchmark.run_from_args(args)

    assert result == ["year", "airline", "taxi"]
    assert seen == [
        ("year", [50, 50], 60_000),
        ("airline", [100, 100], 60_000),
        ("taxi", [100, 100], 120_000),
    ]


def test_airline_dataset_downloads_when_missing(tmp_path, monkeypatch):
    payload = (
        b"ArrDelay,ArrTime,DepTime,Month,DayofMonth,DayOfWeek,plane_age,AirTime,Distance\n"
        b"1,1230,1015,1,1,1,5,120,500\n"
        b"3,1305,1110,1,2,2,6,130,600\n"
    )
    requested_urls = []

    class FakeResponse:
        def __init__(self, data):
            self._stream = BytesIO(data)

        def read(self, size=-1):
            return self._stream.read(size)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(url):
        requested_urls.append(url)
        return FakeResponse(payload)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset_module, "urlopen", fake_urlopen)

    dataset = dataset_module.Airline_Dataset()

    cached_path = tmp_path / "data" / "airline.csv"
    assert requested_urls == [dataset_module.airline_csv_url]
    assert cached_path.read_bytes() == payload
    assert not (tmp_path / "data" / "airline.csv.tmp").exists()
    assert len(dataset) == 2
    assert dataset.train.input_dim == 8
    assert dataset.output_dim == 1
