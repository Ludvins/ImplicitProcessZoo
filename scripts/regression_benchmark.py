"""Large scalar-regression benchmark.

This entrypoint reuses the UCI regression training code for the larger
Variational-LLA-style regression datasets: Year, Airline, and Taxi.

Example:
    python -m scripts.regression_benchmark --model gmvip --dataset year
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.uci_benchmark import parse_args as parse_uci_args
from scripts.uci_benchmark import run_from_args


REGRESSION_DATASETS = ["year", "airline", "taxi"]

DEFAULT_REGRESSION_ITERS = {
    "year": 30_000,
    "airline": 30_000,
    "taxi": 30_000,
}


def parse_args(argv=None):
    return parse_uci_args(
        argv,
        description="Large regression benchmark",
        dataset_names=REGRESSION_DATASETS,
        dataset_group_label="large regression datasets",
        default_output_dir="results/regression",
    )


def main(argv=None):
    args = parse_args(argv)
    return run_from_args(
        args,
        dataset_names=REGRESSION_DATASETS,
        default_iters=DEFAULT_REGRESSION_ITERS,
    )


if __name__ == "__main__":
    main()
