"""Large scalar-regression benchmark.

This entrypoint reuses the UCI regression training code for the larger
Variational-LLA-style regression datasets: Year, Airline, and Taxi.

Example:
    python -m experiments.regression.benchmark --model gmvip --dataset year
"""

from pathlib import Path
import copy
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.uci.benchmark import parse_args as parse_uci_args
from experiments.uci.benchmark import run_from_args as run_uci_from_args


REGRESSION_DATASETS = ["year", "airline", "taxi"]

DEFAULT_REGRESSION_ITERS = {
    "year": 60_000,
    "airline": 60_000,
    "taxi": 120_000,
}

DEFAULT_REGRESSION_HIDDEN_DIMS = {
    "year": [50, 50],
    "airline": [100, 100],
    "taxi": [100, 100],
}


def parse_args(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_uci_args(
        argv,
        description="Large regression benchmark",
        dataset_names=REGRESSION_DATASETS,
        dataset_group_label="large regression datasets",
        default_output_dir="results/regression",
    )
    args._hidden_dims_user_supplied = "--hidden_dims" in raw_argv
    if not args._hidden_dims_user_supplied and args.dataset != "all":
        args.hidden_dims = list(DEFAULT_REGRESSION_HIDDEN_DIMS[args.dataset])
    return args


def run_from_args(
    args,
    *,
    dataset_names=None,
    default_iters=None,
    default_hidden_dims=None,
):
    dataset_names = list(REGRESSION_DATASETS if dataset_names is None else dataset_names)
    default_iters = DEFAULT_REGRESSION_ITERS if default_iters is None else default_iters
    default_hidden_dims = (
        DEFAULT_REGRESSION_HIDDEN_DIMS
        if default_hidden_dims is None
        else default_hidden_dims
    )

    if getattr(args, "_hidden_dims_user_supplied", False):
        return run_uci_from_args(
            args,
            dataset_names=dataset_names,
            default_iters=default_iters,
        )

    datasets = dataset_names if args.dataset == "all" else [args.dataset]
    all_results = []
    for dataset_name in datasets:
        run_args = copy.copy(args)
        run_args.dataset = dataset_name
        run_args.hidden_dims = list(default_hidden_dims[dataset_name])
        all_results.extend(
            run_uci_from_args(
                run_args,
                dataset_names=dataset_names,
                default_iters=default_iters,
            )
        )
    return all_results


def main(argv=None):
    args = parse_args(argv)
    return run_from_args(
        args,
        dataset_names=REGRESSION_DATASETS,
        default_iters=DEFAULT_REGRESSION_ITERS,
    )


if __name__ == "__main__":
    main()
