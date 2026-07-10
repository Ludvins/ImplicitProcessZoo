"""Verify the benchmark entry point and show a reproducible command."""

from __future__ import annotations

import subprocess
import sys

command = [
    sys.executable,
    "-m",
    "experiments.uci.benchmark",
    "--model",
    "gmvip",
    "--dataset",
    "concrete",
    "--iterations",
    "30000",
    "--seed",
    "0",
]

if __name__ == "__main__":
    # CI checks the entry point without downloading a benchmark dataset.
    subprocess.run(
        [sys.executable, "-m", "experiments.uci.benchmark", "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(" ".join(command))
