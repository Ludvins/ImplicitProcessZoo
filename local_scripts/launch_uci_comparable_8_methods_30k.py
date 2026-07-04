"""Launch the comparable 8-method UCI sweep in the background."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
LOG_DIR = REPO / "outputs" / "wandb_logs"
OUT_LOG = LOG_DIR / "uci_comparable_8_methods_30k_sweep.out.log"
ERR_LOG = LOG_DIR / "uci_comparable_8_methods_30k_sweep.err.log"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["APFSVI_DISABLE_TQDM"] = "1"

    cmd = [
        str(PYTHON),
        "local_scripts/run_uci_comparable_8_methods_30k.py",
        *sys.argv[1:],
    ]
    with OUT_LOG.open("wb") as stdout, ERR_LOG.open("wb") as stderr:
        process = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    time.sleep(5)
    print(f"pid={process.pid}")
    print(f"returncode={process.poll()}")
    print(f"stdout={OUT_LOG}")
    print(f"stderr={ERR_LOG}")


if __name__ == "__main__":
    main()
