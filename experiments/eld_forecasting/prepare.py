from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UCI_ELECTRICITY_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
)


def download_raw(root: Path, *, url: str = UCI_ELECTRICITY_ZIP_URL) -> Path:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "electricityloaddiagrams20112014.zip"
    if not zip_path.exists():
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with zip_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.endswith("LD2011_2014.txt")]
        if not members:
            raise FileNotFoundError("Downloaded archive does not contain LD2011_2014.txt.")
        archive.extract(members[0], raw_dir)
        return raw_dir / members[0]


def prepare_raw_file(raw_path: str | Path, root: str | Path) -> Path:
    raw_path = Path(raw_path)
    root = Path(root)
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw ELD file not found: {raw_path}")

    df = pd.read_csv(raw_path, sep=";", decimal=",", index_col=0)
    df = df.dropna(axis=1, how="all")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    values = df.to_numpy(dtype=np.float32, copy=True)
    timestamps_ns = pd.DatetimeIndex(df.index).astype("datetime64[ns]").asi8.astype(np.int64)
    clients = [str(col) for col in df.columns]

    values_path = processed_dir / "values_float32.npy"
    timestamps_path = processed_dir / "timestamps_ns.npy"
    clients_path = processed_dir / "clients.json"
    metadata_path = processed_dir / "metadata.json"
    np.save(values_path, values)
    np.save(timestamps_path, timestamps_ns)
    clients_path.write_text(json.dumps(clients, indent=2), encoding="utf-8")
    metadata = {
        "source_file": str(raw_path),
        "n_timestamps": int(values.shape[0]),
        "n_clients": int(values.shape[1]),
        "dtype": "float32",
        "frequency_minutes": 15,
        "uci_url": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return processed_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UCI ElectricityLoadDiagrams20112014.")
    parser.add_argument("--root", default="data/electricity_load_diagrams")
    parser.add_argument("--raw-path", default=None, help="Path to LD2011_2014.txt.")
    parser.add_argument(
        "--download", action="store_true", help="Download the UCI archive before processing."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    root = Path(args.root)
    raw_path = Path(args.raw_path) if args.raw_path else None
    if args.download:
        raw_path = download_raw(root)
    if raw_path is None:
        candidate = root / "raw" / "LD2011_2014.txt"
        raw_path = candidate
    processed_dir = prepare_raw_file(raw_path, root)
    print(processed_dir)
    return processed_dir


if __name__ == "__main__":
    main()
