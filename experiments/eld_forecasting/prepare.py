from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from implicit_process_zoo.data.downloads import download_source, extract_expected_members
from implicit_process_zoo.data.sources import get_data_source

ELD_SOURCE = get_data_source("eld")
UCI_ELECTRICITY_ZIP_URL = ELD_SOURCE.url


def download_raw(root: Path) -> Path:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_source(ELD_SOURCE, raw_dir / ELD_SOURCE.filename)
    return extract_expected_members(zip_path, raw_dir, ELD_SOURCE.members)[ELD_SOURCE.members[0]]


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
        "source_url": ELD_SOURCE.url,
        "archive_sha256": ELD_SOURCE.sha256,
        "doi": ELD_SOURCE.doi,
        "license": ELD_SOURCE.license,
        "attribution": ELD_SOURCE.attribution,
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
