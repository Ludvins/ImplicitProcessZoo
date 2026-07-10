"""Verified, atomic dataset downloads and safe archive extraction."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import requests

from .sources import DataSource


class ChecksumError(ValueError):
    """Raised when downloaded content does not match its declared digest."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_sha256(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    expected = expected.strip().upper()
    if actual != expected:
        raise ChecksumError(
            f"SHA-256 mismatch for {Path(path)}: expected {expected}, got {actual}."
        )


def download_source(
    source: DataSource,
    destination: str | Path,
    *,
    timeout: tuple[float, float] = (10.0, 120.0),
    retries: int = 3,
    session: requests.Session | None = None,
) -> Path:
    """Download a declared HTTPS source atomically and verify its SHA-256."""
    if not source.url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS dataset URL: {source.url}")
    if not source.sha256:
        raise ValueError(f"Data source {source.name!r} does not declare a SHA-256 digest.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_sha256(destination, source.sha256)
        return destination

    client = session or requests.Session()
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    last_error: Exception | None = None
    try:
        for attempt in range(max(1, int(retries))):
            try:
                with client.get(source.url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                verify_sha256(temporary, source.sha256)
                os.replace(temporary, destination)
                return destination
            except (requests.RequestException, OSError, ChecksumError) as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt + 1 < max(1, int(retries)):
                    time.sleep(0.25 * (2**attempt))
        raise RuntimeError(
            f"Could not download verified data source {source.name!r} after {retries} attempts."
        ) from last_error
    finally:
        temporary.unlink(missing_ok=True)
        if session is None:
            client.close()


def _validate_archive_name(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    if path.parts[0].endswith(":"):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path


def extract_expected_members(
    archive_path: str | Path,
    destination: str | Path,
    expected_members: tuple[str, ...],
) -> dict[str, Path]:
    """Extract only declared ZIP members after rejecting traversal paths."""
    archive_path = Path(archive_path)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    with zipfile.ZipFile(archive_path) as archive:
        validated = {name: _validate_archive_name(name) for name in archive.namelist()}
        for expected in expected_members:
            expected_path = _validate_archive_name(expected)
            matches = [
                name
                for name, member_path in validated.items()
                if member_path == expected_path or member_path.name == expected_path.name
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one archive member matching {expected!r}, found {matches}."
                )
            member_name = matches[0]
            target = (destination / expected_path).resolve()
            if destination not in target.parents:
                raise ValueError(f"Archive member escapes destination: {member_name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
            try:
                with archive.open(member_name) as source_handle, temporary.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            outputs[expected] = target
    return outputs


def fetch_and_extract(source: DataSource, root: str | Path) -> dict[str, Path]:
    if not source.members:
        raise ValueError(f"Data source {source.name!r} does not declare archive members.")
    root = Path(root)
    archive = download_source(source, root / "raw" / source.filename)
    return extract_expected_members(archive, root, source.members)
