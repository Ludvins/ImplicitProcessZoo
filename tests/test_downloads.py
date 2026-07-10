"""Security and reproducibility tests for dataset downloads."""

from __future__ import annotations

import hashlib
import zipfile

import pytest

from implicit_process_zoo.data.downloads import (
    ChecksumError,
    download_source,
    extract_expected_members,
)
from implicit_process_zoo.data.sources import DataSource


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class _Session:
    def __init__(self, content: bytes):
        self.content = content
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        return _Response(self.content)


def _source(content: bytes, *, sha256: str | None = None) -> DataSource:
    return DataSource(
        name="test",
        url="https://example.invalid/data.bin",
        sha256=sha256 or hashlib.sha256(content).hexdigest(),
        filename="data.bin",
    )


def test_download_is_atomic_and_checksum_verified(tmp_path):
    content = b"verified dataset bytes"
    session = _Session(content)
    destination = tmp_path / "nested" / "data.bin"
    result = download_source(_source(content), destination, session=session)
    assert result.read_bytes() == content
    assert session.calls == 1
    assert list(destination.parent.glob("*.part")) == []


def test_checksum_failure_does_not_publish_partial_file(tmp_path):
    content = b"corrupted"
    destination = tmp_path / "data.bin"
    with pytest.raises(RuntimeError) as error:
        download_source(
            _source(content, sha256="0" * 64),
            destination,
            retries=1,
            session=_Session(content),
        )
    assert isinstance(error.value.__cause__, ChecksumError)
    assert not destination.exists()


def test_existing_bad_checksum_fails_before_network(tmp_path):
    destination = tmp_path / "data.bin"
    destination.write_bytes(b"wrong")
    with pytest.raises(ChecksumError):
        download_source(_source(b"right"), destination, session=_Session(b"right"))


def test_archive_traversal_is_rejected(tmp_path):
    archive_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe/data.txt", "safe")
        archive.writestr("../escaped.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe archive member"):
        extract_expected_members(archive_path, tmp_path / "out", ("safe/data.txt",))
    assert not (tmp_path / "escaped.txt").exists()


def test_only_expected_archive_members_are_extracted(tmp_path):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/data.txt", "wanted")
        archive.writestr("nested/extra.txt", "ignored")
    outputs = extract_expected_members(archive_path, tmp_path / "out", ("nested/data.txt",))
    assert outputs["nested/data.txt"].read_text() == "wanted"
    assert not (tmp_path / "out" / "nested" / "extra.txt").exists()
