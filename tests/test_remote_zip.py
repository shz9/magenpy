import io
import re
import zipfile

import pytest

from magenpy.utils.remote_zip import ReadOnlyZipStore


def _zip_bytes(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return output.getvalue()


def _range_server(data):
    requests = []

    def request(range_header):
        requests.append(range_header)
        suffix = re.fullmatch(r"bytes=-(\d+)", range_header)
        if suffix:
            start = max(0, len(data) - int(suffix.group(1)))
            end = len(data) - 1
        else:
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
            assert match is not None
            start, end = map(int, match.groups())
            end = min(end, len(data) - 1)
        return (
            206,
            {"Content-Range": f"bytes {start}-{end}/{len(data)}"},
            data[start : end + 1],
        )

    return request, requests


def test_remote_zip_store_reads_from_prefix_and_is_read_only():
    archive = _zip_bytes(
        [
            ("chr_1/.zmetadata", b"metadata"),
            ("chr_1/metadata/snps/0", b"rs1\nrs2"),
            ("other/file", b"ignored"),
        ]
    )
    request, _ = _range_server(archive)
    store = ReadOnlyZipStore(
        "https://example.com/chr_1.zip",
        prefix="chr_1",
        tail_size=64,
        request=request,
    )

    assert store[".zmetadata"] == b"metadata"
    assert store["metadata/snps/0"] == b"rs1\nrs2"
    assert set(store) == {".zmetadata", "metadata/snps/0"}
    assert "other/file" not in store

    with pytest.raises(TypeError, match="does not support writes"):
        store["new"] = b"value"


def test_remote_zip_store_coalesces_initialization_entries_at_data_barriers():
    matrix_payload = b"m" * 1_024
    archive = _zip_bytes(
        [
            ("chr_1/.zmetadata", b"z" * 128),
            ("chr_1/metadata/snps/0", b"s" * 128),
            ("chr_1/matrix/data/0", matrix_payload),
            ("chr_1/metadata/a1/0", b"a" * 128),
            ("padding", b"p" * 256),
        ]
    )
    request, requests = _range_server(archive)
    store = ReadOnlyZipStore(
        "https://example.com/chr_1.zip",
        prefix="chr_1",
        tail_size=64,
        request=request,
    )
    requests.clear()

    assert store[".zmetadata"] == b"z" * 128
    requests_after_first_entry = len(requests)
    assert store["metadata/snps/0"] == b"s" * 128
    assert len(requests) == requests_after_first_entry

    assert store["matrix/data/0"] == matrix_payload
    matrix_info = store._zip.getinfo("chr_1/matrix/data/0")
    matrix_data_offset = (
        matrix_info.header_offset
        + 30
        + len(matrix_info.orig_filename.encode("utf-8"))
        + len(matrix_info.extra)
    )
    assert not any(
        offset <= matrix_data_offset
        and offset + len(data) >= matrix_data_offset + len(matrix_payload)
        for offset, data in store.reader.ranges
    )

    requests_before_last_entry = len(requests)
    assert store["metadata/a1/0"] == b"a" * 128
    assert len(requests) > requests_before_last_entry

