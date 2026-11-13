"""Round-trip actual --zip CLI output through local and remote LD readers."""

from functools import partial
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
import zarr

import magenpy as mgp
from magenpy.utils import remote_zip

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def zipped_ld(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("compute-ld-zip")
    bfile = mgp.tgp_eur_data_path()
    variants = [
        line.split()[1] for line in Path(bfile + ".bim").read_text().splitlines()[:64]
    ]
    extract = workspace / "variants.txt"
    extract.write_text("\n".join(variants) + "\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin/mgp_compute_ld"),
            "--bfile",
            bfile,
            "--extract",
            str(extract),
            "--estimator",
            "windowed",
            "--ld-window",
            "8",
            "--output-dir",
            str(workspace / "ld"),
            "--temp-dir",
            str(workspace / "temp"),
            "--metadata",
            "Ancestry=EUR,Date=2026-11-13",
            "--zip",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return workspace / "ld/chr_22", workspace / "ld/chr_22.zip"


def assert_same_ld(actual, expected):
    assert actual.sample_size == expected.sample_size
    assert actual.zarr_group.attrs.asdict() == expected.zarr_group.attrs.asdict()
    pd.testing.assert_frame_equal(actual.to_snp_table(), expected.to_snp_table())
    np.testing.assert_array_equal(actual.indptr[:], expected.indptr[:])
    np.testing.assert_array_equal(actual.data[:], expected.data[:])
    np.testing.assert_array_equal(
        actual.to_csr().toarray(), expected.to_csr().toarray()
    )


@pytest.mark.parametrize("reader", [mgp.LDMatrix.from_zip, mgp.LDMatrix.from_path])
@pytest.mark.parametrize("cache_size", [None, 4096])
def test_cli_zip_round_trips_locally(zipped_ld, reader, cache_size):
    directory, archive = zipped_ld
    expected = mgp.LDMatrix.from_directory(str(directory))
    actual = reader(str(archive), cache_size=cache_size)
    assert_same_ld(actual, expected)


def test_cli_zip_is_a_standard_root_level_zarr_store(zipped_ld):
    directory, archive = zipped_ld
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert names[-1] == ".zmetadata"
        assert ".zgroup" in names and "matrix/data/.zarray" in names
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in zipped.infolist()
        )
        expected = {
            p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*")
            if p.is_file()
        }
        assert set(names) == set(expected)
        for name in names:
            assert zipped.read(name) == expected[name]
    with zarr.ZipStore(str(archive), mode="r") as store:
        group = zarr.open_consolidated(store, mode="r")
        assert "matrix/data" in group


@pytest.mark.parametrize("cache_size", [None, 4096])
@pytest.mark.parametrize("protocol", ["http", "hf"])
def test_cli_zip_round_trips_remotely(zipped_ld, monkeypatch, cache_size, protocol):
    directory, archive = zipped_ld
    payload = archive.read_bytes()
    requests = []

    def request(reader, range_header):
        assert reader.url == "https://example.com/renamed.zip?download=1"
        requests.append(range_header)
        suffix = re.fullmatch(r"bytes=-(\d+)", range_header)
        if suffix:
            start, end = max(0, len(payload) - int(suffix[1])), len(payload) - 1
        else:
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
            assert match is not None
            start, end = int(match[1]), min(int(match[2]), len(payload) - 1)
        return (
            206,
            {"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
            payload[start : end + 1],
        )

    monkeypatch.setattr(remote_zip._RemoteRangeReader, "_http_request", request)
    monkeypatch.setattr(
        remote_zip,
        "ReadOnlyZipStore",
        partial(remote_zip.ReadOnlyZipStore, tail_size=64),
    )
    if protocol == "hf":

        def get(url, *, headers, follow_redirects):
            assert follow_redirects
            status, response_headers, data = request(
                SimpleNamespace(url=url), headers["Range"]
            )
            return SimpleNamespace(
                status_code=status, headers=response_headers, content=data
            )

        hub = SimpleNamespace(
            HfFileSystem=lambda: SimpleNamespace(
                token=None,
                url=lambda path: "https://example.com/renamed.zip?download=1",
            ),
            utils=SimpleNamespace(
                build_hf_headers=lambda token: {},
                get_session=lambda: SimpleNamespace(get=get),
            ),
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        actual = mgp.LDMatrix.from_path(
            "hf://datasets/org/repo/renamed.zip", cache_size=cache_size
        )
    else:
        actual = mgp.LDMatrix.from_path(
            "https://example.com/renamed.zip?download=1", cache_size=cache_size
        )
    assert_same_ld(actual, mgp.LDMatrix.from_directory(str(directory)))
    assert len(requests) > 1  # Exercise range reads beyond the initial ZIP tail.


def test_cli_zip_loads_through_gwa_data_loader(zipped_ld, tmp_path):
    directory, archive = zipped_ld
    loader = mgp.GWADataLoader(
        ld_store_files=str(archive), temp_dir=str(tmp_path / "temp")
    )
    try:
        assert list(loader.ld) == [22]
        assert_same_ld(loader.ld[22], mgp.LDMatrix.from_directory(str(directory)))
    finally:
        loader.cleanup()
