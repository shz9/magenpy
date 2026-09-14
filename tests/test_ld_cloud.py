import builtins
import sys
from types import SimpleNamespace

import pytest
import zarr

from magenpy.LDMatrix import LDMatrix


@pytest.mark.parametrize("protocol", ["gs", "gcs"])
def test_from_path_dispatches_google_cloud_urls(monkeypatch, protocol):
    expected = object()
    calls = {}

    def from_gcs(cls, path, cache_size=None):
        calls["args"] = (path, cache_size)
        return expected

    monkeypatch.setattr(LDMatrix, "from_gcs", classmethod(from_gcs))

    path = f"{protocol}://bucket/path/to/store.zarr"
    result = LDMatrix.from_path(path, cache_size=1024)

    assert result is expected
    assert calls["args"] == (path, 1024)


def test_from_gcs_opens_mapper_with_credentials_and_cache(monkeypatch):
    calls = {}
    mapper = object()
    cached_mapper = object()

    class FakeGCSFileSystem:
        def __init__(self, **kwargs):
            calls["filesystem_kwargs"] = kwargs

        def get_mapper(self, path, check):
            calls["mapper_args"] = (path, check)
            return mapper

    monkeypatch.setitem(
        sys.modules,
        "gcsfs",
        SimpleNamespace(GCSFileSystem=FakeGCSFileSystem),
    )
    monkeypatch.setattr(
        zarr,
        "LRUStoreCache",
        lambda store, max_size: (
            calls.update(cache_args=(store, max_size)) or cached_mapper
        ),
    )

    ld_group = zarr.group()
    matrix_group = ld_group.create_group("matrix")
    matrix_group.zeros("data", shape=1)
    matrix_group.zeros("indptr", shape=2)

    def open_group(*, store, mode):
        calls["open_group_args"] = (store, mode)
        return ld_group

    monkeypatch.setattr(zarr, "open_group", open_group)

    result = LDMatrix.from_gcs(
        "gs://bucket/path/to/store.zarr",
        cache_size=2048,
        token="anon",
        project="test-project",
        requester_pays=True,
    )

    assert isinstance(result, LDMatrix)
    assert calls["filesystem_kwargs"] == {
        "token": "anon",
        "project": "test-project",
        "requester_pays": True,
    }
    assert calls["mapper_args"] == (
        "gs://bucket/path/to/store.zarr",
        False,
    )
    assert calls["cache_args"] == (mapper, 2048)
    assert calls["open_group_args"] == (cached_mapper, "r")


def test_from_gcs_explains_how_to_install_dependency(monkeypatch):
    real_import = builtins.__import__

    def import_without_gcsfs(name, *args, **kwargs):
        if name == "gcsfs" or name.startswith("gcsfs."):
            raise ModuleNotFoundError("No module named 'gcsfs'", name="gcsfs")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gcsfs)

    with pytest.raises(ImportError, match=r"magenpy\[cloud\]"):
        LDMatrix.from_gcs("gs://bucket/path/to/store.zarr")
