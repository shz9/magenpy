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


@pytest.mark.parametrize("protocol", ["http", "https"])
def test_from_path_dispatches_http_urls(monkeypatch, protocol):
    expected = object()
    calls = {}

    def from_url(cls, url, cache_size=None):
        calls["args"] = (url, cache_size)
        return expected

    monkeypatch.setattr(LDMatrix, "from_url", classmethod(from_url))

    url = f"{protocol}://example.com/path/to/store.zarr"
    result = LDMatrix.from_path(url, cache_size=1024)

    assert result is expected
    assert calls["args"] == (url, 1024)


def _empty_ld_group():
    ld_group = zarr.group()
    matrix_group = ld_group.create_group("matrix")
    matrix_group.zeros("data", shape=1)
    matrix_group.zeros("indptr", shape=2)
    return ld_group


def _patch_fs_store(monkeypatch, factory):
    class FakeFSStore:
        @staticmethod
        def _fsspec_installed():
            # Zarr consults this class-level capability check while normalizing
            # every store, including the MemoryStore used for the fake LD group.
            # Reporting True here makes a minimal environment import fsspec even
            # though this test double never needs it.
            return False

        def __new__(cls, *args, **kwargs):
            return factory(*args, **kwargs)

    monkeypatch.setattr(zarr.storage, "FSStore", FakeFSStore)


def test_from_url_prefers_consolidated_metadata_and_caches(monkeypatch):
    calls = {}
    store = object()
    cached_store = object()
    ld_group = _empty_ld_group()

    def fs_store(url, **kwargs):
        calls["fs_store_args"] = (url, kwargs)
        return store

    _patch_fs_store(monkeypatch, fs_store)
    monkeypatch.setattr(
        zarr,
        "LRUStoreCache",
        lambda input_store, max_size: (
            calls.update(cache_args=(input_store, max_size)) or cached_store
        ),
    )

    def open_consolidated(*, store, mode):
        calls["open_consolidated_args"] = (store, mode)
        return ld_group

    monkeypatch.setattr(zarr, "open_consolidated", open_consolidated)

    result = LDMatrix.from_url(
        "https://example.com/path/to/store.zarr/",
        cache_size=2048,
        headers={"Authorization": "Bearer token"},
    )

    assert isinstance(result, LDMatrix)
    assert calls["fs_store_args"] == (
        "https://example.com/path/to/store.zarr",
        {
            "mode": "r",
            "check": False,
            "headers": {"Authorization": "Bearer token"},
        },
    )
    assert calls["cache_args"] == (store, 2048)
    assert calls["open_consolidated_args"] == (cached_store, "r")


def test_from_url_falls_back_when_consolidated_metadata_is_absent(monkeypatch):
    calls = {}
    store = object()
    ld_group = _empty_ld_group()

    _patch_fs_store(monkeypatch, lambda *args, **kwargs: store)

    def open_consolidated(*, store, mode):
        calls["open_consolidated_args"] = (store, mode)
        raise KeyError(".zmetadata")

    def open_group(*, store, mode):
        calls["open_group_args"] = (store, mode)
        return ld_group

    monkeypatch.setattr(zarr, "open_consolidated", open_consolidated)
    monkeypatch.setattr(zarr, "open_group", open_group)

    result = LDMatrix.from_url("https://example.com/store.zarr")

    assert isinstance(result, LDMatrix)
    assert calls["open_consolidated_args"] == (store, "r")
    assert calls["open_group_args"] == (store, "r")


def test_open_store_does_not_hide_invalid_consolidated_metadata(monkeypatch):
    monkeypatch.setattr(
        zarr,
        "open_consolidated",
        lambda **kwargs: (_ for _ in ()).throw(KeyError("matrix/.zarray")),
    )
    monkeypatch.setattr(
        zarr,
        "open_group",
        lambda **kwargs: pytest.fail("ordinary metadata should not be used"),
    )

    with pytest.raises(KeyError, match="matrix/.zarray"):
        LDMatrix._open_store(object())


def test_from_url_rejects_non_http_urls():
    with pytest.raises(ValueError, match="http"):
        LDMatrix.from_url("s3://bucket/store.zarr")


def test_from_url_defers_optional_dependency_check_to_zarr(monkeypatch):
    """A mocked/custom Zarr store must not require importing fsspec eagerly."""

    ld_group = object()

    class StubLDMatrix(LDMatrix):
        def __init__(self, zarr_group):
            self._zg = zarr_group

    real_import = builtins.__import__

    def import_without_fsspec(name, *args, **kwargs):
        if name == "fsspec" or name.startswith("fsspec."):
            raise ModuleNotFoundError("No module named 'fsspec'", name="fsspec")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fsspec)
    _patch_fs_store(monkeypatch, lambda *args, **kwargs: object())

    monkeypatch.setattr(zarr, "open_consolidated", lambda **kwargs: ld_group)

    assert isinstance(
        StubLDMatrix.from_url("https://example.com/store.zarr"),
        StubLDMatrix,
    )


def test_from_url_explains_how_to_install_http_dependencies(monkeypatch):
    def unavailable_fs_store(*args, **kwargs):
        raise ImportError("No module named 'aiohttp'")

    _patch_fs_store(monkeypatch, unavailable_fs_store)

    with pytest.raises(ImportError, match=r"magenpy\[http\]"):
        LDMatrix.from_url("https://example.com/store.zarr")


def test_from_s3_prefers_consolidated_metadata(monkeypatch):
    calls = {}
    store = object()
    ld_group = _empty_ld_group()

    class FakeS3FileSystem:
        def __init__(self, **kwargs):
            calls["filesystem_kwargs"] = kwargs

    def s3_map(*, root, s3, check):
        calls["mapper_args"] = (root, s3, check)
        return store

    fake_s3fs = SimpleNamespace(
        S3FileSystem=FakeS3FileSystem,
        S3Map=s3_map,
    )
    monkeypatch.setitem(sys.modules, "s3fs", fake_s3fs)

    def open_consolidated(*, store, mode):
        calls["open_consolidated_args"] = (store, mode)
        return ld_group

    monkeypatch.setattr(zarr, "open_consolidated", open_consolidated)

    result = LDMatrix.from_s3("s3://bucket/path/to/store.zarr")

    assert isinstance(result, LDMatrix)
    assert calls["filesystem_kwargs"] == {
        "anon": True,
        "client_kwargs": {"region_name": "us-east-2"},
    }
    assert calls["mapper_args"][0] == "bucket/path/to/store.zarr"
    assert calls["mapper_args"][2] is False
    assert calls["open_consolidated_args"] == (store, "r")


def test_from_hf_prefers_consolidated_metadata_for_subgroup(monkeypatch):
    calls = {}
    store = object()
    ld_group = _empty_ld_group()

    class StubLDMatrix(LDMatrix):
        def __init__(self, zarr_group):
            self._zg = zarr_group

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace())

    def get_mapper(path):
        calls["mapper_path"] = path
        return store

    monkeypatch.setitem(sys.modules, "fsspec", SimpleNamespace(get_mapper=get_mapper))

    def open_consolidated(*, store, mode, path):
        calls["open_consolidated_args"] = (store, mode, path)
        return ld_group

    monkeypatch.setattr(zarr, "open_consolidated", open_consolidated)

    result = StubLDMatrix.from_hf("hf://datasets/org/repo/store.zarr")

    assert isinstance(result, StubLDMatrix)
    assert calls["mapper_path"] == "hf://datasets/org/repo/store.zarr"
    assert calls["open_consolidated_args"] == (store, "r", "store")


def test_from_hf_uses_read_only_store_for_remote_zip(monkeypatch):
    calls = {}
    store = object()
    ld_group = _empty_ld_group()

    class FakeHfFileSystem:
        token = "token"

        def url(self, path):
            calls["hf_path"] = path
            return "https://example.com/chr_1.zip"

    class FakeResponse:
        status_code = 206
        headers = {"content-range": "bytes 0-0/1"}
        content = b"x"

    class FakeSession:
        def get(self, *args, **kwargs):
            calls["request_args"] = (args, kwargs)
            return FakeResponse()

    fake_hub = SimpleNamespace(
        HfFileSystem=FakeHfFileSystem,
        utils=SimpleNamespace(
            build_hf_headers=lambda token: {"Authorization": f"Bearer {token}"},
            get_session=lambda: FakeSession(),
        ),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import magenpy.utils.remote_zip as remote_zip

    def read_only_zip_store(url, **kwargs):
        calls["store_args"] = (url, kwargs)
        return store

    monkeypatch.setattr(remote_zip, "ReadOnlyZipStore", read_only_zip_store)
    monkeypatch.setattr(LDMatrix, "_open_store", lambda input_store: ld_group)

    result = LDMatrix.from_hf("hf://datasets/org/repo/EUR/chr_1.zip")

    assert isinstance(result, LDMatrix)
    assert calls["hf_path"] == "hf://datasets/org/repo/EUR/chr_1.zip"
    assert calls["store_args"][0] == "https://example.com/chr_1.zip"
    assert calls["store_args"][1]["prefix"] == "chr_1"
    assert calls["store_args"][1]["headers"] == {
        "Authorization": "Bearer token"
    }


def test_ld_matrix_validation_does_not_require_store_listing(monkeypatch):
    ld_group = _empty_ld_group()

    def fail_listing(*args, **kwargs):
        raise AssertionError("listing should not be used")

    monkeypatch.setattr(zarr.hierarchy.Group, "group_keys", fail_listing)
    monkeypatch.setattr(zarr.hierarchy.Group, "array_keys", fail_listing)

    assert isinstance(LDMatrix(ld_group), LDMatrix)


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

    def open_consolidated(*, store, mode):
        calls["open_consolidated_args"] = (store, mode)
        return ld_group

    monkeypatch.setattr(zarr, "open_consolidated", open_consolidated)

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
    assert calls["open_consolidated_args"] == (cached_mapper, "r")


def test_from_gcs_explains_how_to_install_dependency(monkeypatch):
    real_import = builtins.__import__

    def import_without_gcsfs(name, *args, **kwargs):
        if name == "gcsfs" or name.startswith("gcsfs."):
            raise ModuleNotFoundError("No module named 'gcsfs'", name="gcsfs")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gcsfs)

    with pytest.raises(ImportError, match=r"magenpy\[gcs\]"):
        LDMatrix.from_gcs("gs://bucket/path/to/store.zarr")
