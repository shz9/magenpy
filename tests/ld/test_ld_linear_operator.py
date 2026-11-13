"""Dense-reference coverage of compressed LD products (no genotype fixtures)."""

import itertools

import numpy as np
import pytest

from magenpy.LDLinearOperator import LDLinearOperator


def make_operator(storage, full=False, threads=1, flags=(True, True, True), n=29):
    rng = np.random.default_rng(2026)
    # Include empty rows and nonmonotone right boundaries.
    counts = np.array([rng.integers(0, min(11, n - i)) for i in range(n)])
    ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    data = (
        rng.uniform(-0.8, 0.8, ptr[-1])
        * (127 if storage == np.int8 else 32767 if storage == np.int16 else 1)
    ).astype(storage)
    upper = np.zeros((n, n))
    scale = 1 / np.iinfo(storage).max if np.issubdtype(storage, np.integer) else 1
    for i in range(n):
        upper[i, i + 1 : i + 1 + counts[i]] = data[ptr[i] : ptr[i + 1]] * scale
    lower, use_upper, diag = flags
    dense = upper * use_upper + upper.T * lower + np.eye(n) * diag
    left = np.arange(1, n + 1, dtype=np.int32)
    if full:
        dense = upper + upper.T + np.eye(n)
        left = np.zeros(n, dtype=np.int32)
        ptr = np.arange(n + 1, dtype=np.int64) * n
        data = (
            (dense.ravel() / scale).round().astype(storage)
            if np.issubdtype(storage, np.integer)
            else dense.ravel().astype(storage)
        )
        dense = data.reshape(n, n).astype(float) * scale
    op = LDLinearOperator(
        ptr,
        data,
        left,
        symmetric=full,
        threads=threads,
        lower_triangle=lower,
        upper_triangle=use_upper,
        include_diag=diag,
    )
    return op, dense


@pytest.mark.parametrize("storage", [np.int8, np.int16, np.float32, np.float64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize(
    "full,flags",
    [(False, f) for f in itertools.product([False, True], repeat=3)]
    + [(True, (True, True, True))],
)
def test_products(storage, dtype, threads, full, flags):
    op, dense = make_operator(storage, full, threads, flags)
    shift = np.linspace(-0.1, 0.2, dense.shape[0]).astype(dtype)
    op.set_diag_shift(shift)
    dense += np.diag(shift)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(dense.shape[0], 7)).astype(dtype)
    tol = 3e-6 if dtype == np.float32 else 2e-13
    for rhs in (
        x,
        np.asfortranarray(x),
        x[:, ::2],
        x[::-1],
        x[:, :1],
        x[:, 0],
        x[:, :0],
    ):
        np.testing.assert_allclose(op @ rhs, dense @ rhs, rtol=tol, atol=tol)
        np.testing.assert_allclose(op.T @ rhs, dense.T @ rhs, rtol=tol, atol=tol)


@pytest.mark.parametrize("shift", [None, 0.5, np.arange(29) * 0.01])
def test_complex_integer_and_readonly(shift):
    op, dense = make_operator(np.int8)
    op.set_diag_shift(shift)
    if shift is not None:
        dense += np.eye(29) * shift
    for array in (op.ld_data, op.ld_indptr, op.leftmost_idx):
        array.flags.writeable = False
    for x in (
        np.arange(29),
        np.ones((29, 3), dtype=np.int32),
        np.ones((29, 3)) * (1 + 2j),
    ):
        x.flags.writeable = False
        np.testing.assert_allclose(op @ x, dense @ x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(op.H @ x, dense.T @ x, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n", [0, 1])
def test_empty_storage(n):
    op = LDLinearOperator(
        np.zeros(n + 1, dtype=np.int64),
        np.empty(0, dtype=np.int8),
        np.arange(1, n + 1, dtype=np.int32),
    )
    for x in (np.ones(n), np.ones((n, 1)), np.ones((n, 3)), np.ones((n, 0))):
        np.testing.assert_array_equal(op @ x, x)


@pytest.mark.parametrize("threads", [0, -1, 1.5])
def test_invalid_threads(threads):
    with pytest.raises(ValueError, match="positive integer"):
        make_operator(np.float32, threads=threads)


@pytest.mark.parametrize("k", [2, 31, 32, 33, 64, 65])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_rhs_tiles_and_index_width(k, full, index_dtype):
    op, dense = make_operator(np.int16, full=full, threads=3)
    op.ld_indptr = op.ld_indptr.astype(index_dtype)
    x = np.random.default_rng(3).normal(size=(29, k))
    np.testing.assert_allclose(op @ x, dense @ x, rtol=1e-12, atol=1e-12)


def test_full_storage_variable_left_bounds():
    op, dense = make_operator(np.float64)
    dense += np.eye(29) * 0.2
    starts, rows = [], []
    for i in range(29):
        cols = np.flatnonzero(dense[i])
        starts.append(cols[0])
        rows.append(dense[i, cols[0] : cols[-1] + 1])
    op = LDLinearOperator(
        np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64),
        np.concatenate(rows),
        np.array(starts, dtype=np.int32),
        symmetric=True,
        threads=4,
    )
    x = np.random.default_rng(4).normal(size=(29, 33))
    np.testing.assert_allclose(op @ x, dense @ x, atol=1e-12)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("shape", [(29,), (29, 1), (29, 7)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_contiguous_rhs_and_ld_buffers_are_borrowed(monkeypatch, full, shape, dtype):
    """Observe actual kernel arguments, including the vector reshape path."""
    from magenpy.stats.ld import c_utils

    op, _ = make_operator(np.int8, full=full)
    x = np.ones(shape, dtype=dtype)
    x.flags.writeable = False
    original = x.copy()
    kernel_name = (
        "ld_matmat"
        if len(shape) == 2 and shape[1] != 1
        else "ld_dot"
        if full
        else "ut_ld_dot"
    )
    kernel = getattr(c_utils, kernel_name)
    outputs = []

    def checked(*args):
        offset = 0 if kernel_name == "ut_ld_dot" else 1
        if offset:
            assert args[0] is op.leftmost_idx
        assert args[offset] is op.ld_indptr
        assert args[offset + 1] is op.ld_data
        assert np.shares_memory(args[offset + 2], x)
        result = kernel(*args)
        outputs.append(result)
        return result

    monkeypatch.setattr(c_utils, kernel_name, checked)
    result = op @ x
    assert np.shares_memory(result, outputs[0])
    assert not np.shares_memory(result, x)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shift_dtype",
    [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("matrix", [False, True])
def test_diagonal_update_borrows_strided_shift(monkeypatch, dtype, shift_dtype, matrix):
    from magenpy.stats.ld import c_utils

    op, _ = make_operator(np.int8)
    x = np.random.default_rng(7).normal(size=(29, 7) if matrix else 29).astype(dtype)
    expected = op @ x
    shift = np.arange(58).astype(shift_dtype)[::-2]
    shift.flags.writeable = False
    op.set_diag_shift(shift)
    expected += x * (shift[:, None] if matrix else shift)
    kernel = c_utils.add_diag_shift
    seen = []

    def checked(rhs, out, diag):
        assert np.shares_memory(rhs, x)
        assert np.shares_memory(diag, shift)
        seen.append(out)
        return kernel(rhs, out, diag)

    monkeypatch.setattr(c_utils, "add_diag_shift", checked)
    result = op @ x
    assert np.shares_memory(result, seen[0])
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("matrix", [False, True])
def test_complex_uses_one_final_complex_buffer(monkeypatch, dtype, matrix):
    op, dense = make_operator(np.int8)
    x = np.full((29, 7) if matrix else 29, 1 + 2j, dtype=dtype)
    allocate = np.empty
    complex_allocations = []

    def checked(*args, **kwargs):
        result = allocate(*args, **kwargs)
        if np.iscomplexobj(result):
            complex_allocations.append(result)
        return result

    monkeypatch.setattr(np, "empty", checked)
    result = op @ x
    assert len(complex_allocations) == 1
    assert np.shares_memory(result, complex_allocations[0])
    assert result.dtype == dtype
    np.testing.assert_allclose(result, dense @ x, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shift_dtype", [np.float16, np.longdouble, ">f8"])
def test_other_shift_formats(shift_dtype):
    op, _ = make_operator(np.int8)
    x = np.random.default_rng(44).normal(size=(29, 3)).astype(np.float32)
    expected = op @ x
    shift = np.linspace(0.1, 1.0, 29).astype(shift_dtype)
    expected += x * shift[:, None]
    op.set_diag_shift(shift)
    np.testing.assert_allclose(op @ x, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("layout", ["C", "F", "strided", "integer"])
def test_rhs_packed_at_most_once(monkeypatch, layout):
    op, _ = make_operator(np.int8)
    x = np.ones((29, 6), dtype=np.float32)
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "strided":
        x = x[:, ::2]
    elif layout == "integer":
        x = x.astype(np.int32)
    pack = np.ascontiguousarray
    calls = []

    def checked(*args, **kwargs):
        result = pack(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(np, "ascontiguousarray", checked)
    op @ x
    assert len(calls) == 1
    assert np.shares_memory(calls[0], x) == (layout == "C")
