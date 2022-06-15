# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

from libc.math cimport exp
import numpy as np
cimport numpy as np


def zarr_islice(arr, start=None, end=None):

    """
    This is copied from the official, but not yet released implementation of
    i_slice in Zarr codebase:
    https://github.com/zarr-developers/zarr-python/blob/e79e75ca8f07c95a5deede51f7074f699aa41149/zarr/core.py#L463
    :param arr: A Zarr array
    :param start: Start index
    :param end: End index
    """

    if len(arr.shape) == 0:
        # Same error as numpy
        raise TypeError("iteration over a 0-d array")
    if start is None:
        start = 0
    if end is None or end > arr.shape[0]:
        end = arr.shape[0]

    cdef unsigned int j, chunk_size = arr.chunks[0]
    chunk = None

    for j in range(start, end):
        if j % chunk_size == 0:
            chunk = arr[j: j + chunk_size]
        elif chunk is None:
            chunk_start = j - j % chunk_size
            chunk_end = chunk_start + chunk_size
            chunk = arr[chunk_start:chunk_end]
        yield chunk[j % chunk_size]


cpdef find_ld_block_boundaries(long[:] pos, long[:, :] block_boundaries):

    cdef unsigned int i, j, ldb_idx, block_start, block_end, B = len(block_boundaries), M = len(pos)
    cdef long[:] v_min = np.zeros_like(pos, dtype=np.int)
    cdef long[:] v_max = M*np.ones_like(pos, dtype=np.int)

    for i in range(M):

        # Find the positional boundaries for SNP i:
        for ldb_idx in range(B):
            if block_boundaries[ldb_idx, 0] <= pos[i] < block_boundaries[ldb_idx, 1]:
                block_start, block_end = block_boundaries[ldb_idx, 0], block_boundaries[ldb_idx, 1]
                break

        for j in range(i, M):
            if pos[j] >= block_end:
                v_max[i] = j
                break

        for j in range(i, -1, -1):
            if pos[j] < block_start:
                v_min[i] = j + 1
                break

    return np.array((v_min, v_max))


cpdef find_windowed_ld_boundaries(double[:] pos, double max_dist):

    cdef unsigned int i, j, M = len(pos)
    cdef long[:] v_min = np.zeros_like(pos, dtype=np.int)
    cdef long[:] v_max = M*np.ones_like(pos, dtype=np.int)

    for i in range(M):

        for j in range(i, M):
            if pos[j] - pos[i] > max_dist:
                v_max[i] = j
                break

        for j in range(i, -1, -1):
            if pos[i] - pos[j] > max_dist:
                v_min[i] = j + 1
                break

    return np.array((v_min, v_max))


cpdef find_shrinkage_ld_boundaries(double[:] cm_pos,
                                   double genmap_Ne,
                                   int genmap_sample_size,
                                   double cutoff):

    cdef unsigned int i, j, M = len(cm_pos)
    cdef long[:] v_min = np.zeros_like(cm_pos, dtype=np.int)
    cdef long[:] v_max = M*np.ones_like(cm_pos, dtype=np.int)

    # The multiplicative factor for the shrinkage estimator
    cdef double mult_factor = 2. * genmap_Ne / genmap_sample_size

    for i in range(M):

        for j in range(i, M):
            if exp(-mult_factor*(cm_pos[j] - cm_pos[i])) < cutoff:
                v_max[i] = j
                break

        for j in range(i, -1, -1):
            if exp(-mult_factor*(cm_pos[i] - cm_pos[j])) < cutoff:
                v_min[i] = j + 1
                break

    return np.array((v_min, v_max))
