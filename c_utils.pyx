# Author: Shadi Zabad
# Date December 2020

cimport cython
from cython.parallel import prange
from libc.math cimport exp
import numpy as np
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def zarr_islice(arr):

    cdef unsigned int j, chunk_size = arr.chunks[0], end = arr.shape[0]
    chunk = None

    for j in range(end):
        if j % chunk_size == 0:
            chunk = arr[j: j + chunk_size]

        yield chunk[j % chunk_size]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef find_windowed_ld_boundaries(double[:] cm_dist, double max_dist, int n_threads):

    cdef unsigned int i, j, M = len(cm_dist)
    cdef long[:] v_min = np.zeros_like(cm_dist, dtype=np.int)
    cdef long[:] v_max = M*np.ones_like(cm_dist, dtype=np.int)

    for i in prange(M, nogil=True, schedule='static', num_threads=n_threads):

        for j in range(i, M):
            if cm_dist[j] - cm_dist[i] > max_dist:
                v_max[i] = j
                break

        for j in range(i, 0, -1):
            if cm_dist[i] - cm_dist[j] > max_dist:
                v_min[i] = j
                break

    return np.array((v_min, v_max))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef find_shrinkage_ld_boundaries(double[:] cm_dist,
                                   double genmap_Ne,
                                   int genmap_sample_size,
                                   double cutoff,
                                   int n_threads):

    cdef unsigned int i, j, M = len(cm_dist)
    cdef long[:] v_min = np.zeros_like(cm_dist, dtype=np.int)
    cdef long[:] v_max = M*np.ones_like(cm_dist, dtype=np.int)

    # The multiplicative factor for the shrinkage estimator
    cdef double mult_factor = 2. * genmap_Ne / genmap_sample_size

    for i in prange(M, nogil=True, schedule='static', num_threads=n_threads):

        for j in range(i, M):
            if exp(-mult_factor*(cm_dist[j] - cm_dist[i])) < cutoff:
                v_max[i] = j
                break

        for j in range(i, 0, -1):
            if exp(-mult_factor*(cm_dist[i] - cm_dist[j])) < cutoff:
                v_min[i] = j
                break

    return np.array((v_min, v_max))
