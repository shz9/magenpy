# Author: Shadi Zabad
# Date December 2020

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef find_ld_boundaries(double[:] cm_dist, int max_dist, int n_threads):

    cdef int M = len(cm_dist)
    cdef int i, j
    cdef double diff
    cdef long[:] v_min = np.zeros_like(cm_dist, dtype=np.int)
    cdef long[:] v_max = np.zeros_like(cm_dist, dtype=np.int)

    for i in prange(M, nogil=True, schedule='static', num_threads=n_threads):
        for j in range(M):
            if j == M - 1:
                v_max[i] = j + 1
            else:
                diff = cm_dist[i] - cm_dist[j]
                if diff > max_dist:
                    v_min[i] = j
                elif diff < -max_dist:
                    v_max[i] = j
                    break

    return np.array((v_min, v_max))