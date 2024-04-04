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
from libc.stdint cimport int64_t
from cython cimport integral, floating
cimport cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef filter_ut_csr_matrix_low_memory(integral[::1] indptr, char[::1] bool_mask):
    """
    This is a utility function to generate a mask with the purpose of filtering 
    the data array of upper-triangular CSR matrices. The function also generates a new 
    indptr array that reflects the filter requested by the user.

    The reason we have this implementation is to avoid row/column filtering with 
    scipy's native functionality for CSR matrices, which involves using the `indices` 
    array, which can take substantial amounts of memory that is not needed for 
    matrices that have special structure, such as Linkage-Disequilibrium matrices.

    :param indptr: The index pointer array for the CSR matrix to be filtered.
    :param bool_mask: A boolean mask of 0s and 1s represented as int8.
    """


    cdef:
        int64_t i, curr_row, row_bound, new_indptr_idx = 1, curr_shape=indptr.shape[0] - 1
        int64_t[::1] new_indptr = np.zeros(np.count_nonzero(bool_mask) + 1, dtype=np.int64)
        char[::1] data_mask = np.zeros(indptr[curr_shape], dtype=np.int8)

    with nogil:
        # For each row in the current matrix:
        for curr_row in range(curr_shape):

            # If the row is to be included in the new matrix:
            if bool_mask[curr_row]:

                # Useful quantity to convert the data array index `i` to the
                # equivalent row index in the `bool` mask:
                row_bound = curr_row - indptr[curr_row] + 1

                # For the new indptr array, copy the value from the previous row:
                new_indptr[new_indptr_idx] = new_indptr[new_indptr_idx - 1]

                # For each entry for this row in the data array
                for i in range(indptr[curr_row], indptr[curr_row + 1]):

                    # If the entry isn't filtered, make sure it's included in the new matrix
                    # And increase the `indptr` by one unit:
                    if bool_mask[row_bound + i]:
                        data_mask[i] = 1
                        new_indptr[new_indptr_idx] += 1

                new_indptr_idx += 1

    return np.asarray(data_mask).astype(bool), np.asarray(new_indptr)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef expand_ranges(integral[::1] start, integral[::1] end, int64_t output_size):
    """
    Given a set of start and end indices, expand them into one long vector that contains 
    the indices between all start and end positions.
    
    :param start: A vector with the start indices.
    :param end: A vector with the end indices.
    :param output_size: The size of the output vector (equivalent to the sum of the lengths
                        of all ranges).
    """

    cdef:
        integral i, j, size=start.shape[0]
        int64_t out_idx = 0
        integral[::1] output

    if integral is int:
        output = np.empty(output_size, dtype=np.int32)
    else:
        output = np.empty(output_size, dtype=np.int64)

    with nogil:
        for i in range(size):
            for j in range(start[i], end[i]):
                output[out_idx] = j
                out_idx += 1

    return np.asarray(output)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cpdef find_ld_block_boundaries(integral[:] pos, int[:, :] block_boundaries):
    """
    Find the LD boundaries for the blockwise estimator of LD, i.e., the 
    indices of the leftmost and rightmost neighbors for each SNP.
    
    :param pos: A vector with the position of each genetic variant.
    :param block_boundaries: A matrix with the boundaries of each LD block.
    """

    cdef:
        int i, j, ldb_idx, B = block_boundaries.shape[0], M = pos.shape[0]
        integral block_start, block_end
        int[:] v_min = np.zeros_like(pos, dtype=np.int32)
        int[:] v_max = M*np.ones_like(pos, dtype=np.int32)

    with nogil:
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cpdef find_windowed_ld_boundaries(floating[:] pos, double max_dist):
    """
    Find the LD boundaries for the windowed estimator of LD, i.e., the 
    indices of the leftmost and rightmost neighbors for each SNP.
    
    :param pos: A vector with the position of each genetic variant.
    :param max_dist: The maximum distance between SNPs to consider them neighbors.
    """

    cdef:
        int i, j, M = pos.shape[0]
        int[:] v_min = np.zeros_like(pos, dtype=np.int32)
        int[:] v_max = M*np.ones_like(pos, dtype=np.int32)

    with nogil:
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cpdef find_shrinkage_ld_boundaries(floating[:] cm_pos,
                                  double genmap_ne,
                                  int genmap_sample_size,
                                  double cutoff):
    """
    Find the LD boundaries for the shrinkage estimator of Wen and Stephens (2010).
    
    :param cm_pos: A vector with the position of each genetic variant in centi Morgan.
    :param genmap_ne: The effective population size for the genetic map sample.
    :param genmap_sample_size: The sample size used to estimate the genetic map.
    :param cutoff: The threshold below which we set the shrinkage factor to zero.
    """

    cdef:
        int i, j, M = cm_pos.shape[0]
        int[:] v_min = np.zeros_like(cm_pos, dtype=np.int32)
        int[:] v_max = M*np.ones_like(cm_pos, dtype=np.int32)

    # The multiplicative term for the shrinkage factor
    # The shrinkage factor is 4 * Ne * (rho_ij/100) / (2*m)
    # where Ne is the effective population size and m is the sample size
    # for the genetic map and rho_ij is the distance between SNPs i and j
    # in centi Morgan.
    # Therefore, the multiplicative term that we need to apply
    # to the distance between SNPs is: 4*Ne/(200*m), which is equivalent to 0.02*Ne/m
    # See also: https://github.com/stephenslab/rss/blob/master/misc/get_corr.R
    # and Wen and Stephens (2010)
    cdef double mult_term = 0.02 * genmap_ne / genmap_sample_size

    with nogil:
        for i in range(M):

            for j in range(i, M):
                if exp(-mult_term*(cm_pos[j] - cm_pos[i])) < cutoff:
                    v_max[i] = j
                    break

            for j in range(i, -1, -1):
                if exp(-mult_term*(cm_pos[i] - cm_pos[j])) < cutoff:
                    v_min[i] = j + 1
                    break

    return np.array((v_min, v_max))
