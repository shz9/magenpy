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
from libc.stdint cimport int64_t, int32_t
from libcpp.string cimport string
from libcpp cimport bool as cpp_bool
from cython cimport floating
cimport cython
import numpy as np
cimport numpy as cnp


ctypedef fused int_dtype:
    cnp.int32_t
    cnp.int64_t

ctypedef fused noncomplex_numeric:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.float32_t
    cnp.float64_t


cdef extern from "ld_utils.hpp" nogil:
    bint blas_supported() noexcept nogil
    bint omp_supported() noexcept nogil

    void ld_matrix_dot[T, U, I](int c_size,
              int* ld_left_bound,
              I* ld_indptr,
              U* ld_data,
              T* vec,
              T* out,
              T dq_scale,
              int threads) noexcept nogil

    void ut_ld_matrix_dot[T, U, I](int c_size,
              I* ld_indptr,
              U* ld_data,
              T* vec,
              T* out,
              T dq_scale,
              bint include_lower_triangle,
              bint include_upper_triangle,
              bint include_diag,
              int threads) noexcept nogil

    void ld_rank_one_update[T, U, I](int c_size,
                   int* ld_left_bound,
                   I* ld_indptr,
                   U* ld_data,
                   T* vec,
                   T* out,
                   T alpha,
                   T dq_scale,
                   int threads) noexcept nogil

    void cpp_compute_ld_from_bed "compute_ld_from_bed"[T](string bed_filename,
                    const int* ref_snp_indices,
                    const int* alt_snp_indices,
                    int num_pairs,
                    const int* sample_indices,
                    int total_samples,
                    int num_samples,
                    const T* allele_frequencies,
                    cpp_bool impute_missing,
                    T* ld_data,
                    int threads) except + nogil

    void cpp_compute_ut_ld_from_bed "compute_ut_ld_from_bed"[T](string bed_filename,
                    const int* snp_indices,
                    const int* ld_boundaries_end,
                    const int64_t* ld_indptr,
                    int num_snps,
                    const int* sample_indices,
                    int total_samples,
                    int num_samples,
                    const T* allele_frequencies,
                    cpp_bool impute_missing,
                    T* ld_data,
                    int threads) except + nogil


cpdef ld_dot(int[::1] ld_left_bound,
             int_dtype[::1] ld_indptr,
             noncomplex_numeric[::1] ld_data,
             floating[::1] vec,
             floating dq_scale = 1.,
             int threads = 1):

    cdef:
        floating[::1] out = np.zeros_like(vec)

    ld_matrix_dot(vec.shape[0],
                  &ld_left_bound[0],
                  &ld_indptr[0],
                  &ld_data[0],
                  &vec[0],
                  &out[0],
                  dq_scale,
                  threads)

    return np.asarray(out)


cpdef ut_ld_dot(int_dtype[::1] ld_indptr,
             noncomplex_numeric[::1] ld_data,
             floating[::1] vec,
             floating dq_scale = 1.,
             bint include_lower_triangle = 1,
             bint include_upper_triangle = 1,
             bint include_diag = 1,
             int threads = 1):

    cdef:
        floating[::1] out = np.zeros_like(vec)

    ut_ld_matrix_dot(vec.shape[0],
                  &ld_indptr[0],
                  &ld_data[0],
                  &vec[0],
                  &out[0],
                  dq_scale,
                  include_lower_triangle,
                  include_upper_triangle,
                  include_diag,
                  threads)

    return np.asarray(out)


cpdef rank_one_update(int[::1] ld_left_bound,
                      int_dtype[::1] ld_indptr,
                      noncomplex_numeric[::1] ld_data,
                      floating[::1] vec,
                      floating[::1] out,
                      floating alpha = 1.,
                      floating dq_scale = 1.,
                      int threads = 1):

    ld_rank_one_update(
        vec.shape[0],
        &ld_left_bound[0],
        &ld_indptr[0],
        &ld_data[0],
        &vec[0],
        &out[0],
        alpha,
        dq_scale,
        threads
    )


cpdef compute_ld_from_bed(bed_filename,
                          const int[::1] ref_snp_indices,
                          const int[::1] alt_snp_indices,
                          const int[::1] sample_indices,
                          int total_samples,
                          allele_frequencies,
                          bint impute_missing=False,
                          int threads=1,
                          dtype=np.float32):
    """
    Compute LD directly from a PLINK BED file.

    `ref_snp_indices[i]` and `alt_snp_indices[i]` define one pair. The returned
    array has length `len(ref_snp_indices)`, with entries stored as correlations
    between standardized dosages. The `allele_frequencies` array is indexed by
    original BED variant index.
    """

    cdef:
        string c_bed_filename
        const int* ref_ptr = NULL
        const int* alt_ptr = NULL
        const int* sample_ptr = NULL
        const float* allele_frequency_float_ptr = NULL
        const double* allele_frequency_double_ptr = NULL
        float* ld_float_ptr = NULL
        double* ld_double_ptr = NULL
        const float[::1] allele_frequency_float_view
        const double[::1] allele_frequency_double_view
        float[::1] ld_float
        double[::1] ld_double
        int num_pairs
        int num_samples
        cpp_bool c_impute_missing

    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")

    if ref_snp_indices.shape[0] != alt_snp_indices.shape[0]:
        raise ValueError("ref_snp_indices and alt_snp_indices must have the same length.")

    if ref_snp_indices.shape[0] > 2147483647:
        raise ValueError("Too many SNP pairs for the C++ LD backend.")
    if sample_indices.shape[0] > 2147483647:
        raise ValueError("Too many selected samples for the C++ LD backend.")

    num_pairs = <int> ref_snp_indices.shape[0]
    num_samples = <int> sample_indices.shape[0]
    c_impute_missing = <cpp_bool> impute_missing

    if num_pairs > 0:
        ref_ptr = &ref_snp_indices[0]
        alt_ptr = &alt_snp_indices[0]
    if num_samples > 0:
        sample_ptr = &sample_indices[0]

    c_bed_filename = bed_filename.encode()
    dtype = np.dtype(dtype)

    if dtype == np.dtype(np.float32):
        allele_frequencies = np.ascontiguousarray(allele_frequencies, dtype=np.float32)
        allele_frequency_float_view = allele_frequencies
        if allele_frequency_float_view.shape[0] == 0:
            raise ValueError("allele_frequencies must not be empty.")
        allele_frequency_float_ptr = &allele_frequency_float_view[0]

        ld_float = np.zeros(num_pairs, dtype=np.float32)
        if num_pairs > 0:
            ld_float_ptr = &ld_float[0]

        cpp_compute_ld_from_bed[float](c_bed_filename,
                                       ref_ptr,
                                       alt_ptr,
                                       num_pairs,
                                       sample_ptr,
                                       total_samples,
                                       num_samples,
                                       allele_frequency_float_ptr,
                                       c_impute_missing,
                                       ld_float_ptr,
                                       threads)

        return np.asarray(ld_float)

    elif dtype == np.dtype(np.float64):
        allele_frequencies = np.ascontiguousarray(allele_frequencies, dtype=np.float64)
        allele_frequency_double_view = allele_frequencies
        if allele_frequency_double_view.shape[0] == 0:
            raise ValueError("allele_frequencies must not be empty.")
        allele_frequency_double_ptr = &allele_frequency_double_view[0]

        ld_double = np.zeros(num_pairs, dtype=np.float64)
        if num_pairs > 0:
            ld_double_ptr = &ld_double[0]

        cpp_compute_ld_from_bed[double](c_bed_filename,
                                        ref_ptr,
                                        alt_ptr,
                                        num_pairs,
                                        sample_ptr,
                                        total_samples,
                                        num_samples,
                                        allele_frequency_double_ptr,
                                        c_impute_missing,
                                        ld_double_ptr,
                                        threads)

        return np.asarray(ld_double)

    else:
        raise ValueError("dtype must be float32 or float64.")


cpdef compute_ut_ld_from_bed(bed_filename,
                             const int[::1] snp_indices,
                             const int[::1] ld_boundaries_end,
                             const int64_t[::1] ld_indptr,
                             const int[::1] sample_indices,
                             int total_samples,
                             allele_frequencies,
                             bint impute_missing=True,
                             int threads=1,
                             dtype=np.float32):
    """
    Compute the flat data array for an upper-triangular LD matrix directly from
    a PLINK BED file.

    For selected SNP row `j`, this computes columns
    `j + 1 : ld_boundaries_end[j]` and stores them at
    `ld_indptr[j] : ld_indptr[j + 1]`.
    """

    cdef:
        string c_bed_filename
        const int* snp_ptr = NULL
        const int* boundary_end_ptr = NULL
        const int64_t* indptr_ptr = NULL
        const int* sample_ptr = NULL
        const float* allele_frequency_float_ptr = NULL
        const double* allele_frequency_double_ptr = NULL
        float* ld_float_ptr = NULL
        double* ld_double_ptr = NULL
        const float[::1] allele_frequency_float_view
        const double[::1] allele_frequency_double_view
        float[::1] ld_float
        double[::1] ld_double
        int num_snps
        int num_samples
        int64_t num_values
        cpp_bool c_impute_missing

    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")

    if snp_indices.shape[0] != ld_boundaries_end.shape[0]:
        raise ValueError("snp_indices and ld_boundaries_end must have the same length.")

    if ld_indptr.shape[0] != snp_indices.shape[0] + 1:
        raise ValueError("ld_indptr must have length len(snp_indices) + 1.")

    if snp_indices.shape[0] > 2147483647:
        raise ValueError("Too many SNPs for the C++ LD backend.")
    if sample_indices.shape[0] > 2147483647:
        raise ValueError("Too many selected samples for the C++ LD backend.")

    num_snps = <int> snp_indices.shape[0]
    num_samples = <int> sample_indices.shape[0]
    num_values = ld_indptr[num_snps]
    c_impute_missing = <cpp_bool> impute_missing

    if num_values < 0:
        raise ValueError("ld_indptr must be non-negative.")

    if num_snps > 0:
        snp_ptr = &snp_indices[0]
        boundary_end_ptr = &ld_boundaries_end[0]
        indptr_ptr = &ld_indptr[0]
    if num_samples > 0:
        sample_ptr = &sample_indices[0]

    c_bed_filename = bed_filename.encode()
    dtype = np.dtype(dtype)

    if dtype == np.dtype(np.float32):
        allele_frequencies = np.ascontiguousarray(allele_frequencies, dtype=np.float32)
        allele_frequency_float_view = allele_frequencies
        if allele_frequency_float_view.shape[0] == 0:
            raise ValueError("allele_frequencies must not be empty.")
        allele_frequency_float_ptr = &allele_frequency_float_view[0]

        ld_float = np.zeros(num_values, dtype=np.float32)
        if num_values > 0:
            ld_float_ptr = &ld_float[0]

        cpp_compute_ut_ld_from_bed[float](c_bed_filename,
                                          snp_ptr,
                                          boundary_end_ptr,
                                          indptr_ptr,
                                          num_snps,
                                          sample_ptr,
                                          total_samples,
                                          num_samples,
                                          allele_frequency_float_ptr,
                                          c_impute_missing,
                                          ld_float_ptr,
                                          threads)

        return np.asarray(ld_float)

    elif dtype == np.dtype(np.float64):
        allele_frequencies = np.ascontiguousarray(allele_frequencies, dtype=np.float64)
        allele_frequency_double_view = allele_frequencies
        if allele_frequency_double_view.shape[0] == 0:
            raise ValueError("allele_frequencies must not be empty.")
        allele_frequency_double_ptr = &allele_frequency_double_view[0]

        ld_double = np.zeros(num_values, dtype=np.float64)
        if num_values > 0:
            ld_double_ptr = &ld_double[0]

        cpp_compute_ut_ld_from_bed[double](c_bed_filename,
                                           snp_ptr,
                                           boundary_end_ptr,
                                           indptr_ptr,
                                           num_snps,
                                           sample_ptr,
                                           total_samples,
                                           num_samples,
                                           allele_frequency_double_ptr,
                                           c_impute_missing,
                                           ld_double_ptr,
                                           threads)

        return np.asarray(ld_double)

    else:
        raise ValueError("dtype must be float32 or float64.")

# -------------------------------------------------------------------------
# Utilities for manipulating/interacting with generic numeric types:

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef noncomplex_numeric numeric_abs(noncomplex_numeric x) noexcept nogil:
    """
    Return the absolute value of a numeric type.
    """
    if x < 0:
        return -x
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef noncomplex_numeric numeric_max(noncomplex_numeric x, noncomplex_numeric y) noexcept nogil:
    """
    Return maximum of two numeric values.
    """
    if x > y:
        return x
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef noncomplex_numeric numeric_min(noncomplex_numeric x, noncomplex_numeric y) noexcept nogil:
    """
    Return minimum of two numeric values.
    """
    if x < y:
        return x
    return y

# -------------------------------------------------------------------------
# Utilities for slicing, filtering, symmetrizing and interacting with the LD matrix:

cpdef find_tagging_variants(int[::1] variant_indices,
                           int_dtype[::1] indptr,
                           noncomplex_numeric[::1] data,
                           noncomplex_numeric threshold):
    """
    Find variants that are in LD (in absolute value) with a set of focal variants.

    The function assumes that LD is stored in upper-triangular CSR format without the diagonal,
    with row entries contiguous around the diagonal.

    :param variant_indices: Indices of focal variants to start the search from.
    Assumed to be valid indices in `[0, n_variants)`.
    :param indptr: Index pointer array for the upper-triangular CSR matrix.
    :param data: Data array for the upper-triangular CSR matrix.
    :param threshold: Absolute LD threshold.
    :return: A boolean mask of length `n_variants` marking tagging variants.
    """

    cdef:
        int64_t i, focal, n_variants = indptr.shape[0] - 1
        int64_t neigh_row, row_size, data_idx
        int32_t neigh_offset
        int32_t[::1] leftmost_idx = get_leftmost_index(indptr)
        char[::1] tagging = np.zeros(n_variants, dtype=np.int8)

    with nogil:
        # Tag all valid seeds first.
        for i in range(variant_indices.shape[0]):
            focal = variant_indices[i]
            tagging[focal] = 1

        for i in range(variant_indices.shape[0]):
            focal = variant_indices[i]

            # Backward neighbors: rows whose upper-triangular entries can reach `focal`.
            for neigh_row in range(leftmost_idx[focal], focal):
                if tagging[neigh_row] == 0:
                    row_size = indptr[neigh_row + 1] - indptr[neigh_row]
                    neigh_offset = focal - neigh_row - 1

                    if neigh_offset < row_size:
                        data_idx = indptr[neigh_row] + neigh_offset
                        if numeric_abs(data[data_idx]) >= threshold:
                            tagging[neigh_row] = 1

            # Forward neighbors: directly stored in focal row.
            row_size = indptr[focal + 1] - indptr[focal]
            for neigh_offset in range(row_size):
                neigh_row = focal + neigh_offset + 1
                if tagging[neigh_row] == 0:
                    data_idx = indptr[focal] + neigh_offset
                    if numeric_abs(data[data_idx]) >= threshold:
                        tagging[neigh_row] = 1

    return np.asarray(tagging).view(bool)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef get_leftmost_index(int_dtype[::1] indptr):
    """
    Given the index pointer array from an upper triangular CSR matrix (without the diagonal),
    return the index of the leftmost neighbor for each row in the equivalent symmetric matrix.

    :param indptr: The index pointer array for the upper triangular CSR matrix.
    :return: An integer array with the leftmost index for each row.
    """

    # Determine the rightmost element for every row:
    rightmost = np.diff(indptr).astype(np.int32) + np.arange(indptr.shape[0] - 1, dtype=np.int32)

    # Get unique boundaries:
    uniq_res = np.unique(rightmost, return_index=True)

    # Loop over rows to get the leftmost index:
    cdef:
        int32_t curr_row, shape=rightmost.shape[0]
        int32_t[::1] leftmost = np.zeros(shape, dtype=np.int32)
        int32_t[::1] uniq_rightmost = uniq_res[0]
        int32_t[::1] rightmost_first_idx = uniq_res[1].astype(np.int32)
        int32_t rightmost_idx = 0, uniq_rightmost_size = uniq_res[0].shape[0]

    with nogil:
        for curr_row in range(shape):

            if curr_row > uniq_rightmost[rightmost_idx] and rightmost_idx < uniq_rightmost_size - 1:
                rightmost_idx += 1

            leftmost[curr_row] = rightmost_first_idx[rightmost_idx]

    return np.asarray(leftmost)

cpdef prune_ld_ut(int_dtype[::1] indptr,
                  noncomplex_numeric[::1] data,
                  noncomplex_numeric r_threshold,
                  int[::1] variant_order=None):
    """
    Pass over the LD matrix once and prune it so that variants whose absolute correlation coefficient is above
    or equal to a certain threshold are filtered away. If two variants are highly correlated,
    this function keeps the one that occurs earlier in the pruning order.

    This function works with LD matrices in any data type
    (quantized to integers or floats), but it is the user's responsibility to set the appropriate
    threshold for the data type used.

    !!! note
        This function assumes that the LD matrix is in upper triangular form and doesn't include the
        diagonal and that entries are contiguous around the diagonal.

    :param indptr: The index pointer array for the CSR matrix to be pruned.
    :param data: The data array for the CSR matrix to be pruned.
    :param r_threshold: The Pearson Correlation coefficient threshold above which to prune variants.
    :param variant_order: Optional pruning order. If not provided, matrix order is used.

    :return: An boolean array of which variants are kept after pruning.
    """

    cdef:
        int64_t i, curr_row, curr_row_size, curr_data_idx, curr_shape=indptr.shape[0]-1
        int64_t neigh_row, neigh_row_size, neigh_data_idx
        int32_t neigh_offset
        bint use_variant_order = variant_order is not None
        char[::1] keep = np.ones(curr_shape, dtype=np.int8)
        int32_t[::1] leftmost_idx = np.zeros(curr_shape, dtype=np.int32)

    if use_variant_order:
        leftmost_idx = get_leftmost_index(indptr)

    with nogil:
        if not use_variant_order:
            for curr_row in range(curr_shape):

                if keep[curr_row] == 1:

                    curr_row_size = indptr[curr_row + 1] - indptr[curr_row]

                    for i in range(curr_row_size):
                        curr_data_idx = indptr[curr_row] + i

                        if numeric_abs(data[curr_data_idx]) >= r_threshold:
                            keep[curr_row + i + 1] = 0
        else:
            for i in range(curr_shape):
                curr_row = variant_order[i]

                if keep[curr_row] == 1:
                    # Prune backward neighbors of curr_row:
                    for neigh_row in range(leftmost_idx[curr_row], curr_row):
                        neigh_row_size = indptr[neigh_row + 1] - indptr[neigh_row]
                        neigh_offset = curr_row - neigh_row - 1

                        if neigh_offset < neigh_row_size:
                            neigh_data_idx = indptr[neigh_row] + neigh_offset

                            if numeric_abs(data[neigh_data_idx]) >= r_threshold:
                                keep[neigh_row] = 0

                    # Prune forward neighbors of curr_row:
                    curr_row_size = indptr[curr_row + 1] - indptr[curr_row]

                    for neigh_offset in range(curr_row_size):
                        curr_data_idx = indptr[curr_row] + neigh_offset

                        if numeric_abs(data[curr_data_idx]) >= r_threshold:
                            keep[curr_row + neigh_offset + 1] = 0

    return np.asarray(keep).view(bool)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef get_symmetrized_indptr_with_mask(int_dtype[::1] indptr,
                                       cnp.ndarray[cnp.npy_bool, ndim=1] mask):
    """
    Given an index pointer array from an upper triangular CSR matrix, this function
    computes the equivalent indptr for the symmetric matrix and returns also the
    column index of the leftmost element for each row. This is a utility function
    mainly used to help symmetrize upper triangular and block-diagonal CSR matrices with minimal
    memory overhead. The function also supports filtering the matrix by using a boolean mask.

    :param indptr: The index pointer array for the CSR matrix to be symmetrized.
    :param mask: A boolean mask indicating which elements (rows) of the matrix to keep.

    :return: A tuple with the new indptr array and the leftmost column index for each row.
    """

    # Compute the cumulative number of skipped rows:
    cum_skipped_rows = np.cumsum(~mask, dtype=np.int32)

    # Determine the rightmost element for every row:
    rightmost = np.diff(indptr).astype(np.int32) + np.arange(indptr.shape[0] - 1, dtype=np.int32)
    # Update the index by taking into account the number of skipped rows:
    rightmost -= cum_skipped_rows[rightmost]
    # Keep rows indicated by the mask:
    rightmost = rightmost[mask]

    # Get unique boundaries:
    uniq_res = np.unique(rightmost, return_index=True)

    # Loop over the remaining rows to get the leftmost index:
    cdef:
        int32_t curr_row, shape=rightmost.shape[0]
        int32_t[::1] leftmost = np.zeros(shape, dtype=np.int32)
        int32_t[::1] uniq_rightmost = uniq_res[0]
        int32_t[::1] rightmost_first_idx = uniq_res[1].astype(np.int32)
        int32_t rightmost_idx = 0, uniq_rightmost_size = uniq_res[0].shape[0]

    with nogil:
        for curr_row in range(shape):

            if curr_row > uniq_rightmost[rightmost_idx] and rightmost_idx < uniq_rightmost_size - 1:
                rightmost_idx += 1

            leftmost[curr_row] = rightmost_first_idx[rightmost_idx]


    leftmost = np.asarray(leftmost)

    # Compute the new indptr:
    new_indptr = np.zeros(leftmost.shape[0] + 1, dtype=np.int64)
    new_indptr[1:] += rightmost + 1 - leftmost
    np.cumsum(new_indptr, out=new_indptr)

    return new_indptr, leftmost


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef get_symmetrized_indptr(int_dtype[::1] indptr):
    """
    Given an index pointer array from an upper triangular CSR matrix, this function
    computes the equivalent indptr for the symmetric matrix and returns also the
    column index of the leftmost element for each row. This is a utility function
    mainly used to help symmetrize upper triangular and block-diagonal CSR matrices with minimal
    memory overhead.

    :param indptr: The index pointer array for the CSR matrix to be symmetrized.

    :return: A tuple with the new indptr array and the leftmost column index for each row.
    """

    # Determine the rightmost element for every row:
    rightmost = np.diff(indptr).astype(np.int32) + np.arange(indptr.shape[0] - 1, dtype=np.int32)
    # Reuse helper for leftmost indices:
    leftmost = get_leftmost_index(indptr)

    # Compute the new indptr:
    new_indptr = np.zeros(leftmost.shape[0] + 1, dtype=np.int64)
    new_indptr[1:] += rightmost + 1 - leftmost
    np.cumsum(new_indptr, out=new_indptr)

    return new_indptr, leftmost


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef symmetrize_ut_csr_matrix_with_mask(int_dtype[::1] indptr,
                                         noncomplex_numeric[::1] data,
                                         cnp.ndarray[cnp.npy_bool, ndim=1] mask,
                                         noncomplex_numeric diag_fill_value):
    """
    Given an upper triangular CSR matrix, this function symmetrizes it by adding the
    transpose of the upper triangular matrix to itself. This function assumes the following:

        1. The non-zero elements are contiguous along the diagonal of each row (starting from
        the diagonal + 1).
        2. The diagonal elements aren't present in the upper triangular matrix.

    The function also supports filtering the matrix by using a boolean mask.

    :param indptr: The index pointer array for the CSR matrix to be symmetrized.
    :param data: The data array for the CSR matrix to be symmetrized.
    :param mask: A boolean mask indicating which elements (rows) of the matrix to keep.
    :param diag_fill_value: The value to fill the diagonal with (equivalent to 1 for the various data types).

    :return: A tuple with the new data array, the new indptr array, and the leftmost column index for each row.

    """

    new_idx = get_symmetrized_indptr_with_mask(indptr, mask)

    cdef:
        int64_t[::1] new_indptr = new_idx[0]
        int32_t[::1] leftmost_col = new_idx[1]
        int32_t[::1] cum_skipped_rows = np.cumsum(~mask, dtype=np.int32)
        int32_t curr_row, curr_col, curr_row_size, filt_curr_row, filt_curr_col, curr_shape=indptr.shape[0]-1
        int64_t curr_data_idx, new_idx_1, new_idx_2
        noncomplex_numeric[::1] new_data = np.empty_like(data, shape=(new_idx[0][new_indptr.shape[0] - 1], ))


    with nogil:

        # For each row in the current matrix:
        for curr_row in range(curr_shape):
            if mask[curr_row]:

                # Determine the row size for the upper triangular matrix:
                curr_row_size = indptr[curr_row + 1] - indptr[curr_row]

                filt_curr_row = curr_row - cum_skipped_rows[curr_row]

                # First, add the identity to the diagonal
                new_idx_1 = new_indptr[filt_curr_row] + filt_curr_row - leftmost_col[filt_curr_row]
                new_data[new_idx_1] = diag_fill_value

                # Then, add and reflect the off-diagonal entries:
                for curr_col in range(curr_row + 1, curr_row_size + curr_row + 1):
                    if mask[curr_col]:

                        filt_curr_col = curr_col - cum_skipped_rows[curr_col]

                        curr_data_idx = indptr[curr_row] + curr_col - curr_row - 1

                        new_idx_1 = new_indptr[filt_curr_row] + filt_curr_col - leftmost_col[filt_curr_row]
                        new_idx_2 = new_indptr[filt_curr_col] + filt_curr_row - leftmost_col[filt_curr_col]

                        new_data[new_idx_1] = data[curr_data_idx]
                        new_data[new_idx_2] = data[curr_data_idx]

    return np.asarray(new_data), np.asarray(new_idx[0]), np.asarray(new_idx[1])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef symmetrize_ut_csr_matrix(int_dtype[::1] indptr,
                                noncomplex_numeric[::1] data,
                                noncomplex_numeric diag_fill_value):
    """
    Given an upper triangular CSR matrix, this function symmetrizes it by adding the
    transpose of the upper triangular matrix to itself. This function assumes the following:

        1. The non-zero elements are contiguous along the diagonal of each row (starting from
        the diagonal + 1).
        2. The diagonal elements aren't present in the upper triangular matrix.

    :param indptr: The index pointer array for the CSR matrix to be symmetrized.
    :param data: The data array for the CSR matrix to be symmetrized.
    :param diag_fill_value: The value to fill the diagonal with (equivalent to 1 for the various data types).

    :return: A tuple with the new data array, the new indptr array, and the leftmost column index for each row.

    """

    new_idx = get_symmetrized_indptr(indptr)

    cdef:
        int64_t[::1] new_indptr = new_idx[0]
        int32_t[::1] leftmost_col = new_idx[1]
        int32_t curr_row, curr_col, curr_row_size, curr_shape=indptr.shape[0] - 1
        int64_t curr_data_idx, new_idx_1, new_idx_2
        noncomplex_numeric[::1] new_data = np.empty_like(data, shape=(new_idx[0][new_indptr.shape[0] - 1], ))

    with nogil:

        # For each row in the current matrix:
        for curr_row in range(curr_shape):

            # Determine the row size for the upper triangular matrix:
            curr_row_size = indptr[curr_row + 1] - indptr[curr_row]

            # First, add the identity to the diagonal
            new_idx_1 = new_indptr[curr_row] + curr_row - leftmost_col[curr_row]
            new_data[new_idx_1] = diag_fill_value

            # Then, add and reflect the off-diagonal entries:
            for curr_col in range(curr_row + 1, curr_row_size + curr_row + 1):

                curr_data_idx = indptr[curr_row] + curr_col - curr_row - 1

                new_idx_1 = new_indptr[curr_row] + curr_col - leftmost_col[curr_row]
                new_data[new_idx_1] = data[curr_data_idx]

                # To allow for non-square matrices, we need to check if the column index is within bounds:
                if curr_col < curr_shape:
                    new_idx_2 = new_indptr[curr_col] + curr_row - leftmost_col[curr_col]
                    new_data[new_idx_2] = data[curr_data_idx]

    return np.asarray(new_data), np.asarray(new_idx[0]), np.asarray(new_idx[1])


cpdef filter_ut_csr_matrix_inplace(int_dtype[::1] indptr,
                                  noncomplex_numeric[::1] data,
                                  char[::1] bool_mask,
                                  int new_size):
    """
    Given an upper triangular CSR matrix represented by the data and indptr arrays, this function filters
    its entries with a boolean mask.

        1. The non-zero elements are contiguous along the diagonal of each row (starting from
        the diagonal + 1).
        2. The diagonal elements aren't present in the upper triangular matrix.

    !!! warning
        This function modifies the input data array inplace.

    :param indptr: The index pointer array for the CSR matrix to be filtered.
    :param data: The data array for the CSR matrix to be filtered.
    :param bool_mask: A boolean mask indicating which elements (rows) of the matrix to keep.
    :param new_size: The new size of the filtered matrix.

    :return: A tuple with the new filtered data array and the new indptr array.
    """

    cdef:
        int64_t i, curr_row, row_bound, new_indptr_idx = 1, new_data_idx = 0, curr_shape=indptr.shape[0] - 1
        int64_t[::1] new_indptr = np.zeros(new_size + 1, dtype=np.int64)

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
                        data[new_data_idx] = data[i]
                        new_data_idx += 1
                        new_indptr[new_indptr_idx] += 1

                new_indptr_idx += 1

    return np.asarray(data)[:new_data_idx], np.asarray(new_indptr)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef extract_block_from_ld_data(int[::1] ld_left_bound,
                                 int_dtype[::1] indptr,
                                 noncomplex_numeric[::1] data,
                                 int block_start,
                                 int block_end,
                                 floating dq_scale):
    """
    Given LD data in the form of a CSR matrix represented by LDLinearOperator arrays, this function extracts
    a block from the matrix. The block is defined by the start and end indices of the rows to be included.
    This implementation assumes that the non-zero elements are contiguous along the diagonal of each row.

    The function returns a symmetric numpy matrix of size
    (block_end - block_start) x (block_end - block_start) containing the data from this block.

    :param ld_left_bound: The leftmost column index for each row in the matrix.
    :param indptr: The index pointer array for the CSR matrix to be filtered.
    :param data: The data array for the CSR matrix to be filtered.
    :param block_start: The start index of the block to be extracted.
    :param block_end: The end index of the block to be extracted.
    :param dq_scale: The scaling factor to apply to dequantize the data.

    :return: A symmetric numpy matrix containing the data from the block.
    """

    cdef:
        int row_idx, col_idx, row, row_size, col_offset_start, col_offset_end, col_slice_size, block_size = block_end - block_start
        int64_t data_offset
        float[:, ::1] out = np.zeros((block_size, block_size), dtype=np.float32)

    with nogil:

        for row_idx in range(block_size):

            # Determine the row in the original matrix:
            row = block_start + row_idx

            # Determine the size of the row:
            row_size = indptr[row + 1] - indptr[row]

            col_offset_start = row + 1 - ld_left_bound[row]
            col_offset_end = numeric_min(ld_left_bound[row] + row_size, block_end) - ld_left_bound[row]

            col_slice_size = col_offset_end - col_offset_start

            # Compute the offset in the original data array:
            data_offset = indptr[row] + (row + 1) - ld_left_bound[row] + col_offset_start

            # Set the diagonal entry to 1:
            out[row_idx, row_idx] = 1.

            # Set the off-diagonal entries:
            for j in range(col_slice_size):

                # Determine the column index:
                col_idx = row_idx + 1 + j

                # Populate the output array:
                out[row_idx, col_idx] = dq_scale*data[data_offset + j]
                out[col_idx, row_idx] = out[row_idx, col_idx]

    return np.asarray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef slice_ld_data(int[::1] ld_left_bound,
                    int_dtype[::1] indptr,
                    noncomplex_numeric[::1] data,
                    int row_start,
                    int row_end,
                    int col_start,
                    int col_end):
    """
    A general method to slice LD data as represented in LDLinearOperator format, primarily
    the leftmost column index, the index pointer array, and the data array. The function
    take these input arrays and returns equivalent arrays with the sliced data.

    :param ld_left_bound: The leftmost column index for each row in the matrix.
    :param indptr: The index pointer array for the CSR matrix to be filtered.
    :param data: The data array for the CSR matrix to be filtered.
    :param row_start: The start index of the rows to be included.
    :param row_end: The end index of the rows to be included.
    :param col_start: The start index of the columns to be included.
    :param col_end: The end index of the columns to be included.

    :return: A tuple with the sliced leftmost column index,
    the sliced index pointer array, and the sliced data array.
    """

    cdef:
        int row_idx, col_idx, row_size, col_offset_start, col_offset_end, col_slice_size, new_indptr_idx = 1
        int_dtype row_offset
        int64_t new_data_idx = 0
        int32_t[::1] new_ld_left_bound = np.zeros(row_end - row_start, dtype=np.int32)
        int64_t[::1] new_indptr = np.zeros(row_end - row_start + 1, dtype=np.int64)
        noncomplex_numeric[::1] new_data = np.empty_like(data, shape=(indptr[row_end] - indptr[row_start], ))

    with nogil:

        for row_idx in range(row_start, row_end):

            # The row offset in the old data array:
            row_offset = indptr[row_idx]
            # The row size in the old data array:
            row_size = indptr[row_idx + 1] - row_offset

            # Determine the offset for the columns depending on the column slice:
            col_offset_start = numeric_max(ld_left_bound[row_idx], col_start) - ld_left_bound[row_idx]
            col_offset_end = numeric_min(ld_left_bound[row_idx] + row_size, col_end) - ld_left_bound[row_idx]

            col_slice_size = col_offset_end - col_offset_start

            # Update the leftmost column index for the row:
            new_ld_left_bound[new_indptr_idx - 1] = col_offset_start + ld_left_bound[row_idx] - row_start
            # Copy the value from the previous row and add the number of columns in the slice:
            new_indptr[new_indptr_idx] = new_indptr[new_indptr_idx - 1] + col_slice_size

            for col_idx in range(col_offset_start, col_offset_end):
                new_data[new_data_idx] = data[row_offset + col_idx]
                new_data_idx += 1

            new_indptr_idx += 1

    return np.asarray(new_data)[:new_data_idx], np.asarray(new_indptr), np.asarray(new_ld_left_bound)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cpdef expand_ranges(int_dtype[::1] start, int_dtype[::1] end, int64_t output_size):
    """
    Given a set of start and end indices, expand them into one long vector that contains
    the indices between all start and end positions.

    :param start: A vector with the start indices.
    :param end: A vector with the end indices.
    :param output_size: The size of the output vector (equivalent to the sum of the lengths
                        of all ranges).
    """

    cdef:
        int_dtype i, j, size=start.shape[0]
        int64_t out_idx = 0
        int_dtype[::1] output = np.empty_like(start, shape=(output_size, ))

    with nogil:
        for i in range(size):
            for j in range(start[i], end[i]):
                output[out_idx] = j
                out_idx += 1

    return np.asarray(output)


# -------------------------------------------------------------------------
# LD Block Boundaries (for LD matrix computation)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cpdef find_ld_block_boundaries(
    const int_dtype[:] pos,
    int[:, :] block_boundaries
):
    """
    Find the LD boundaries for the blockwise estimator of LD, i.e., the
    indices of the leftmost and rightmost neighbors for each SNP.

    :param pos: A vector with the position of each genetic variant.
    :param block_boundaries: A matrix with the boundaries of each LD block.
    """

    cdef:
        int i, j, ldb_idx, B = block_boundaries.shape[0], M = pos.shape[0]
        int_dtype block_start, block_end
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
cpdef find_windowed_ld_boundaries(
    const floating[::1] pos,
    double max_dist
):
    """
    Find the LD boundaries for the windowed estimator of LD, i.e., the
    indices of the leftmost and rightmost neighbors for each SNP.

    .. note::
        To match plink's behavior, the bounds here are inclusive, i.e.,
        if the distance between two SNPs is exactly equal to the maximum distance,
        they are considered neighbors.

    :param pos: A vector with the position of each genetic variant.
    :param max_dist: The maximum distance between SNPs to consider them neighbors.
    """

    cdef:
        int i, j, M = pos.shape[0]
        int[::1] v_min = np.zeros_like(pos, dtype=np.int32)
        int[::1] v_max = M*np.ones_like(pos, dtype=np.int32)

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
cpdef find_shrinkage_ld_boundaries(const floating[::1] cm_pos,
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
        int[::1] v_min = np.zeros_like(cm_pos, dtype=np.int32)
        int[::1] v_max = M*np.ones_like(cm_pos, dtype=np.int32)

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
