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
    TODO: Implement function to find tagging variants.
    """
    pass

cpdef prune_ld_ut(int_dtype[::1] indptr,
                  noncomplex_numeric[::1] data,
                  noncomplex_numeric r_threshold):
    """
    Pass over the LD matrix once and prune it so that variants whose absolute correlation coefficient is above 
    or equal to a certain threshold are filtered away. If two variants are highly correlated, 
    this function keeps the one that occurs earlier in the matrix. 
    
    This function works with LD matrices in any data type 
    (quantized to integers or floats), but it is the user's responsibility to set the appropriate 
    threshold for the data type used.
    
    !!! note 
        This function assumes that the LD matrix is in upper triangular form and doesn't include the 
        diagonal. We will try to generalize this implementation later.
    
    :param indptr: The index pointer array for the CSR matrix to be pruned.
    :param data: The data array for the CSR matrix to be pruned.
    :param r_threshold: The Pearson Correlation coefficient threshold above which to prune variants.
    
    :return: An boolean array of which variants are kept after pruning.
    """

    cdef:
        int64_t i, curr_row, curr_row_size, curr_data_idx, curr_shape=indptr.shape[0]-1
        char[::1] keep = np.ones(curr_shape, dtype=np.int8)

    with nogil:
        for curr_row in range(curr_shape):

            if keep[curr_row] == 1:

                curr_row_size = indptr[curr_row + 1] - indptr[curr_row]

                for i in range(curr_row_size):
                    curr_data_idx = indptr[curr_row] + i

                    if numeric_abs(data[curr_data_idx]) >= r_threshold:
                        keep[curr_row + i + 1] = 0

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
cpdef find_ld_block_boundaries(int_dtype[:] pos, int[:, :] block_boundaries):
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
cpdef find_windowed_ld_boundaries(floating[::1] pos, double max_dist):
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
cpdef find_shrinkage_ld_boundaries(floating[::1] cm_pos,
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
