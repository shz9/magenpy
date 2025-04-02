from scipy.sparse.linalg import LinearOperator
import numpy as np


class LDLinearOperator(LinearOperator):
    """
    A class that represents a LinearOperator to facilitate performing matrix-vector
    multiplication with a Linkage-Disequilibrium (LD) matrix without representing the
    entire matrix in memory. The object is initialized with the data array of an LD matrix,
    the index pointer array, and other parameters to specify the behavior of the operator.

    For instance, you may customize the operator to exclude the diagonal entries of the LD
    matrix by setting `include_diag=False`, or exclude the lower triangle by setting
    `lower_triangle=False`.

    .. note::
        The LD linear operator assumes that the underlying LD matrix is square and symmetric.
        Currently, does not support non-square matrices (which may arise in certain situations).
        This is left for future work.

    :ivar ld_indptr: The index pointer array for the CSR matrix.
    :ivar ld_data: The data array for the CSR matrix (supported data types are float32, float64, int8, int16).
    :param diag_shift: A scalar or vector of the same shape as the matrix with quantities to add to
        the diagonal of the matrix being represented by the linear operator. This is used to support
        cases where a diagonal matrix needs to be added before the matrix-vector multiplication.
    :ivar dtype: The data type for the entries of the LD matrix.
    :ivar shape: The shape of the LD matrix.
    :ivar include_diag: If True, the diagonal entries of the LD matrix are included in the computation.
    :ivar lower_triangle: If True, the lower triangle of the LD matrix is included in the computation.
    :ivar upper_triangle: If True, the upper triangle of the LD matrix is included in the computation.
    :ivar threads: The number of threads to use for the computation (experimental).

    """
    def __init__(self,
                 ld_indptr,
                 ld_data,
                 leftmost_idx,
                 shape=None,
                 symmetric=False,
                 diag_shift=None,
                 dtype='float32',
                 include_diag=True,
                 lower_triangle=True,
                 upper_triangle=True,
                 threads=1):
        """
        Initialize an LDLinearOperator object by passing the index pointer array and data array.

        :param ld_indptr: The index pointer array for the CSR matrix.
        :param ld_data: The data array for the CSR matrix.
        :param diag_shift: A scalar or vector of the same shape as the matrix with quantities to add to
        the diagonal of the matrix being represented by the linear operator.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64.
        and integer quantized data types int8 and int16).
        :param include_diag: If True, the diagonal entries of the LD matrix are included in the computation.
        :param lower_triangle: If True, the lower triangle of the LD matrix is included in the computation.
        :param upper_triangle: If True, the upper triangle of the LD matrix is included in the computation.
        :param threads: The number of threads to use for the computation (experimental).

        """

        super().__init__(dtype=np.dtype(dtype), shape=(ld_indptr.shape[0] - 1, ld_indptr.shape[0] - 1))

        self.ld_indptr = ld_indptr
        self.ld_data = ld_data
        self.leftmost_idx = leftmost_idx
        self.shape = shape or (ld_indptr.shape[0] - 1, ld_indptr.shape[0] - 1)
        self.dtype = np.dtype(dtype)

        # Initialize the diagonal shift term:
        self.diag_shift = None
        self.set_diag_shift(diag_shift)

        # Set the behavior of the operator:
        self.symmetric = symmetric

        if symmetric:
            self.include_diag = True
            self.lower_triangle = True
            self.upper_triangle = True
        else:
            self.include_diag = include_diag
            self.upper_triangle = upper_triangle
            self.lower_triangle = lower_triangle

        # Set the number of threads to use for the computation:
        self.threads = threads

    @property
    def ld_data_type(self) -> np.dtype:
        """
        :return: The data type of the LD data.
        """
        return self.ld_data.dtype

    @property
    def dequantization_scale(self) -> float:
        """
        :return: The dequantization scale for the LD data (if quantized to integer data types).
        If the data is not quantized, returns 1.
        """
        if np.issubdtype(self.ld_data_type, np.integer):
            return 1./np.iinfo(self.ld_data.dtype).max
        else:
            return 1.

    @property
    def is_square(self) -> bool:
        """
        :return: True if the LD matrix is square, False otherwise.
        """
        return self.shape[0] == self.shape[1]

    def set_diag_shift(self, diag_shift):
        """
        Set the elements of the diagonal matrix to be added to the LD matrix.

        :param diag_shift: A scalar or vector of the same shape as the matrix with quantities to add to
        the diagonal of the matrix being represented by the linear operator.
        """

        if diag_shift is not None:
            assert np.isscalar(diag_shift) or diag_shift.shape[0] == self.shape[0]

        if self.diag_shift is None or diag_shift is None:
            self.diag_shift = diag_shift
        else:
            self.diag_shift += diag_shift

    def _matvec(self, x):
        """
        Perform matrix-vector multiplication with the LD matrix.

        :param x: The input vector to multiply with the LD matrix.
        :return: A numpy array representing the result of the matrix-vector multiplication.
        """

        assert self.is_square or self.symmetric, "Only square or symmetric matrices are supported for now."
        assert x.shape[0] == self.shape[1]

        from .stats.ld.c_utils import ld_dot, ut_ld_dot

        flatten = len(x.shape) > 1

        if self.symmetric:
            out_vec = ld_dot(self.leftmost_idx,
                             self.ld_indptr,
                             self.ld_data,
                             x.flatten() if flatten else x,
                             x.dtype.type(self.dequantization_scale),
                             self.threads)
        else:
            out_vec = ut_ld_dot(self.ld_indptr,
                                self.ld_data,
                                x.flatten() if flatten else x,
                                x.dtype.type(self.dequantization_scale),
                                self.lower_triangle,
                                self.upper_triangle,
                                self.include_diag,
                                self.threads)

        if self.diag_shift is not None:
            out_vec += x*self.diag_shift

        if flatten:
            return out_vec.reshape(-1, 1)
        else:
            return out_vec

    def _rmatvec(self, x):
        return self._matvec(x)

    def rank_one_update(self, x, alpha=None, inplace=False):
        """
        Perform a rank-one update/perturbation on the LD matrix by
        adding the outer product of a vector x with the matrix itself:

        LD = LD + alpha * x * x^T

        Note that this performs the update on the active (non-zero) entries of
        the matrix only!

        !!! warning
            Only supported for square matrices for now.

        :param x: The vector to use for the rank-one update.
        :param alpha: The scaling factor for the rank-one update.
        :param inplace: If True, the operation is performed in-place.

        :return: A new `LDLinearOperator` object representing the updated matrix.
        """

        if len(x.shape) > 1:
            x = x.flatten()

        assert self.is_square
        assert x.shape[0] == self.shape[0]

        # If the user requests to do the update in-place, make sure that
        # the LD data is floating data type and the input vector is cast to
        # the same floating data type as the data:
        if inplace:
            assert np.issubdtype(self.ld_data_type, np.floating)
            x = x.astype(self.ld_data_type)
            out = self.ld_data
        else:
            out = np.zeros_like(self.ld_data, dtype=x.dtype)

        alpha = alpha or 1.

        from .stats.ld.c_utils import rank_one_update

        rank_one_update(
            self.leftmost_idx,
            self.ld_indptr,
            self.ld_data,
            x,
            out,
            alpha,
            self.dequantization_scale,
            self.threads
        )

        # Adjustment to the diagonal:
        diag_shift = alpha * x * x

        if inplace:
            self.set_diag_shift(diag_shift)
            return self
        else:
            new_ld_lop = LDLinearOperator(
                self.ld_indptr,
                out,
                self.leftmost_idx,
                diag_shift=self.diag_shift,
                dtype=self.dtype,
                include_diag=self.include_diag,
                lower_triangle=self.lower_triangle,
                upper_triangle=self.upper_triangle,
                threads=self.threads
            )

            new_ld_lop.set_diag_shift(diag_shift)

            return new_ld_lop

    def to_csr(self, keep_sparse=False):
        """
        Convert the LDLinearOperator object to a `scipy.csr_matrix` object that contains the same
        data and structure as the original LD matrix.

        :param keep_sparse: If True, the data is kept in the original sparse format, meaning that
        the CSR matrix will also be upper triangular and the data will remain quantized if it was
        quantized in the original LD matrix

        :return: A `scipy.csr_matrix` object representing the LD matrix.
        """

        if not keep_sparse and not self.symmetric and self.lower_triangle:
            from .stats.ld.c_utils import symmetrize_ut_csr_matrix

            if np.issubdtype(self.ld_data_type, np.integer):
                fill_val = np.iinfo(self.ld_data_type).max
            else:
                fill_val = 1.

            data, indptr, leftmost_idx = symmetrize_ut_csr_matrix(self.ld_indptr, self.ld_data, fill_val)
        else:
            data = self.ld_data
            indptr = self.ld_indptr
            leftmost_idx = self.leftmost_idx

        # ---------------------------
        # Step 2: Dequantize the LD data (if needed):
        if not keep_sparse:
            dtype = self.dtype
            if np.issubdtype(self.ld_data_type, np.integer):
                from .utils.model_utils import dequantize
                data = dequantize(data, float_dtype=self.dtype)
            else:
                data = data.astype(self.dtype, copy=False)
        else:
            dtype = self.ld_data_type

        # ---------------------------
        # Step 3: Prepare indices for the CSR data structure:
        from .stats.ld.c_utils import expand_ranges
        from scipy.sparse import csr_matrix, identity, diags

        indices = expand_ranges(leftmost_idx,
                                (np.diff(indptr) + leftmost_idx).astype(np.int32),
                                data.shape[0])

        mat = csr_matrix(
            (
                data,
                indices,
                indptr
            ),
            shape=self.shape,
            dtype=dtype
        )

        # ---------------------------
        # Step 4: Add the diagonal shift (if instantiated):

        if not keep_sparse and self.diag_shift is not None:

            if np.isscalar(self.diag_shift):
                mat += identity(self.shape[0], format='csr', dtype=self.dtype) * self.diag_shift
            else:
                mat += diags(self.diag_shift, format='csr', dtype=self.dtype)

        return mat

    def __getitem__(self, item):
        """
        Extract a sub-matrix from the LD matrix represented by the linear operator.
        This method supports both row and column slicing.

        :param item: The index or slice to extract from the LD matrix. If a tuple is passed, the first
        element is the row index or slice, and the second element is the column index or slice. If a single
        element is passed, then we just slice the rows.

        :return: A new `LDLinearOperator` object representing the extracted sub-matrix.
        """

        start_row = 0
        end_row = self.shape[0]
        start_col = 0
        end_col = self.shape[1]

        def check_normalize_slice(in_slice, dim=0):
            # Check that the slice is with step 1:
            assert in_slice.step is None or in_slice.step == 1, "Slice with step is not supported."
            # Check that the start/end are within bounds:
            assert in_slice.start is None or 0 <= in_slice.start < self.shape[dim], "Index out of bounds."
            assert in_slice.stop is None or 0 < in_slice.stop <= self.shape[dim], "Index out of bounds."

            return in_slice.start or 0, in_slice.stop or self.shape[dim]

        def check_normalize_index(index, dim=0):
            assert 0 <= index < self.shape[dim], "Index out of bounds."
            return index, index + 1

        if isinstance(item, tuple):

            # --- Row slicing ---
            if isinstance(item[0], slice):
                start_row, end_row = check_normalize_slice(item[0])
            elif isinstance(item[0], int):
                start_row, end_row = check_normalize_index(item[0])
            else:
                raise ValueError("Invalid row index.")

            # --- Column slicing ---
            if isinstance(item[1], slice):
                start_col, end_col = check_normalize_slice(item[1], dim=1)
            elif isinstance(item[1], int):
                start_col, end_col = check_normalize_index(item[1], dim=1)
            else:
                raise ValueError("Invalid column index.")
        elif isinstance(item, slice):
            start_row, end_row = check_normalize_slice(item)
        elif isinstance(item, int):
            start_row, end_row = check_normalize_index(item)
        else:
            raise ValueError("Invalid slice/index for the LD matrix.")

        new_shape = (end_row - start_row, end_col - start_col)

        if new_shape == self.shape:
            return self

        from .stats.ld.c_utils import slice_ld_data

        new_data, new_indptr, new_leftmost_idx = slice_ld_data(
            self.leftmost_idx,
            self.ld_indptr,
            self.ld_data,
            start_row,
            end_row,
            start_col,
            end_col
        )

        return LDLinearOperator(
            new_indptr,
            new_data,
            new_leftmost_idx,
            shape=new_shape,
            symmetric=self.symmetric,
            diag_shift=None,
            dtype=self.dtype,
            include_diag=self.include_diag,
            lower_triangle=self.lower_triangle,
            upper_triangle=self.upper_triangle,
            threads=self.threads
        )

    def to_numpy(self, block_start=None, block_end=None):
        """
        Extract a square numpy matrix from the LD matrix represented by the linear operator.

        :param block_start: The starting index of the square sub-matrix.
        :param block_end: The ending index of the squared sub-matrix.

        :return: A numpy matrix representing the extracted block matrix.
        """

        # Process block start:
        block_start = block_start or 0
        block_start = max(block_start, 0)

        # Process block end:
        block_end = block_end or self.shape[0]
        block_end = min(block_end, self.shape[0])

        # Sanity checks:
        assert block_start >= 0 and block_end <= self.shape[0]
        assert block_start < block_end

        from .stats.ld.c_utils import extract_block_from_ld_data

        return extract_block_from_ld_data(
            self.leftmost_idx,
            self.ld_indptr,
            self.ld_data,
            block_start,
            block_end,
            self.dequantization_scale
        )

    def getrow(self, row_idx, symmetric=False, return_indices=False):
        """
        Extract a row from the LD matrix represented by the linear operator.

        :param row_idx: The index of the row to extract.
        :param symmetric: If True, the row is extracted from the symmetric part of the matrix.
        :param return_indices: If True, the indices of the non-zero entries are also returned.

        :return: A numpy array representing the extracted row.
        """

        if symmetric:
            if self.symmetric:
                data = self.ld_data[self.ld_indptr[row_idx]:self.ld_indptr[row_idx + 1]]
            else:
                raise NotImplementedError("Symmetric extraction is only supported for symmetric matrices for now.")
        else:
            if self.symmetric:
                offset = row_idx - self.leftmost_idx[row_idx] + 1
                data = self.ld_data[self.ld_indptr[row_idx] + offset:self.ld_indptr[row_idx + 1]]
            else:
                data = self.ld_data[self.ld_indptr[row_idx]:self.ld_indptr[row_idx + 1]]

        if return_indices:
            if symmetric:
                indices = np.arange(self.leftmost_idx[row_idx], self.leftmost_idx[row_idx] + len(data))
            else:
                indices = np.arange(row_idx + 1, row_idx + len(data))
            return data, indices
        else:
            return data
