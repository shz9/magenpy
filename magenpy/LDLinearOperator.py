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

        self.ld_indptr = ld_indptr
        self.ld_data = ld_data
        self.shape = (ld_indptr.shape[0] - 1, ld_indptr.shape[0] - 1)
        self.dtype = np.dtype(dtype)

        self.diag_shift = None
        self.set_diag_shift(diag_shift)

        self.include_diag = include_diag
        self.upper_triangle = upper_triangle
        self.lower_triangle = lower_triangle
        self.threads = threads

    @property
    def ld_data_type(self):
        """
        :return: The data type of the LD data.
        """
        return self.ld_data.dtype

    @property
    def dequantization_scale(self):
        """
        :return: The dequantization scale for the LD data (if quantized to integer data types).
        If the data is not quantized, returns 1.
        """
        if np.issubdtype(self.ld_data_type, np.integer):
            return 1./np.iinfo(self.ld_data.dtype).max
        else:
            return 1.

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

        assert x.shape[0] == self.shape[0]

        from .stats.ld.c_utils import ld_dot

        flatten = len(x.shape) > 1

        out_vec = ld_dot(self.ld_indptr,
                         self.ld_data,
                         x.flatten() if flatten else x,
                         x.dtype.type(self.dequantization_scale),
                         self.lower_triangle,
                         self.upper_triangle,
                         self.include_diag,
                         self.threads
                         )

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

        :param x: The vector to use for the rank-one update.
        :param alpha: The scaling factor for the rank-one update.
        :param inplace: If True, the operation is performed in-place.

        :return: A new `LDLinearOperator` object representing the updated matrix.
        """

        if len(x.shape) > 1:
            x = x.flatten()

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
            np.arange(1, self.shape[0] + 1, dtype=np.int32),
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
                diag_shift=self.diag_shift,
                dtype=self.dtype,
                include_diag=self.include_diag,
                lower_triangle=self.lower_triangle,
                upper_triangle=self.upper_triangle,
                threads=self.threads
            )

            new_ld_lop.set_diag_shift(diag_shift)

            return new_ld_lop

    def to_csr(self):
        """
        Convert the LDLinearOperator to a symmetric CSR matrix.

        !!! warning
            Use this only for testing/debugging purposes on small matrices.

        :return: The CSR matrix.
        """

        # Step 1: Symmetrize the LD matrix:

        from .stats.ld.c_utils import symmetrize_ut_csr_matrix

        if np.issubdtype(self.ld_data_type, np.integer):
            fill_val = np.iinfo(self.ld_data_type).max
        else:
            fill_val = 1.

        data, indptr, leftmost_idx = symmetrize_ut_csr_matrix(self.ld_indptr, self.ld_data, fill_val)

        # Step 2: Dequantize the LD data (if needed):
        if np.issubdtype(self.ld_data_type, np.integer):
            from .utils.model_utils import dequantize
            data = dequantize(data, float_dtype=self.dtype)

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
            dtype=self.dtype
        )

        # Step 4: Add the diagonal shift (if instantiated):

        if self.diag_shift is not None:

            if np.isscalar(self.diag_shift):
                mat += identity(self.shape[0], format='csr', dtype=self.dtype) * self.diag_shift
            else:
                mat += diags(self.diag_shift, format='csr', dtype=self.dtype)

        return mat
