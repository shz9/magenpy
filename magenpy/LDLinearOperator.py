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
    :param diag_add: A scalar or vector of the same shape as the matrix with quantities to add to
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
                 diag_add=None,
                 dtype='float32',
                 include_diag=True,
                 lower_triangle=True,
                 upper_triangle=True,
                 threads=1):
        """
        Initialize an LDLinearOperator object by passing the index pointer array and data array.

        :param ld_indptr: The index pointer array for the CSR matrix.
        :param ld_data: The data array for the CSR matrix.
        :param diag_add: A scalar or vector of the same shape as the matrix with quantities to add to
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

        self.diag_add = None
        self.set_diag_add(diag_add)

        self.include_diag = include_diag
        self.upper_triangle = upper_triangle
        self.lower_triangle = lower_triangle
        self.threads = threads

    @property
    def dequantization_scale(self):
        """
        :return: The dequantization scale for the LD data (if quantized to integer data types).
        If the data is not quantized, returns 1.
        """
        if np.issubdtype(self.ld_data.dtype, np.integer):
            return 1./np.iinfo(self.ld_data.dtype).max
        else:
            return 1.

    def set_diag_add(self, diag_add):
        """
        Set the elements of the diagonal matrix to be added to the LD matrix.

        :param diag_add: A scalar or vector of the same shape as the matrix with quantities to add to
        the diagonal of the matrix being represented by the linear operator.

        """

        if diag_add is not None:
            assert np.isscalar(diag_add) or diag_add.shape[0] == self.shape[0]

        self.diag_add = diag_add

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

        if self.diag_add is not None:
            out_vec += x*self.diag_add

        if flatten:
            return out_vec.reshape(-1, 1)
        else:
            return out_vec

    def _rmatvec(self, x):
        return self._matvec(x)
