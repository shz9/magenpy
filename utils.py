from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import numpy as np
import sys
import collections
import six


def iterable(arg):
    return (
        isinstance(arg, collections.Iterable)
        and not isinstance(arg, six.string_types)
    )


def sparsify_chunked_matrix(arr, arr_ranges):
    """
    A utility to sparsifying chunked matrices
    :param arr: the LD matrix
    :param arr_ranges: an 2xM array of start and end position for each row
    :return: A sparsified array of the same format
    """

    def update_prev_chunk(j):
        chunk_start = (j - 1) - (j - 1) % chunk_size
        chunk_end = chunk_start + chunk_size
        arr[chunk_start:chunk_end] = chunk

    chunk_size = arr.chunks[0]
    chunk = None

    for j in range(arr_ranges.shape[1]):
        if j % chunk_size == 0:
            if j > 0:
                update_prev_chunk(j)

            chunk = arr[j: j + chunk_size]

        chunk[j % chunk_size, :arr_ranges[0, j]] = 0
        chunk[j % chunk_size, arr_ranges[1, j]:] = 0

    update_prev_chunk(j)

    return arr


def sparse_cholesky(A):
    """
    from: https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d
    """

    n = A.shape[0]
    LU = splinalg.splu(A, diag_pivot_thresh=0)  # sparse LU decomposition

    # check the matrix A is positive definite.

    if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():
        return LU.L.dot(sparse.diags(LU.U.diagonal() ** 0.5))
    else:
        raise Exception('Matrix is not positive definite')
