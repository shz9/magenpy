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
