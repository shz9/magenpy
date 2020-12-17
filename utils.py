from scipy.sparse import linalg as splinalg
from rechunker import rechunk
import scipy.sparse as ss
import sparse
import dask.array as da
import numpy as np
import zarr
import numcodecs
import psutil
import errno
import os
import glob
import collections
import six

from c_utils import zarr_islice


def makedir(cdir):
    try:
        os.makedirs(cdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_filenames(path, extension=None):

    if os.path.isdir(path):
        if extension == '.zarr':
            if os.path.isfile(os.path.join(path, '.zarray')):
                return [path]
            else:
                return glob.glob(os.path.join(path, '*/'))
        return glob.glob(os.path.join(path, '*'))
    else:
        if extension is None:
            return glob.glob(path + '*')
        elif extension in path:
            return [path]
        elif os.path.isfile(path + extension):
            return [path + extension]
        else:
            return glob.glob(path + '*' + extension)


def iterable(arg):
    return (
        isinstance(arg, collections.Iterable)
        and not isinstance(arg, six.string_types)
    )


def sparsify_chunked_matrix(arr, bounds):
    """
    A utility to sparsify chunked matrices
    :param arr: the LD matrix
    :param bounds: an 2xM array of start and end position for each row
    :return: A sparsified array of the same format
    """

    def update_prev_chunk(j):
        chunk_start = (j - 1) - (j - 1) % chunk_size
        chunk_end = chunk_start + chunk_size
        arr[chunk_start:chunk_end] = chunk

    chunk_size = arr.chunks[0]
    chunk = None

    for j in range(bounds.shape[1]):
        if j % chunk_size == 0:
            if j > 0:
                update_prev_chunk(j)

            chunk = arr[j: j + chunk_size]

        chunk[j % chunk_size, :bounds[0, j]] = 0
        chunk[j % chunk_size, bounds[1, j]:] = 0

    update_prev_chunk(j)

    return arr


def rechunk_zarr(arr, target_chunks, target_store, intermediate_store='temp/intermediate_ld_rechunk.zarr', **kwargs):

    if os.path.isdir(target_store):
        try:
            z = zarr.open(target_store)
            z.store.rmdir()
        except Exception as e:
            raise e

    rechunked = rechunk(arr,
                        target_chunks=target_chunks,
                        target_store=target_store,
                        temp_store=intermediate_store,
                        max_mem=psutil.virtual_memory().available / psutil.cpu_count(),
                        **kwargs)

    try:
        rechunked.execute()
        # Delete the older stores:
        zarr.open(intermediate_store).store.rmdir()
        arr.store.rmdir()
    except Exception as e:
        raise e

    return zarr.open(target_store)


def zarr_to_ragged(z, bounds):
    """
    TODO: Figure out a way to automatically reconfigure the chunking of the
    ragged array to optimize performance (Maybe aim for ~50-100MB per chunk).
    rough idea: approximate size of new data structure would be :

    n_rows * avg(n_cols) * dtype / 8 / 1024. ** sizes[out]
    https://gist.github.com/dimalik/f4609661fb83e3b5d22e7550c1776b90

    divide by pre-specified chunk MB size to obtain # of the chunks.

    NOTE: May need to re-write this method if we change chunk size.

    :param z:
    :param bounds:
    :return:
    """

    dir_store = os.path.join(os.path.dirname(z.chunk_store.path) + '_ragged',
                             os.path.basename(z.chunk_store.path))

    z_rag = zarr.open(dir_store, mode='w',
                      shape=z.shape[0], chunks=z.chunks[:1], dtype=object,
                      object_codec=numcodecs.VLenArray(float))

    chunk_size = z.chunks[0]

    for i in range(int(np.ceil(z.shape[0] / chunk_size))):

        start = i * chunk_size
        end = min((i + 1) * chunk_size, z.shape[0])

        z_chunk = z[start: end]
        r_chunk = z_rag[start: end]

        for j in range(z_chunk.shape[0]):
            r_chunk[j] = z_chunk[j][bounds[0, start + j]:bounds[1, start + j]]

        z_rag[start: end] = r_chunk

    z_rag.attrs.update(z.attrs.asdict())

    return z_rag


def zarr_to_sparse(mat, to_csr=True):

    d_mat = da.from_zarr(mat)
    sp_mat = d_mat.map_blocks(sparse.COO).compute()

    if to_csr:
        return sp_mat.tocsr()
    else:
        return sp_mat


def sparse_cholesky(A):
    """
    from: https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d
    """

    n = A.shape[0]
    LU = splinalg.splu(A, diag_pivot_thresh=0)  # sparse LU decomposition

    # check the matrix A is positive definite.

    if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():
        return LU.L.dot(ss.diags(LU.U.diagonal() ** 0.5))
    else:
        raise Exception('Matrix is not positive definite')
