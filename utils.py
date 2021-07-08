from rechunker import rechunk
import pandas as pd
import numpy as np
import zarr
import numcodecs
import psutil
import dask.array as da
import errno
import os
import subprocess
import glob
import collections
import six


def read_snp_filter_file(filename, snp_id_col=0):

    try:
        keep_list = pd.read_csv(filename, sep="\t", header=None).values[:, snp_id_col]
    except Exception as e:
        raise e

    return keep_list


def read_individual_filter_file(filename, iid_col=1):

    try:
        keep_list = pd.read_csv(filename, sep="\t", header=None).values[:, iid_col]
    except Exception as e:
        raise e

    return keep_list


def standardize_genotype_matrix(g_mat, fill_na=True):

    sg_mat = (g_mat - g_mat.mean(axis=0)) / g_mat.std(axis=0)

    if fill_na:
        sg_mat = sg_mat.fillna(0.)

    return sg_mat


def get_shared_distance_matrix(tree, tips=None):
    """
    This function takes a Biopython tree and returns the
    shared distance matrix (time to most recent common ancestor - MRCA)
    """

    tips = tree.get_terminals() if tips is None else tips
    n_tips = len(tips)  # Number of terminal species
    sdist_matrix = np.zeros((n_tips, n_tips))  # Shared distance matrix

    for i in range(n_tips):
        for j in range(i, n_tips):
            if i == j:
                sdist_matrix[i, j] = tree.distance(tree.root, tips[i])
            else:
                mrca = tree.common_ancestor(tips[i], tips[j])
                sdist_matrix[i, j] = sdist_matrix[j, i] = tree.distance(tree.root, mrca)

    return sdist_matrix


def tree_to_rho(tree, min_corr):
    """
    This function takes a Biopython tree and a minimum correlation
    parameter and returns the correlation matrix for the effect sizes
    across populations.

    :param tree: a Biopython Phylo object
    :param min_corr: minimum correlation
    :return:
    """

    max_depth = max(tree.depths().values())
    tree.root.branch_length = min_corr*max_depth / (1. - min_corr)
    max_depth = max(tree.depths().values())

    for c in tree.find_clades():
        c.branch_length /= (max_depth)

    return tree.root.branch_length + get_shared_distance_matrix(tree)


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


def shrink_ld_matrix(arr, cm_dist, genmap_Ne, genmap_sample_size, shrinkage_cutoff=1e-3):

    # The multiplicative factor for the shrinkage estimator
    mult_factor = 2.*genmap_Ne / genmap_sample_size

    def update_prev_chunk(j):
        chunk_start = (j - 1) - (j - 1) % chunk_size
        chunk_end = chunk_start + chunk_size
        arr[chunk_start:chunk_end] = chunk

    chunk_size = arr.chunks[0]
    chunk = None

    for j in range(arr.shape[0]):

        if j % chunk_size == 0:
            if j > 0:
                update_prev_chunk(j)

            chunk = arr[j: j + chunk_size]

        # Compute the shrinkage factor the entries in row j
        shrink_mult = np.exp(-mult_factor * np.abs(cm_dist - cm_dist[j]))
        # Set any shrinkage factor below the cutoff value to zero:
        shrink_mult[shrink_mult < shrinkage_cutoff] = 0.

        # Shrink the entries of the LD matrix:
        chunk[j % chunk_size] *= shrink_mult

    update_prev_chunk(j)

    return arr


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


def rechunk_zarr(arr, target_chunks, target_store, intermediate_store, **kwargs):

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
                        max_mem="128MiB",
                        **kwargs)

    try:
        rechunked.execute()
        # Delete the older stores:
        zarr.open(intermediate_store).store.rmdir()
        arr.store.rmdir()
    except Exception as e:
        raise e

    return zarr.open(target_store)


def optimize_chunks_for_memory(chunked_array, cpus=None, max_mem=None):
    """
    Modified from: Sergio Hleap
    Determine optimal chunks that fit in max_mem. Max_mem should be numerical in GiB
    """

    if cpus is None:
        cpus = psutil.cpu_count()

    if max_mem is None:
        max_mem = psutil.virtual_memory().available / (1024.0 ** 3)

    chunk_mem = max_mem / cpus
    chunks = da.core.normalize_chunks(f"{chunk_mem}GiB", shape=chunked_array.shape, dtype=chunked_array.dtype)

    return chunked_array.chunk(chunks)


def estimate_row_chunk_size(rows, cols, dtype=np.float64, chunk_size=128):
    """

    :param rows: Number of rows.
    :param cols: Number of columns. If a ragged array, provide average size of arrays
    :param dtype: data type
    :param chunk_size: chunk size in MB
    :return:
    """

    matrix_size = rows * cols * np.dtype(dtype).itemsize / 1024 ** 2
    n_chunks = matrix_size // chunk_size

    if n_chunks < 1:
        return None, None
    else:
        return int(rows / n_chunks), None


def zarr_to_ragged(z, keep_snps=None, bounds=None, rechunk=True):
    """
    This function takes a chunked Zarr matrix (e.g. sparse LD matrix)
    and returns a ragged array matrix.
    The function allows filtering down the original matrix by passing
    a list of SNPs to keep. It also allows the user to re-chunk
    the ragged array for optimized read/write performance.

    :param z: the original Zarr matrix (implementation assumes 2D matrix)
    :param keep_snps: A list of SNP IDs to keep.
    :param rechunk: Whether to re-chunk the ragged array (for optimized read/write performance)
    :return:
    """

    dir_store = os.path.join(os.path.dirname(z.chunk_store.path) + '_ragged',
                             os.path.basename(z.chunk_store.path))

    if keep_snps is None:
        n_rows = z.shape[0]

        idx_map = pd.DataFrame({'SNP': z.attrs['SNPs']}).reset_index()
        idx_map.columns = ['index_x', 'SNP']
        idx_map['index_y'] = idx_map['index_x']

    else:
        idx_map = pd.DataFrame({'SNP': keep_snps}).reset_index().merge(
            pd.DataFrame({'SNP': z.attrs['SNPs']}).reset_index(),
            on='SNP',
            suffixes=('_y', '_x')
        )
        idx_map['chunk_x'] = (idx_map['index_x'] // z.chunks[0]).astype(int)
        n_rows = len(keep_snps)

    idx_map['chunk_x'] = (idx_map['index_x'] // z.chunks[0]).astype(int)

    if bounds is None:
        bounds = np.array(z.attrs['LD Boundaries'])

    if rechunk:
        avg_ncol = int((bounds[1, :] - bounds[0, :]).mean())
        n_chunks = estimate_row_chunk_size(n_rows, avg_ncol)
    else:
        n_chunks = z.chunks

    z_rag = zarr.open(dir_store, mode='w',
                      shape=n_rows,
                      chunks=n_chunks[:1],
                      dtype=object,
                      object_codec=numcodecs.VLenArray(float))

    z_rag_mem = z_rag[:]

    chunk_size = z.chunks[0]

    for i in range(int(np.ceil(z.shape[0] / chunk_size))):

        start = i * chunk_size
        end = min((i + 1) * chunk_size, z.shape[0])

        z_chunk = z[start: end]

        for _, (k, _, j, _) in idx_map.loc[idx_map['chunk_x'] == i].iterrows():
            if keep_snps is None:
                z_rag_mem[k] = z_chunk[j - start][bounds[0, j]:bounds[1, j]]
            else:
                z_rag_mem[k] = z_chunk[j - start, idx_map['index_x']][bounds[0, k]:bounds[1, k]]

    z_rag[:] = z_rag_mem
    z_rag.attrs.update(z.attrs.asdict())

    if keep_snps is not None:
        z_rag.attrs['SNPs'] = list(keep_snps)
        z_rag.attrs['LD Boundaries'] = bounds.tolist()

    return z_rag


def run_shell_script(cmd):

    result = subprocess.run(cmd, shell=True, capture_output=True)

    if result.stderr:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=result.args,
            stderr=result.stderr
        )

    return result


def delete_temp_files(prefix):
    for f in glob.glob(f"{prefix}*"):
        try:
            os.remove(f)
        except Exception as e:
            continue
