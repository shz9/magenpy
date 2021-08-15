import os
import shutil
import psutil

import pandas as pd
import numpy as np
import dask.array as da
import numcodecs
import zarr


def move_ld_store(z_arr, target_path, overwrite=True):

    source_path = z_arr.store.dir_path()

    if overwrite or not any(os.scandir(target_path)):
        shutil.rmtree(target_path, ignore_errors=True)
        shutil.move(source_path, target_path)

    return zarr.open(target_path)


def delete_ld_store(z_arr):

    try:
        z_arr.store.rmdir()
    except Exception as e:
        print(e)


def clump_snps(ldw, stat, rsq_threshold=.9, extract=True):
    """
    This function takes an LDWrapper object and clumps the SNPs based
    on the `stat` vector (usually p-value) and the provided r-squared threshold.
    If two SNPs have an r-squared greater than the threshold,
    the SNP with the higher `stat` value is excluded.
    :param ldw: LDWrapper object
    :param stat: A vector of statistics (e.g. p-values) for the SNPs
    :param rsq_threshold: The r^2 threshold to use for filtering
    :param extract: if True, return remaining SNPs. If False, return removed SNPs.
    :return: A list of SNPs passing the specified filter
    """

    snps = ldw.snps
    ld_bounds = ldw.ld_boundaries
    remove_snps = set()

    for idx, ld in enumerate(ldw):

        if snps[idx] in remove_snps:
            continue

        rsq = np.array(ld)**2

        for s_idx in np.where(rsq > rsq_threshold)[0]:
            real_idx = s_idx + ld_bounds[0, idx]
            if idx == real_idx or snps[real_idx] in remove_snps:
                continue

            if stat[idx] < stat[real_idx]:
                remove_snps.add(snps[real_idx])
            else:
                remove_snps.add(snps[idx])

    if extract:
        return list(set(snps) - remove_snps)
    else:
        return list(remove_snps)


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


def sparsify_ld_matrix(arr, bounds):
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

    from rechunker import rechunk

    rechunked = rechunk(arr,
                        target_chunks=target_chunks,
                        target_store=target_store,
                        temp_store=intermediate_store,
                        max_mem="128MiB",
                        **kwargs)

    try:
        rechunked.execute()
    except Exception as e:
        raise e

    # Delete the older/intermediate stores:
    delete_ld_store(zarr.open(intermediate_store))
    delete_ld_store(arr)

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


def zarr_array_to_ragged(z,
                         dir_store,
                         keep_snps=None,
                         bounds=None,
                         rechunk=True,
                         delete_original=False):
    """
    This function takes a chunked Zarr matrix (e.g. sparse LD matrix)
    and returns a ragged array matrix.
    The function allows filtering down the original matrix by passing
    a list of SNPs to keep. It also allows the user to re-chunk
    the ragged array for optimized read/write performance.

    TODO: Optimize this for large chromosomes/LD matrices!

    :param z: the original Zarr matrix (implementation assumes 2D matrix)
    :param keep_snps: A list of SNP IDs to keep.
    :param rechunk: Whether to re-chunk the ragged array (for optimized read/write performance)
    :param dir_store: The path to the new Zarr matrix store
    :param delete_original: Delete the original store after transformation.
    :return:
    """

    if keep_snps is None:
        n_rows = z.shape[0]

        idx_map = pd.DataFrame({'SNP': z.attrs['SNP']}).reset_index()
        idx_map.columns = ['index_x', 'SNP']
        idx_map['index_y'] = idx_map['index_x']

    else:
        idx_map = pd.DataFrame({'SNP': keep_snps}).reset_index().merge(
            pd.DataFrame({'SNP': z.attrs['SNP']}).reset_index(),
            on='SNP',
            suffixes=('_y', '_x')
        )
        idx_map['chunk_x'] = (idx_map['index_x'] // z.chunks[0]).astype(int)
        n_rows = len(keep_snps)

    idx_map['chunk_x'] = (idx_map['index_x'] // z.chunks[0]).astype(int)

    if bounds is None:
        orig_bounds = bounds = np.array(z.attrs['LD boundaries'])
    else:
        orig_bounds = np.array(z.attrs['LD boundaries'])

    avg_ncol = int((bounds[1, :] - bounds[0, :]).mean())

    if rechunk:
        n_chunks = estimate_row_chunk_size(n_rows, avg_ncol)
    else:
        n_chunks = z.chunks

    if avg_ncol == n_rows:
        z_rag = zarr.open(dir_store,
                          mode='w',
                          shape=(n_rows, n_rows),
                          chunks=n_chunks,
                          dtype=float)
    else:
        z_rag = zarr.open(dir_store,
                          mode='w',
                          shape=n_rows,
                          chunks=n_chunks[:1],
                          dtype=object,
                          object_codec=numcodecs.VLenArray(float))

    idx_x = idx_map['index_x'].values
    chunk_size = z.chunks[0]

    for i in range(int(np.ceil(z.shape[0] / chunk_size))):

        start = i * chunk_size
        end = min((i + 1) * chunk_size, z.shape[0])

        z_chunk = z[start: end]

        z_rag_index = []
        z_rag_rows = []

        for _, (k, _, j, _) in idx_map.loc[idx_map['chunk_x'] == i].iterrows():

            z_rag_index.append(k)

            if keep_snps is None:
                row_val = z_chunk[j - start][bounds[0, j]:bounds[1, j]]
            else:
                # Find the index of SNPs in the original LD matrix that
                # remain after matching with the `keep_snps` variable.
                orig_idx = idx_x[(orig_bounds[0, j] <= idx_x) & (idx_x < orig_bounds[1, j])] - orig_bounds[0, j]
                row_val = z_chunk[j - start][orig_idx]

            z_rag_rows.append(row_val)

        if avg_ncol == n_rows:
            z_rag.oindex[z_rag_index] = np.array(z_rag_rows)
        else:
            z_rag.oindex[z_rag_index] = np.array(z_rag_rows + [None], dtype=object)[:-1]

    z_rag.attrs.update(z.attrs.asdict())

    if bounds is not None:
        z_rag.attrs['LD boundaries'] = bounds.tolist()

    if keep_snps is not None:
        z_rag.attrs['SNP'] = list(keep_snps)
        z_rag.attrs['BP'] = list(map(int, np.array(z.attrs['BP'])[idx_x]))
        z_rag.attrs['cM'] = list(map(float, np.array(z.attrs['cM'])[idx_x]))

    if delete_original:
        delete_ld_store(z)

    return z_rag
