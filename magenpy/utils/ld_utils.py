import os
import shutil
import psutil

from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import dask.array as da
import numcodecs
import zarr

from magenpy.utils.compute_utils import generate_slice_dictionary


def _validate_ld_matrix(ld_mat):
    """
    Takes an `LDMatrix` object and checks its contents for validity.
    Specifically, we check that:
     - The dimensions of the matrix and its associated attributes are matching.
     - The LD boundaries are correct.
     - The masking is working properly.
    :param ld_mat: An instance of `LDMatrix`
    :return: True if `ld_mat` has the correct structure, False otherwise.
    """

    attributes = ['snps', 'a1', 'maf', 'bp_position', 'cm_position', 'ld_score']

    for attr in attributes:
        attribute = getattr(ld_mat, attr)
        if attribute is None:
            continue
        if len(attribute) != ld_mat.n_elements:
            raise ValueError("Invalid LD Matrix: Attribute dimensions are not aligned!")

    # Check LD bounds:
    ld_bounds = ld_mat.get_masked_boundaries()

    if ld_bounds.shape != (2, ld_mat.n_elements):
        raise ValueError("Invalid LD Matrix: LD boundaries have the wrong dimensions!")

    ld_block_lengths = ld_bounds[1, :] - ld_bounds[0, :]

    # Iterate over the stored LD data to check its dimensions:
    i = 0

    for i, d in enumerate(ld_mat):
        if len(d) != ld_block_lengths[i]:
            raise ValueError(f"Invalid LD Matrix: Element {i} does not have matching LD boundaries!")

    if i != (ld_mat.n_elements - 1):
        raise ValueError(f"Invalid LD Matrix: Conflicting total number of elements!")

    return True


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


def from_plink_ld_bin_to_zarr(bin_file, dir_store, ld_boundaries):
    """
    This method takes an LD binary file from PLINK and converts it to
    a chunked Zarr matrix
    :param bin_file: The path to the LD binary file
    :param dir_store: The directory store where the Zarr array will be stored
    :param ld_boundaries: The boundaries for the desired LD matrix.
    """

    n_rows = ld_boundaries.shape[1]
    avg_ncol = int((ld_boundaries[1, :] - ld_boundaries[0, :]).mean())

    n_chunks = estimate_row_chunk_size(n_rows, avg_ncol)

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

    chunk_size = z_rag.chunks[0]

    for i in range(int(np.ceil(z_rag.shape[0] / chunk_size))):

        n_chunk_rows = min(chunk_size, n_rows - i*chunk_size)
        ld_chunk = np.fromfile(bin_file,
                               offset=i*chunk_size*n_rows,
                               count=n_chunk_rows*n_rows).reshape(n_chunk_rows, n_rows)

        for j, ld in enumerate(ld_chunk):
            idx = i*chunk_size + j
            start, end = ld_boundaries[:, idx]
            ld_chunk[j] = ld_chunk[j][start: end]

        z_rag[i*chunk_size:(i+1)*chunk_size] = ld_chunk

    return z_rag


def write_csr_to_zarr(csr_mat, z_arr, start_row=None, end_row=None, ld_boundaries=None, purge_data=False):
    """
    Write from Scipy's csr matrix to Zarr array
    :param csr_mat: A scipy compressed sparse row matrix `csr_matrix`
    :param z_arr: A ragged zarr array with the same row dimension as the csr matrix
    :param start_row: The start row
    :param end_row: The end row
    :param purge_data: If `True`, delete the data that was written to Zarr from `csr_mat`
    :param ld_boundaries: If provided, we'd only write the elements within the provided boundaries.
    """

    if start_row is None:
        start_row = 0

    if end_row is None:
        end_row = csr_mat.shape[0]
    else:
        end_row = min(end_row, csr_mat.shape[0])

    ld_rows = []
    for i in range(start_row, end_row):
        if ld_boundaries is None:
            ld_rows.append(np.nan_to_num(csr_mat[i, :].data))
        else:
            ld_rows.append(np.nan_to_num(csr_mat[i, ld_boundaries[0, i]:ld_boundaries[1, i]].data))

    z_arr.oindex[np.arange(start_row, end_row)] = np.array(ld_rows + [None], dtype=object)[:-1]

    if purge_data:
        # Delete data from csr matrix:
        csr_mat.data[csr_mat.indptr[start_row]:csr_mat.indptr[end_row - 1]] = 0.
        csr_mat.eliminate_zeros()


def from_plink_ld_table_to_zarr_chunked(ld_file, dir_store, ld_boundaries, snps):
    """
    Transform a PLINK LD table to Zarr ragged array.
    PLINK LD tables are of the format:
    CHR_A   BP_A   SNP_A  CHR_B   BP_B  SNP_B  R
    This function deploys a chunked implementation so it only requires
    modest memory.

    :param dir_store: A path to the new Zarr store
    :param ld_file: A path to the plink LD table
    :param ld_boundaries: LD boundaries matrix
    :param snps: A list of SNPs
    """

    # Preliminaries:

    # Estimate row chunk-size for the Zarr array:
    rows, avg_ncols = len(snps), int((ld_boundaries[1, :] - ld_boundaries[0, :]).mean())
    chunks = estimate_row_chunk_size(rows, avg_ncols)

    # Create a ragged Zarr array:
    z_arr = zarr.open(dir_store,
                      mode='w',
                      shape=rows,
                      chunks=chunks[:1],
                      dtype=object,
                      object_codec=numcodecs.VLenArray(float))

    row_chunk_size = z_arr.chunks[0]

    # Create a chunked iterator with pandas:
    # Chunk size will correspond to the average chunk size for the Zarr array:
    ld_chunks = pd.read_csv(ld_file,
                            delim_whitespace=True,
                            usecols=['SNP_A', 'SNP_B', 'R'],
                            engine='c',
                            chunksize=row_chunk_size*avg_ncols // 2)

    # Create a ragged Zarr array:
    z_arr = zarr.open(dir_store,
                      mode='w',
                      shape=rows,
                      chunks=(row_chunk_size,),
                      dtype=object,
                      object_codec=numcodecs.VLenArray(float))

    # Create a dictionary mapping SNPs to their indices:
    snp_dict = dict(zip(snps, np.arange(len(snps))))

    # The sparse matrix will help us convert from triangular
    # sparse matrix to square sparse matrix:
    sp_mat = None

    curr_chunk = 0

    # For each chunk in the LD file:
    for ld_chunk in ld_chunks:

        # Create an indexed LD chunk:
        ld_chunk['index_A'] = ld_chunk['SNP_A'].map(snp_dict)
        ld_chunk['index_B'] = ld_chunk['SNP_B'].map(snp_dict)

        ld_chunk['R'].values[ld_chunk['R'].values == 0.] = np.nan

        # Create a compressed sparse row matrix:
        chunk_mat = csr_matrix((ld_chunk['R'].values,
                               (ld_chunk['index_A'].values, ld_chunk['index_B'].values)),
                               shape=(rows, rows))

        if sp_mat is None:
            sp_mat = chunk_mat + chunk_mat.T
            sp_mat.setdiag(1.)
        else:
            sp_mat = sp_mat + (chunk_mat + chunk_mat.T)

        # The chunk of the snp of largest index:
        max_index_chunk = ld_chunk['index_A'].max() // row_chunk_size

        if max_index_chunk > curr_chunk:
            write_csr_to_zarr(sp_mat, z_arr,
                              start_row=curr_chunk*row_chunk_size,
                              end_row=max_index_chunk*row_chunk_size,
                              ld_boundaries=ld_boundaries,
                              purge_data=True)
            curr_chunk = max_index_chunk

    write_csr_to_zarr(sp_mat, z_arr,
                      start_row=curr_chunk * row_chunk_size,
                      ld_boundaries=ld_boundaries,
                      purge_data=True)

    return z_arr


def from_plink_ld_table_to_zarr(ld_file, dir_store, ld_boundaries=None, snps=None):
    """
    Transform a PLINK LD table to Zarr ragged array.
    PLINK LD tables are of the format:
    CHR_A   BP_A   SNP_A  CHR_B   BP_B  SNP_B  R
    :param dir_store: A path to the new Zarr store
    :param ld_file: A path to the plink LD table
    :param ld_boundaries: LD boundaries matrix
    :param snps: A list of SNPs
    """

    ld_df = pd.read_csv(ld_file,
                        delim_whitespace=True,
                        usecols=['BP_A', 'SNP_A', 'BP_B', 'SNP_B', 'R'],
                        engine='c')

    # Assume that PLINK's table is already sorted by BP_A, BP_B:
    a_snps_slice = generate_slice_dictionary(ld_df.SNP_A.values)
    r_a = ld_df.R.values

    ld_sort_b = ld_df.sort_values(['BP_B', 'BP_A'])
    b_snps_slice = generate_slice_dictionary(ld_sort_b.SNP_B.values)
    r_b = ld_sort_b.R.values

    if snps is None:
        snp_df = pd.DataFrame(np.concatenate([ld_df[['SNP_A', 'BP_A']].drop_duplicates().values,
                                              ld_df[['SNP_B', 'BP_B']].drop_duplicates().values]),
                              columns=['SNP', 'BP'])
        snps = snp_df.drop_duplicates().sort_values('BP')['SNP'].values

    if ld_boundaries is None:
        before_bound = np.repeat([None], len(snps))
        after_bound = np.repeat([None], len(snps))
    else:
        before_bound = ld_boundaries[0, :] - np.arange(ld_boundaries.shape[1])
        after_bound = ld_boundaries[1, :] - np.arange(ld_boundaries.shape[1]) - 1

    ld_array = []
    avg_ncol = 0

    for i, snp in enumerate(snps):

        try:
            if before_bound[i] < 0:
                before = r_b[b_snps_slice[snp]][before_bound[i]:]
            else:
                before = []
        except KeyError:
            before = []

        try:
            after = r_a[a_snps_slice[snp]][:after_bound[i]]
        except KeyError:
            after = []

        ld_array.append(np.concatenate([before, [1.], after]))

        avg_ncol += (len(ld_array[-1]) - avg_ncol) / (i + 1)

    n_chunks = estimate_row_chunk_size(len(ld_array), int(avg_ncol))

    z_arr = zarr.open(dir_store,
                      mode='w',
                      shape=len(ld_array),
                      chunks=n_chunks[:1],
                      dtype=object,
                      object_codec=numcodecs.VLenArray(float))

    z_arr[:] = np.array(ld_array, dtype=object)

    return z_arr


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


def shrink_ld_matrix(arr, cm_dist, genmap_Ne, genmap_sample_size, shrinkage_cutoff=1e-3, ld_boundaries=None):

    if ld_boundaries is None:
        ld_boundaries = np.array([np.repeat(None, arr.shape[0]), np.repeat(None, arr.shape[0])])

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
        shrink_mult = np.exp(-mult_factor * np.abs(cm_dist - cm_dist[j])[ld_boundaries[0, j]: ld_boundaries[1, j]])
        # Set any shrinkage factor below the cutoff value to zero:
        shrink_mult[shrink_mult < shrinkage_cutoff] = 0.

        # Shrink the entries of the LD matrix:
        try:
            chunk[j % chunk_size] = chunk[j % chunk_size]*shrink_mult
        except ValueError:
            print(j)
            print(shrink_mult)
            print(chunk[j % chunk_size])
            raise ValueError

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

        if len(z_rag_index) == 0:
            continue

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
