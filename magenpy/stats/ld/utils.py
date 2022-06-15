import os
import os.path as osp

from scipy.sparse import csr_matrix
import dask.array as da
import pandas as pd
import numpy as np
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
        import shutil
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


def clump_snps(ldm, stat, rsq_threshold=.9, extract=True):
    """
    This function takes an LDMatrix object and clumps the SNPs based
    on the `stat` vector (usually p-value) and the provided r-squared threshold.
    If two SNPs have an r-squared greater than the threshold,
    the SNP with the higher `stat` value is excluded.
    :param ldm: LDMatrix object
    :param stat: A vector of statistics (e.g. p-values) for the SNPs
    :param rsq_threshold: The r^2 threshold to use for filtering
    :param extract: If True, return remaining SNPs. If False, return removed SNPs.
    :return: A list of SNPs passing the specified filter
    """

    snps = ldm.snps
    ld_bounds = ldm.ld_boundaries
    remove_snps = set()

    for idx, ld in enumerate(ldm):

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


def shrink_ld_matrix(arr,
                     cm_pos,
                     genmap_Ne,
                     genmap_sample_size,
                     shrinkage_cutoff=1e-5,
                     ld_boundaries=None):
    """
    Shrink the entries of the LD matrix using the shrinkage estimator
    described in Lloyd-Jones (2019).

    :param arr: The Zarr array containing the original LD matrix.
    :param cm_pos: The position of each variant in the LD matrix in centi Morgan.
    :param genmap_Ne: The effective population size for the genetic map.
    :param genmap_sample_size: The sample size used to estimate the genetic map.
    :param shrinkage_cutoff: The cutoff value below which we assume that the LD is zero.
    :param ld_boundaries: The LD boundaries to use when shrinking the LD matrix.
    """

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
        shrink_mult = np.exp(-mult_factor * np.abs(cm_pos - cm_pos[j])[ld_boundaries[0, j]: ld_boundaries[1, j]])
        # Set any shrinkage factor below the cutoff value to zero:
        shrink_mult[shrink_mult < shrinkage_cutoff] = 0.

        # Shrink the entries of the LD matrix:
        try:
            chunk[j % chunk_size] = chunk[j % chunk_size]*shrink_mult
        except ValueError:
            print("chunk row:", chunk[j % chunk_size])
            print("chunk shape:", chunk[j % chunk_size].shape)
            print("shrinkage:", shrink_mult)
            raise ValueError(f'Failed to apply shrinkage to row number: {j}')

    update_prev_chunk(j)

    return arr


def sparsify_ld_matrix(arr, bounds):
    """
    A utility function to sparsify chunked LD matrices
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
    """
    Rechunk a Zarr matrix using utilities from `rechunker`.
    """

    if osp.isdir(target_store):
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
    Determine optimal chunks that fit in max_mem. Max_mem should be numerical in GiB
    Modified from: Sergio Hleap
    """

    import psutil

    if cpus is None:
        cpus = psutil.cpu_count()

    if max_mem is None:
        max_mem = psutil.virtual_memory().available / (1024.0 ** 3)

    chunk_mem = max_mem / cpus
    chunks = da.core.normalize_chunks(f"{chunk_mem}GiB", shape=chunked_array.shape, dtype=chunked_array.dtype)

    return chunked_array.chunk(chunks)


def estimate_row_chunk_size(rows, cols, dtype=np.float64, chunk_size=128):
    """
    Estimate the chunk size for ragged arrays, given the number of rows, columns, and data type.

    :param rows: Number of rows.
    :param cols: Number of columns. If a ragged array, provide average size of arrays
    :param dtype: Data type
    :param chunk_size: chunk size in MB
    """

    matrix_size = rows * cols * np.dtype(dtype).itemsize / 1024 ** 2
    n_chunks = matrix_size // chunk_size

    if n_chunks < 1:
        return None, None
    else:
        return int(rows / n_chunks), None


def dense_zarr_array_to_ragged(z,
                               dir_store,
                               ld_boundaries,
                               rechunk=True,
                               delete_original=True):
    """
    This function takes a dense chunked Zarr matrix
    and, given an array of window sizes for each row, returns a sparse ragged array matrix.
    This is a utility function that works with `dask` or `xarray` generated Linkage-Disequilibrium (LD) matrices
    and aims to create compact LD matrices that are easier to manipulate and work with.

    :param z: The original LD matrix in Zarr format.
    :param dir_store: The path to the new store where the sparse LD matrix will be stored.
    :param ld_boundaries: The LD boundaries or window around each SNP. This is a 2xM array where
    the first row contains the start and the second row contains the end of each window.
    :param rechunk: If True, re-chunk the ragged array for optimized read/write performance.
    :param delete_original: Delete the original store after creating the ragged array.

    """

    avg_ncol = int((ld_boundaries[1, :] - ld_boundaries[0, :]).mean())

    if rechunk:
        n_chunks = estimate_row_chunk_size(z.shape[0], avg_ncol)
    else:
        n_chunks = z.chunks

    if avg_ncol == z.shape[0]:
        z_rag = zarr.open(dir_store,
                          mode='w',
                          shape=z.shape,
                          chunks=n_chunks,
                          dtype=float)
    else:
        z_rag = zarr.open(dir_store,
                          mode='w',
                          shape=z.shape[0],
                          chunks=n_chunks[:1],
                          dtype=object,
                          object_codec=numcodecs.VLenArray(float))

    chunk_size = z.chunks[0]

    for i in range(int(np.ceil(z.shape[0] / chunk_size))):

        start = i * chunk_size
        end = min((i + 1) * chunk_size, z.shape[0])

        z_chunk = z[start: end]

        z_rag_rows = []

        for j in range(start, end):
            z_rag_rows.append(
                z_chunk[j - start][ld_boundaries[0, j]:ld_boundaries[1, j]]
            )

        if avg_ncol == z.shape[0]:
            z_rag.oindex[np.arange(start, end)] = np.array(z_rag_rows)
        else:
            z_rag.oindex[np.arange(start, end)] = np.array(z_rag_rows + [None], dtype=object)[:-1]

    if delete_original:
        delete_ld_store(z)

    return z_rag


def filter_zarr_array(z,
                      dir_store,
                      extract_snps,
                      ld_boundaries,
                      rechunk=True,
                      delete_original=False):
    """
    This function takes a chunked Zarr matrix (dense or sparse LD matrix)
    and, given a list of SNPs to extract, returns a filtered ragged array matrix.

    TODO: Optimize this for large chromosomes/LD matrices!

    :param z: the original Zarr matrix (implementation assumes 2D matrix)
    :param dir_store: The path to the new Zarr matrix store
    :param extract_snps: A list or vector of SNP IDs to keep.
    :param ld_boundaries: The LD boundaries or window around each SNP. This is a 2xM array where
    the first row contains the start and the second row contains the end of each window.
    :param rechunk: If True, re-chunk the filtered array for optimized read/write performance.
    :param delete_original: If True, delete the original store after transformation.
    """

    idx_map = pd.DataFrame({'SNP': extract_snps}).reset_index().merge(
        pd.DataFrame({'SNP': z.attrs['SNP']}).reset_index(),
        on='SNP',
        suffixes=('_y', '_x')
    )
    idx_map['chunk_x'] = (idx_map['index_x'] // z.chunks[0]).astype(int)
    n_rows = len(extract_snps)

    idx_map['chunk_x'] = (idx_map['index_x'] // z.chunks[0]).astype(int)

    orig_bounds = np.array(z.attrs['LD boundaries'])

    avg_ncol = int((ld_boundaries[1, :] - ld_boundaries[0, :]).mean())

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

    # Update the attributes associated with the new matrix:
    z_rag.attrs['LD boundaries'] = ld_boundaries.tolist()

    try:
        z_rag.attrs['SNP'] = list(extract_snps)
    except Exception:
        pass

    try:
        z_rag.attrs['BP'] = list(map(int, np.array(z.attrs['BP'])[idx_x]))
    except Exception:
        pass

    try:
        z_rag.attrs['cM'] = list(map(float, np.array(z.attrs['cM'])[idx_x]))
    except Exception:
        pass

    try:
        z_rag.attrs['MAF'] = list(map(float, np.array(z.attrs['MAF'])[idx_x]))
    except Exception:
        pass

    try:
        z_rag.attrs['A1'] = list(np.array(z.attrs['A1'])[idx_x])
    except Exception:
        pass

    try:
        z_rag.attrs['LDScore'] = list(map(float, np.array(z.attrs['LDScore'])[idx_x]))
    except Exception:
        pass

    if delete_original:
        delete_ld_store(z)

    return z_rag


def compute_ld_plink1p9(genotype_matrix,
                        ld_boundaries,
                        output_dir,
                        temp_dir='temp'):

    from magenpy.utils.executors import plink1Executor
    from magenpy.GenotypeMatrix import plinkBEDGenotypeMatrix

    assert isinstance(genotype_matrix, plinkBEDGenotypeMatrix)

    plink1 = plink1Executor()

    keep_file = osp.join(temp_dir, 'samples.keep')
    keep_table = genotype_matrix.sample_table.get_individual_table()
    keep_table.to_csv(keep_file, index=False, header=False, sep="\t")

    snp_keepfile = osp.join(temp_dir, 'variants.keep')
    pd.DataFrame({'SNP': genotype_matrix.snps}).to_csv(
        snp_keepfile, index=False, header=False
    )

    plink_output = osp.join(temp_dir, f'chr_{str(genotype_matrix.chromosome)}')

    # Set the window sizes in various units:
    # (1) Number of neighboring SNPs:
    window_size = (ld_boundaries - np.arange(genotype_matrix.m)).max() + 10

    # (2) Kilobases:

    positional_bounds = np.clip(np.array([ld_boundaries[0, :] - 1, ld_boundaries[1, :]]),
                                a_min=0, a_max=ld_boundaries.shape[1] - 1)

    kb_pos = .001*genotype_matrix.bp_pos
    kb_bounds = kb_pos[positional_bounds]
    kb_window_size = (kb_bounds - kb_pos).max() + .01

    # (3) centi Morgan:
    try:
        cm_pos = genotype_matrix.cm_pos
        cm_bounds = genotype_matrix.cm_pos[positional_bounds]
        cm_window_size = (cm_bounds - cm_pos).max() + .01
    except Exception:
        cm_window_size = None

    cmd = [
        f"--bfile {genotype_matrix.bed_file.replace('.bed', '')}",
        f"--keep {keep_file}",
        f"--extract {snp_keepfile}",
        "--keep-allele-order",
        f"--out {plink_output}",
        "--r gz",
        f"--ld-window {window_size}",
        f"--ld-window-kb {kb_window_size}"
    ]

    if cm_window_size is not None:
        cmd.append(f"--ld-window-cm {cm_window_size}")

    plink1.execute(cmd)

    # Convert from PLINK LD files to Zarr:
    fin_ld_store = osp.join(output_dir, 'ld', 'chr_' +
                            str(genotype_matrix.chromosome))

    z_ld_mat = from_plink_ld_table_to_zarr_chunked(f"{plink_output}.ld.gz",
                                                   fin_ld_store,
                                                   ld_boundaries,
                                                   genotype_matrix.snps)

    return z_ld_mat


def compute_ld_xarray(genotype_matrix,
                      ld_boundaries,
                      output_dir,
                      temp_dir='temp'):
    """
    Compute the Linkage Disequilibrium matrix or snp-by-snp
    correlation matrix assuming that the genotypes are represented
    by `xarray` or `dask`-like matrix. This function computes the
    entire X'X/N and stores the result in Zarr arrays.
    To create sparse matrices out of this, consult the
    LD estimators and their implementations.

    NOTE: We don't recommend using this for large-scale genotype matrices.
    Use `compute_ld_plink` instead if you have plink installed on your system.

    :param genotype_matrix: An `xarrayGenotypeMatrix` object
    :param temp_dir: A temporary directory where to store intermediate results.

    """

    from magenpy.GenotypeMatrix import xarrayGenotypeMatrix
    assert isinstance(genotype_matrix, xarrayGenotypeMatrix)

    g_data = genotype_matrix.xr_mat

    # Re-chunk the array
    g_data = g_data.chunk((min(1024, g_data.shape[0]),
                           min(1024, g_data.shape[1])))

    from ..transforms.genotype import standardize

    # Standardize the genotype matrix and fill missing data with zeros:
    g_mat = standardize(g_data)

    # Compute the LD matrix:
    ld_mat = (da.dot(g_mat.T, g_mat) / genotype_matrix.sample_size).astype(np.float64)
    ld_mat.to_zarr(temp_dir, overwrite=True)

    z_ld_mat = zarr.open(temp_dir)
    z_ld_mat = rechunk_zarr(z_ld_mat,
                            ld_mat.rechunk({0: 'auto', 1: None}).chunksize,
                            temp_dir + '_rechunked',
                            temp_dir + '_intermediate')

    fin_ld_store = osp.join(output_dir, 'ld', 'chr_' +
                            str(genotype_matrix.chromosome))

    # If the matrix is sparse/thresholded, then convert to a ragged zarr array:
    if (ld_boundaries[1, :] - ld_boundaries[0, :]).min() < genotype_matrix.n_snps:

        z_ld_mat = dense_zarr_array_to_ragged(z_ld_mat,
                                              fin_ld_store,
                                              ld_boundaries)

    else:
        z_ld_mat = move_ld_store(z_ld_mat, fin_ld_store)

    return z_ld_mat
