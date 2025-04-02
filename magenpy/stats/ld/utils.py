import os
import os.path as osp
import copy
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import zarr

from ...LDMatrix import LDMatrix


def move_ld_store(z_arr, target_path, overwrite=True):
    """
    Move an LD store from its current path to the `target_path`
    :param z_arr: An LDMatrix object
    :param target_path: The target path where to move the LD store
    :param overwrite: If True, overwrites the target path if it exists.

    :return: A Zaarr array object pointing to the new location of the LD store.
    """

    source_path = z_arr.store.dir_path()

    if overwrite or not any(os.scandir(target_path)):
        import shutil
        shutil.rmtree(target_path, ignore_errors=True)
        shutil.move(source_path, target_path)

    return zarr.open(target_path)


def delete_ld_store(ld_mat):
    """
    Delete the LD store from disk.
    :param ld_mat: An `LDMatrix` object
    """

    try:
        ld_mat.store.rmdir()
    except Exception as e:
        print(e)


def combine_ld_matrices(ld_matrices, output_dir, overwrite=True, delete_original=False):
    """
    Combine the Zarr arrays underlying multiple `LDMatrix` objects into
    a single store, which will reside in `output_dir`. This function combines
    the `matrix/data/` and `matrix/indptr` arrays of the Zarr stores, in addition
    to the metadata, such as SNP rsIDs, basepair position, centi Morgan distance, etc.

    !!! warning
        This function does not copy any of the attributes of the individual `LDMatrix` objects.

    !!! warning
        This implementation assumes there is no overlap in the set of variants present in
        the LD matrices.

    :param ld_matrices: A list of `LDMatrix` objects to combine.
    :param output_dir: The output directory where to store the combined LD matrix.
    :param overwrite: If True, overwrites the output directory if it exists.
    :param delete_original: If True, deletes the original LD matrices after combining them.

    :return: An `LDMatrix` object pointing to the combined LD matrix.
    """

    # Determine the total number of variants and total number of elements in the `data` array:
    total_n_snps = sum([ld.stored_n_snps for ld in ld_matrices])
    total_elements = sum([ld.data.shape[0] for ld in ld_matrices])
    compressor = ld_matrices[0].compressor
    metadata = list(ld_matrices[0].zarr_group['metadata'].keys())

    # Create hierarchical storage with zarr groups:
    store = zarr.DirectoryStore(output_dir)
    z = zarr.group(store=store, overwrite=overwrite)

    # Create sub-hierarchy to store the data of the sparse LD matrix:
    mat = z.create_group('matrix')
    mat.empty('data', shape=total_elements,
              dtype=ld_matrices[0].stored_dtype,
              compressor=compressor)
    mat.zeros('indptr', shape=total_n_snps + 1,
              dtype=np.int64, compressor=compressor)

    # Create sub-hierarchy to store the metadata for the LD matrix:
    meta = z.create_group('metadata')
    for meta_col, meta_zarr in ld_matrices[0].zarr_group['metadata'].items():

        if meta_zarr.dtype == object:
            meta.empty(meta_col, shape=total_n_snps, dtype=str, compressor=compressor)
        else:
            meta.empty(meta_col, shape=total_n_snps,
                       dtype=meta_zarr.dtype, compressor=compressor)

    curr_n_snps = 0
    curr_n_entries = 0

    for ld in ld_matrices:

        # Copy the matrix data:
        mat['data'][curr_n_entries:curr_n_entries + ld.data.shape[0]] = ld.data[:]
        mat['indptr'][curr_n_snps + 1:curr_n_snps + ld.stored_n_snps + 1] = curr_n_entries + ld.indptr[1:]

        # Copy the metadata:

        for meta_col in metadata:
            meta[meta_col][curr_n_snps:curr_n_snps + ld.stored_n_snps] = ld.zarr_group[f'metadata/{meta_col}'][:]

        # Update the counters:
        curr_n_snps += ld.stored_n_snps
        curr_n_entries += ld.data.shape[0]

        # Delete the original store (if requested):

        if delete_original:
            delete_ld_store(ld)

    return LDMatrix(z)


def clump_snps(ldm,
               statistic=None,
               rsq_threshold=.9,
               extract=True,
               sort_key=None):
    """
    This function takes an LDMatrix object and clumps SNPs based
    on the `stat` vector (usually p-value) and the provided r-squared threshold.
    If two SNPs have an r-squared greater than the threshold,
    the SNP with the higher `stat` value is excluded.

    :param ldm: An LDMatrix object
    :param statistic: A vector of statistics (e.g. p-values) for each SNP that will determine which SNPs to discard.
    :param rsq_threshold: The r^2 threshold to use for filtering variants.
    :param extract: If True, return remaining SNPs. If False, return removed SNPs.
    :param sort_key: The key function for the sorting algorithm that will decide how to sort the `statistic`.
    By default, we select the SNP with the minimum value for the `statistic` (e.g. smaller p-value).

    :return: A list of SNP rsIDs that are left after clumping (or discarded if `extract=False`).
    """

    snps = ldm.snps

    if statistic is None:
        # if a statistic is not provided, then clump SNPs based on their base pair order,
        # meaning that if two SNPs are highly correlated, we keep the one with smaller base pair position.
        statistic = ldm.bp_position
    else:
        assert len(statistic) == len(snps)

    if sort_key is not None:
        sort_key = lambda x: sort_key(statistic[x])

    sorted_idx = sorted(range(len(ldm)), key=sort_key)

    snps = ldm.snps
    keep_snps_dict = dict(zip(snps, np.ones(len(snps), dtype=bool)))

    for idx in sorted_idx:

        if not keep_snps_dict[snps[idx]]:
            continue

        r, indices = ldm.get_row(idx, return_indices=True)
        # Find the SNPs that we need to remove:
        # We remove SNPs whose squared correlation coefficient with the index SNP is
        # greater than the specified rsq_threshold:
        snps_to_remove = snps[indices[np.where(r**2 > rsq_threshold)[0]]]

        # Update the `keep_snps_dict` dictionary:
        keep_snps_dict.update(dict(zip(snps_to_remove, np.zeros(len(snps_to_remove), dtype=bool))))

    if extract:
        return [snp for snp, cond in keep_snps_dict.items() if cond]
    else:
        return [snp for snp, cond in keep_snps_dict.items() if not cond]


def expand_snps(seed_snps, ldm, rsq_threshold=0.9):
    """
    Given an initial set of SNPs, expand the set by adding
    "neighbors" whose squared correlation with the is higher than
    a user-specified threshold.

    :param seed_snps: An iterable containing initial set of SNP rsIDs.
    :param ldm: An `LDMatrix` object containing SNP-by-SNP correlations.
    :param rsq_threshold: The r^2 threshold to use for including variants.

    """

    ldm_snps = ldm.snps
    snp_seed_idx = np.where(np.isin(seed_snps, ldm_snps))

    if len(snp_seed_idx) < 1:
        print("Warning: None of the seed SNPs are present in the LD matrix object!")
        return seed_snps

    final_set = set(seed_snps)

    for idx in snp_seed_idx:
        r, indices = ldm.get_row(idx, return_indices=True)
        final_set = final_set.union(set(ldm_snps[indices[np.where(r**2 > rsq_threshold)[0]]]))

    return list(final_set)


def compute_extremal_eigenvalues(mat, k=1, which='both', **eigsh_kwargs):
    """
    A helper function to compute the extremal eigenvalues (minimum and maximum) of the LD matrix efficiently.
    This function uses a trick to shift the eigenvalues to the right by the maximum eigenvalue
    and then computes the smallest eigenvalue of the shifted matrix. This is a more stable
    and efficient approach for computing the smallest eigenvalue of the LD matrix.

    :param mat: A `scipy` sparse CSR matrix object or an LDLinearOperator object for which to compute the
    extremal eigenvalues.
    :param k: The number of eigenvalues to compute.
    :param which: Which extremal eigenalue to return. Can be 'both', 'min', or 'max'.
    :param eigsh_kwargs: Additional keyword arguments to pass to the `scipy.sparse.linalg.eigsh` function.

    :return: A dictionary containing the extremal eigenvalues of the LD matrix.
    """

    assert which in ('both', 'min', 'max')

    if 'maxiter' not in eigsh_kwargs:
        eigsh_kwargs['maxiter'] = 10000

    if 'tol' not in eigsh_kwargs:
        eigsh_kwargs['tol'] = 1e-6

    from scipy.sparse.linalg import eigsh, ArpackNoConvergence
    from scipy.sparse import csr_matrix, identity

    eig_result = {}

    # In all cases we need to compute the largest eigenvalue:
    eig_result['max'] = eigsh(mat, k=k, which='LM', return_eigenvectors=False, **eigsh_kwargs)
    max_eig = np.max(eig_result['max'])

    # Compute the minimum eigenvalue (if requested):
    if which in ('min', 'both'):

        if isinstance(mat, csr_matrix):
            diag_shift = max_eig*identity(mat.shape[0], format='csr', dtype=mat.dtype)
            mat -= diag_shift
        else:
            if mat.diag_shift is not None:
                orig_diag_shift = copy.copy(mat.diag_shift)
                mat.set_diag_shift(mat.diag_shift-max_eig)
            else:
                orig_diag_shift = None
                mat.set_diag_shift(-max_eig)

        # TODO: Try other solvers here?
        try:
            lm_max_hat = eigsh(mat, k=k, which='LM', return_eigenvectors=False, **eigsh_kwargs)
        except ArpackNoConvergence:
            # If the solver does not converge, we can try to increase the number of iterations and the tolerance:
            if 'maxiter' in eigsh_kwargs:
                eigsh_kwargs['maxiter'] *= 2

            if 'tol' in eigsh_kwargs:
                eigsh_kwargs['tol'] *= 10

            lm_max_hat = eigsh(mat, k=k, which='LM', return_eigenvectors=False, **eigsh_kwargs)

        if isinstance(mat, csr_matrix):
            mat += diag_shift
        else:
            mat.set_diag_shift(orig_diag_shift)

        eig_result['min'] = lm_max_hat + max_eig

    if k == 1:
        for key in eig_result:
            eig_result[key] = eig_result[key][0]

    if which == 'both':
        return eig_result
    else:
        return eig_result[which]


def multivariate_normal_conditional_sampling(ldm,
                                             mean=None,
                                             cov_scale=None,
                                             size=None,
                                             method='eigh',
                                             threads=1,
                                             seed=None,
                                             max_block_size=500):
    """
    Perform conditional sampling from a multivariate normal distribution where the covariance
    matrix is a linear scaling of the LD matrix. This function is useful for imputation or
    generating synthetic data, with realistic LD structure.

    The function generates a vector of dimension `n_snps` (i.e. the number of variants represented
    in the LD matrix) with covariance structure determined by the LD matrix, according to the
    following formula:

    x ~ MVN(mean, cov_scale * LD)

    where `mean` is the mean vector of the multivariate normal distribution, `cov_scale` is a scaling factor,
    and `LD` is the LD matrix.

    :param ldm: An `LDMatrix` object containing the LD matrix.
    :param mean: The mean of the multivariate normal distribution.
    :param cov_scale: The scaling factor for the covariance matrix.
    :param size: The number of samples to generate.
    :param method: The method to use for computing the factor matrix (see documentation for
    `numpy.random.Generator.multivariate_normal`).
    :param threads: The number of threads to use for parallel computation.
    :param seed: The random seed for reproducibility.
    :param max_block_size: The maximum block size to use for computing the conditional samples.

    :return: A vector of samples from the conditional multivariate normal distribution.
    """

    if mean is not None:
        assert mean.shape[0] == ldm.n_snps

    if cov_scale is None:
        cov_scale = 1.

    rng = np.random.default_rng(seed)

    def sample_block(cov):

        try:
            result = rng.multivariate_normal(np.zeros(cov.shape[0]),
                                             cov,
                                             check_valid='ignore',
                                             size=size,
                                             method=method)
        except np.linalg.LinAlgError as e:
            if method == 'cholesky':
                # If cholesky, try again with eigen decomposition:
                result = rng.multivariate_normal(np.zeros(cov.shape[0]),
                                                 cov,
                                                 size=size,
                                                 method='eigh')
            else:
                raise e

        del cov  # Free up memory
        return result

    parallel = Parallel(n_jobs=threads, backend='threading')

    with parallel:
        x = parallel(
            delayed(sample_block)(cov_scale*cov_block) for cov_block in ldm.iter_blocks(
                return_type='numpy',
                return_symmetric=True,
                dtype='float32',
                max_block_size=max_block_size,
            )
        )

    if size is not None:
        x = np.concatenate(x, axis=1)
    else:
        x = np.concatenate(x)

    if mean is not None:
        return mean + x
    else:
        return x


# -------------------------------------------------------------------------


def shrink_ld_matrix(ld_mat_obj,
                     cm_pos,
                     maf_var,
                     genmap_ne,
                     genmap_sample_size,
                     shrinkage_cutoff=1e-3,
                     phased_haplotype=False,
                     chunk_size=1000):

    """
    Shrink the entries of the LD matrix using the shrinkage estimator
    described in Lloyd-Jones (2019) and Wen and Stephens (2010). The estimator
    is also implemented in the RSS software by Xiang Zhu:

    https://github.com/stephenslab/rss/blob/master/misc/get_corr.R

    TODO: Re-write this function without using CSR data structures.
    Also, consider saving shrinkage factor per pair of variants instead of shrunk LD entries?

    :param ld_mat_obj: An `LDMatrix` object encapsulating the LD matrix whose entries we wish to shrink.
    :param cm_pos: The position of each variant in the LD matrix in centi Morgan.
    :param maf_var: A vector of the variance in minor allele frequency (MAF) for each SNP in the LD matrix. Should be
    equivalent to 2*pj*(1. - pj), where pj is the MAF of SNP j.
    :param genmap_ne: The effective population size for the genetic map.
    :param genmap_sample_size: The sample size used to estimate the genetic map.
    :param shrinkage_cutoff: The cutoff value below which we assume that the shrinkage factor is zero.
    :param phased_haplotype: A flag indicating whether the LD was calculated from phased haplotypes.
    :param chunk_size: An optional parameter that sets the maximum number of rows processed simultaneously. The smaller
    the `chunk_size`, the less memory requirements needed for this step.
    """

    # The multiplicative term for the shrinkage factor
    # The shrinkage factor is 4 * Ne * (rho_ij/100) / (2*m)
    # where Ne is the effective population size and m is the sample size
    # for the genetic map and rho_ij is the distance between SNPs i and j
    # in centi Morgan.
    # Therefore, the multiplicative term that we need to apply
    # to the distance between SNPs is: 4*Ne/(200*m), which is equivalent to 0.02*Ne/m
    # See also: https://github.com/stephenslab/rss/blob/master/misc/get_corr.R
    # and Wen and Stephens (2010)

    mult_term = .02*genmap_ne / genmap_sample_size

    def harmonic_series_sum(n):
        """
        A utility function to compute the sum of the harmonic series
        found in Equation 2.8 in Wen and Stephens (2010)
        Acknowledgement: https://stackoverflow.com/a/27683292
        """
        from scipy.special import digamma
        return digamma(n + 1) + np.euler_gamma

    # Compute theta according to Eq. 2.8 in Wen and Stephens (2010)

    h_sum = harmonic_series_sum(2*genmap_sample_size - 1)  # The sum of the harmonic series in Eq. 2.8
    theta = (1. / h_sum) / (2. * genmap_sample_size + 1. / h_sum)  # The theta parameter (related to mutation rate)
    theta_factor = (1. - theta)**2  # The theta factor that we'll multiply all elements of the covariance matrix with
    theta_diag_factor = .5 * theta * (1. - .5 * theta)  # The theta factor for the diagonal elements

    # Phased haplotype/unphased genotype multiplicative factor:
    # Wen and Stephens (2010), Section 2.4
    phased_mult = [.5, 1.][phased_haplotype]

    # We need to turn the correlation matrix into a covariance matrix to
    # apply the shrinkage factor. For this, we have to multiply each row
    # by the product of standard deviations:
    maf_sd = np.sqrt(phased_mult*maf_var)

    # According to Eqs. 2.6 and 2.7 in Wen and Stephens (2010), the shrunk standard deviation should be:
    shrunk_sd = np.sqrt(theta_factor*maf_var*phased_mult + theta_diag_factor)

    global_indptr = ld_mat_obj.indptr

    for chunk_idx in range(int(np.ceil(len(ld_mat_obj) / chunk_size))):

        start_row = chunk_idx*chunk_size
        end_row = min((chunk_idx+1)*chunk_size, len(ld_mat_obj))

        # Load the subset of the LD matrix specified by chunk_size.
        csr_mat = ld_mat_obj.load_data(
            start_row=start_row,
            end_row=end_row,
            return_symmetric=False,
            return_square=False,
            keep_original_shape=True,
            return_as_csr=True,
            dtype=np.float32
        )

        # Get the relevant portion of indices and pointers from the CSR matrix:
        indptr = global_indptr[start_row:end_row+1]

        row_indices = np.concatenate([
            (start_row + r_idx)*np.ones(indptr[r_idx+1] - indptr[r_idx], dtype=int)
            for r_idx in range(len(indptr) - 1)
        ])

        # Compute the shrinkage factor for entries in the current block:
        shrink_factor = np.exp(-mult_term*np.abs(cm_pos[csr_mat.indices] - cm_pos[row_indices]))
        # Set shrinkage factors below the cutoff value to 0.:
        shrink_factor[shrink_factor < shrinkage_cutoff] = 0.
        # Compute the theta multiplicative factor following Eq. 2.6 in Wen and Stephens (2010)
        shrink_factor *= theta_factor

        # The factor to convert the entries of the correlation matrix into corresponding covariances:
        to_cov_factor = maf_sd[row_indices]*maf_sd[csr_mat.indices]

        # Compute the new denominator for the Pearson correlation:
        # The shrunk standard deviation of SNP j multiplied by the shrunk standard deviations of each neighbor:
        shrunk_sd_prod = shrunk_sd[row_indices]*shrunk_sd[csr_mat.indices]

        # Finally, compute the shrunk LD matrix entries:
        csr_mat.data *= to_cov_factor*shrink_factor / shrunk_sd_prod

        # Update the LD matrix object inplace:
        ld_mat_obj.update_rows_inplace(csr_mat, start_row=start_row, end_row=end_row)

    return ld_mat_obj


def estimate_rows_per_chunk(rows, cols, dtype='int16', mem_size=128):
    """
    Estimate the number of rows per chunk for matrices conditional on the desired size of the chunk in MB.
    The estimator takes as input the number of rows, columns, data type, and projected size of the chunk in memory.

    :param rows: Total number of rows in the matrix.
    :param cols: Total number of columns. If sparse matrix with uneven columns, provide average column size.
    :param dtype: The data type for the matrix entries.
    :param mem_size: Size of the chunk in memory (MB)

    :return: The estimated number of rows per chunk.
    """

    matrix_size = rows * cols * np.dtype(dtype).itemsize / 1024 ** 2
    n_chunks = max(1, matrix_size // mem_size)

    return rows // n_chunks


def compute_ld_plink1p9(genotype_matrix,
                        ld_boundaries,
                        output_dir,
                        window_size_thresholds,
                        trim_boundaries=False,
                        temp_dir='temp',
                        overwrite=True,
                        dtype='int16',
                        compressor_name='zstd',
                        compression_level=7):

    """
    Compute LD matrices using plink 1.9.

    :param genotype_matrix: A plinkBEDGenotypeMatrix object
    :param ld_boundaries: An 2xM matrix of LD boundaries for every variant.
    :param output_dir: The output directory for the final LD matrix file (after processing).
    :param window_size_thresholds: A dictionary of the window size thresholds to pass to plink1.9.
    :param trim_boundaries: If True, trims the LD rows computed by plink. This is useful in cases where the
    window size varies per variant, but we ask plink to compute the LD for a fixed (maximum) window size.
    :param temp_dir: A temporary directory to store intermediate files (e.g. files created for and by plink).
    :param overwrite: If True, it overwrites any LD matrices in `output_dir`.
    :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
    :param compressor_name: The name of the compressor to use for the Zarr arrays.
    :param compression_level: The compression level to use for the Zarr arrays (1-9).

    :return: An LDMatrix object
    """

    from ...utils.executors import plink1Executor
    from ...GenotypeMatrix import plinkBEDGenotypeMatrix

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

    cmd = [
        f"--bfile {genotype_matrix.bed_file.replace('.bed', '')}",
        f"--keep {keep_file}",
        f"--extract {snp_keepfile}",
        "--keep-allele-order",
        f"--out {plink_output}",
        "--r gz",
        f"--ld-window {window_size_thresholds['window_size']}",
        f"--ld-window-kb {window_size_thresholds['kb_window_size']}"
    ]

    if 'cm_window_size' in window_size_thresholds:
        cmd.append(f"--ld-window-cm {window_size_thresholds['cm_window_size']}")

    # ---------------------------------------------------------
    # Test if plink1.9 version is compatible with setting the --ld-window-r2 flag:
    # This is important to account for due to differences in the behavior of plink1.9
    # across different versions.
    # See here for discussion of this behavior: https://github.com/shz9/viprs/issues/3

    plink1.verbose = False

    r2_flag_compatible = True

    from subprocess import CalledProcessError

    try:
        plink1.execute(["--r gz", "--ld-window-r2 0"])
    except CalledProcessError as e:
        if "--ld-window-r2 flag cannot be used with --r" in e.stderr.decode():
            r2_flag_compatible = False

    if r2_flag_compatible:
        cmd += ["--ld-window-r2 0"]

    plink1.verbose = True

    # ---------------------------------------------------------

    plink1.execute(cmd)

    # Convert from PLINK LD files to Zarr:
    fin_ld_store = osp.join(output_dir, 'chr_' + str(genotype_matrix.chromosome))

    # Compute the pandas chunk_size
    # The goal of this is to process chunks of the LD table without overwhelming memory resources:
    avg_ncols = int((ld_boundaries[1, :] - ld_boundaries[0, :]).mean())
    # NOTE: Estimate the rows per chunk using float32 because that's how we'll read the data from plink:
    rows_per_chunk = estimate_rows_per_chunk(ld_boundaries.shape[1], avg_ncols, dtype=np.float32)

    if rows_per_chunk > 0.1*ld_boundaries.shape[1]:
        pandas_chunksize = None
    else:
        pandas_chunksize = rows_per_chunk*avg_ncols // 2

    return LDMatrix.from_plink_table(f"{plink_output}.ld.gz",
                                     genotype_matrix.snps,
                                     fin_ld_store,
                                     ld_boundaries=[None, ld_boundaries][trim_boundaries],
                                     pandas_chunksize=pandas_chunksize,
                                     overwrite=overwrite,
                                     dtype=dtype,
                                     compressor_name=compressor_name,
                                     compression_level=compression_level)


def compute_ld_xarray(genotype_matrix,
                      ld_boundaries,
                      output_dir,
                      temp_dir='temp',
                      overwrite=True,
                      delete_original=True,
                      dtype='int16',
                      compressor_name='zstd',
                      compression_level=7):

    """
    Compute the Linkage Disequilibrium matrix or snp-by-snp
    correlation matrix assuming that the genotypes are represented
    by `xarray` or `dask`-like matrix objects. This function computes the
    entire X'X/N and stores the result on-disk in Zarr arrays. Then, we call the utilities
    from the `LDMatrix` class to sparsify the dense matrix according to the parameters
    specified by the `ld_boundaries` matrix.


    !!! Warning
        We don't recommend using this for large-scale genotype matrices.
        Use `compute_ld_plink1p9` instead if you have plink1.9 installed on your system.

    :param genotype_matrix: An `xarrayGenotypeMatrix` object
    :param ld_boundaries: An array of LD boundaries for every SNP
    :param output_dir: The output directory for the final LD matrix file.
    :param temp_dir: A temporary directory where to store intermediate results.
    :param overwrite: If True, overwrites LD matrices in `temp_dir` and `output_dir`, if they exist.
    :param delete_original: If True, it deletes the original dense matrix after generating the sparse alternative.
    :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
    :param compressor_name: The name of the compressor to use for the Zarr arrays.
    :param compression_level: The compression level to use for the Zarr arrays (1-9).

    :return: An LDMatrix object
    """

    from ...GenotypeMatrix import xarrayGenotypeMatrix

    assert isinstance(genotype_matrix, xarrayGenotypeMatrix)

    g_data = genotype_matrix.xr_mat.data

    # Re-chunk the array to optimize computational speed and efficiency:
    # New chunksizes:
    new_chunksizes = (min(1024, g_data.shape[0]), min(1024, g_data.shape[1]))
    g_data = g_data.rechunk(new_chunksizes)

    from ..transforms.genotype import standardize
    import dask.array as da

    # Standardize the genotype matrix and fill missing data with zeros:
    g_mat = standardize(g_data)
    g_mat = da.nan_to_num(g_mat)

    # Compute the full LD matrix and store to a temporary directory in the form of Zarr arrays:
    import warnings

    # Ignore performance-related warnings from Dask:
    with warnings.catch_warnings():

        if np.issubdtype(np.dtype(dtype), np.integer):
            # If the requested data type is integer, we need to convert
            # the data to `float32` to avoid overflow errors when computing the dot product:
            dot_dtype = np.float32
        else:
            dot_dtype = dtype

        warnings.simplefilter("ignore")
        ld_mat = (da.dot(g_mat.T, g_mat) / genotype_matrix.sample_size).astype(dot_dtype)
        ld_mat.to_zarr(temp_dir, overwrite=overwrite)

    fin_ld_store = osp.join(output_dir, 'chr_' + str(genotype_matrix.chromosome))

    # Load the dense matrix and transform it to a sparse matrix using utilities implemented in the
    # `LDMatrix` class:
    return LDMatrix.from_dense_zarr_matrix(temp_dir,
                                           ld_boundaries,
                                           fin_ld_store,
                                           overwrite=overwrite,
                                           delete_original=delete_original,
                                           dtype=dtype,
                                           compressor_name=compressor_name,
                                           compression_level=compression_level)
