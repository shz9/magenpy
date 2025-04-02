from tqdm import tqdm
import numpy as np
import pandas as pd


def match_chromosomes(chrom_1, chrom_2, check_patterns=('chr_', 'chr:', 'chr'), return_both=False):
    """
    Given two lists of chromosome IDs, this function returns the
    chromosomes that are common to both lists. By default, the returned chromosomes
    follow the data type and order of the first list. If `return_both` is set to True,
    the function returns the common chromosomes in both lists.

    The function also accounts for common ways to encode chromosomes, such as
    chr18, chr_18, 18, etc.

    :param chrom_1: A list or numpy array of chromosome IDs
    :param chrom_2: A list or numpy array of chromosome IDs
    :param check_patterns: A list of patterns to check for and replace in the chromosome IDs
    :param return_both: If True, return the common chromosomes in both lists
    """

    chrom_1 = np.array(list(chrom_1))
    chrom_2 = np.array(list(chrom_2))

    # First, convert the data types to strings:
    chr1_str = chrom_1.astype(str)
    chr2_str = chrom_2.astype(str)

    _, chr1_idx, chr2_idx = np.intersect1d(chr1_str, chr2_str, return_indices=True)

    if len(chr1_idx) < 1:
        # Replace patterns
        for pattern in check_patterns:
            chr1_str = np.char.replace(chr1_str, pattern, '')
            chr2_str = np.char.replace(chr2_str, pattern, '')

        _, chr1_idx, chr2_idx = np.intersect1d(chr1_str, chr2_str, return_indices=True)

    if len(chr1_idx) < 1:
        if return_both:
            return [], []
        else:
            return []
    else:
        if return_both:
            return chrom_1[chr1_idx], chrom_2[chr2_idx]
        else:
            return chrom_1[chr1_idx]


def merge_snp_tables(ref_table,
                     alt_table,
                     how='inner',
                     on='auto',
                     signed_statistics=('BETA', 'STD_BETA', 'Z'),
                     drop_duplicates=True,
                     correct_flips=True,
                     return_ref_indices=False,
                     return_alt_indices=False):
    """
    This function takes a reference SNP table with at least 3 columns ('SNP', 'A1', `A2`)
    and matches it with an alternative table that also has these 3 columns defined. In the most recent
    implementation, we allow users to merge on any set of columns that they wish by specifying the `on`
    parameter. For example, instead of `SNP`, the user can join the SNP tables on `CHR` and `POS`, the
    chromosome number and base pair position of the SNP.

    The manner in which the join operation takes place depends on the `how` argument.
    Currently, the function supports `inner` and `left` joins.

    The function removes duplicates if `drop_duplicates` parameter is set to True

    If `correct_flips` is set to True, the function will correct summary statistics in
    the alternative table `alt_table` (e.g. BETA, MAF) based whether the A1 alleles agree between the two tables.

    :param ref_table: The reference table (pandas dataframe).
    :param alt_table: The alternative table (pandas dataframe)
    :param how: `inner` or `left`
    :param on: Which columns to use as anchors when merging. By default, we automatically infer which columns
    to use, but the user can specify this directly. When `on='auto'`, we try to use `SNP` (i.e. rsID) if available.
    If not, we use `['CHR', 'POS']`. If neither are available, we raise a ValueError.
    :param signed_statistics: The columns with signed statistics to flip if `correct_flips` is set to True.
    :param drop_duplicates: Drop duplicate SNPs
    :param correct_flips: Correct SNP summary statistics that depend on status of alternative allele
    :param return_ref_indices: Return the indices of the remaining entries in the reference table before merging.
    :param return_alt_indices: Return the indices of the remaining entries in the alternative table before merging.
    """

    # Sanity checking steps:
    assert how in ('left', 'inner')
    for tab in (ref_table, alt_table):
        assert isinstance(tab, pd.DataFrame)
        if not all([col in tab.columns for col in ('A1', 'A2')]):
            raise ValueError("To merge SNP tables, we require that the columns `A1` and `A2` are present.")

    # Determine which columns to merge on:
    if on == 'auto':
        # Check that the `SNP` column is present in both tables:
        if all(['SNP' in tab.columns for tab in (ref_table, alt_table)]):
            on = ['SNP']
        # Check that the `CHR`, `POS` columns are present in both tables:
        elif all([col in tab.columns for col in ('CHR', 'POS') for tab in (ref_table, alt_table)]):
            on = ['CHR', 'POS']
        else:
            raise ValueError("Cannot merge SNP tables without specifying which columns to merge on.")
    elif isinstance(on, str):
        on = [on]

    ref_include = on + ['A1', 'A2']

    if return_ref_indices:
        ref_table.reset_index(inplace=True, names='REF_IDX')
        ref_include += ['REF_IDX']
    if return_alt_indices:
        alt_table.reset_index(inplace=True, names='ALT_IDX')

    merged_table = ref_table[ref_include].merge(alt_table, how=how, on=on)

    if drop_duplicates:
        merged_table.drop_duplicates(inplace=True, subset=on)

    if how == 'left':
        merged_table['A1_y'] = merged_table['A1_y'].fillna(merged_table['A1_x'])
        merged_table['A2_y'] = merged_table['A2_y'].fillna(merged_table['A2_x'])

    # Assign A1 to be the one derived from the reference table:
    merged_table['A1'] = merged_table['A1_x']
    merged_table['A2'] = merged_table['A2_x']

    # Detect cases where the correct allele is specified in both tables:
    matching_allele = np.all(merged_table[['A1_x', 'A2_x']].values == merged_table[['A1_y', 'A2_y']].values, axis=1)

    # Detect cases where the effect and reference alleles are flipped:
    flip = np.all(merged_table[['A2_x', 'A1_x']].values == merged_table[['A1_y', 'A2_y']].values, axis=1)

    if how == 'inner':
        # Variants to keep:
        if correct_flips:
            keep_snps = matching_allele | flip
        else:
            keep_snps = matching_allele

        # Keep only SNPs with matching alleles or SNPs with flipped alleles:
        merged_table = merged_table.loc[keep_snps, ]
        flip = flip[keep_snps]

    if correct_flips:

        flip = flip.astype(int)
        num_flips = flip.sum()

        if num_flips > 0:

            # If the user provided a single signed statistic as a string, convert to list first:
            if isinstance(signed_statistics, str):
                signed_statistics = [signed_statistics]

            # Loop over the signed statistics and correct them:
            for s_stat in signed_statistics:
                if s_stat in merged_table:
                    merged_table[s_stat] = (-2. * flip + 1.) * merged_table[s_stat]

            # Correct MAF:
            if 'MAF' in merged_table:
                merged_table['MAF'] = np.abs(flip - merged_table['MAF'])

    merged_table.drop(['A1_x', 'A1_y', 'A2_x', 'A2_y'], axis=1, inplace=True)

    return merged_table


def sumstats_train_test_split(gdl, prop_train=0.9, **kwargs):
    """
    Perform a train-test split on the GWAS summary statistics data.
    This function implemented the PUMAS procedure described in

    > Zhao, Z., Yi, Y., Song, J. et al. PUMAS: fine-tuning polygenic risk scores with GWAS summary statistics.
    Genome Biol 22, 257 (2021). https://doi.org/10.1186/s13059-021-02479-9

    Specifically, the function takes harmonized LD and summary statistics data (in the form of a
    `GWADataLoader` object) and samples the marginal beta values for the training set, conditional
    on the LD matrix and the proportion of training data specified by `prop_train`.

    :param gdl: A `GWADataLoader` object containing the harmonized GWAS summary statistics and LD data.
    :param prop_train: The proportion of training data to sample.
    :param kwargs: Additional keyword arguments to pass to the
    `multivariate_normal_conditional_sampling` function.

    :return: A dictionary with the sampled beta values for the training and test sets.
    """

    # Sanity checks:
    assert 0. < prop_train < 1., "The proportion of training data must be between 0 and 1."
    assert gdl.sumstats_table is not None, "The GWADataLoader object must have summary statistics initialized."
    assert gdl.ld is not None, "The GWADataLoader object must have LD matrices initialized."

    from ..stats.ld.utils import multivariate_normal_conditional_sampling

    prop_test = 1. - prop_train

    results = {}

    for chrom in gdl.chromosomes:

        assert gdl.ld[chrom].n_snps == gdl.sumstats_table[chrom].n_snps, (
            "The number of SNPs in the LD matrix and the summary statistics table must match. Invoke the "
            "`harmonize_data` method on the GWADataLoader object to ensure that the data is harmonized."
        )

        n_per_snp = gdl.sumstats_table[chrom].n_per_snp
        n = n_per_snp.max()  # Get the GWAS sample size

        # The covariance scale is computed based on the proportion of training data:
        cov_scale = 1. / (prop_train * n) - (1. / n)
        # Use the standardized marginal beta as the mean:
        mean = gdl.sumstats_table[chrom].standardized_marginal_beta

        # Sample the training beta values:
        sampled_train_beta = multivariate_normal_conditional_sampling(
            gdl.ld[chrom],
            mean=mean,
            cov_scale=cov_scale,
            **kwargs
        )

        # Calculate the test beta values:
        sampled_test_beta = mean * (1. / prop_test) - sampled_train_beta * (prop_train / prop_test)

        # Store the results:
        results[chrom] = {
            'train_beta': sampled_train_beta,
            'train_n': n_per_snp * prop_train,
            'test_beta': sampled_test_beta,
            'test_n': n_per_snp * prop_test
        }

    return results


def map_variants_to_genomic_blocks(variant_table,
                                   block_table,
                                   variant_pos_col='POS',
                                   block_boundary_cols=('block_start', 'block_end'),
                                   filter_unmatched=False):
    """
    Merge a variant table with a genomic block table. This function assumes that the
    variant table includes a column with the positions of the variants `POS` and the block table
    contains the start and end positions of each block.

     !!! warning
         This function assumes that the tables contains data for a single chromosome only.

    :param variant_table: A pandas dataframe with variant information
    :param block_table: A pandas dataframe with block information
    :param variant_pos_col: The name of the column in the variant table that contains the positions. By default,
    this is set to `POS`.
    :param block_boundary_cols: A tuple of two strings specifying the column names in the block table that contain
    the start and end positions of the blocks. By default, this is set to `('block_start', 'block_end')`.
    :param filter_unmatched: If True, filter out variants that were not matched to a block.
    """

    # Sanity checks:
    assert variant_pos_col in variant_table.columns
    assert all([col in block_table.columns for col in block_boundary_cols])

    # Sort the variant table by position:
    variant_table.sort_values(variant_pos_col, inplace=True)

    # Merge the two dataframes to assign each SNP to its corresponding block:
    merged_df = pd.merge_asof(variant_table, block_table,
                              left_on=variant_pos_col, right_on=block_boundary_cols[0],
                              direction='backward')

    if filter_unmatched:
        # Filter merged_df to only include variants that were matched properly with a block:
        merged_df = merged_df.loc[(merged_df[variant_pos_col] >= merged_df[block_boundary_cols[0]]) &
                                  (merged_df[variant_pos_col] < merged_df[block_boundary_cols[1]])]

    return merged_df


def identify_mismatched_snps(gdl,
                             chrom=None,
                             n_iter=10,
                             G=100,
                             p_dentist_threshold=5e-8,
                             p_gwas_threshold=1e-2,
                             rsq_threshold=.95,
                             max_removed_per_iter=.005):
    """
    This function implements a simple quality control procedures
    that checks that the GWAS summary statistics (Z-scores)
    are consistent with the LD reference panel. This is done
    using a simplified version of the framework outlined in the DENTIST paper:

    Improved analyses of GWAS summary statistics by reducing data heterogeneity and errors
    Chen et al. 2021

    Compared to DENTIST, the simplifications we make are:
        -   For each SNP, we sample one neighboring SNP at a time and compute the T statistic
            using that neighbor's information. The benefit of this is that we don't need to
            invert any matrices, so it's a fast operation to run.
        -   To arrive at a more robust estimate, we sample up to `k` neighbors and average
            the T-statistic across those `k` neighbors.

    NOTE: May need to re-implement this to apply some of the constraints genome-wide
    rather than on a per-chromosome basis.

    :param gdl: A `GWADataLoader` object
    :param chrom: Perform checking only on chromosome `chrom`
    :param n_iter: Number of iterations
    :param G: The number of neighboring SNPs to sample (default: 100)
    :param p_dentist_threshold: The Bonferroni-corrected P-value threshold (default: 5e-8)
    :param p_gwas_threshold: The nominal GWAS P-value threshold for partitioning variants (default: 1e-2)
    :param rsq_threshold: The R^2 threshold to select neighbors (neighbor's squared
    correlation coefficient must be less than specified threshold).
    :param max_removed_per_iter: The maximum proportion of variants removed in each iteration
    """

    # Import required modules / functions:
    from scipy import stats

    # Data preparation:
    if chrom is None:
        chromosomes = gdl.chromosomes
    else:
        chromosomes = [chrom]

    shapes = gdl.shapes
    mismatched_dict = {c: np.repeat(False, gdl.shapes[c])
                       for c in chromosomes}

    p_gwas_above_thres = {c: gdl.sumstats_table[c].p_value > p_gwas_threshold for c in chromosomes}
    gwas_thres_size = {c: p.sum() for c, p in p_gwas_above_thres.items()}
    converged = {c: False for c in chromosomes}

    for j in tqdm(range(n_iter),
                  total=n_iter,
                  desc="Identifying mismatched SNPs..."):

        for chrom in chromosomes:

            if converged[chrom]:
                continue

            ld_bounds = gdl.ld[chrom].get_masked_boundaries()
            z = gdl.z_scores[chrom]  # Obtain the z-scores
            t = np.zeros_like(z)

            # Loop over the LD matrix:
            for i, r in enumerate(gdl.ld[chrom]):

                # If the number of neighbors is less than 10, skip...
                if mismatched_dict[chrom][i] or len(r) < 10:
                    continue

                start_idx = ld_bounds[0, i]
                # Select neighbors randomly
                # Note: We are excluding neighbors whose squared correlation coefficient
                # is greater than pre-specified threshold:
                p = (np.array(r)**2 < rsq_threshold).astype(float)
                p /= p.sum()

                neighbor_idx = np.random.choice(len(r), p=p, size=G)
                neighbor_r = np.array(r)[neighbor_idx]

                # Predict the z-score of snp i, given the z-scores of its neighbors:
                pred_z = neighbor_r*z[start_idx + neighbor_idx]

                # Compute the Td statistic for each neighbor and average:
                t[i] = ((z[i] - pred_z) ** 2 / (1. - neighbor_r**2)).mean()

            # Compute the DENTIST p-value assuming a Chi-Square distribution with 1 dof.
            dentist_pval = 1. - stats.chi2.cdf(t, 1)
            # Use a Bonferroni correction to select mismatched SNPs:
            mismatched_snps = dentist_pval < p_dentist_threshold

            if mismatched_snps.sum() < 1:
                # If no new mismatched SNPs are identified, stop iterating...
                converged[chrom] = True
            elif j == n_iter - 1:
                # If this is the last iteration, take all identified SNPs
                mismatched_dict[chrom] = (mismatched_dict[chrom] | mismatched_snps)
            else:

                # Otherwise, we will perform the iterative filtering procedure
                # by splitting variants based on their GWAS p-values:

                # (1) Group S1: SNPs to remove from P_GWAS > threshold:
                mismatch_above_thres = mismatched_snps & p_gwas_above_thres[chrom]
                n_mismatch_above_thres = mismatch_above_thres.sum()
                prop_mismatch_above_thres = n_mismatch_above_thres / gwas_thres_size[chrom]

                if n_mismatch_above_thres < 1:
                    # If no mismatches are detected above the threshold, filter
                    # the mismatches below the threshold and continue...
                    mismatched_dict[chrom] = (mismatched_dict[chrom] | mismatched_snps)
                    continue

                # Sort the DENTIST p-values by index:
                sort_d_pval_idx = np.argsort(dentist_pval)

                if prop_mismatch_above_thres > max_removed_per_iter:
                    idx_to_keep = sort_d_pval_idx[mismatch_above_thres][
                                  int(gwas_thres_size[chrom]*max_removed_per_iter):]
                    mismatch_above_thres[idx_to_keep] = False

                # (2) Group S2: SNPs to remove from P_GWAS < threshold

                # Find mismatched variants below the threshold:
                mismatch_below_thres = mismatched_snps & (~p_gwas_above_thres[chrom])
                n_mismatch_below_thres = mismatch_below_thres.sum()
                prop_mismatch_below_thres = n_mismatch_below_thres / (shapes[chrom] - gwas_thres_size[chrom])

                # For the mismatched variants below the threshold,
                # we remove the same proportion as the variants above the threshold:
                prop_keep_below_thres = min(max_removed_per_iter, prop_mismatch_above_thres)

                if prop_mismatch_below_thres > prop_keep_below_thres:
                    idx_to_keep = sort_d_pval_idx[mismatch_below_thres][
                                  int((shapes[chrom] - gwas_thres_size[chrom]) * prop_keep_below_thres):
                                  ]
                    mismatch_below_thres[idx_to_keep] = False

                # Update the number of variants above the threshold:
                gwas_thres_size[chrom] -= mismatch_above_thres.sum()

                # Update the mismatched dictionary:
                mismatched_dict[chrom] = (mismatched_dict[chrom] | mismatch_below_thres | mismatch_above_thres)

    return mismatched_dict


def get_shared_distance_matrix(tree, tips=None):
    """
    This function takes a Biopython tree and returns the
    shared distance matrix, i.e. for a pair of clades or populations,
    time to most recent common ancestor of the pair minus the time of
    the most recent common ancestor (MRCA).
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
        c.branch_length /= max_depth

    return tree.root.branch_length + get_shared_distance_matrix(tree)


def quantize(floats, int_dtype=np.int8):
    """
    Quantize floating point numbers to the specified integer type.
    NOTE: Assumes that the floats are in the range [-1, 1].
    :param floats: A numpy array of floats
    :param int_dtype: The integer type to quantize to.
    """

    # Infer the boundaries from the integer type
    info = np.iinfo(int_dtype)

    # NOTE: We add 1 to the info.min here to force the zero point to be exactly at 0.
    # See discussions on Scale Quantization Mapping.

    # Use as much in-place operations as possible
    # (Currently, we copy the data twice)
    scaled_floats = floats*info.max
    np.round(scaled_floats, out=scaled_floats)
    np.clip(scaled_floats, info.min + 1, info.max, out=scaled_floats)

    return scaled_floats.astype(int_dtype)


def dequantize(ints, float_dtype=np.float32):
    """
    Dequantize integers to the specified floating point type.
    NOTE: Assumes original floats are in the range [-1, 1].
    :param ints: A numpy array of integers
    :param float_dtype: The floating point data type to dequantize the integers to.
    """

    # Infer the boundaries from the integer type
    info = np.iinfo(ints.dtype)

    dq = ints.astype(float_dtype)
    dq /= info.max  # in-place multiplication

    return dq


def multinomial_rvs(n, p):
    """
    Copied from Warren Weckesser:
    https://stackoverflow.com/a/55830796

    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out
