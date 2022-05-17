from tqdm import tqdm
import numpy as np
from scipy import stats


def merge_snp_tables(ref_table, alt_table,
                     how='inner', drop_duplicates=True,
                     correct_flips=True):
    """
    This function takes a reference SNP table with at least 2 columns ('SNP', 'A1')
    and matches it with an alternative table that also has these 2 columns defined.
    The manner in which the join operation takes place depends on the `how` argument.
    Currently, the function supports `inner` and `left` joins.

    The function removes duplicates if `drop_dupicates` parameter is set to True

    If `correct_flips` is set to True, the function will correct summary statistics in
    the alternative table (e.g. BETA, MAF) based whether the A1 alleles agree between the two tables.


    :param ref_table: The reference table (pandas dataframe)
    :param alt_table: The alternative table (pandas dataframe)
    :param how: `inner` or `left`
    :param drop_duplicates: Drop duplicate SNPs
    :param correct_flips: Correct SNP summary statistics that depend on status of alternative allele
    """

    assert how in ('left', 'inner')

    merged_table = ref_table.merge(alt_table, how=how, on='SNP')

    if drop_duplicates:
        merged_table.drop_duplicates(inplace=True, subset=['SNP'])

    if how == 'left':
        merged_table['A1_y'] = merged_table['A1_y'].fillna(merged_table['A1_x'])

    # Assign A1 to be the one derived from the reference table:
    merged_table['A1'] = merged_table['A1_x']

    if correct_flips:

        merged_table['flip'] = np.not_equal(merged_table['A1_x'].values, merged_table['A1_y'].values).astype(int)

        if merged_table['flip'].sum() > 0:

            # Correct betas:
            if 'BETA' in merged_table:
                merged_table['BETA'] = (-2.*merged_table['flip'] + 1.) * merged_table['BETA']

            # Correct Z-scores:
            if 'Z' in merged_table:
                merged_table['Z'] = (-2.*merged_table['flip'] + 1.) * merged_table['Z']

            # Correct MAF:
            if 'MAF' in merged_table:
                merged_table['MAF'] = np.abs(merged_table['flip'] - merged_table['MAF'])

    return merged_table


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

    :param gdl: A GWASDataLoader object
    :param chrom: A chromosome
    :param n_iter: Number of iterations
    :param G: The number of neighboring SNPs to sample (default: 100)
    :param p_dentist_threshold: The Bonferroni-corrected P-value threshold (default: 5e-8)
    :param p_gwas_threshold: The nominal GWAS P-value threshold for partitioning variants (default: 1e-2)
    :param rsq_threshold: The R^2 threshold to select neighbors (neighbor's squared
    correlation coefficient must be less than specified threshold).
    :param max_removed_per_iter: The maximum proportion of variants removed in each iteration
    """

    if chrom is None:
        chromosomes = gdl.chromosomes
    else:
        chromosomes = [chrom]

    shapes = gdl.shapes
    mismatched_dict = {c: np.repeat(False, c_size)
                       for c, c_size in gdl.shapes.items()}

    p_gwas_above_thres = {c: p_val > p_gwas_threshold for c, p_val in gdl.p_values.items()}
    gwas_thres_size = {c: p.sum() for c, p in p_gwas_above_thres.items()}
    converged = {c: False for c in gdl.chromosomes}

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
        c.branch_length /= max_depth

    return tree.root.branch_length + get_shared_distance_matrix(tree)


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
