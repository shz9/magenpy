import numpy as np


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
