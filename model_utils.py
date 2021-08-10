import numpy as np


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
