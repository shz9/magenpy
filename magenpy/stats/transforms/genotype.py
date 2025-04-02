
def standardize(g_mat):
    """
    Standardize the genotype matrix, such that the columns (i.e. variants)
    have zero mean and unit variance.
    :param g_mat: A two-dimensional matrix (numpy, dask, xarray, etc.) where the rows are samples (individuals)
    and the columns are genetic variants.

    :return: The standardized genotype matrix.

    """

    return (g_mat - g_mat.mean(axis=0)) / g_mat.std(axis=0)
