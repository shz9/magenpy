
def standardize(g_mat, fill_na=True):
    """
    Standardize the genotype matrix, such that the columns (i.e. snps)
    have zero mean and unit variance.
    :param g_mat: A two-dimensional matrix (numpy, dask, xarray, etc.) where the rows are samples (individuals)
    and the columns are genetic variants.
    :param fill_na: If true, fill the missing values with zero after standardizing.

    :return: The standardized genotype matrix.

    """
    sg_mat = (g_mat - g_mat.mean(axis=0)) / g_mat.std(axis=0)

    if fill_na:
        sg_mat = sg_mat.fillna(0.)

    return sg_mat
