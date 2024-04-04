from ..LDMatrix import LDMatrix
import matplotlib.pylab as plt
import numpy as np


def plot_ld_matrix(ldm: LDMatrix,
                   row_subset=None,
                   display='full',
                   cmap='OrRd',
                   include_colorbar=True):
    """
    Plot a heatmap representing the LD matrix or portions of it.

    :param ldm: An instance of `LDMatrix`.
    :param row_subset: A boolean or integer index array for the subset of rows/columns to extract from the LD matrix.
    :param display: A string indicating what part of the matrix to display. Can be 'full', 'upper', 'lower'.
    If upper, only the upper triangle of the matrix will be displayed.
    If lower, only the lower triangle will be displayed.
    :param cmap: The color map for the LD matrix plot.
    :param include_colorbar: If True, include a colorbar in the plot.
    """

    if row_subset is None:
        row_subset = np.arange(ldm.shape[0])

    # TODO: Figure out a way to do this without loading the entire matrix:
    ldm.load(return_symmetric=True, fill_diag=True, dtype='float32')

    mat = ldm.csr_matrix[row_subset, :][:, row_subset].toarray()

    if display == 'upper':
        mat = np.triu(mat, k=1)
    elif display == 'lower':
        mat = np.tril(mat, k=1)

    plt.imshow(mat, cmap=cmap, vmin=-1., vmax=1.)

    if include_colorbar:
        plt.colorbar()

    plt.axis('off')
