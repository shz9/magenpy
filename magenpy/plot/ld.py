from ..LDMatrix import LDMatrix
import matplotlib.pylab as plt
import numpy as np


def plot_ld_matrix(ldm: LDMatrix,
                   variant_subset=None,
                   start_row=None,
                   end_row=None,
                   symmetric=False,
                   cmap='RdBu',
                   include_colorbar=True):
    """
    Plot a heatmap representing the LD matrix or portions of it.

    :param ldm: An instance of `LDMatrix`.
    :param variant_subset: A list of variant rsIDs to subset the LD matrix.
    :param start_row: The starting row index for the LD matrix plot.
    :param end_row: The ending row index (not inclusive) for the LD matrix plot.
    :param symmetric: If True, plot the symmetric version of the LD matrix.
    Otherwise, plot the upper triangular part.
    :param cmap: The color map for the LD matrix plot.
    :param include_colorbar: If True, include a colorbar in the plot.
    """

    curr_mask = None

    if variant_subset is not None:
        curr_mask = ldm.get_mask()
        ldm.reset_mask()
        ldm.filter_snps(variant_subset)

    ld_mat = ldm.load_data(start_row=start_row,
                           end_row=end_row,
                           return_symmetric=symmetric,
                           return_square=True,
                           return_as_csr=True,
                           dtype=np.float32).todense()

    plt.imshow(ld_mat, cmap=cmap, vmin=-1., vmax=1., interpolation='none')

    if include_colorbar:
        plt.colorbar()

    # Reset the original mask:
    if curr_mask is not None:
        ldm.set_mask(curr_mask)

    plt.axis('off')
