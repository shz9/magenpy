"""
Author: Shadi Zabad
Date: March 2021
"""

import numpy as np
from .c_utils import zarr_islice


class LDWrapper(object):
    """
    This class provides a wrapper for LD data structures
    stored in Zarr array formats.
    """

    def __init__(self, ld_zarr_store):

        self.store = ld_zarr_store
        self.in_memory = False
        self.data = None

    @property
    def size(self):
        """
        Returns the size of the LD matrix in MB
        :return:
        """
        ld_bounds = self.ld_boundaries

        n_rows = ld_bounds.shape[1]
        n_cols = np.mean(ld_bounds[:, 1] - ld_bounds[:, 1])

        return n_rows * n_cols * np.dtype(np.float64).itemsize / 1024 ** 2

    @property
    def chunks(self):
        return self.store.chunks

    @property
    def chromosome(self):
        return self.get_store_attr('Chromosome')

    @property
    def ld_estimator(self):
        return self.get_store_attr('LD Estimator')

    @property
    def snps(self):
        return self.get_store_attr('SNPs')

    @property
    def n_snps(self):
        return len(self.snps)

    @property
    def ld_boundaries(self):
        return self.get_store_attr('LD Boundaries')

    def get_store_attr(self, attr):
        try:
            return self.store.attrs[attr]
        except KeyError:
            return None

    def load(self):
        if not self.in_memory:
            self.data = self.store[:]
            self.in_memory = True

    def release(self):
        self.data = None
        self.in_memory = False

    def iterate(self):
        if self.data is None:
            for Di in zarr_islice(self.store):
                yield Di
        else:
            for i in range(self.data.shape[0]):
                yield self.data[i]

