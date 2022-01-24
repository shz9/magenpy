# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True


import zarr
import numpy as np
cimport numpy as np
import pandas as pd
from .c_utils import zarr_islice


cdef class LDMatrix:

    cdef public:
        object _zarr
        bint in_memory
        list _data
        unsigned int index
        long[:, ::1] _ld_boundaries  # For caching
        np.ndarray _mask

    def __init__(self, ld_zarr_arr):

        self._zarr = ld_zarr_arr

        self._data = None
        self.in_memory = False
        self.index = 0

        self._ld_boundaries = None
        self._mask = None

    @classmethod
    def from_path(cls, ld_store_path):
        ldw = zarr.open(ld_store_path)
        return cls(ldw)

    @property
    def shape(self):
        return self._zarr.shape

    @property
    def store(self):
        return self._zarr.store

    @property
    def z_array(self):
        return self._zarr

    @property
    def chunks(self):
        return self._zarr.chunks

    @property
    def chunk_size(self):
        return self.chunks[0]

    @property
    def chromosome(self):
        return self.get_store_attr('Chromosome')

    @property
    def ld_estimator(self):
        return self.get_store_attr('LD estimator')

    @property
    def estimator_properties(self):
        return self.get_store_attr('Estimator properties')

    @property
    def snps(self):

        z_snps = np.array(self.get_store_attr('SNP'))

        if self._mask is not None:
            return z_snps[self._mask]
        else:
            return z_snps

    @property
    def ld_boundaries(self):

        if self._ld_boundaries is None:
            self._ld_boundaries = np.array(self.get_store_attr('LD boundaries'))

        return self._ld_boundaries

    @property
    def sample_size(self):
        return self.get_store_attr('Sample size')

    @property
    def a1(self):

        a1 = np.array(self.get_store_attr('A1'))

        if self._mask is not None:
            return a1[self._mask]
        else:
            return a1

    @property
    def maf(self):

        maf = np.array(self.get_store_attr('MAF'))

        if self._mask is not None:
            return maf[self._mask]
        else:
            return maf

    @property
    def bp_position(self):

        bp = np.array(self.get_store_attr('BP'))

        if self._mask is not None:
            return bp[self._mask]
        else:
            return bp

    @property
    def cm_position(self):

        cm = np.array(self.get_store_attr('cM'))

        if self._mask is not None:
            return cm[self._mask]
        else:
            return cm

    @property
    def ld_score(self):

        ld_score = self.get_store_attr('LDScore')

        if ld_score is None:
            ld_score = self.compute_ld_scores()
            self.set_store_attr('LDScore', ld_score.tolist())
        else:
            ld_score = np.array(ld_score)

        if self._mask is not None:
            return ld_score[self._mask]
        else:
            return ld_score

    def get_mask(self):
        if self._mask is None:
            return None
        else:
            return np.array(self._mask)

    def set_mask(self, mask):
        self._mask = mask
        # Load the LD boundaries:
        ld_bounds = self.ld_boundaries

    def get_masked_boundaries(self):
        """
        Return the LD boundaries after applying the mask
        If the mask is not set, return the original boundaries
        """

        curr_ld_bounds = np.array(self.ld_boundaries)

        if self._mask is None:
            return curr_ld_bounds
        else:
            # Number of excluded elements up to (and including) position i
            n_excluded = np.cumsum(~self._mask)
            # Number of excluded elements up to (not including) position i
            n_excluded_before = n_excluded - (~self._mask).astype(int)

            # New start position:
            start_pos = curr_ld_bounds[0, :] - n_excluded_before[curr_ld_bounds[0, :]]
            # New end position:
            end_pos = curr_ld_bounds[1, :] - n_excluded[curr_ld_bounds[1, :] - 1]

            # Return masked boundaries array:
            return np.array([start_pos[self._mask], end_pos[self._mask]])

    def to_snp_table(self, col_subset=('CHR', 'SNP', 'POS', 'A1', 'MAF')):

        table = pd.DataFrame({'SNP': self.snps})

        for col in col_subset:
            if col == 'CHR':
                table['CHR'] = self.chromosome
            if col == 'POS':
                table['POS'] = self.bp_position
            if col == 'cM':
                table['cM'] = self.cm_position
            if col == 'A1':
                table['A1'] = self.a1
            if col == 'MAF':
                table['MAF'] = self.maf
            if col == 'LDScore':
                table['LDScore'] = self.ld_score

        return table[list(col_subset)]

    def compute_ld_scores(self, corrected=True):
        """
        Computes the LD scores for all SNPs in the LD matrix.
        :param corrected: Use the sample-size corrected estimator (Bulik-Sullivan et al. 2015)
        """

        ld_scores = []
        cdef int n = self.sample_size

        for snp_ld in self:

            ldsc = np.array(snp_ld) ** 2

            if corrected:
                ldsc = ldsc - (1. - ldsc) / (n - 2)

            ld_scores.append(ldsc.sum())

        return np.array(ld_scores)

    def store_size(self):
        """
        Returns the size of the compressed LD store in MB
        """
        return self.store.getsize() / 1024 ** 2

    def estimate_uncompressed_size(self):
        """
        Returns an estimate of size of the uncompressed LD matrix in MB
        """
        ld_bounds = np.array(self.ld_boundaries)

        return (ld_bounds[1, :] - ld_bounds[0, :]).sum() * np.dtype(np.float64).itemsize / 1024 ** 2

    def get_store_attr(self, attr):
        try:
            return self._zarr.attrs[attr]
        except KeyError:
            print(f"Warning: Attribute '{attr}' is not set!")
            return None

    def set_store_attr(self, attr, value):
        try:
            self._zarr.attrs[attr] = value
        except Exception as e:
            raise e

    def load(self, start=0, end=None):

        if end is None:
            end = len(self)
            if start == 0:
                self.in_memory = True

        self._data = []

        for d in zarr_islice(self._zarr, start, end):
            # TODO: Figure out a way to get around copying
            self._data.append(d.copy())

    def release(self):
        self._data = None
        self.in_memory = False
        self.index = 0

    def __getstate__(self):
        return self.store.path, self.in_memory, self._mask

    def __setstate__(self, state):

        path, in_mem, mask = state

        self._zarr = zarr.open(path)
        self.set_mask(mask)

        if in_mem:
            self.load()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        if self.in_memory:
            return self._data[item]
        else:
            return self._zarr[item]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):

        if self.index == len(self):
            self.index = 0
            raise StopIteration

        cdef double[::1] next_item

        if self._mask is None:

            if self.in_memory:
                next_item = self._data[self.index]
            else:
                if self.index % self.chunk_size == 0:
                    self.load(start=self.index, end=self.index + self.chunk_size)

                next_item = self._data[self.index % self.chunk_size]

        else:
            # If there's a mask, we need to fetch the next unmasked element
            # and set the index to point to the following one.

            if self.index == 0:
                curr_chunk = -1
            else:
                curr_chunk = (self.index - 1) // self.chunk_size

            # Keep iterating until finding a SNP that is included in the mask:
            while not self._mask[self.index]:
                self.index += 1

                if self.index == len(self):
                    self.index = 0
                    raise StopIteration

            # Extract the LD boundaries:
            bound_start, bound_end = self._ld_boundaries[:, self.index]

            if self.in_memory:
                next_item = self._data[self.index][self._mask[bound_start: bound_end]]
            else:

                index_chunk = self.index // self.chunk_size
                if index_chunk > curr_chunk:
                    self.load(start=index_chunk*self.chunk_size , end=(index_chunk + 1)*self.chunk_size)

                next_item = self._data[self.index % self.chunk_size][self._mask[bound_start: bound_end]]

        self.index += 1

        return next_item