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
import os.path as osp
import numpy as np
cimport numpy as np
import pandas as pd
from magenpy.stats.ld.c_utils import zarr_islice


cdef class LDMatrix:

    cdef public:
        object _zarr
        bint in_memory
        list _data
        unsigned int index, _n_elements
        np.ndarray _ld_boundaries  # For caching
        np.ndarray _mask

    def __init__(self, ld_zarr_arr):

        self._zarr = ld_zarr_arr

        self._data = None
        self.in_memory = False
        self.index = 0

        self._ld_boundaries = None
        self._mask = None
        self._n_elements = self.shape[0]

    @classmethod
    def from_path(cls, ld_store_path):

        if '.zarray' in ld_store_path:
            ld_store_path = osp.dirname(ld_store_path)

        if osp.isfile(osp.join(ld_store_path, '.zarray')):
            ldm = zarr.open(ld_store_path)
            return cls(ldm)
        else:
            raise FileNotFoundError

    @property
    def n_elements(self):
        return self._n_elements

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
        return np.array(self._ld_boundaries)

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
            if self._mask is None:
                self.set_store_attr('LDScore', ld_score.tolist())
        else:
            ld_score = np.array(ld_score)

        if self._mask is not None:
            return ld_score[self._mask]
        else:
            return ld_score

    def filter_snps(self, extract_snps=None, extract_file=None):
        """
       Filter the LDMatrix to a subset of SNPs.
       :param extract_snps: A list or array of SNP IDs to keep.
       :param extract_file: A file containing the SNP IDs to keep.
       """

        assert extract_snps is not None or extract_file is not None

        if extract_snps is None:
            from .parsers.misc_parsers import read_snp_filter_file
            extract_snps = read_snp_filter_file(extract_file)

        from .utils.compute_utils import intersect_arrays

        extract_index = intersect_arrays(np.array(self.get_store_attr('SNP')),
                                         extract_snps,
                                         return_index=True)

        new_mask = np.zeros(self.shape[0], dtype=bool)
        new_mask[extract_index] = True

        self.set_mask(new_mask)

    def get_mask(self):
        if self._mask:
            return np.array(self._mask)

    def set_mask(self, mask):

        self._mask = mask

        if mask is None:
            # Update the number of elements:
            self._n_elements = self.shape[0]
        else:
            # Update the number of elements:
            self._n_elements = mask.sum()

            # Load the LD boundaries:
            ld_bounds = self.ld_boundaries

            # If the data is already in memory, reload:
            if self.in_memory:
                self.load(force_reload=True)

    def get_masked_boundaries(self):
        """
        Return the LD boundaries after applying the mask
        If the mask is not set, return the original boundaries
        """

        curr_ld_bounds = self.ld_boundaries

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

    def to_snp_table(self, col_subset=None):

        col_subset = col_subset or ['CHR', 'SNP', 'POS', 'A1', 'MAF', 'LDScore']

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

        return table[col_subset]

    def to_csr_matrix(self):
        """
        Convert the Zarr-formatted LD matrix into a sparse CSR matrix.
        """

        # Concatenate the data (entries of the LD matrix):
        data = np.concatenate([np.array(x) for x in self])

        # Stitch together the rows and columns for each data point:
        bounds = self.get_masked_boundaries()
        window_sizes = bounds[1, :] - bounds[0, :]

        rows = np.concatenate([np.repeat(i, ws) for i, ws in enumerate(window_sizes)])
        cols = np.concatenate([np.arange(bounds[0, i], bounds[1, i]) for i in range(bounds.shape[1])])

        from scipy.sparse import csr_matrix

        return csr_matrix((data, (rows, cols)), shape=(self._n_elements, self._n_elements))

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
        If the array is masked, it returns a (rough) estimate of the size of the
        elements that will be loaded into memory.
        """
        ld_bounds = self.get_masked_boundaries()

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

    def load(self, start=0, end=None, force_reload=False):

        if self.in_memory and not force_reload:
            return

        if end is None:
            end = len(self)
            if start == 0:
                self.in_memory = True

        cdef:
            unsigned int i
            long[:, ::1] ld_bounds = self.ld_boundaries

        self._data = []

        if self._mask is None:

            for d in zarr_islice(self._zarr, start, end):
                # TODO: Figure out a way to get around copying
                self._data.append(d.copy())

        else:

            for i, d in enumerate(zarr_islice(self._zarr, start, end), start):
                if self._mask[i]:
                    bound_start, bound_end = ld_bounds[:, i]
                    self._data.append(d[self._mask[bound_start: bound_end]].copy())
                else:
                    self._data.append(np.array([np.nan]))

    def release(self):
        self._data = None
        self.in_memory = False
        self.index = 0

    def iterate_chunks(self):
        # TODO: Incorporate the mask into the chunk iterator?
        for i in range(len(self) // self.chunk_size + 1):
            yield self.z_array[i*self.chunk_size:(i + 1)*self.chunk_size]

    def __getstate__(self):
        return self.store.path, self.in_memory, self._mask

    def __setstate__(self, state):

        path, in_mem, mask = state

        self._zarr = zarr.open(path)

        if mask is None:
            if in_mem:
                self.load()
        else:
            self.in_memory = in_mem
            self.set_mask(mask)

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

        cdef int i, curr_chunk, index_chunk

        if self.index == 0:
            curr_chunk = -1
        else:
            curr_chunk = (self.index - 1) // self.chunk_size

        if self._mask is not None:

            try:
                if not self._mask[self.index]:
                    for i in range(0, len(self) - self.index + 1):
                        if self._mask[self.index + i]:
                            break

                    self.index += i
            except IndexError:
                # Reached the end of the array
                self.index = len(self)
                pass

        if self.index == len(self):
            self.index = 0
            raise StopIteration

        cdef double[::1] next_item

        if self.in_memory:
            next_item = self._data[self.index]
        else:
            index_chunk = self.index // self.chunk_size
            if index_chunk > curr_chunk:
                self.load(start=index_chunk * self.chunk_size, end=(index_chunk + 1) * self.chunk_size)

            next_item = self._data[self.index % self.chunk_size]

        self.index += 1

        return next_item
