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


cdef class LDWrapper:

    cdef:
        object _zarr
        bint in_memory
        list _data
        unsigned int index

    def __init__(self, ld_zarr_arr):

        self._zarr = ld_zarr_arr

        self._data = None
        self.in_memory = False
        self.index = 0

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
        return np.array(self.get_store_attr('SNP'))

    @property
    def ld_boundaries(self):
        return np.array(self.get_store_attr('LD boundaries'))

    @property
    def sample_size(self):
        return self.get_store_attr('Sample size')

    @property
    def maf(self):
        return self.get_store_attr('MAF')

    @property
    def bp_position(self):
        return np.array(self.get_store_attr('BP'))

    @property
    def cm_position(self):
        return np.array(self.get_store_attr('cM'))

    def store_size(self):
        """
        Returns the size of the compressed LD store in MB
        :return:
        """
        return self.store.getsize() / 1024 ** 2

    def estimate_uncompressed_size(self):
        """
        Returns an estimate of size of the uncompressed LD matrix in MB
        """
        ld_bounds = self.ld_boundaries

        n_rows = ld_bounds.shape[1]
        n_cols = n_cols = np.mean(ld_bounds[1, :] - ld_bounds[0, :])

        return n_rows * n_cols * np.dtype(np.float64).itemsize / 1024 ** 2

    def get_store_attr(self, attr):
        try:
            return self._zarr.attrs[attr]
        except KeyError:
            print(f"Warning: Attribute {attr} is not set!")
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
        cdef double[::1] v

        for d in self._zarr[start:end]:
            v = d.copy()
            self._data.append(v)

    def release(self):
        self._data = None
        self.in_memory = False
        self.index = 0

    def iterate(self):
        """
        May be useful in some contexts...
        """

        cdef unsigned int idx

        for idx in range(len(self)):
            if self.in_memory:
                curr_item = self._data[idx]
            else:
                if idx % self.chunk_size == 0:
                    self.load(start=idx, end=idx + self.chunk_size)

                curr_item = self._data[idx % self.chunk_size]

            yield curr_item

    def __getstate__(self):
        return self.store.path, self.in_memory

    def __setstate__(self, state):
        path, in_mem = state
        self._zarr = zarr.open(path)
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

        if self.in_memory:
            curr_item = self._data[self.index]
        else:
            if self.index % self.chunk_size == 0:
                self.load(start=self.index, end=self.index + self.chunk_size)

            curr_item = self._data[self.index % self.chunk_size]

        self.index += 1

        return curr_item
