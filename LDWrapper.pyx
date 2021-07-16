# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True


import numpy as np
cimport numpy as np


cdef class LDWrapper:

    cdef:
        public object _store
        bint in_memory
        public list _data
        public unsigned int index, size, chunk_size

    def __init__(self, ld_zarr_arr):

        self._zarr = ld_zarr_arr
        self.size = self._zarr.shape[0]

        self._data = None
        self.chunk_size = self.chunks[0]
        self.in_memory = False
        self.index = 0

    @property
    def zarr_store(self):
        return self._zarr.store

    @property
    def zarr_array(self):
        return self._zarr

    @property
    def chunks(self):
        return self._zarr.chunks

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

    def position(self, units='cM'):
        if units == 'cM':
            return np.array(self.get_store_attr('cM'))
        elif units == 'BP':
            return np.array(self.get_store_attr('BP'))
        else:
            raise KeyError(f"Position with {units} units is not available!")

    def mem_size(self):
        """
        Returns an estimate of size of the LD matrix in MB
        """
        ld_bounds = self.ld_boundaries

        n_rows = ld_bounds.shape[1]
        n_cols = np.mean(ld_bounds[:, 1] - ld_bounds[:, 1])

        return n_rows * n_cols * np.dtype(np.float64).itemsize / 1024 ** 2

    def get_store_attr(self, attr):
        try:
            return self._zarr.attrs[attr]
        except KeyError:
            print(f"Warning: Attribute {attr} is not set!")
            return None

    def load(self, start=0, end=None):

        if end is None:
            end = self.size
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

        for idx in range(self.size):
            if self.in_memory:
                curr_item = self._data[idx]
            else:
                if idx % self.chunk_size == 0:
                    self.load(start=idx, end=idx + self.chunk_size)

                curr_item = self._data[idx % self.chunk_size]

            yield curr_item

    def __getitem__(self, item):
        if self.in_memory:
            return self._data[item]
        else:
            return self._zarr[item]

    def __iter__(self):
        return self

    def __next__(self):

        if self.index == self.size:
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
