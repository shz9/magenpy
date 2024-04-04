import zarr
import os.path as osp
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity, triu, diags
from .utils.model_utils import quantize, dequantize


class LDMatrix(object):
    """
    A class that represents Linkage-Disequilibrium (LD) matrices, which record
    the SNP-by-SNP pairwise correlations in a sample of genetic data. The class
    provides various functionalities for initializing, storing, loading, and
    performing computations with LD matrices. The LD matrices are stored in a
    hierarchical format using the `Zarr` library, which allows for efficient
    storage and retrieval of the data.

    The class provides the following functionalities:

    * Initialize an `LDMatrix` object from plink's LD table files.
    * Initialize an `LDMatrix` object from a sparse CSR matrix.
    * Initialize an `LDMatrix` object from a Zarr array store.
    * Compute LD scores for each SNP in the LD matrix.
    * Filter the LD matrix based on SNP indices or ranges.

    The Zarr hierarchy is structured as follows:

    * `chr_22.zarr`: The Zarr group.
        * `matrix`: The subgroup containing the data of the LD matrix in Scipy Sparse CSR matrix format.
            * `data`: The array containing the non-zero entries of the LD matrix.
            * `indptr`: The array containing the index pointers for the CSR matrix.
        * `metadata`: The subgroup containing the metadata for variants included in the LD matrix.
            * `snps`: The array containing the SNP rsIDs.
            * `a1`: The array containing the alternative alleles.
            * `a2`: The array containing the reference alleles.
            * `maf`: The array containing the minor allele frequencies.
            * `bp`: The array containing the base pair positions.
            * `cm`: The array containing the centi Morgan positions.
            * `ldscore`: The array containing the LD scores.
        * `attrs`: A JSON-style metadata object containing general information about how the LD matrix
        was calculated, including the chromosome number, sample size, genome build, LD estimator,
        and estimator properties.

    :ivar _zg: The Zarr group object that stores the LD matrix and its metadata.
    :ivar _mat: The in-memory CSR matrix object.
    :ivar in_memory: A boolean flag indicating whether the LD matrix is in memory.
    :ivar is_symmetric: A boolean flag indicating whether the LD matrix is symmetric.
    :ivar index: An integer index for the current SNP in the LD matrix (useful for iterators).
    :ivar _mask: A boolean mask for filtering the LD matrix.

    """

    def __init__(self, zarr_group, symmetric=False):
        """
        Initialize an `LDMatrix` object from a Zarr group store.

        :param zarr_group: The Zarr group object that stores the LD matrix.
        :param symmetric: A boolean flag indicating whether to represent the LD matrix as symmetric.
        """

        # Checking the input for correct formatting:
        # First, it has to be a Zarr group:
        assert isinstance(zarr_group, zarr.hierarchy.Group)
        # Second, it has to have a group called `matrix`:
        assert 'matrix' in list(zarr_group.group_keys())

        # Third, all the sparse array keys must be present:
        arr_keys = list(zarr_group['matrix'].array_keys())
        assert all([arr in arr_keys
                    for arr in ('data', 'indptr')])

        self._zg = zarr_group

        self._mat = None
        self.in_memory = False
        self.is_symmetric = symmetric
        self.index = 0

        self._mask = None

    @classmethod
    def from_path(cls, ld_store_path):
        """
        Initialize an `LDMatrix` object from a pre-computed Zarr group store.
        :param ld_store_path: The path to the Zarr array store on the filesystem.

        !!! seealso "See Also"
            * [from_dir][magenpy.LDMatrix.LDMatrix.from_dir]

        """

        for level in range(2):
            try:
                ld_group = zarr.open_group(ld_store_path, mode='r')
                return cls(ld_group)
            except zarr.hierarchy.GroupNotFoundError as e:
                if level < 1:
                    ld_store_path = osp.dirname(ld_store_path)
                else:
                    raise e

    @classmethod
    def from_dir(cls, ld_store_path):
        """
        Initialize an `LDMatrix` object from a Zarr array store.
        :param ld_store_path: The path to the Zarr array store on the filesystem.

        !!! seealso "See Also"
            * [from_path][magenpy.LDMatrix.LDMatrix.from_path]
        """
        return cls.from_path(ld_store_path)

    @classmethod
    def from_csr(cls,
                 csr_mat,
                 store_path,
                 overwrite=False,
                 dtype='int16',
                 compressor_name='lz4',
                 compression_level=5):
        """
        Initialize an LDMatrix object from a sparse CSR matrix.

        :param csr_mat: The sparse CSR matrix.
        :param store_path: The path to the Zarr LD store where the data will be stored.
        :param overwrite: If True, it overwrites the LD store at `store_path`.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compressor or compression algorithm to use with Zarr.
        :param compression_level: The compression level to use with the compressor (1-9).
        """

        dtype = np.dtype(dtype)

        # Get the upper triangular part of the matrix:
        triu_mat = triu(csr_mat, k=1, format='csr')

        # Check that the non-zeros are contiguous around the diagonal with no gaps.
        # If there are gaps, eliminate them or raise an error.
        if np.diff(triu_mat.indices).max() > 1:
            # TODO: Figure out a way to fix this automatically for the user?
            raise ValueError("The non-zero entries of the LD matrix are not contiguous around the diagonal.")

        # Create hierarchical storage with zarr groups:
        store = zarr.DirectoryStore(store_path)
        z = zarr.group(store=store, overwrite=overwrite)

        # Create a compressor object:
        compressor = zarr.Blosc(cname=compressor_name, clevel=compression_level)

        # First sub-hierarchy stores the information for the sparse LD matrix:
        mat = z.create_group('matrix')
        if np.issubdtype(dtype, np.integer):
            mat.array('data', quantize(triu_mat.data, int_dtype=dtype), dtype=dtype, compressor=compressor)
        else:
            mat.array('data', triu_mat.data.astype(dtype), dtype=dtype, compressor=compressor_name)

        # Store the index pointer:
        mat.array('indptr', triu_mat.indptr,
                  dtype=np.int32, compressor=compressor)

        return cls(z)

    @classmethod
    def from_plink_table(cls,
                         plink_ld_file,
                         snps,
                         store_path,
                         pandas_chunksize=None,
                         overwrite=False,
                         dtype='int16',
                         compressor_name='lz4',
                         compression_level=5):
        """
        Construct a Zarr LD matrix using output tables from plink1.9.
        This class method takes the following inputs:

        :param plink_ld_file: The path to the plink LD table file.
        :param snps: An iterable containing the list of SNPs in the LD matrix.
        :param store_path: The path to the Zarr LD store.
        :param pandas_chunksize: If the LD table is large, provide chunk size
        (i.e. number of rows to process at each step) to keep memory footprint manageable.
        :param overwrite: If True, it overwrites the LD store at `store_path`.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compressor or compression algorithm to use with Zarr.
        :param compression_level: The compression level to use with the compressor (1-9).
        """

        dtype = np.dtype(dtype)

        # Create hierarchical storage with zarr groups:
        store = zarr.DirectoryStore(store_path)
        z = zarr.group(store=store, overwrite=overwrite)

        # Create a compressor object:
        compressor = zarr.Blosc(cname=compressor_name, clevel=compression_level)

        # First sub-hierarchy stores the information for the sparse LD matrix:
        mat = z.create_group('matrix')
        mat.empty('data', shape=len(snps)**2, dtype=dtype, compressor=compressor)

        # Create a chunked iterator with pandas:
        # Chunk size will correspond to the average chunk size for the Zarr array:
        ld_chunks = pd.read_csv(plink_ld_file,
                                sep=r'\s+',
                                usecols=['SNP_A', 'R'],
                                engine='c',
                                chunksize=pandas_chunksize,
                                dtype={'SNP_A': str, 'R': np.float32})

        if pandas_chunksize is None:
            ld_chunks = [ld_chunks]

        # Create a dictionary mapping SNPs to their indices:
        snp_dict = dict(zip(snps, np.arange(len(snps))))

        indptr_counts = np.zeros(len(snps), dtype=np.int32)

        total_len = 0

        # For each chunk in the LD file:
        for ld_chunk in ld_chunks:

            # Create an indexed LD chunk:
            ld_chunk['row_index'] = ld_chunk['SNP_A'].map(snp_dict)

            # Add LD data to the zarr array:
            if np.issubdtype(dtype, np.integer):
                mat['data'][total_len:total_len + len(ld_chunk)] = quantize(ld_chunk['R'].values, int_dtype=dtype)
            else:
                mat['data'][total_len:total_len + len(ld_chunk)] = ld_chunk['R'].values.astype(dtype)

            total_len += len(ld_chunk)

            # Group by the row index:
            grouped_ridx = ld_chunk.groupby('row_index').size()

            # Add the number of entries to indptr_counts:
            indptr_counts[grouped_ridx.index] += grouped_ridx.values

        # Get the final indptr by computing cumulative sum:
        indptr = np.insert(np.cumsum(indptr_counts), 0, 0)
        # Store indptr in the zarr group:
        mat.array('indptr', indptr, dtype=np.int32, compressor=compressor)

        # Resize the data array:
        mat['data'].resize(total_len)

        return cls(z)

    @classmethod
    def from_dense_zarr_matrix(cls,
                               dense_zarr,
                               ld_boundaries,
                               store_path,
                               overwrite=False,
                               delete_original=False,
                               dtype='int16',
                               compressor_name='lz4',
                               compression_level=5):
        """
         Initialize a new LD matrix object using a Zarr array object. This method is
         useful for converting a dense LD matrix computed using Dask (or other distributed computing
         software) to a sparse or banded one.

         :param dense_zarr: The path to the dense Zarr array object.
         :param ld_boundaries: The LD boundaries for each SNP in the LD matrix (delineates the indices of
            the leftmost and rightmost neighbors of each SNP).
         :param store_path: The path where to store the new LD matrix.
         :param overwrite: If True, it overwrites the LD store at `store_path`.
         :param delete_original: If True, it deletes the original dense LD matrix.
         :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
            and integer quantized data types int8 and int16).
         :param compressor_name: The name of the compressor or compression algorithm to use with Zarr.
         :param compression_level: The compression level to use with the compressor (1-9).
        """

        dtype = np.dtype(dtype)

        # If dense_zarr is a path, rather than a Zarr Array object, then
        # open it as a Zarr array object before proceeding:
        if isinstance(dense_zarr, str):
            if osp.isfile(osp.join(dense_zarr, '.zarray')):
                dense_zarr = zarr.open(dense_zarr)
            else:
                raise FileNotFoundError

        # Create hierarchical storage with zarr groups:
        store = zarr.DirectoryStore(store_path)
        z = zarr.group(store=store, overwrite=overwrite)

        # Create a compressor object:
        compressor = zarr.Blosc(cname=compressor_name, clevel=compression_level)

        # First sub-hierarchy stores the information for the sparse LD matrix:
        mat = z.create_group('matrix')
        mat.empty('data', shape=dense_zarr.shape[0]**2, dtype=dtype, compressor=compressor)

        num_rows = dense_zarr.shape[0]
        chunk_size = dense_zarr.chunks[0]

        indptr_counts = np.zeros(num_rows, dtype=int)

        total_len = 0

        for chunk_idx in range(int(np.ceil(num_rows / chunk_size))):

            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, num_rows)

            z_chunk = dense_zarr[chunk_start: chunk_end]

            data = []

            chunk_len = 0

            for j in range(chunk_start, chunk_end):

                data.append(
                    z_chunk[j - chunk_start][j + 1:ld_boundaries[1, j]]
                )
                indptr_counts[j] = len(data[-1])
                chunk_len += int(ld_boundaries[1, j] - (j+1))

            # Add data + columns indices to zarr array:
            concat_data = np.concatenate(data)

            if np.issubdtype(dtype, np.integer):
                mat['data'][total_len:total_len + chunk_len] = quantize(concat_data, int_dtype=dtype)
            else:
                mat['data'][total_len:total_len + chunk_len] = concat_data.astype(dtype)

            total_len += chunk_len

        # Get the final indptr by computing cumulative sum:
        indptr = np.insert(np.cumsum(indptr_counts), 0, 0)
        # Store indptr in the zarr array:
        mat.array('indptr', indptr, compressor=compressor)

        # Resize the data and indices arrays:
        mat['data'].resize(total_len)

        if delete_original:
            from .stats.ld.utils import delete_ld_store
            delete_ld_store(dense_zarr)

        return cls(z)

    @classmethod
    def from_ragged_zarr_matrix(cls,
                                ragged_zarr,
                                store_path,
                                overwrite=False,
                                delete_original=False,
                                dtype='int16',
                                compressor_name='lz4',
                                compression_level=5):
        """
        Initialize a new LD matrix object using a Zarr array object
        conforming to the old LD Matrix format from magenpy v<=0.0.12.

        This utility function will also copy some of the stored attributes
        associated with the matrix in the old format.

        :param ragged_zarr: The path to the ragged Zarr array object.
        :param store_path: The path where to store the new LD matrix.
        :param overwrite: If True, it overwrites the LD store at `store_path`.
        :param delete_original: If True, it deletes the original ragged LD matrix.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compressor or compression algorithm to use with Zarr.
        :param compression_level: The compression level to use with the compressor (1-9).
        """

        dtype = np.dtype(dtype)

        # If ragged_zarr is a path, rather than a Zarr Array object, then
        # open it as a Zarr array object before proceeding:
        if isinstance(ragged_zarr, str):
            if osp.isfile(osp.join(ragged_zarr, '.zarray')):
                ragged_zarr = zarr.open(ragged_zarr)
            else:
                raise FileNotFoundError

        num_rows = ragged_zarr.shape[0]
        chunk_size = ragged_zarr.chunks[0]

        # Create hierarchical storage with zarr groups:
        store = zarr.DirectoryStore(store_path)
        z = zarr.group(store=store, overwrite=overwrite)

        # Create a compressor object:
        compressor = zarr.Blosc(cname=compressor_name, clevel=compression_level)

        # First sub-hierarchy stores the information for the sparse LD matrix:
        mat = z.create_group('matrix')
        mat.empty('data', shape=num_rows ** 2, dtype=dtype, compressor=compressor)

        indptr_counts = np.zeros(num_rows, dtype=int)

        # Get the LD boundaries from the Zarr array attributes:
        ld_boundaries = np.array(ragged_zarr.attrs['LD boundaries'])

        total_len = 0

        for chunk_idx in range(int(np.ceil(num_rows / chunk_size))):

            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, num_rows)

            z_chunk = ragged_zarr[chunk_start: chunk_end]

            data = []
            chunk_len = 0

            for j in range(chunk_start, chunk_end):

                start, end = ld_boundaries[:, j]
                new_start = (j - start) + 1

                data.append(
                    z_chunk[j - chunk_start][new_start:]
                )
                indptr_counts[j] = end - (j + 1)
                chunk_len += int(end - (j + 1))

            # Add data + columns indices to zarr array:
            concat_data = np.concatenate(data)

            if np.issubdtype(dtype, np.integer):
                mat['data'][total_len:total_len + chunk_len] = quantize(concat_data, int_dtype=dtype)
            else:
                mat['data'][total_len:total_len + chunk_len] = concat_data.astype(dtype)

            total_len += chunk_len

        # Get the final indptr by computing cumulative sum:
        indptr = np.insert(np.cumsum(indptr_counts), 0, 0)
        # Store indptr in the zarr array:
        mat.array('indptr', indptr, compressor=compressor)

        # Resize the data and indices arrays:
        mat['data'].resize(total_len)

        # ============================================================
        # Transfer the attributes/metadata from the old matrix format:

        ld_mat = cls(z)

        ld_mat.set_metadata('snps', np.array(ragged_zarr.attrs['SNP']))
        ld_mat.set_metadata('a1', np.array(ragged_zarr.attrs['A1']))
        ld_mat.set_metadata('a2', np.array(ragged_zarr.attrs['A2']))
        ld_mat.set_metadata('maf', np.array(ragged_zarr.attrs['MAF']))
        ld_mat.set_metadata('bp', np.array(ragged_zarr.attrs['BP']))
        ld_mat.set_metadata('cm', np.array(ragged_zarr.attrs['cM']))

        try:
            ld_mat.set_metadata('ldscore', np.array(ragged_zarr.attrs['LDScore']))
        except KeyError:
            print("Did not find LD scores in old LD matrix format! Skipping...")

        # Set matrix attributes:
        ld_mat.set_store_attr('Chromosome', ragged_zarr.attrs['Chromosome'])
        ld_mat.set_store_attr('LD estimator', ragged_zarr.attrs['LD estimator'])
        ld_mat.set_store_attr('Estimator properties', ragged_zarr.attrs['Estimator properties'])
        ld_mat.set_store_attr('Sample size', ragged_zarr.attrs['Sample size'])

        if delete_original:
            from .stats.ld.utils import delete_ld_store
            delete_ld_store(ragged_zarr)

        return ld_mat

    @property
    def n_snps(self):
        """
        :return: The number of variants in the LD matrix. If the matrix is loaded and filtered,
        we return the number of variants remaining after applying the filter.
        """
        if self._mat is not None:
            return self._mat.shape[0]
        else:
            return self.stored_n_snps

    @property
    def shape(self):
        """

        !!! seealso "See Also"
            * [n_snps][magenpy.LDMatrix.LDMatrix.n_snps]

        :return: The shape of the square LD matrix.
        """
        return self.n_snps, self.n_snps

    @property
    def store(self):
        """
        :return: The Zarr group store object.
        """
        return self._zg.store

    @property
    def compressor(self):
        """
        :return: The `numcodecs` compressor object for the LD data.
        """
        return self._zg['matrix/data'].compressor

    @property
    def zarr_group(self):
        """
        :return: The Zarr group object that stores the LD matrix and its metadata.
        """
        return self._zg

    @property
    def chunks(self):
        """
        :return: The chunks for the data array of the LD matrix.
        """
        return self._zg['matrix/data'].chunks

    @property
    def chunk_size(self):
        """
        :return: The chunk size for the data array of the LD matrix.
        """
        return self.chunks[0]

    @property
    def stored_n_snps(self):
        """
        :return: The number of variants stored in the LD matrix (irrespective of any masks / filters).
        """
        return self._zg['matrix/indptr'].shape[0] - 1

    @property
    def stored_dtype(self):
        """
        :return: The data type for the stored entries of `data` array of the LD matrix.
        """
        return self._zg['matrix/data'].dtype

    @property
    def stored_shape(self):
        """
        :return: The shape of the stored LD matrix (irrespective of any masks / filters).
        """
        n_snps = self.stored_n_snps
        return n_snps, n_snps

    @property
    def dtype(self):
        """
        :return: The data type for the entries of the `data` array of the LD matrix. If the matrix is
        in memory, return the dtype of the CSR matrix. Otherwise, return the
        dtype of the entries in the Zarr array.
        """
        if self.in_memory:
            return self.csr_matrix.dtype
        else:
            return self.stored_dtype

    @property
    def chromosome(self):
        """
        :return: The chromosome for which this LD matrix was calculated.
        """
        return self.get_store_attr('Chromosome')

    @property
    def ld_estimator(self):
        """
        :return: The LD estimator used to compute the LD matrix. Examples include: `block`, `windowed`, `shrinkage`.
        """
        return self.get_store_attr('LD estimator')

    @property
    def estimator_properties(self):
        """
        :return: The properties of the LD estimator used to compute the LD matrix.
        """
        return self.get_store_attr('Estimator properties')

    @property
    def sample_size(self):
        """
        :return: The sample size used to compute the LD matrix.
        """
        return self.get_store_attr('Sample size')

    @property
    def genome_build(self):
        """
        :return: The genome build based on which the base pair coordinates are defined.
        """
        return self.get_store_attr('Genome build')

    @property
    def snps(self):
        """
        :return: rsIDs of the variants included in the LD matrix.
        """
        return self.get_metadata('snps')

    @property
    def a1(self):
        """
        :return: The alternative alleles of the variants included in the LD matrix.
        """
        return self.get_metadata('a1')

    @property
    def a2(self):
        """
        :return: The reference alleles of the variants included in the LD matrix.
        """
        return self.get_metadata('a2')

    @property
    def maf(self):
        """
        :return: The minor allele frequency (MAF) of the alternative allele (A1) in the LD matrix.
        """
        try:
            return self.get_metadata('maf')
        except KeyError:
            return None

    @property
    def bp_position(self):
        """
        !!! seealso "See Also"
            * [genome_build][magenpy.LDMatrix.LDMatrix.genome_build]

        :return: The base pair position of each SNP in the LD matrix.
        """
        return self.get_metadata('bp')

    @property
    def cm_position(self):
        """
        :return: The centi Morgan (cM) position of each variant in the LD matrix.
        """
        try:
            return self.get_metadata('cm')
        except KeyError:
            return None

    @property
    def ld_score(self):
        """
        :return: The LD score of each variant in the LD matrix.
        """
        try:
            return self.get_metadata('ldscore')
        except KeyError:

            ld_score = self.compute_ld_scores()

            if self._mask is None:
                self.set_metadata('ldscore', ld_score, overwrite=True)

            return ld_score

    @property
    def ld_boundaries(self):
        """
        The LD boundaries associated with each variant.
        The LD boundaries are defined as the index of the leftmost neighbor
        (lower boundary) and the rightmost neighbor (upper boundary) of for each variant.
        If the LD matrix is upper triangular, then the boundaries for variant `i` go from `i + 1` to `i + k_i`,
        where `k_i` is the number of neighbors that SNP `i` is in LD with.

        :return: A matrix of shape `(2, n_snps)` where the first row contains the lower boundaries and the second row
        contains the upper boundaries.

        """

        indptr = self.indptr

        if self.in_memory and self.is_symmetric:

            # Check that the matrix has canonical format (indices are sorted / no duplicates):
            assert self.csr_matrix.has_canonical_format

            return np.vstack([self.indices[indptr[:-1]], self.indices[indptr[1:] - 1] + 1]).astype(np.int32)

        else:

            # If the matrix is not in memory, then the format is upper triangular.
            # Therefore, it goes from diagonal + 1 to the end of the row.
            left_bound = np.arange(1, len(indptr) - 1)  # The leftmost neighbor of each SNP (diagonal + 1)
            return np.vstack([left_bound, left_bound + np.diff(indptr[:-1])]).astype(np.int32)

    @property
    def window_size(self):
        """
        !!! seealso "See Also"
            * [n_neighbors][magenpy.LDMatrix.LDMatrix.n_neighbors]

        !!! note
            This includes the variant itself if the matrix is in memory and is symmetric.

        :return: The number of variants in the LD window for each SNP.

        """
        return np.diff(self.indptr)

    @property
    def n_neighbors(self):
        """
        The number of variants in the LD window for each SNP.

        !!! seealso "See Also"
            * [window_size][magenpy.LDMatrix.LDMatrix.window_size]

        !!! note
            This includes the variant itself if the matrix is in memory and is symmetric.

        """
        return self.window_size()

    @property
    def csr_matrix(self):
        """
        :return: The in-memory CSR matrix object.

        ..note ::
            If the LD matrix is not in-memory, then it'll be loaded using default settings.

        """
        if self._mat is None:
            self.load()
        return self._mat

    @property
    def data(self):
        """
        :return: The `data` array of the sparse `CSR` matrix, containing the entries of the LD matrix.
        """
        if self.in_memory:
            return self.csr_matrix.data
        else:
            return self._zg['matrix/data']

    @property
    def indices(self):
        """
        :return: The column indices of the non-zero elements of the sparse, CSR representation of the LD matrix.
        """
        if self.in_memory:
            return self.csr_matrix.indices
        else:
            ld_bounds = self.ld_boundaries

            from .stats.ld.c_utils import expand_ranges

            return expand_ranges(ld_bounds[0], ld_bounds[1], self.data.shape[0])

    @property
    def row_indices(self):
        """
        :return: The row indices of the non-zero elements of the sparse, CSR representation of the LD matrix
        """
        if self.in_memory:
            # TODO: Check that this behaves correctly if some entries are zero but not eliminated.
            return self.csr_matrix.nonzero()[0]
        else:
            indptr = self.indptr
            return np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))

    @property
    def indptr(self):
        """
        :return: The index pointers `indptr` delineating where the data for each row of the flattened,
        sparse CSR representation of the lD matrix.
        """
        if self.in_memory:
            return self.csr_matrix.indptr
        else:
            return self._zg['matrix/indptr']

    def filter_snps(self, extract_snps=None, extract_file=None):
        """
        Filter the LDMatrix to keep a subset of variants. This mainly sets
        the mask for the LD matrix, which is used to hide/remove some SNPs from the LD matrix,
        without altering the stored objects on-disk.

        :param extract_snps: A list or array of SNP rsIDs to keep.
        :param extract_file: A plink-style file containing the SNP rsIDs to keep.
        """

        assert extract_snps is not None or extract_file is not None

        if extract_snps is None:
            from .parsers.misc_parsers import read_snp_filter_file
            extract_snps = read_snp_filter_file(extract_file)

        from .utils.compute_utils import intersect_arrays

        new_mask = intersect_arrays(self.get_metadata('snps', apply_mask=False),
                                    extract_snps,
                                    return_index=True)

        self.set_mask(new_mask)

    def get_mask(self):
        """
        :return: The mask (a boolean flag array) used to hide/remove some SNPs from the LD matrix.
        """
        return self._mask

    def set_mask(self, mask):
        """
        Set the mask (a boolean array) to hide/remove some SNPs from the LD matrix.
        :param mask: An array of indices or boolean mask for SNPs to retain.
        """

        # If the mask is equivalent to the current mask, return:
        if np.array_equal(mask, self._mask):
            return

        # If the mask is boolean, convert to indices (should we?):
        if mask.dtype == bool:
            self._mask = np.where(mask)[0]
        else:
            self._mask = mask

        # If the data is already in memory, reload:
        if self.in_memory:
            self.load(force_reload=True,
                      return_symmetric=self.is_symmetric,
                      fill_diag=self.is_symmetric)

    def to_snp_table(self, col_subset=None):
        """
        :param col_subset: The subset of columns to add to the table. If None, it returns
        all available columns.

        :return: A `pandas` dataframe of the SNP attributes and metadata for variants
        included in the LD matrix.
        """

        col_subset = col_subset or ['CHR', 'SNP', 'POS', 'A1', 'A2', 'MAF', 'LDScore']

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
            if col == 'A2':
                table['A2'] = self.a2
            if col == 'MAF':
                table['MAF'] = self.maf
            if col == 'LDScore':
                table['LDScore'] = self.ld_score
            if col == 'WindowSize':
                table['WindowSize'] = self.window_size

        return table[list(col_subset)]

    def compute_ld_scores(self,
                          annotation_matrix=None,
                          corrected=True,
                          chunk_size=10_000):
        """

        Computes the LD scores for variants in the LD matrix. LD Scores are defined
        as the sum of the squared pairwise Pearson Correlation coefficient between the focal SNP and
        all its neighboring SNPs. See Bulik-Sullivan et al. (2015) for details.

        :param annotation_matrix: A matrix of annotations for each variant for which to aggregate the LD scores.
        :param corrected: Use the sample-size corrected estimator for the squared Pearson correlation coefficient.
            See Bulik-Sullivan et al. (2015).
        :param chunk_size: Specify the number of rows (i.e. SNPs) to compute the LD scores for simultaneously.
            Smaller chunk sizes should require less memory resources. If set to None, we compute LD scores
            for all SNPs in the LD matrix in one go.

        :return: An array of LD scores for each variant in the LD matrix.
        """

        if chunk_size is None:
            chunk_size = self.stored_n_snps

        if annotation_matrix is None:
            annotation_matrix = np.ones((self.n_snps, 1), dtype=np.float32)

        ld_scores = np.zeros((self.n_snps, annotation_matrix.shape[1]))

        for chunk_idx in range(int(np.ceil(self.stored_n_snps / chunk_size))):

            start_row = chunk_idx*chunk_size
            end_row = (chunk_idx + 1)*chunk_size

            csr_mat = self.load_rows(start_row=start_row,
                                     end_row=end_row,
                                     return_symmetric=False,
                                     fill_diag=False,
                                     dtype=np.float32)

            # If a mask is set, apply it to the matrix:
            if self._mask is not None:
                csr_mat = csr_mat[self._mask, :][:, self._mask]

            mat_sq = csr_mat.power(2)

            if corrected:
                mat_sq.data -= (1. - mat_sq.data) / (self.sample_size - 2)

            ld_scores += mat_sq.dot(annotation_matrix)
            ld_scores += mat_sq.T.dot(annotation_matrix)

        # Add the contribution of the diagonal:
        ld_scores += identity(self.n_snps, dtype=np.float32).dot(annotation_matrix)

        # Set floating type to float32:
        ld_scores = ld_scores.astype(np.float32)

        if ld_scores.shape[1] == 1:
            return ld_scores.flatten()
        else:
            return ld_scores

    def multiply(self, vec):
        """
        Multiply the LD matrix with an input vector `vec`.

        !!! seealso "See Also"
            * [dot][magenpy.LDMatrix.LDMatrix.dot]

        :return: The product of the LD matrix with the input vector.
        """
        return self.csr_matrix.dot(vec)

    def dot(self, vec):
        """
        Multiply the LD matrix with an input vector `vec`.

        !!! seealso "See Also"
            * [multiply][magenpy.LDMatrix.LDMatrix.multiply]

        :return: The product of the LD matrix with the input vector.

        """
        return self.multiply(vec)

    def estimate_uncompressed_size(self, dtype=None):
        """
        Provide an estimate of size of the uncompressed LD matrix in megabytes (MB).
        This is only a rough estimate. Depending on how the LD matrix is loaded, the actual size
        may be much larger than this estimate.

        :return: The estimated size of the uncompressed LD matrix in MB.

        """

        if dtype is None:
            dtype = self.stored_dtype

        return 2.*self._zg['matrix/data'].shape[0]*np.dtype(dtype).itemsize / 1024 ** 2

    def get_metadata(self, key, apply_mask=True):
        """
        Get the metadata associated with each variant in the LD matrix.
        :param key: The key for the metadata item.
        :param apply_mask: If True, apply the mask (e.g. filter) to the metadata.

        :return: The metadata item for each variant in the LD matrix.
        :raises KeyError: if the metadata item is not set.
        """
        try:
            if self._mask is not None and apply_mask:
                return self._zg[f'metadata/{key}'][self._mask]
            else:
                return self._zg[f'metadata/{key}'][:]
        except KeyError:
            raise KeyError(f"LD matrix metadata item {key} is not set!")

    def get_store_attr(self, attr):
        """
        Get the attribute or metadata `attr` associated with the LD matrix.
        :param attr: The attribute name.

        :return: The value for the attribute.
        :raises KeyError: if the attribute is not set.
        """
        try:
            return self._zg.attrs[attr]
        except KeyError:
            print(f"Warning: Attribute '{attr}' is not set!")
            return None

    def set_store_attr(self, attr, value):
        """
        Set the attribute `attr` associated with the LD matrix. This is used
        to set high-level information, such as information about the sample from which
        the matrix was computed, the LD estimator used, its properties, etc.

        :param attr: The attribute name.
        :param value: The value for the attribute.
        """

        self._zg.attrs[attr] = value

    def set_metadata(self, key, value, overwrite=False):
        """
        Set the metadata field associated with variants the LD matrix.
        :param key: The key for the metadata item.
        :param value: The value for the metadata item (an array with the same length as the number of variants).
        :param overwrite: If True, overwrite the metadata item if it already exists.
        """

        if 'metadata' not in list(self._zg.group_keys()):
            meta = self._zg.create_group('metadata')
        else:
            meta = self._zg['metadata']

        value = np.array(value)

        if np.issubdtype(value.dtype, np.floating):
            dtype = np.float32
        elif np.issubdtype(value.dtype, np.integer):
            dtype = np.int32
        else:
            dtype = str

        meta.array(key, value, overwrite=overwrite, dtype=dtype, compressor=self.compressor)

    def update_rows_inplace(self, new_csr, start_row=None, end_row=None):
        """
        A utility function to perform partial updates to a subset of rows in the
        LD matrix. The function takes a new CSR matrix and, optionally, a start
        and end row delimiting the chunk of the LD matrix to update with the `new_csr`.

        !!! note
            Current implementation assumes that the update does not change the sparsity
            structure of the original matrix. Updating the matrix with new sparsity structure
            is a harder problem that we will try to tackle later on.

        !!! note
            Current implementation assumes `new_csr` is upper triangular.

        :param new_csr: A sparse CSR matrix (`scipy.sparse.csr_matrix`) where the column dimension
        matches the column dimension of the LD matrix.
        :param start_row: The start row for the chunk to update.
        :param end_row: The end row for the chunk to update.

        :raises AssertionError: if the column dimension of `new_csr` does not match the column dimension
        """

        assert new_csr.shape[1] == self.stored_n_snps

        start_row = start_row or 0
        end_row = end_row or self.stored_n_snps

        # Sanity checking:
        assert start_row >= 0
        assert end_row <= self.stored_n_snps

        indptr = self._zg['matrix/indptr'][:]

        data_start = indptr[start_row]
        data_end = indptr[end_row]

        # TODO: Check that this covers most cases and would not result in unexpected behavior
        if np.issubdtype(self.stored_dtype, np.integer) and np.issubdtype(new_csr.dtype, np.floating):
            self._zg['matrix/data'][data_start:data_end] = quantize(new_csr.data, int_dtype=self.stored_dtype)
        else:
            self._zg['matrix/data'][data_start:data_end] = new_csr.data.astype(self.stored_dtype)

    def low_memory_load(self, dtype=None):
        """
        A utility method to load the LD matrix in low-memory mode.
        The method will load the entries of the upper triangular portion of the matrix,
        perform filtering based on the mask (if set), and return the filtered data
        and index pointer (`indptr`) arrays.

        This is useful for some application, such as the `low_memory` version of
        the `viprs` method, because it avoids reconstructing the `indices` array for the CSR matrix,
        which can potentially be a very long array of large integers.

        !!! note
            The method, by construction, does not support loading the full symmetric matrix. If
            that's the goal, use the `.load()` or `.load_rows()` methods.

        !!! seealso "See Also"
            * [load_rows][magenpy.LDMatrix.LDMatrix.load_rows]
            * [load][magenpy.LDMatrix.LDMatrix.load]

        :param dtype: The data type for the entries of the LD matrix.

        :return: A tuple of the data and index pointer arrays for the LD matrix.

        """

        # Determine the final data type for the LD matrix entries
        # and whether we need to perform dequantization or not depending on
        # the stored data type and the requested data type.

        if dtype is None:
            dtype = self.stored_dtype
            dequantize_data = False
        else:
            dtype = np.dtype(dtype)
            if np.issubdtype(self.stored_dtype, np.integer) and np.issubdtype(dtype, np.floating):
                dequantize_data = True
            else:
                dequantize_data = False

        # Get the index pointer array:
        indptr = self._zg['matrix/indptr'][:]

        # Filter the index pointer array based on the mask:
        if self._mask is not None:

            if np.issubdtype(self._mask.dtype, np.integer):
                mask = np.zeros(self.stored_n_snps, dtype=np.int8)
                mask[self._mask] = 1
            else:
                mask = self._mask

            from .stats.ld.c_utils import filter_ut_csr_matrix_low_memory

            data_mask, indptr = filter_ut_csr_matrix_low_memory(indptr, mask)
            # Unfortunately, .vindex is very slow in Zarr right now (~order of magnitude)
            # So for now, we load the entire data array before performing the mask selection:
            data = self._zg['matrix/data'][:][data_mask]
        else:
            data = self._zg['matrix/data'][:]

        if dequantize_data:
            return dequantize(data, float_dtype=dtype), indptr
        else:
            return data.astype(dtype), indptr

    def load_rows(self,
                  start_row=None,
                  end_row=None,
                  return_symmetric=False,
                  fill_diag=False,
                  keep_shape=True,
                  dtype=None):
        """
        A utility function to allow for loading a subset of the LD matrix.
        By specifying `start_row` and `end_row`, the user can process or inspect small
        blocks of the LD matrix without loading the whole thing into memory.

        TODO: Consider using `low_memory_load` internally to avoid reconstructing the `indices` array.

        !!! note
            This method does not perform any filtering on the stored data.
            To access the LD matrix with filtering, use `.load()` or `low_memory_load`.

        !!! seealso "See Also"
            * [low_memory_load][magenpy.LDMatrix.LDMatrix.low_memory_load]
            * [load][magenpy.LDMatrix.LDMatrix.load]

        :param start_row: The start row to load to memory
        :param end_row: The end row (not inclusive) to load to memory
        :param return_symmetric: If True, return a full symmetric representation of the LD matrix.
        :param fill_diag: If True, fill the diagonal of the LD matrix with ones.
        :param keep_shape: If True, return the LD matrix with the same shape as the original. Here,
        entries that are outside the requested start_row:end_row region will be zeroed out.
        :param dtype: The data type for the entries of the LD matrix.

        :return: The requested sub-matrix of the LD matrix.
        """

        # Determine the final data type for the LD matrix entries
        # and whether we need to perform dequantization or not depending on
        # the stored data type and the requested data type.
        if dtype is None:
            dtype = self.stored_dtype
            dequantize_data = False
        else:
            dtype = np.dtype(dtype)
            if np.issubdtype(self.stored_dtype, np.integer) and np.issubdtype(dtype, np.floating):
                dequantize_data = True
            else:
                dequantize_data = False

        # Sanity checking + forming the dimensions of the
        # requested sub-matrix:
        n_snps = self.stored_n_snps

        start_row = start_row or 0
        end_row = end_row or n_snps

        # Sanity checking:
        assert start_row >= 0
        end_row = min(end_row, n_snps)

        # Load the index pointer from disk:
        indptr = self._zg['matrix/indptr'][:]

        # Determine the start and end positions in the data matrix
        # based on the requested start and end rows:
        data_start = indptr[start_row]
        data_end = indptr[end_row]

        # If the user is requesting a subset of the matrix, then we need to adjust
        # the index pointer accordingly:
        if start_row > 0 or end_row < n_snps:
            # Zero out all index pointers before `start_row`:
            indptr = np.clip(indptr - data_start, a_min=0, a_max=None)
            # Adjust all index pointers after `end_row`:
            indptr[end_row+1:] = (data_end - data_start)

        # Extract the data for the requested rows:
        csr_data = self._zg['matrix/data'][data_start:data_end]

        # If we need to de-quantize the data, do it now:
        if dequantize_data:
            csr_data = dequantize(csr_data, float_dtype=dtype)

        # Construct a CSR matrix from the loaded data, updated indptr, and indices:

        # Get the indices array:
        if self.in_memory:
            # If the matrix (or a version of it) is already loaded,
            # then set the `in_memory` flag to False before fetching the indices.
            self.in_memory = False
            indices = self.indices
            self.in_memory = True
        else:
            indices = self.indices

        mat = csr_matrix(
            (
                csr_data,
                indices[data_start:data_end],
                indptr
            ),
            shape=(n_snps, n_snps),
            dtype=dtype
        )

        # Determine the "invalid" value for the purposes of reconstructing
        # the symmetric matrix:
        if np.issubdtype(dtype, np.integer):
            # For integers, we don't use the minimum value during quantization
            # because we would like to have the zero point at exactly zero. So,
            # we can use this value as our alternative to `nan`.
            invalid_value = np.iinfo(dtype).min
            identity_val = np.iinfo(dtype).max
        else:
            invalid_value = np.nan
            identity_val = 1

        if return_symmetric:

            # First, replace explicit zeros with invalid value (this is a hack to prevent scipy
            # from eliminating those zeros when making the matrix symmetric):
            mat.data[mat.data == 0] = invalid_value

            # Add the matrix transpose to make it symmetric:
            mat = (mat + mat.T).astype(dtype)

            # If the user requested filling the diagonals, do it here:
            if fill_diag:
                diag_vals = np.concatenate([np.zeros(start_row, dtype=dtype),
                                            identity_val*np.ones(end_row - start_row, dtype=dtype),
                                            np.zeros(n_snps - end_row, dtype=dtype)])
                mat += diags(diag_vals, dtype=dtype, shape=mat.shape)

            # Replace the invalid values with zeros again:
            if np.isnan(invalid_value):
                mat.data[np.isnan(mat.data)] = 0
            else:
                mat.data[mat.data == invalid_value] = 0

            return mat
        elif fill_diag:
            diag_vals = np.concatenate([np.zeros(start_row, dtype=dtype),
                                        identity_val*np.ones(end_row - start_row, dtype=dtype),
                                        np.zeros(n_snps - end_row, dtype=dtype)])
            mat += diags(diag_vals, dtype=dtype, shape=mat.shape)

        # If the shape remains the same, return the matrix as is.
        # Otherwise, return the requested sub-matrix:
        if keep_shape:
            return mat
        else:
            return mat[start_row:end_row, :]

    def load(self,
             force_reload=False,
             return_symmetric=True,
             fill_diag=True,
             dtype=None):

        """
        Load the LD matrix from on-disk storage in the form of Zarr arrays to memory,
        in the form of sparse CSR matrices.

        !!! seealso "See Also"
            * [low_memory_load][magenpy.LDMatrix.LDMatrix.low_memory_load]
            * [load_rows][magenpy.LDMatrix.LDMatrix.load_rows]

        :param force_reload: If True, it will reload the data even if it is already in memory.
        :param return_symmetric: If True, return a full symmetric representation of the LD matrix.
        :param fill_diag: If True, fill the diagonal elements of the LD matrix with ones.
        :param dtype: The data type for the entries of the LD matrix.

        :return: The LD matrix as a sparse CSR matrix.
        """

        if dtype is not None:
            dtype = np.dtype(dtype)

        if self.in_memory:
            # If the LD matrix is already in memory:

            if (return_symmetric == self.is_symmetric) and not force_reload:
                # If the requested symmetry is the same as the one already loaded,
                # and the user asked not to force a reload, then do nothing.

                # If the currently loaded LD matrix has float entries and the user wants
                # the return type to be another floating point, then just cast and return.
                # Otherwise, we have to reload the matrix:
                if np.issubdtype(self._mat.data.dtype, np.floating) and np.issubdtype(dtype, np.floating):
                    self._mat.data = self._mat.data.astype(dtype)
                    return
                elif self._mat.data.dtype == dtype:
                    return

        # If we are re-loading the matrix, make sure to release the current one:
        self.release()

        self._mat = self.load_rows(return_symmetric=return_symmetric,
                                   fill_diag=fill_diag,
                                   dtype=dtype)

        # If a mask is set, apply it:
        if self._mask is not None:
            self._mat = self._mat[self._mask, :][:, self._mask]

        # Update the flags:
        self.in_memory = True
        self.is_symmetric = return_symmetric

    def release(self):
        """
        Release the LD data from memory.
        """
        self._mat = None
        self.in_memory = False
        self.is_symmetric = False
        self.index = 0

    def get_row(self, index, return_indices=False):
        """
        Extract a single row from the LD matrix.

        :param index: The index of the row to extract.
        :param return_indices: If True, return the indices of the non-zero elements of that row.

        :return: The requested row of the LD matrix.
        """

        if self.in_memory:
            row = self.csr_matrix.getrow(index)
            if return_indices:
                return row.data, row.indices
            else:
                return row.data
        else:
            indptr = self.indptr[:]
            start_idx, end_idx = indptr[index], indptr[index + 1]
            if return_indices:
                return self.data[start_idx:end_idx], np.arange(index + 1,
                                                               index + 1 + (indptr[index + 1] - indptr[index]))
            else:
                return self.data[start_idx:end_idx]

    def validate_ld_matrix(self):
        """
        Checks that the `LDMatrix` object has correct structure and
        checks its contents for validity.

        Specifically, we check that:
        * The dimensions of the matrix and its associated attributes are matching.
        * The masking is working properly.

        :return: True if the matrix has the correct structure.
        :raises ValueError: if the matrix is not valid.
        """

        class_attrs = ['snps', 'a1', 'a2', 'maf', 'bp_position', 'cm_position', 'ld_score']

        for attr in class_attrs:
            attribute = getattr(self, attr)
            if attribute is None:
                continue
            if len(attribute) != len(self):
                raise ValueError(f"Invalid LD Matrix: Dimensions for attribute {attr} are not aligned!")

        # TODO: Add other sanity checks here?

        return True

    def __getstate__(self):
        return self.store.path, self.in_memory, self.is_symmetric, self._mask

    def __setstate__(self, state):

        path, in_mem, is_symmetric, mask = state

        self._zg = zarr.open_group(path, mode='r')
        self.in_memory = in_mem
        self.is_symmetric = is_symmetric
        self._mat = None
        self.index = 0
        self._mask = None

        if mask is not None:
            self.set_mask(mask)

        if in_mem:
            self.load(return_symmetric=is_symmetric, fill_diag=is_symmetric)

    def __len__(self):
        return self.n_snps

    def __getitem__(self, index):
        return self.get_row(index)

    def __iter__(self):
        """
        TODO: Add a flag to allow for chunked iterator, with limited memory footprint.
        """
        self.index = 0
        self.load(return_symmetric=self.is_symmetric)
        return self

    def __next__(self):

        if self.index == len(self):
            self.index = 0
            raise StopIteration

        next_item = self.get_row(self.index)
        self.index += 1

        return next_item
