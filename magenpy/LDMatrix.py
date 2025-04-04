import zarr
import os.path as osp
import numpy as np
import pandas as pd
from typing import Union
from .utils.model_utils import quantize, dequantize
from .LDLinearOperator import LDLinearOperator


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
    * Perform linear algebra operations on LD matrices, including SVD, estimating extremal eigenvalues,
    and efficient matrix-vector multiplication.

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
            * `cm`: The array containing the centi Morgan distance along the chromosome.
            * `ldscore`: The array containing the LD scores.
        * `attrs`: A JSON-style metadata object containing general information about how the LD matrix
        was calculated, including the chromosome number, sample size, genome build, LD estimator,
        and estimator properties.

    :ivar _zg: The Zarr group object containing the LD matrix and its metadata.
    :ivar _cached_lop: A cached `LDLinearOperator` object for performing linear algebra operations.
    :ivar index: An integer index for the current SNP in the LD matrix (useful for iterators).
    :ivar _mask: A boolean mask for filtering the LD matrix.
    :ivar _n_masked: The number of SNPs masked (i.e. discarded) by the current mask.

    """

    def __init__(self, zarr_group):
        """
        Initialize an `LDMatrix` object from a Zarr group store.

        :param zarr_group: The Zarr group object that stores the LD matrix.
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

        # The Zarr storage hierarchy:
        self._zg: zarr.hierarchy.Group = zarr_group

        # Caching loaded/processed data of the LD matrix:
        self._cached_lop: Union[LDLinearOperator, None] = None

        # To support dynamically filtering the LD matrix:
        self._mask = None
        self._n_masked = 0

        # To support iteration over the LD matrix:
        self.index: int = 0

    @classmethod
    def from_path(cls, ld_store_path, cache_size=None):
        """
        Initialize an `LDMatrix` object from a pre-computed Zarr group store. This is a genetic method
        that can work with both cloud-based stores (e.g. s3 storage) or local filesystems.

        :param ld_store_path: The path to the Zarr array store.
        :param cache_size: The size of the cache for the Zarr store (in bytes). Default is `None` (no caching).

        !!! seealso "See Also"
            * [from_directory][magenpy.LDMatrix.LDMatrix.from_directory]
            * [from_s3][magenpy.LDMatrix.LDMatrix.from_s3]

        :return: An `LDMatrix` object.
        """

        if 's3://' in ld_store_path:
            return cls.from_s3(ld_store_path, cache_size)
        else:
            return cls.from_directory(ld_store_path, cache_size)

    @classmethod
    def from_s3(cls, s3_path, cache_size=None):
        """
        Initialize an `LDMatrix` object from a Zarr group store hosted on AWS s3 storage.

        :param s3_path: The path to the Zarr group store on AWS s3. s3 paths are expected
        to be of the form `s3://bucket-name/path/to/zarr-store`.
        :param cache_size: The size of the cache for the Zarr store (in bytes). Default is 16MB.

        .. note::
            Requires installing the `s3fs` package to access the Zarr store on AWS s3.

        !!! seealso "See Also"
            * [from_path][magenpy.LDMatrix.LDMatrix.from_path]
            * [from_directory][magenpy.LDMatrix.LDMatrix.from_directory]

        :return: An `LDMatrix` object.
        """

        import s3fs

        s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='us-east-2'))
        store = s3fs.S3Map(root=s3_path.replace('s3://', ''), s3=s3, check=False)
        if cache_size is not None:
            store = zarr.LRUStoreCache(store, max_size=cache_size)
        ld_group = zarr.open_group(store=store, mode='r')

        return cls(ld_group)

    @classmethod
    def from_directory(cls, dir_path, cache_size=None):
        """
        Initialize an `LDMatrix` object from a Zarr array store.
        :param dir_path: The path to the Zarr array store on the local filesystem.
        :param cache_size: The size of the cache for the Zarr store (in bytes).  Default is `None` (no caching).

        !!! seealso "See Also"
            * [from_s3][magenpy.LDMatrix.LDMatrix.from_s3]
            * [from_path][magenpy.LDMatrix.LDMatrix.from_path]

        :return: An `LDMatrix` object.
        """

        for level in range(2):
            try:
                dir_store = zarr.storage.DirectoryStore(dir_path)
                if cache_size is not None:
                    dir_store = zarr.LRUStoreCache(dir_store, max_size=cache_size)
                ld_group = zarr.open_group(dir_store, mode='r')
                return cls(ld_group)
            except zarr.hierarchy.GroupNotFoundError as e:
                if level < 1:
                    dir_path = osp.dirname(dir_path)
                else:
                    raise e

    @classmethod
    def from_csr(cls,
                 csr_mat,
                 store_path,
                 overwrite=False,
                 dtype='int16',
                 compressor_name='zstd',
                 compression_level=7):
        """
        Initialize an LDMatrix object from a sparse CSR matrix.

        TODO: Determine the chunksize based on the avg neighborhood size?

        :param csr_mat: The sparse CSR matrix.
        :param store_path: The path to the Zarr LD store where the data will be stored.
        :param overwrite: If True, it overwrites the LD store at `store_path`.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compressor or compression algorithm to use with Zarr.
        :param compression_level: The compression level to use with the compressor (1-9).

        :return: An `LDMatrix` object.
        """

        from scipy.sparse import triu

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
        mat.array('indptr', triu_mat.indptr, dtype=np.int64, compressor=compressor)

        return cls(z)

    @classmethod
    def from_plink_table(cls,
                         plink_ld_file,
                         snps,
                         store_path,
                         ld_boundaries=None,
                         pandas_chunksize=None,
                         overwrite=False,
                         dtype='int16',
                         compressor_name='zstd',
                         compression_level=7):
        """
        Construct a Zarr LD matrix using LD tables generated by plink1.9.

        TODO: Determine the chunksize based on the avg neighborhood size?

        :param plink_ld_file: The path to the plink LD table file.
        :param snps: An iterable containing the list of ordered SNP rsIDs to be included in the LD matrix.
        :param store_path: The path to the Zarr LD store.
        :param ld_boundaries: The LD boundaries for each SNP in the LD matrix (delineates the indices of
        the leftmost and rightmost neighbors of each SNP). If not provided, the LD matrix will be constructed
        using the full LD table from plink.
        :param pandas_chunksize: If the LD table is large, provide chunk size
        (i.e. number of rows to process at each step) to keep memory footprint manageable.
        :param overwrite: If True, it overwrites the LD store at `store_path`.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compressor or compression algorithm to use with Zarr.
        :param compression_level: The compression level to use with the compressor (1-9).

        :return: An `LDMatrix` object.
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

        if ld_boundaries is not None:
            use_cols = ['SNP_A', 'SNP_B', 'R']
            bounds_df = pd.DataFrame(np.column_stack((np.arange(len(snps)).reshape(-1, 1),
                                                      ld_boundaries[1:, :].T)),
                                     columns=['SNP_idx', 'end'])
        else:
            use_cols = ['SNP_A', 'R']

        # Create a chunked iterator with pandas:
        # Chunk size will correspond to the average chunk size for the Zarr array:
        ld_chunks = pd.read_csv(plink_ld_file,
                                sep=r'\s+',
                                usecols=use_cols,
                                engine='c',
                                chunksize=pandas_chunksize,
                                dtype={'SNP_A': str, 'R': np.float32})

        if pandas_chunksize is None:
            ld_chunks = [ld_chunks]

        # Create a dictionary mapping SNPs to their indices:
        snp_idx = pd.Series(np.arange(len(snps), dtype=np.int32), index=snps)

        indptr_counts = np.zeros(len(snps), dtype=np.int32)

        total_len = 0

        # For each chunk in the LD file:
        for ld_chunk in ld_chunks:

            # Fill N/A in R before storing it:
            ld_chunk.fillna({'R': 0.}, inplace=True)

            # If LD boundaries are provided, filter the LD table accordingly:
            if ld_boundaries is not None:

                row_index = snp_idx[ld_chunk['SNP_A'].values]

                ld_chunk['SNP_A_index'] = snp_idx[ld_chunk['SNP_A'].values].values
                ld_chunk['SNP_B_index'] = snp_idx[ld_chunk['SNP_B'].values].values

                ld_chunk = ld_chunk.merge(bounds_df, left_on='SNP_A_index', right_on='SNP_idx')
                ld_chunk = ld_chunk.loc[(ld_chunk['SNP_B_index'] >= ld_chunk['SNP_A_index'] + 1) &
                                        (ld_chunk['SNP_B_index'] < ld_chunk['end'])]

            # Create an indexed LD chunk:
            row_index = snp_idx[ld_chunk['SNP_A'].values]

            # Add LD data to the zarr array:
            if np.issubdtype(dtype, np.integer):
                mat['data'][total_len:total_len + len(ld_chunk)] = quantize(ld_chunk['R'].values, int_dtype=dtype)
            else:
                mat['data'][total_len:total_len + len(ld_chunk)] = ld_chunk['R'].values.astype(dtype)

            total_len += len(ld_chunk)

            # Count the number of occurrences of each SNP in the chunk:
            snp_counts = row_index.value_counts()

            # Add the number of entries to indptr_counts:
            indptr_counts[snp_counts.index] += snp_counts.values

        # Get the final indptr by computing cumulative sum:
        indptr = np.insert(np.cumsum(indptr_counts, dtype=np.int64), 0, 0)
        # Store indptr in the zarr group:
        mat.array('indptr', indptr, dtype=np.int64, compressor=compressor)

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
                               compressor_name='zstd',
                               compression_level=7):
        """
         Initialize a new LD matrix object using a Zarr array object. This method is
         useful for converting a dense LD matrix computed using Dask (or other distributed computing
         software) to a sparse or banded one.

         TODO: Determine the chunksize based on the avg neighborhood size?

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

         :return: An `LDMatrix` object.
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

        indptr_counts = np.zeros(num_rows, dtype=np.int32)

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
        indptr = np.insert(np.cumsum(indptr_counts, dtype=np.int64), 0, 0)
        # Store indptr in the zarr array:
        mat.array('indptr', indptr, dtype=np.int64, compressor=compressor)

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
                                compressor_name='zstd',
                                compression_level=7):
        """
        Initialize a new LD matrix object using a Zarr array object
        conforming to the old LD Matrix format from magenpy v<=0.0.12.

        This utility function will also copy some of the stored attributes
        associated with the matrix in the old format.

        TODO: Determine the chunksize based on the avg neighborhood size?

        :param ragged_zarr: The path to the ragged Zarr array object.
        :param store_path: The path where to store the new LD matrix.
        :param overwrite: If True, it overwrites the LD store at `store_path`.
        :param delete_original: If True, it deletes the original ragged LD matrix.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compressor or compression algorithm to use with Zarr.
        :param compression_level: The compression level to use with the compressor (1-9).

        :return: An `LDMatrix` object.
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

        indptr_counts = np.zeros(num_rows, dtype=np.int64)

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
        indptr = np.insert(np.cumsum(indptr_counts, dtype=np.int64), 0, 0)
        # Store indptr in the zarr array:
        mat.array('indptr', indptr, dtype=np.int64, compressor=compressor)

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
            pass

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
        :return: The number of variants in the LD matrix. If a mask is set, we return the
        number of variants included in the mask.
        """
        if self._cached_lop is not None:
            return self._cached_lop.shape[0]
        else:
            return self.stored_n_snps - self._n_masked

    @property
    def shape(self):
        """

        !!! seealso "See Also"
            * [n_snps][magenpy.LDMatrix.LDMatrix.n_snps]

        :return: The shape of the square LD matrix.
        """
        return self.n_snps, self.n_snps

    @property
    def in_memory(self):
        """
        :return: A boolean flag indicating whether the LD matrix is in memory.
        """
        return self._cached_lop is not None

    @property
    def is_symmetric(self):
        """
        :return: A boolean flag indicating whether the loaded LD matrix is symmetric.
        """
        if self.in_memory:
            return self._cached_lop.symmetric
        else:
            return False

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
            return self._cached_lop.ld_data_type
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
    def dequantization_scale(self):
        """
        :return: The dequantization scale for the quantized LD matrix. If the matrix is not quantized, returns 1.
        """
        if np.issubdtype(self.stored_dtype, np.integer):
            return 1./np.iinfo(self.stored_dtype).max
        else:
            return 1.

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
        leftmost_idx = self.leftmost_index

        return np.vstack([leftmost_idx, leftmost_idx + np.diff(indptr)]).astype(np.int32)

    @property
    def window_size(self):
        """
        !!! seealso "See Also"
            * [n_neighbors][magenpy.LDMatrix.LDMatrix.n_neighbors]

        :return: The number of variants in the LD window for each SNP.

        """

        if self.in_memory and self.is_symmetric:
            indptr = self.indptr
        else:
            from .stats.ld.c_utils import get_symmetrized_indptr
            indptr, _ = get_symmetrized_indptr(self.indptr[:])

        return np.diff(indptr)

    @property
    def n_neighbors(self):
        """
        !!! seealso "See Also"
            * [window_size][magenpy.LDMatrix.LDMatrix.window_size]

        :return: The number of variants in the LD window for each SNP.

        """
        return self.window_size

    @property
    def data(self):
        """
        :return: The `data` array of the sparse `CSR` matrix, containing the entries of the LD matrix.
        """
        if self.in_memory:
            return self._cached_lop.ld_data
        else:
            return self._zg['matrix/data']

    @property
    def leftmost_index(self):
        """
        :return: The index of the leftmost neighbor of each variant in the LD matrix.
        """
        if self.in_memory:
            return self._cached_lop.leftmost_idx
        else:
            return np.arange(1, len(self.indptr) - 1)

    @property
    def indices(self):
        """
        :return: The column indices of the non-zero elements of the sparse, CSR representation of the LD matrix.
        """

        ld_bounds = self.ld_boundaries

        from .stats.ld.c_utils import expand_ranges

        return expand_ranges(ld_bounds[0], ld_bounds[1], self.data.shape[0])

    @property
    def row_indices(self):
        """
        :return: The row indices of the non-zero elements of the sparse, CSR representation of the LD matrix
        """
        indptr = self.indptr
        return np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))

    @property
    def indptr(self):
        """
        :return: The index pointers `indptr` delineating where the data for each row of the flattened,
        sparse CSR representation of the lD matrix.
        """
        if self.in_memory:
            return self._cached_lop.ld_indptr
        else:
            return self._zg['matrix/indptr']

    @property
    def is_mask_set(self):
        """
        :return: A boolean flag indicating whether a mask is set for the LD matrix.
        """
        return self._mask is not None

    def get_long_range_ld_variants(self, return_value='snps'):
        """
        A utility method to exclude variants that are in long-range LD regions. The
        boundaries of those regions are derived from here:

        https://genome.sph.umich.edu/wiki/Regions_of_high_linkage_disequilibrium_(LD)

        Which is based on the work of

        > Anderson, Carl A., et al. "Data quality control in genetic case-control association studies."
        Nature protocols 5.9 (2010): 1564-1573.

        :param return_value: The value to return. Options are 'mask', 'index', 'snps'. If `mask`, then
        it returns a boolean array of which variants are in long-range LD regions. If `index`, then it returns
        the index of those variants. If `snps`, then it returns the rsIDs of those variants.

        :return: An array of the variants that are in long-range LD regions.
        """

        assert return_value in ('mask', 'index', 'snps')

        from .parsers.annotation_parsers import parse_annotation_bed_file
        from .utils.data_utils import lrld_path

        bed_df = parse_annotation_bed_file(lrld_path())

        # Filter to only regions specific to the chromosome of this matrix:
        bed_df = bed_df.loc[bed_df['CHR'] == self.chromosome]

        bp_pos = self.bp_position
        snp_mask = np.zeros(len(bp_pos), dtype=bool)

        # Loop over the LRLD region on this chromosome and include the SNPs in these regions:
        for _, row in bed_df.iterrows():
            start, end = row['Start'], row['End']
            snp_mask |= ((bp_pos >= start) & (bp_pos <= end))

        if return_value == 'mask':
            return snp_mask
        elif return_value == 'index':
            return np.where(snp_mask)[0]
        else:
            return self.snps[snp_mask]

    def filter_long_range_ld_regions(self):
        """
        A utility method to exclude variants that are in long-range LD regions. The
        boundaries of those regions are derived from here:

        https://genome.sph.umich.edu/wiki/Regions_of_high_linkage_disequilibrium_(LD)

        Which is based on the work of

        > Anderson, Carl A., et al. "Data quality control in genetic case-control association studies."
        Nature protocols 5.9 (2010): 1564-1573.
        """

        # Filter the SNP to only those not in the LRLD regions:
        self.filter_snps(self.snps[~self.get_long_range_ld_variants(return_value='mask')])

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

        # Check that mask is a numpy array:
        if not isinstance(mask, np.ndarray):
            raise ValueError("Mask must be a numpy array.")

        # Check that mask is either a boolean array or an array of indices:
        if mask.dtype != bool and not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("Mask must be a boolean array or an array of indices.")

        # If mask is a boolean array, ensure that it matches the number of stored SNPs:
        if mask.dtype == bool and len(mask) != self.stored_n_snps:
            raise ValueError("Boolean mask must have the same length as the number of stored SNPs.")

        # If the mask is equivalent to the current mask, return:
        if np.array_equal(mask, self._mask):
            return

        # If the mask consists of indices, convert to boolean mask:
        if mask.dtype != bool:
            self._mask = np.zeros(self.stored_n_snps, dtype=bool)
            self._mask[mask] = True
        else:
            self._mask = mask

        self._n_masked = np.sum(~self._mask)

        # If the data has already been loaded to memory, reload:
        if self.in_memory:
            self.load(force_reload=True,
                      return_symmetric=self.is_symmetric,
                      dtype=self.dtype)

    def reset_mask(self):
        """
        Reset the mask to its default value (None).
        """

        self._mask = None
        self._n_masked = 0

        if self.in_memory:
            self.load(force_reload=True,
                      return_symmetric=self.is_symmetric,
                      dtype=self.dtype)

    def prune(self, threshold):
        """
        Perform LD pruning to remove variants that are in high LD with other variants.
        If two variants are in high LD, this function keeps the variant that occurs
        earlier in the matrix. This behavior will be updated in the future to allow
        for arbitrary ordering of variants.

        !!! note
            Experimental for now. Needs further testing & improvement.

        :param threshold: The absolute value of the Pearson correlation coefficient above which to prune variants. A
        positive floating point number between 0. and 1.
        :return: A boolean array indicating whether a variant is kept after pruning.
        """

        from .stats.ld.c_utils import prune_ld_ut

        assert 0. < threshold <= 1.

        if np.issubdtype(self.dtype, np.integer):
            threshold = quantize(np.array([threshold]), int_dtype=self.dtype)[0]

        return prune_ld_ut(self.indptr[:], self.data[:], threshold)

    def to_snp_table(self, col_subset=None, use_original_index=False):
        """
        :param col_subset: The subset of columns to add to the table. If None, it returns
        all available columns.
        :param use_original_index: If True, it uses the original index of the SNPs in the LD matrix (
        before applying any filters).

        :return: A `pandas` dataframe of the SNP attributes and metadata for variants
        included in the LD matrix.
        """

        col_subset = col_subset or ['CHR', 'SNP', 'POS', 'A1', 'A2', 'MAF', 'LDScore']

        # Create the index according to the original SNP order:
        if use_original_index:
            original_index = np.arange(self.stored_n_snps)
            if self._mask is not None:
                original_index = original_index[self._mask]
        else:
            original_index = None

        table = pd.DataFrame({'SNP': self.snps}, index=original_index)

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
            chunk_size = self.n_snps

        if annotation_matrix is None:
            annotation_matrix = np.ones((self.n_snps, 1), dtype=np.float32)
        else:
            assert annotation_matrix.shape[0] == self.n_snps, ("Annotation matrix must have the same "
                                                               "number of rows as the LD matrix.")

        ld_scores = np.zeros((self.n_snps, annotation_matrix.shape[1]), dtype=np.float32)

        for chunk_idx in range(int(np.ceil(self.n_snps / chunk_size))):

            start_row = chunk_idx*chunk_size
            end_row = min((chunk_idx + 1)*chunk_size, self.n_snps)

            csr_mat = self.load_data(start_row=start_row,
                                     end_row=end_row,
                                     return_symmetric=False,
                                     return_square=False,
                                     keep_original_shape=True,
                                     return_as_csr=True,
                                     dtype=np.float32)

            mat_sq = csr_mat.power(2)

            if corrected:
                mat_sq.data -= (1. - mat_sq.data) / (self.sample_size - 2)

            ld_scores += mat_sq.dot(annotation_matrix)
            ld_scores += mat_sq.T.dot(annotation_matrix)

        from scipy.sparse import identity

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

        :param vec: The input vector to multiply with the LD matrix.

        !!! seealso "See Also"
            * [dot][magenpy.LDMatrix.LDMatrix.dot]

        :return: The product of the LD matrix with the input vector.
        """

        if self.in_memory:
            return self._cached_lop.dot(vec)
        else:
            ld_opr = self.to_linear_operator()
            return ld_opr.dot(vec)

    def dot(self, vec):
        """
        Multiply the LD matrix with an input vector `vec`.

        :param vec: The input vector to multiply with the LD matrix.

        !!! seealso "See Also"
            * [multiply][magenpy.LDMatrix.LDMatrix.multiply]

        :return: A numpy array that is the result of the matrix multiplication between
        the LD matrix with the input vector.

        """
        return self.multiply(vec)

    def perform_svd(self, **svds_kwargs):
        """
        Perform the Singular Value Decomposition (SVD) on the LD matrix.
        This method is a wrapper around the `scipy.sparse.linalg.svds` function and provides
        utilities to perform SVD with a LinearOperator for large-scale LD matrix, so that
        the matrices don't need to be fully represented in memory.

        :param svds_kwargs: Additional keyword arguments to pass to the `scipy.sparse.linalg.svds` function.

        :return: The result of the SVD decomposition of the LD matrix.
        """

        from scipy.sparse.linalg import svds

        if self.in_memory:
            mat = self._cached_lop
        else:
            mat = self.to_linear_operator()

        return svds(mat, **svds_kwargs)

    def estimate_extremal_eigenvalues(self,
                                      block_size=None,
                                      block_size_cm=None,
                                      block_size_kb=None,
                                      blocks=None,
                                      which='both',
                                      return_block_boundaries=False,
                                      assign_to_variants=False):
        """
        Estimate the smallest/largest algebraic eigenvalues of the LD matrix. This is useful for
        analyzing the spectral properties of the LD matrix and detecting potential
        issues for downstream applications that leverage the LD matrix. For instance, many LD
        matrices are not positive semi-definite (PSD) and this manifests in having negative eigenvalues.
        This function can be used to detect such issues.

        To perform this computation efficiently, we leverage fast ARPACK routines provided by `scipy` to
        compute only the extremal eigenvalues of the LD matrix. Another advantage of this implementation
        is that it doesn't require symmetric or dequantized LD matrices. The `LDLinearOperator` class
        can be used to perform all the computations without symmetrizing or dequantizing the matrix beforehand,
        which should make it more efficient in terms of memory and CPU resources.

        Furthermore, this function supports computing eigenvalues for sub-blocks of the LD matrix,
        by simply providing one of the following parameters:
            * `block_size`: Number of variants per block
            * `block_size_cm`: Block size in centi-Morgans
            * `block_size_kb` Block size in kilobases

        :param block_size: An integer specifying the block size or number of variants to
        compute the minimum eigenvalue for. If provided, we compute minimum eigenvalues for each block in the
        LD matrix separately, instead of the minimum eigenvalue for the entire matrix. This can be useful for
        large LD matrices that don't fit in memory or in cases where information about local blocks is needed.
        :param block_size_cm: The block size in centi-Morgans (cM) to compute the minimum eigenvalue for.
        :param block_size_kb: The block size in kilo-base pairs (kb) to compute the minimum eigenvalue for.
        :param blocks: An array or list specifying the block boundaries to compute the minimum eigenvalue for.
        If there are B blocks, then the array should be of size B + 1, with the entries specifying the start position
        of each block.
        :param which: The extremal eigenvalues to compute. Options are 'min', 'max', or 'both'.
        :param return_block_boundaries: If True, return the block boundaries used to compute the minimum eigenvalue.
        :param assign_to_variants: If True, assign the minimum eigenvalue to the variants used to compute it.

        :return: The extremal eigenvalue(s) of the LD matrix or sub-blocks of the LD matrix. If `assign_to_variants`
        is set to True, then return an array of size `n_snps` mapping the extremal eigenvalues to each variant.
        """

        assert which in ('min', 'max', 'both')

        if assign_to_variants:
            if which == 'both':
                eigs_per_var = np.zeros((self.stored_n_snps, 2), dtype=np.float32)
            else:
                eigs_per_var = np.zeros(self.stored_n_snps, dtype=np.float32)
        else:
            eigs = []

        block_boundaries = []

        from .stats.ld.utils import compute_extremal_eigenvalues

        for mat, (start, end) in self.iter_blocks(block_size=block_size,
                                                  block_size_cm=block_size_cm,
                                                  block_size_kb=block_size_kb,
                                                  blocks=blocks,
                                                  return_type='linop',
                                                  return_block_boundaries=True):

            block_boundaries.append({'block_start': start, 'block_end': end})
            eig = compute_extremal_eigenvalues(mat, which=which)

            if assign_to_variants:
                if which == 'both':
                    eigs_per_var[start:end, 0] = eig['min']
                    eigs_per_var[start:end, 1] = eig['max']
                else:
                    eigs_per_var[start:end] = eig
            else:
                eigs.append(eig)

        block_boundaries = pd.DataFrame(block_boundaries).to_dict(orient='list')

        if assign_to_variants:

            if self._mask is not None:
                eigs_per_var = eigs_per_var[self._mask, :]

            if which == 'both':
                eigs_per_var = {
                    'min': eigs_per_var[:, 0],
                    'max': eigs_per_var[:, 1]
                }

            if return_block_boundaries:
                return eigs_per_var, block_boundaries
            else:
                return eigs_per_var

        elif return_block_boundaries:
            if which == 'both':
                return pd.DataFrame(eigs).to_dict(orient='list'), block_boundaries
            else:
                return eigs, block_boundaries
        else:
            if len(eigs) == 1:
                return eigs[0]
            else:
                if which == 'both':
                    return pd.DataFrame(eigs).to_dict(orient='list')
                else:
                    return eigs

    def get_lambda_min(self, aggregate=None, min_max_ratio=0.):
        """
        A utility method to compute the `lambda_min` value for the LD matrix. `lambda_min` is the smallest
        algebraic eigenvalue of the LD matrix. This quantity is useful to know in some applications.
        The function retrieves minimum eigenvalue (if pre-computed and stored) per block and maps it
        to each variant in the corresponding block. If minimum eigenvalues per block are not available,
         we use global minimum eigenvalue (either from matrix attributes or we compute it on the spot).

        Before returning the `lambda_min` value to the user, we apply the following transformation:

        abs(min(lambda_min, 0.))

        This implies that if the minimum eigenvalue is non-negative, we just return 0. for `lambda_min`. We are mainly
        interested in negative eigenvalues here (if they exist).

        :param aggregate: A summary of the minimum eigenvalue across variants or across blocks (if available).
        Supported aggregation functions are `min_block` and `min`. If `min` is selected,
        we return the minimum eigenvalue for the entire matrix (rather than sub-blocks of it). If `min_block` is
        selected, we return the minimum eigenvalue for each block separately (mapped to variants within that block).

        :param min_max_ratio: The ratio between the absolute values of the minimum and maximum eigenvalues.
        This could be used to target a particular threshold for the minimum eigenvalue.

        :return: The absolute value of the minimum eigenvalue for the LD matrix. If the minimum
        eigenvalue is non-negative, we return zero.
        """

        if aggregate is not None:
            assert aggregate in ('min_block', 'min')

        # Get the attributes of the LD store:
        store_attrs = self.list_store_attributes()

        def threshold_lambda_min(eigs):
            return np.abs(np.minimum(eigs['min'] + min_max_ratio*eigs['max'], 0.)) / (1. + min_max_ratio)

        lambda_min = 0.

        if 'Spectral properties' not in store_attrs:
            if aggregate in ('mean_block', 'median_block', 'min_block'):
                raise ValueError('Aggregating lambda_min across blocks '
                                 'requires that these blocks are pre-defined.')
            else:
                lambda_min = threshold_lambda_min(self.estimate_extremal_eigenvalues())

        else:

            spectral_props = self.get_store_attr('Spectral properties')

            if aggregate == 'min_block':
                assert 'Eigenvalues per block' in spectral_props, (
                    'Aggregating lambda_min across blocks '
                    'requires that these blocks are pre-defined.')

            if aggregate == 'min' or 'Eigenvalues per block' not in spectral_props:

                if 'Extremal' in spectral_props:
                    lambda_min = threshold_lambda_min(spectral_props['Extremal'])
                else:
                    lambda_min = threshold_lambda_min(self.estimate_extremal_eigenvalues())

            elif 'Eigenvalues per block' in spectral_props:

                # If we have eigenvalues per block, map the block value to each variant:
                block_eigs = spectral_props['Eigenvalues per block']

                if aggregate is None:

                    # Create a dataframe with the block information:
                    block_df = pd.DataFrame(block_eigs)
                    block_df['add_lam'] = block_df.apply(threshold_lambda_min, axis=1)

                    merged_df = pd.merge_asof(pd.DataFrame({'SNP_idx': np.arange(self.stored_n_snps)}),
                                              block_df,
                                              right_on='block_start', left_on='SNP_idx', direction='backward')
                    # Filter merged_df to only include variants that were matched properly with a block:
                    merged_df = merged_df.loc[
                        (merged_df.SNP_idx >= merged_df.block_start) & (merged_df.SNP_idx < merged_df.block_end)
                    ]

                    if len(merged_df) < 1:
                        raise ValueError('No variants were matched to blocks. '
                                         'This could be due to incorrect block boundaries.')

                    lambda_min = np.zeros(self.stored_n_snps)
                    lambda_min[merged_df['SNP_idx'].values] = merged_df['add_lam'].values

                    if self.is_mask_set:
                        lambda_min = lambda_min[self._mask]

                elif aggregate == 'min_block':
                    lambda_min = np.min(block_eigs['min'])

        return lambda_min

    def estimate_uncompressed_size(self, dtype=None):
        """
        Provide an estimate of size of the uncompressed LD matrix in megabytes (MB).
        This is only a rough estimate. Depending on how the LD matrix is loaded, actual memory
        usage may be larger than this estimate.

        :param dtype: The data type for the entries of the LD matrix. If None, the stored data type is used
        to determine the size of the data in memory.

        :return: The estimated size of the uncompressed LD matrix in MB.

        """

        if dtype is None:
            dtype = self.stored_dtype

        return 2.*self._zg['matrix/data'].shape[0]*np.dtype(dtype).itemsize / 1024 ** 2

    def get_total_storage_size(self):
        """
        Estimate the storage size for all elements of the `LDMatrix` hierarchy,
        including the LD data arrays, metadata arrays, and attributes.

        :return: The estimated size of the stored and compressed LDMatrix object in MB.
        """

        total_bytes = 0

        # Estimate contribution of matrix arrays
        for arr_name, array in self.zarr_group.matrix.arrays():
            total_bytes += array.nbytes_stored

        # Estimate contribution of metadata arrays
        for arr_name, array in self.zarr_group.metadata.arrays():
            total_bytes += array.nbytes_stored

        # Estimate the contribution of the attributes:
        if hasattr(self.zarr_group, 'attrs'):
            total_bytes += len(str(dict(self.zarr_group.attrs)).encode('utf-8'))

        return total_bytes / 1024**2

    def get_metadata(self, key, apply_mask=True):
        """
        Get the metadata associated with each variant in the LD matrix.

        :param key: The key for the metadata item.
        :param apply_mask: If True, apply the mask (e.g. filter) to the metadata.

        :return: The metadata item for each variant in the LD matrix.
        :raises KeyError: if the metadata item is not set.
        """
        try:
            metadata = self._zg[f'metadata/{key}'][:]
        except KeyError:
            raise KeyError(f"LD matrix metadata item {key} is not set!")

        if self.is_mask_set and apply_mask:
            metadata = metadata[self._mask]

        return metadata

    def get_store_attr(self, attr):
        """
        Get the attribute or metadata `attr` associated with the LD matrix.
        :param attr: The attribute name.

        :return: The value for the attribute.
        :raises KeyError: if the attribute is not set.
        """
        return self._zg.attrs[attr]

    def list_store_attributes(self):
        """
        Get all the attributes associated with the LD matrix.
        :return: A list of all the attributes.
        """
        return list(self._zg.attrs.keys())

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
        Set the metadata field associated with variants in the LD matrix.
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

        meta.array(key,
                   value,
                   overwrite=overwrite,
                   dtype=dtype,
                   compressor=self.compressor)

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
            self._zg['matrix/data'][data_start:data_end] = new_csr.data.astype(self.stored_dtype, copy=False)

    def to_linear_operator(self, **load_kwargs):
        """
        Get the LD data as a `LDLinearOperator` object. This is useful for performing
        linear algebra operations on the LD matrix efficiently.

        !!! seealso "See Also"
        * [LDLinearOperator][magenpy.LDMatrix.LDLinearOperator]

        :param load_kwargs: Additional keyword arguments to pass to the `load_data` method.

        :return: An `LDLinearOperator` object containing the LD data.
        """

        if self.in_memory:
            return self._cached_lop
        else:
            return self.load_data(**load_kwargs)

    def to_csr(self, **load_kwargs):
        """
        Get the LD data as a `scipy.csr_matrix` object.

        :param load_kwargs: Additional keyword arguments to pass to the `load_data` method.
        :return: A `scipy.csr_matrix` object representing the LD matrix.
        """

        if self.in_memory:
            return self._cached_lop.to_csr()
        else:
            return self.load_data(return_as_csr=True, **load_kwargs)

    def load_data(self,
                  start_row=None,
                  end_row=None,
                  dtype=None,
                  return_square=True,
                  keep_original_shape=False,
                  return_symmetric=False,
                  return_as_csr=False):
        """
        A utility function to load and process the LD matrix data.
        This function is particularly useful for filtering, symmetrizing, and dequantizing the LD matrix
        after it's loaded to memory.

        .. note ::
            Start and end row positions are with reference to the stored on-disk LD matrix. This means
            that the mask is not considered when defining the boundaries
            based on the start and end row positions.

        :param start_row: The start row to load to memory (if loading a subset of the matrix).
        :param end_row: The end row (not inclusive) to load to memory (if loading a subset of the matrix).
        :param dtype: The data type for the entries of the LD matrix.
        :param return_square: If True, return a square representation of the LD matrix. This flag is used in
        conjunction with the `start_row` and `end_row` parameters. In particular, if `end_row` is less than the
        number of variants and `return_square=False`, then we return a rectangular slice of the LD matrix
        corresponding to the rows requested by the user.
        :param keep_original_shape: If True, keep the original shape of the LD matrix. This is useful when
        returning a subset of the matrix, but keeping the original shape.
        :param return_symmetric: If True, return a full symmetric representation of the LD matrix.
        :param return_as_csr: If True, return the data in the CSR format.

        :return: An LDLinearOperator object containing the LD data or a scipy CSR matrix (if
        `return_as_csr` is set to `True`.
        """

        # Sanity checking:

        if start_row is not None:
            assert 0. <= start_row < self.stored_n_snps
        if end_row is not None:
            assert 0. < end_row <= self.stored_n_snps

        if keep_original_shape:
            assert return_as_csr, "If keeping the original shape, the data must be returned as a CSR matrix."

        # -------------- Step 1: Preparing input data type --------------
        if dtype is None:
            dtype = self.stored_dtype
            dequantize_data = False
        else:
            dtype = np.dtype(dtype)
            if np.issubdtype(self.stored_dtype, np.integer) and np.issubdtype(dtype, np.floating):
                dequantize_data = True
            else:
                dequantize_data = False

        # -------------- Step 2: Pre-process data boundaries (if provided) --------------
        n_snps = self.stored_n_snps

        start_row = start_row or 0
        end_row = end_row or n_snps

        end_row = min(end_row, n_snps)

        # -------------- Step 2: Fetch the indptr array --------------

        # Get the index pointer array:
        indptr = self._zg['matrix/indptr'][start_row:end_row + 1]

        # Determine the start and end positions in the data matrix
        # based on the requested start and end rows:
        data_start = indptr[0]
        data_end = indptr[-1]

        # If the user is requesting a subset of the matrix, then we need to adjust
        # the index pointer accordingly:
        if start_row > 0:
            # Zero out all index pointers before `start_row`:
            indptr = np.clip(indptr - data_start, a_min=0, a_max=None)

        # -------------- Step 3: Loading and filtering data array --------------

        data = self._zg['matrix/data'][data_start:data_end]

        # Filter the data and index pointer arrays based on the mask (if set):
        if self.is_mask_set or (end_row < n_snps and return_square):

            mask = np.zeros(n_snps, dtype=np.int8)

            # Two cases to consider:

            # 1) If the mask is not set:
            if not self.is_mask_set:

                # If the returned matrix should be square:
                if return_square:
                    mask[start_row:end_row] = 1
                else:
                    mask[start_row:] = 1

                new_nrows = end_row - start_row
            else:
                # If the mask is set:

                mask[self._mask] = 1

                # If return square, ensure that elements after end row are set to 0 in the mask:
                if return_square:
                    mask[end_row:] = 0

                # Compute new size:
                new_nrows = mask[start_row:end_row].sum()

            from .stats.ld.c_utils import filter_ut_csr_matrix_inplace

            data, indptr = filter_ut_csr_matrix_inplace(indptr, data, mask[start_row:], new_nrows)

        # -------------- Step 4: Symmetrizing input matrix --------------

        if return_symmetric:

            from .stats.ld.c_utils import symmetrize_ut_csr_matrix

            if np.issubdtype(self.stored_dtype, np.integer):
                fill_val = np.iinfo(self.stored_dtype).max
            else:
                fill_val = 1.

            data, indptr, leftmost_idx = symmetrize_ut_csr_matrix(indptr, data, fill_val)
        else:
            leftmost_idx = np.arange(1, indptr.shape[0], dtype=np.int32)

        # -------------- Step 5: Dequantizing/type cast requested data --------------

        if dequantize_data:
            data = dequantize(data, float_dtype=dtype)
        else:
            data = data.astype(dtype, copy=False)

        # ---------------------------------------------------------------------------
        # Determine the shape of the data matrix:

        if keep_original_shape:
            shape = (n_snps, n_snps)
        elif return_square or end_row == n_snps:
            shape = (indptr.shape[0] - 1, indptr.shape[0] - 1)
        else:
            shape = (indptr.shape[0] - 1, n_snps)

        # ---------------------------------------------------------------------------
        # Return the requested data:

        if return_as_csr:
            # If the user requested the data as CSR matrix:

            from .stats.ld.c_utils import expand_ranges
            from scipy.sparse import csr_matrix

            indices = expand_ranges(leftmost_idx,
                                    (np.diff(indptr) + leftmost_idx).astype(np.int32),
                                    data.shape[0])

            if keep_original_shape:
                # TODO: Consider incorporating this in `LDLinearOperator.to_csr`
                indices += start_row
                indptr = np.concatenate([np.zeros(start_row, dtype=indptr.dtype),
                                         indptr,
                                         np.ones(n_snps - end_row, dtype=indptr.dtype) * indptr[-1]])

            return csr_matrix(
                (
                    data,
                    indices,
                    indptr
                ),
                shape=shape,
                dtype=dtype
            )

        else:
            # Otherwise, return as a linear operator:

            return LDLinearOperator(
                indptr,
                data,
                leftmost_idx,
                symmetric=return_symmetric,
                shape=shape
            )

    def load(self,
             force_reload=False,
             return_symmetric=False,
             dtype=None) -> LDLinearOperator:

        """
        Load the LD matrix from on-disk storage in the form of Zarr arrays to memory,
        in the form of sparse CSR matrices.

        :param force_reload: If True, it will reload the data even if it is already in memory.
        :param return_symmetric: If True, return a full symmetric representation of the LD matrix.
        :param dtype: The data type for the entries of the LD matrix.

        !!! seealso "See Also"
            * [load_data][magenpy.LDMatrix.LDMatrix.load_data]

        :return: The LD matrix as a `scipy` sparse CSR matrix.
        """

        if dtype is not None:
            dtype = np.dtype(dtype)
        else:
            dtype = self.dtype

        if not force_reload and self.in_memory and return_symmetric == self._cached_lop.symmetric:
            # If the LD matrix is already in memory and the requested symmetry is the same,
            # then we don't need to reload the matrix. Here, we only transform its entries it to
            # conform to the requested data types of the user:

            # If the requested data type differs from the stored one, we need to cast the data:
            if dtype is not None and self._cached_lop.ld_data_type != np.dtype(dtype):

                if np.issubdtype(self._cached_lop.ld_data_type, np.floating) and np.issubdtype(dtype, np.floating):
                    # The user requested casting the data to different floating point precision:
                    self._cached_lop.ld_data = self._cached_lop.ld_data.astype(dtype)
                elif np.issubdtype(self._cached_lop.ld_data_type, np.integer) and np.issubdtype(dtype, np.integer):
                    # The user requested casting the data to different integer format:
                    self._cached_lop.ld_data = quantize(dequantize(self._cached_lop.ld_data), int_dtype=dtype)
                elif np.issubdtype(self._cached_lop.ld_data_type, np.floating) and np.issubdtype(dtype, np.integer):
                    # The user requested quantizing the data from floats to integers:
                    self._cached_lop.ld_data = quantize(self._cached_lop.ld_data, int_dtype=dtype)
                else:
                    # The user requested dequantizing the data from integers to floats:
                    self._cached_lop.ld_data = dequantize(self._cached_lop.ld_data, float_dtype=dtype)

        else:
            # If we are re-loading the matrix, make sure to release the current one:
            self.release()

            self._cached_lop = self.load_data(return_symmetric=return_symmetric,
                                              dtype=dtype)

        return self._cached_lop

    def release(self):
        """
        Release the LD data and associated arrays from memory.
        """
        self._cached_lop = None
        self.index = 0

    def validate_ld_matrix(self):
        """
        Checks that the `LDMatrix` object has correct structure and
        checks its contents for validity.

        Specifically, we check that:
        * The dimensions of the matrix and its associated attributes are matching.
        * The masking is working properly.
        * Index pointer is valid and its contents make sense.

        :return: True if the matrix has the correct structure.
        :raises ValueError: If the matrix or some of its entries are not valid.
        """

        class_attrs = ['snps', 'a1', 'a2', 'maf', 'bp_position', 'cm_position', 'ld_score']

        for attr in class_attrs:
            attribute = getattr(self, attr)
            if attribute is None:
                continue
            if len(attribute) != len(self):
                raise ValueError(f"Invalid LD Matrix: Dimensions for attribute {attr} are not aligned!")

        # -------------------- Index pointer checks --------------------
        # Check that the entries of the index pointer are all positive or zero:
        indptr = self.indptr[:]

        if indptr.min() < 0:
            raise ValueError("The index pointer contains negative entries!")

        # Check that the entries don't decrease:
        indptr_diff = np.diff(indptr)
        if indptr_diff.min() < 0:
            raise ValueError("The index pointer entries are not increasing!")

        # Check that the last entry of the index pointer matches the shape of the data:
        if indptr[-1] != self.data.shape[0]:
            raise ValueError("The last entry of the index pointer "
                             "does not match the shape of the data!")

        # TODO: Add other sanity checks here?

        return True

    def iter_blocks(self,
                    block_size=None,
                    block_size_cm=None,
                    block_size_kb=None,
                    blocks=None,
                    min_block_size=2,
                    max_block_size=None,
                    return_type='csr',
                    return_block_boundaries=False,
                    dry_run=False,
                    **return_type_kwargs
                    ):
        """
        Iterator over blocks of the LD matrix.

        This function allows for iterating over blocks of the LD matrix, either based on the number of variants
        per block, or based on the physical distance between variants. The function yields the requested data
        in the form of a sparse CSR matrix, a `LinearOperator`, or a dense numpy array.

        .. note::
            For now, all block-related information is with reference to the original, unfiltered LD matrix.
            In future releases, we may consider supporting block iterators based on subsets of the
            matrix.

        :param block_size: An integer specifying the block size in terms of the number of variants.
        :param block_size_cm: The block size in centi-Morgans (cM).
        :param block_size_kb: The block size in kilo-base pairs (kb).
        :param blocks: An array or list specifying the block boundaries to iterate over.
        :param min_block_size: The minimum block size.
        :param return_type: The type of data to return. Options are 'csr', 'linop', or 'numpy'.
        :param return_block_boundaries: If True, return the boundaries of the generated blocks along with the
        LD data itself.
        :param dry_run: If True, do not load the data, just return the block boundaries. Useful for debugging.
        :param return_type_kwargs: Additional keyword arguments to pass to the return type constructor.

        """

        # Sanity checks:
        assert return_type in ('csr', 'linop', 'numpy')
        assert min_block_size >= 1

        # Determine the block boundaries based on the input parameters:
        if self.in_memory:
            n_snps = self.n_snps
        else:
            n_snps = self.stored_n_snps

        from .utils.compute_utils import generate_overlapping_windows

        if blocks is not None:
            block_iter = blocks
        elif block_size is not None:

            windows = generate_overlapping_windows(np.arange(n_snps),
                                                   block_size - 1, block_size,
                                                   min_window_size=min_block_size)
            block_iter = np.insert(windows[:, 1], 0, 0)

        elif block_size_cm is not None or block_size_kb is not None:

            if block_size_cm is not None:
                dist = self.get_metadata('cm', apply_mask=self.in_memory)
                block_size = block_size_cm
            else:
                dist = self.get_metadata('bp', apply_mask=self.in_memory) / 1000
                block_size = block_size_kb

            windows = generate_overlapping_windows(dist,
                                                   block_size,
                                                   block_size,
                                                   min_window_size=min_block_size)

            block_iter = np.insert(windows[:, 1], 0, 0)
        elif self.ld_estimator == 'windowed':

            est_properties = self.estimator_properties

            if 'Window size (cM)' in est_properties:
                block_size = est_properties['Window size (cM)']
                dist = self.get_metadata('cm', apply_mask=self.in_memory)
            elif 'Window size (kb)' in est_properties:
                block_size = est_properties['Window size (kb)']
                dist = self.get_metadata('bp', apply_mask=self.in_memory) / 1000
            else:
                block_size = est_properties['Window size']
                dist = np.arange(n_snps)

            windows = generate_overlapping_windows(dist,
                                                   block_size,
                                                   block_size,
                                                   min_window_size=min_block_size)

            block_iter = np.insert(windows[:, 1], 0, 0)

        elif self.ld_estimator == 'block':

            from .utils.model_utils import map_variants_to_genomic_blocks

            variants_to_blocks = map_variants_to_genomic_blocks(
                pd.DataFrame({
                    'POS': self.get_metadata('bp', apply_mask=self.in_memory)
                }).reset_index(),
                pd.DataFrame(np.array(self.estimator_properties['LD blocks']),
                             columns=['block_start', 'block_end'],
                             dtype=np.int32),
                filter_unmatched=True
            )

            block_iter = [0] + list(variants_to_blocks.groupby('block_end')['index'].max().values + 1)
        else:
            block_iter = [0, n_snps]

        # If the maximum block size is specified by the user,
        # then use the `split_block_boundaries` utility function to split
        # blocks to conform to this constraint:
        if max_block_size is not None:

            from .utils.compute_utils import split_block_boundaries

            block_iter = split_block_boundaries(
                block_iter,
                max_block_size,
                mask=[self._mask, None][self.in_memory]
            )

        mat = None

        # Loop over the blocks and yield the requested data:
        for bidx in range(len(block_iter) - 1):

            start = block_iter[bidx]
            end = block_iter[bidx + 1]

            if dry_run:
                yield start, end
            else:
                # If the data is in memory, subset the data for the requested block:
                if self.in_memory:

                    if return_type == 'numpy':
                        mat = self._cached_lop.to_numpy(start, end)
                    elif return_type in ('csr', 'linop'):
                        mat = self._cached_lop[start:end, start:end]
                        if return_type == 'csr':
                            mat = mat.to_csr()

                else:

                    mat = self.load_data(start_row=start,
                                         end_row=end,
                                         return_as_csr=return_type in ('csr', 'numpy'),
                                         **return_type_kwargs)
                    if return_type == 'numpy':
                        mat = mat.todense()

                if return_block_boundaries:
                    yield mat, (start, end)
                else:
                    yield mat

    def getrow(self, index, symmetric=False, return_indices=False):
        """
        Extract a single row from the LD matrix.

        # TODO: Support extracting rows from the symmetric LD matrix.
        # TODO: Verify that this is correct.

        :param index: The index of the row to extract.
        :param symmetric: If True, return a symmetric representation of the row (i.e. LD with
        variants before and after the index variant).
        :param return_indices: If True, return the indices of the non-zero elements of that row.

        :return: The requested row of the LD matrix.
        """

        if symmetric:
            raise NotImplementedError("Symmetric row extraction is not yet supported.")

        if self.in_memory:
            return self._cached_lop.getrow(index, symmetric=symmetric, return_indices=return_indices)
        else:
            start_idx, end_idx = self.indptr[index:index + 2]
            data = self.data[start_idx:end_idx]

            if return_indices:
                return data, np.arange(
                    index + 1,
                    index + len(data)
                )
            else:
                return data

    def summary(self):
        """
        :return: A `pandas` dataframe with summary of the main attributes of the LD matrix.
        """

        return pd.DataFrame([
            {'LD Matrix property': 'Chromosome', 'Value': self.chromosome},
            {'LD Matrix property': 'Stored shape', 'Value': self.stored_shape},
            {'LD Matrix property': 'Stored data type', 'Value': self.stored_dtype},
            {'LD Matrix property': 'Stored entries', 'Value': self._zg['matrix/data'].shape[0]},
            {'LD Matrix property': 'Path', 'Value': self.store.path},
            {'LD Matrix property': 'In memory?', 'Value': self.in_memory},
            {'LD Matrix property': 'Mask set?', 'Value': self.is_mask_set},
            {'LD Matrix property': 'On-disk storage', 'Value': f'{self.get_total_storage_size():.3} MB'},
            {'LD Matrix property': 'Estimated uncompressed size', 'Value': f'{self.estimate_uncompressed_size():.3} MB'}
        ]).set_index('LD Matrix property')

    def __repr__(self):
        """
        :return: A summary of the LDMatrix object as a string.
        """
        return self.summary().to_string()

    def _repr_html_(self):
        """
        :return: A summary of the LDMatrix object as an HTML table.
        """

        styled_df = self.summary().style.set_table_attributes(
            'class="dataframe" style="width: 50%"'
        ).set_properties(**{
            'text-align': 'left',
            'white-space': 'normal',  # Allow text wrapping
            'word-wrap': 'break-word',  # Break words at any character
            'word-break': 'break-word',  # Allow breaking words when necessary
            'font-size': '13px',
            'padding': '10px 15px'
        }).set_table_styles([
            # Table border and design
            {'selector': 'table', 'props': [
                ('border-collapse', 'separate'),
                ('border-spacing', '0px'),
                ('border-radius', '5px'),
                ('overflow', 'hidden'),
                ('box-shadow', '0 2px 3px rgba(0,0,0,0.1)'),
                ('margin', '20px 0'),
                ('table-layout', 'fixed'),  # Fixed layout helps with word wrapping
                ('width', '50%')  # Ensure table takes full width
            ]},
            # Header styling
            {'selector': 'thead th', 'props': [
                ('background-color', '#3b5f9e'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'left'),
                ('padding', '12px 15px')
            ]},
            # Property name cells (index)
            {'selector': 'tbody th', 'props': [
                ('background-color', '#f8f9fa'),
                ('color', '#333'),
                ('font-weight', 'bold'),
                ('border-bottom', '1px solid #eaeaea'),
                ('text-align', 'left'),
                ('padding', '10px 15px'),
                ('width', '30%'),  # Control width of the property column
                ('word-wrap', 'break-word'),
                ('white-space', 'normal')
            ]},
            # Value cells
            {'selector': 'tbody td', 'props': [
                ('background-color', 'white'),
                ('border-bottom', '1px solid #eaeaea'),
                ('color', '#444'),
                ('padding', '10px 15px'),
                ('width', '70%'),  # Control width of the value column
                ('word-wrap', 'break-word'),
                ('white-space', 'normal'),
                ('max-width', '0')  # Forces the cell to respect width constraints
            ]},
            # Alternating rows
            {'selector': 'tbody tr:nth-of-type(odd) td', 'props': [
                ('background-color', '#f8f9fa'),
            ]},
            # Hover effect on rows
            {'selector': 'tbody tr:hover td, tbody tr:hover th', 'props': [
                ('background-color', '#e8f0fe'),
            ]}
        ]).hide(axis='columns')

        return styled_df._repr_html_()

    def __getstate__(self):
        return self.store.path, self.in_memory, self.is_symmetric, self._mask, self.dtype

    def __setstate__(self, state):

        path, in_mem, is_symmetric, mask, dtype = state

        self._zg = zarr.open_group(path, mode='r')
        self._cached_lop = None
        self.index = 0
        self._mask = None

        if mask is not None:
            self.set_mask(mask)

        if in_mem:
            self.load(return_symmetric=is_symmetric, dtype=dtype)

    def __len__(self):
        return self.n_snps

    def __getitem__(self, item):
        """
        Access the LD matrix entries via the `[]` operator.
        This implementation supports the following types of indexing:
        * Accessing a single row of the LD matrix by specifying numeric index or SNP rsID.
        * Accessing a single entry of the LD matrix by specifying numeric indices or SNP rsIDs.

        Example usages:

            >>> ldm[0]
            >>> ldm['rs123']
            >>> ldm['rs123', 'rs456']
        """

        dq_scale = self.dequantization_scale

        if isinstance(item, tuple):
            assert len(item) == 2
            assert type(item[0]) is type(item[1])

            if isinstance(item[0], str):

                # If they're the same variant:
                if item[0] == item[1]:
                    return 1.

                # Extract the indices of the two variants:
                snps = self.snps.tolist()

                try:
                    index_1 = snps.index(item[0])
                except ValueError:
                    raise ValueError(f"Invalid variant rsID: {item[0]}")

                try:
                    index_2 = snps.index(item[1])
                except ValueError:
                    raise ValueError(f"Invalid variant rsID: {item[1]}")

            else:
                index_1, index_2 = item

            index_1, index_2 = sorted([index_1, index_2])

            if index_1 == index_2:
                return 1.
            if index_2 - index_1 > self.window_size[index_1]:
                return 0.
            else:
                row = self.getrow(index_1)
                return dq_scale*row[index_2 - index_1 - 1]

        if isinstance(item, int):
            return dq_scale*self.getrow(item)
        elif isinstance(item, str):
            try:
                index = self.snps.tolist().index(item)
            except ValueError:
                raise ValueError(f"Invalid variant rsID: {item}")

            return dq_scale*self.getrow(index)

    def __iter__(self):
        """
        Iterate over the rows of the LD matrix.
        This iterator supports fetching the rows of the LD matrix one by one from on-disk storage,
        or, if the data is already loaded, it subsets the data for each row.
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Get the next row of the LD matrix.
        """

        if self.index == len(self):
            self.index = 0
            raise StopIteration

        next_item = self.getrow(self.index)
        self.index += 1

        return next_item
