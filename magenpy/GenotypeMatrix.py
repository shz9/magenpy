from typing import Union
import tempfile
import pandas as pd
import numpy as np
from .utils.system_utils import makedir
from .SampleTable import SampleTable


class GenotypeMatrix(object):
    """
    A class to represent a genotype matrix. The genotype matrix is a matrix of
    where the rows represent samples and the columns represent genetic variants.
    In general, genotype matrices are assumed to reside on disk and this class
    provides a convenient interface to interact with and perform computations
    on the genotype matrix.

    Currently, we assume that the genotype matrix is stored using plink's BED
    file format, with associated tables for the samples (i.e. FAM file) and genetic
    variants (i.e. BIM file). Classes that inherit from this generic class support
    various backends to access and performing computations on this genotype data.

    !!! seealso "See Also"
            * [xarrayGenotypeMatrix][magenpy.GenotypeMatrix.xarrayGenotypeMatrix]
            * [plinkBEDGenotypeMatrix][magenpy.GenotypeMatrix.plinkBEDGenotypeMatrix]

    :ivar sample_table: A table containing information about the samples in the genotype matrix
    (initially read from the FAM file).
    :ivar snp_table: A table containing information about the genetic variants in the genotype matrix
    (initially read from the BIM file).
    :ivar bed_file: The path to the plink BED file containing the genotype matrix.
    :ivar _genome_build: The genome build or assembly under which the SNP coordinates are defined.
    :ivar temp_dir: The directory where temporary files will be stored (if needed).
    :ivar threads: The number of threads to use for parallel computations.

    """

    def __init__(self,
                 sample_table: Union[pd.DataFrame, SampleTable, None] = None,
                 snp_table: Union[pd.DataFrame, None] = None,
                 temp_dir: str = 'temp',
                 bed_file: str = None,
                 genome_build=None,
                 threads=1,
                 **kwargs):
        """
        Initialize a GenotypeMatrix object.

        :param sample_table: A table containing information about the samples in the genotype matrix.
        :param snp_table: A table containing information about the genetic variants in the genotype matrix.
        :param temp_dir: The directory where temporary files will be stored (if needed).
        :param bed_file: The path to the plink BED file containing the genotype matrix.
        :param genome_build: The genome build or assembly under which the SNP coordinates are defined.
        :param threads: The number of threads to use for parallel computations.
        :param kwargs: Additional keyword arguments.
        """

        self.sample_table: Union[pd.DataFrame, SampleTable, None] = None
        self.snp_table: Union[pd.DataFrame, None] = snp_table

        if sample_table is not None:
            self.set_sample_table(sample_table)

        if snp_table is not None and 'original_index' not in self.snp_table.columns:
            self.snp_table['original_index'] = np.arange(len(self.snp_table))

        temp_dir_prefix = 'gmat_'

        if self.chromosome is not None:
            temp_dir_prefix += f'chr{self.chromosome}_'

        self.temp_dir = temp_dir
        self.temp_dir_prefix = temp_dir_prefix

        makedir(self.temp_dir)

        self.bed_file = bed_file
        self._genome_build = genome_build

        self.threads = threads

    @classmethod
    def from_file(cls, file_path, temp_dir='temp', **kwargs):
        """
        Initialize a genotype matrix object by passing a file path + other keyword arguments.
        :param file_path: The path to the plink BED file.
        :type file_path: str
        :param temp_dir: The directory where temporary files will be stored.
        :type temp_dir: str
        :param kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    @property
    def shape(self):
        """
        :return: The shape of the genotype matrix. Rows correspond to the
        number of samples and columns to the number of SNPs.
        """
        return self.n, self.m

    @property
    def n(self):
        """
        !!! seealso "See Also"
            * [sample_size][magenpy.GenotypeMatrix.GenotypeMatrix.sample_size]

        :return: The sample size or number of individuals in the genotype matrix.
        """
        return self.sample_table.n

    @property
    def sample_size(self):
        """
        !!! seealso "See Also"
            * [n][magenpy.GenotypeMatrix.GenotypeMatrix.n]

        :return: The sample size or number of individuals in the genotype matrix.
        """
        return self.n

    @property
    def samples(self):
        """
        :return: An array of sample IDs in the genotype matrix.
        """
        return self.sample_table.iid

    @property
    def sample_index(self):
        return self.sample_table.table['original_index'].values

    @property
    def snp_index(self):
        return self.snp_table['original_index'].values

    @property
    def m(self):
        """

        !!! seealso "See Also"
            * [n_snps][magenpy.GenotypeMatrix.GenotypeMatrix.n_snps]

        :return: The number of variants in the genotype matrix.
        """
        if self.snp_table is not None:
            return len(self.snp_table)

    @property
    def n_snps(self):
        """
        !!! seealso "See Also"
            * [m][magenpy.GenotypeMatrix.GenotypeMatrix.m]

        :return: The number of variants in the genotype matrix.
        """
        return self.m

    @property
    def genome_build(self):
        """
        :return: The genome build or assembly under which the SNP coordinates are defined.
        """
        return self._genome_build

    @property
    def chromosome(self):
        """
        ..note::
        This is a convenience method that assumes that the genotype matrix contains variants
        from a single chromosome. If there are multiple chromosomes, the method will return `None`.

        :return: The chromosome associated with the variants in the genotype matrix.
        """
        chrom = self.chromosomes
        if chrom is not None and len(chrom) == 1:
            return chrom[0]

    @property
    def chromosomes(self):
        """
        :return: The unique set of chromosomes comprising the genotype matrix.
        """
        chrom = self.get_snp_attribute('CHR')
        if chrom is not None:
            return np.unique(chrom)

    @property
    def snps(self):
        """
        :return: The SNP rsIDs for variants in the genotype matrix.
        """
        return self.get_snp_attribute('SNP')

    @property
    def bp_pos(self):
        """
        :return: The basepair position for the genetic variants in the genotype matrix.
        """
        return self.get_snp_attribute('POS')

    @property
    def cm_pos(self):
        """
        :return: The position of genetic variants in the genotype matrix in units of Centi Morgan.
        :raises KeyError: If the genetic distance is not set in the genotype file.
        """
        cm = self.get_snp_attribute('cM')
        if len(set(cm)) == 1:
            raise KeyError("Genetic distance in centi Morgan (cM) is not "
                           "set in the genotype file!")
        return cm

    @property
    def a1(self):
        """
        !!! seealso "See Also"
            * [alt_allele][magenpy.GenotypeMatrix.GenotypeMatrix.alt_allele]
            * [effect_allele][magenpy.GenotypeMatrix.GenotypeMatrix.effect_allele]

        :return: The effect allele `A1` for each genetic variant.

        """
        return self.get_snp_attribute('A1')

    @property
    def a2(self):
        """

        !!! seealso "See Also"
            * [ref_allele][magenpy.GenotypeMatrix.GenotypeMatrix.ref_allele]

        :return: The reference allele `A2` for each genetic variant.

        """
        return self.get_snp_attribute('A2')

    @property
    def ref_allele(self):
        """

        !!! seealso "See Also"
            * [a2][magenpy.GenotypeMatrix.GenotypeMatrix.a2]

        :return: The reference allele `A2` for each genetic variant.
        """
        return self.a2

    @property
    def alt_allele(self):
        """
        !!! seealso "See Also"
            * [effect_allele][magenpy.GenotypeMatrix.GenotypeMatrix.effect_allele]
            * [a1][magenpy.GenotypeMatrix.GenotypeMatrix.a1]

        :return: The effect allele `A1` for each genetic variant.

        """
        return self.a1

    @property
    def effect_allele(self):
        """

        !!! seealso "See Also"
            * [alt_allele][magenpy.GenotypeMatrix.GenotypeMatrix.alt_allele]
            * [a1][magenpy.GenotypeMatrix.GenotypeMatrix.a1]

        :return: The effect allele `A1` for each genetic variant.

        """
        return self.a1

    @property
    def n_per_snp(self):
        """
        :return: Sample size per genetic variant (accounting for potential missing values).
        """
        n = self.get_snp_attribute('N')
        if n is not None:
            return n
        else:
            self.compute_sample_size_per_snp()
            return self.get_snp_attribute('N')

    @property
    def maf(self):
        """
        :return: The minor allele frequency (MAF) of each variant in the genotype matrix.
        """
        maf = self.get_snp_attribute('MAF')
        if maf is not None:
            return maf
        else:
            self.compute_allele_frequency()
            return self.get_snp_attribute('MAF')

    @property
    def maf_var(self):
        """
        :return: The variance in minor allele frequency (MAF) of each variant in the genotype matrix.
        """
        return 2. * self.maf * (1. - self.maf)

    def estimate_memory_allocation(self, dtype=np.float32):
        """
        :return: An estimate of the memory allocation for the genotype matrix in megabytes.
        """
        return self.n * self.m * np.dtype(dtype).itemsize / 1024 ** 2

    def get_snp_table(self, col_subset=None):
        """
        A convenience method to extract SNP-related information from the genotype matrix.
        :param col_subset: A list of columns to extract from the SNP table.

        :return: A `pandas` DataFrame with the requested columns.
        """

        if col_subset is None:
            return self.snp_table.copy()
        else:
            present_cols = list(set(col_subset).intersection(set(self.snp_table.columns)))
            non_present_cols = list(set(col_subset) - set(present_cols))

            if len(present_cols) > 0:
                table = self.snp_table[present_cols].copy()
            else:
                table = pd.DataFrame({c: [] for c in non_present_cols})

            for col in non_present_cols:

                if col == 'MAF':
                    table['MAF'] = self.maf
                elif col == 'MAF_VAR':
                    table['MAF_VAR'] = self.maf_var
                elif col == 'N':
                    table['N'] = self.n_per_snp
                else:
                    raise KeyError(f"Column '{col}' is not available in the SNP table!")

            return table[list(col_subset)]

    def get_snp_attribute(self, attr):
        """

        :param attr: The name of the attribute to extract from the SNP table.
        :return: The values of a specific attribute for each variant in the genotype matrix.
        """
        if self.snp_table is not None and attr in self.snp_table.columns:
            return self.snp_table[attr].values

    def compute_ld(self,
                   estimator,
                   output_dir,
                   dtype='int16',
                   compressor_name='zstd',
                   compression_level=7,
                   compute_spectral_properties=False,
                   **ld_kwargs):
        """

        Compute the Linkage-Disequilibrium (LD) or SNP-by-SNP correlation matrix
        for the variants defined in the genotype matrix.

        :param estimator: The estimator for the LD matrix. We currently support
        4 different estimators: `sample`, `windowed`, `shrinkage`, and `block`.
        :param output_dir: The output directory where the Zarr array containing the
        entries of the LD matrix will be stored.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compressor to use for the Zarr array.
        :param compression_level: The compression level for the Zarr array (1-9)
        :param ld_kwargs: keyword arguments for the various LD estimators. Consult
        the implementations of `WindowedLD`, `ShrinkageLD`, and `BlockLD` for details.
        :param compute_spectral_properties: If True, compute and store information about the eigenvalues of
        the LD matrix.
        """

        from .stats.ld.estimator import SampleLD, WindowedLD, ShrinkageLD, BlockLD

        if estimator == 'sample':
            ld_est = SampleLD(self)
        elif estimator == 'windowed':
            ld_est = WindowedLD(self, **ld_kwargs)
        elif estimator == 'shrinkage':
            ld_est = ShrinkageLD(self, **ld_kwargs)
        elif estimator == 'block':
            ld_est = BlockLD(self, **ld_kwargs)
        else:
            raise KeyError(f"LD estimator {estimator} is not recognized!")

        return ld_est.compute(output_dir,
                              dtype=dtype,
                              compressor_name=compressor_name,
                              compression_level=compression_level,
                              compute_spectral_properties=compute_spectral_properties)

    def set_sample_table(self, sample_table):
        """
        A convenience method set the sample table for the genotype matrix.
        This may be useful for syncing sample tables across different Genotype matrices
        corresponding to different chromosomes or genomic regions.

        :param sample_table: An instance of SampleTable or a pandas dataframe containing
        information about the samples in the genotype matrix.

        """

        if isinstance(sample_table, SampleTable):
            self.sample_table = sample_table
        elif isinstance(sample_table, pd.DataFrame):
            self.sample_table = SampleTable(sample_table)
        else:
            raise ValueError("The sample table is invalid! "
                             "Has to be either an instance of "
                             "SampleTable or pandas DataFrame.")

    def filter_snps(self, extract_snps=None, extract_file=None):
        """
        Filter variants from the genotype matrix. User must specify
        either a list of variants to extract or the path to a plink-style file
        with the list of variants to extract.

        :param extract_snps: A list (or array) of SNP IDs to keep in the genotype matrix.
        :param extract_file: The path to a file with the list of variants to extract.
        """

        assert extract_snps is not None or extract_file is not None

        if extract_snps is None:
            from .parsers.misc_parsers import read_snp_filter_file
            extract_snps = read_snp_filter_file(extract_file)

        self.snp_table = self.snp_table.merge(pd.DataFrame({'SNP': extract_snps}))

    def filter_by_allele_frequency(self, min_maf=None, min_mac=1):
        """
        Filter variants by minimum minor allele frequency or allele count cutoffs.

        :param min_maf: Minimum minor allele frequency
        :param min_mac: Minimum minor allele count (1 by default)
        """

        if min_mac or min_maf:

            maf = self.maf
            n = self.n_per_snp

            keep_flag = None

            if min_mac:
                mac = (2*maf*n).astype(np.int64)
                keep_flag = (mac >= min_mac) & ((2*n - mac) >= min_mac)

            if min_maf:

                maf_cond = (maf >= min_maf) & (1. - maf >= min_maf)
                if keep_flag is not None:
                    keep_flag = keep_flag & maf_cond
                else:
                    keep_flag = maf_cond

            if keep_flag is not None:
                self.filter_snps(extract_snps=self.snps[keep_flag])

    def drop_duplicated_snps(self):
        """
        A convenience method to drop variants with duplicated SNP rsIDs.
        """

        u_snps, counts = np.unique(self.snps, return_counts=True)
        if len(u_snps) < self.n_snps:
            # Keep only SNPs which occur once in the sequence:
            self.filter_snps(u_snps[counts == 1])

    def filter_samples(self, keep_samples=None, keep_file=None):
        """
        Filter samples from the genotype matrix. User must specify
        either a list of samples to keep or the path to a plink-style file
        with the list of samples to keep.

        :param keep_samples: A list (or array) of sample IDs to keep in the genotype matrix.
        :param keep_file: The path to a file with the list of samples to keep.
        """

        self.sample_table.filter_samples(keep_samples=keep_samples, keep_file=keep_file)

        # IMPORTANT: After filtering samples, update SNP attributes that depend on the
        # samples, such as MAF and N:
        if 'N' in self.snp_table:
            self.compute_sample_size_per_snp()
        if 'MAF' in self.snp_table:
            self.compute_allele_frequency()

    def score(self, beta, standardize_genotype=False):
        """
        Perform linear scoring, i.e. multiply the genotype matrix by the vector of effect sizes, `beta`.

        :param beta: A vector of effect sizes for each variant in the genotype matrix.
        :param standardize_genotype: If True, standardized the genotype matrix when computing the score.
        """
        raise NotImplementedError

    def perform_gwas(self, **gwa_kwargs):
        """
        Perform genome-wide association testing of all variants against the phenotype.

        :param gwa_kwargs: Keyword arguments to pass to the GWA functions. Consult `stats.gwa.utils`
        for relevant keyword arguments for each backend.

        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def compute_allele_frequency(self):
        """
        Compute the allele frequency of each variant or SNP in the genotype matrix.

        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def compute_sample_size_per_snp(self):
        """
        Compute the sample size for each variant in the genotype matrix, accounting for
        potential missing values.

        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def split_by_chromosome(self):
        """
        Split the genotype matrix by chromosome, so that we would
        have a separate `GenotypeMatrix` objects for each chromosome.
        This method returns a dictionary where the key is the chromosome number
        and the value is an object of `GenotypeMatrix` for that chromosome.

        :return: A dictionary of `GenotypeMatrix` objects, one for each chromosome.
        """

        chromosome = self.chromosome

        if chromosome:
            return {chromosome: self}
        else:
            chrom_tables = self.snp_table.groupby('CHR')

            return {
                c: self.__class__(sample_table=self.sample_table,
                                  snp_table=chrom_tables.get_group(c),
                                  bed_file=self.bed_file,
                                  temp_dir=self.temp_dir,
                                  genome_build=self.genome_build,
                                  threads=self.threads)
                for c in chrom_tables.groups
            }

    def split_by_variants(self, variant_group_dict):
        """
        Split the genotype matrix by variants into separate `GenotypeMatrix` objects
        based on the groups defined in `variant_group_dict`. The dictionary should have
        the group name as the key and the list of SNP rsIDs in that group as the value.

        :param variant_group_dict: A dictionary where the key is the group name and the value
        is a list of SNP rsIDs to group together.

        :return: A dictionary of `GenotypeMatrix` objects, one for each group.
        """

        if isinstance(variant_group_dict, dict):

            variant_group_dict = pd.concat([
                pd.DataFrame({'group': group, 'SNP': snps})
                for group, snps in variant_group_dict.items()
            ])
        elif isinstance(variant_group_dict, pd.DataFrame):
            assert 'SNP' in variant_group_dict.columns and 'group' in variant_group_dict.columns
        else:
            raise ValueError("The variant group dictionary is invalid!")

        grouped_table = self.snp_table.merge(variant_group_dict, on='SNP').groupby('group')

        return {
            group: self.__class__(sample_table=self.sample_table,
                                  snp_table=grouped_table.get_group(group).drop(columns='group'),
                                  bed_file=self.bed_file,
                                  temp_dir=self.temp_dir,
                                  genome_build=self.genome_build,
                                  threads=self.threads)
            for group in grouped_table.groups
        }

    def cleanup(self):
        """
        Clean up all temporary files and directories
        """

        pass


class xarrayGenotypeMatrix(GenotypeMatrix):
    """
    A class that defines methods and interfaces for interacting with genotype matrices
    using the `xarray` library. In particular, the class leverages functionality provided by
    the `pandas-plink` package to represent on-disk genotype matrices as chunked multidimensional
    arrays that can be queried and manipulated efficiently and in parallel.

    This class inherits all the attributes of the `GenotypeMatrix` class.

    :ivar xr_mat: The `xarray` object representing the genotype matrix.

    """

    def __init__(self,
                 sample_table=None,
                 snp_table=None,
                 bed_file=None,
                 temp_dir='temp',
                 xr_mat=None,
                 genome_build=None,
                 threads=1):
        """
        Initialize an xarrayGenotypeMatrix object.

        :param sample_table: A table containing information about the samples in the genotype matrix.
        :param snp_table: A table containing information about the genetic variants in the genotype matrix.
        :param bed_file: The path to the plink BED file containing the genotype matrix.
        :param temp_dir: The directory where temporary files will be stored (if needed).
        :param xr_mat: The xarray object representing the genotype matrix.
        :param genome_build: The genome build or assembly under which the SNP coordinates are defined.
        :param threads: The number of threads to use for parallel computations.
        """

        super().__init__(sample_table=sample_table,
                         snp_table=snp_table,
                         temp_dir=temp_dir,
                         bed_file=bed_file,
                         genome_build=genome_build,
                         threads=threads)

        # xarray matrix object, as defined by pandas-plink:
        self.xr_mat = xr_mat

    @classmethod
    def from_file(cls, file_path, temp_dir='temp', **kwargs):
        """
        Create a GenotypeMatrix object using a PLINK BED file with the help
        of the data structures defined in `pandas_plink`. The genotype matrix
        will be represented implicitly in an `xarray` object, and we will use it
        to perform various computations. This method is a utility function to
        construct the genotype matrix object from a plink BED file.

        :param file_path: Path to the plink BED file.
        :param temp_dir: The directory where the temporary files will be stored.
        :param kwargs: Additional keyword arguments.
        """

        from pandas_plink import read_plink1_bin
        import warnings

        # Ignore FutureWarning for now
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            try:
                xr_gt = read_plink1_bin(file_path + ".bed", ref="a0", verbose=False)
            except ValueError:
                xr_gt = read_plink1_bin(file_path, ref="a0", verbose=False)

        # Set the sample table:
        sample_table = xr_gt.sample.coords.to_dataset().to_dataframe()
        sample_table.columns = ['FID', 'IID', 'fatherID', 'motherID', 'sex', 'phenotype']
        sample_table.reset_index(inplace=True, drop=True)
        sample_table = sample_table.astype({
            'FID': str,
            'IID': str,
            'fatherID': str,
            'motherID': str,
            'sex': float,
            'phenotype': float
        })

        sample_table['phenotype'] = sample_table['phenotype'].replace({-9.: np.nan})

        # Set the snp table:
        snp_table = xr_gt.variant.coords.to_dataset().to_dataframe()
        snp_table.columns = ['CHR', 'SNP', 'cM', 'POS', 'A1', 'A2']
        snp_table.reset_index(inplace=True, drop=True)
        snp_table = snp_table.astype({
            'CHR': int,
            'SNP': str,
            'cM': np.float32,
            'POS': np.int32,
            'A1': str,
            'A2': str
        })

        g_mat = cls(sample_table=SampleTable(sample_table),
                    snp_table=snp_table,
                    temp_dir=temp_dir,
                    bed_file=file_path,
                    xr_mat=xr_gt,
                    **kwargs)

        return g_mat

    def set_sample_table(self, sample_table):
        """
        A convenience method set the sample table for the genotype matrix.
        This is useful for cases when we need to sync the sample table across chromosomes.

        :param sample_table: An instance of SampleTable or a pandas dataframe containing
        information about the samples in the genotype matrix.
        """

        super().set_sample_table(sample_table)

        try:
            if self.n != self.xr_mat.shape[0]:
                self.xr_mat = self.xr_mat.sel(sample=self.samples)
        except AttributeError:
            pass

    def filter_snps(self, extract_snps=None, extract_file=None):
        """
        Filter variants from the genotype matrix. User must specify either a list of variants to
        extract or the path to a file with the list of variants to extract.

        :param extract_snps: A list or array of SNP rsIDs to keep in the genotype matrix.
        :param extract_file: The path to a file with the list of variants to extract.
        """

        super().filter_snps(extract_snps=extract_snps, extract_file=extract_file)

        from .utils.compute_utils import intersect_arrays

        idx = intersect_arrays(self.xr_mat.variant.coords['snp'].values, self.snps, return_index=True)

        self.xr_mat = self.xr_mat.isel(variant=idx)

    def filter_samples(self, keep_samples=None, keep_file=None):
        """
        Filter samples from the genotype matrix.
        User must specify either a list of samples to keep or the path to a file with the list of samples to keep.

        :param keep_samples: A list (or array) of sample IDs to keep in the genotype matrix.
        :param keep_file: The path to a file with the list of samples to keep.
        """

        super().filter_samples(keep_samples=keep_samples, keep_file=keep_file)
        self.xr_mat = self.xr_mat.sel(sample=self.samples)

    def to_numpy(self, dtype=np.int8):
        """
        Convert the genotype matrix to a numpy array.
        :param dtype: The data type of the numpy array. Default: Int8

        :return: A numpy array representation of the genotype matrix.
        """

        return self.xr_mat.data.astype(dtype).compute()

    def to_csr(self, dtype=np.int8):
        """
        Convert the genotype matrix to a scipy sparse CSR matrix.
        :param dtype: The data type of the scipy array. Default: Int8

        :return: A `scipy` sparse CSR matrix representation of the genotype matrix.
        """

        mat = self.to_numpy(dtype=dtype)

        from scipy.sparse import csr_matrix

        return csr_matrix(mat)

    def score(self, beta, standardize_genotype=False, skip_na=True):
        """
        Perform linear scoring on the genotype matrix.
        :param beta: A vector or matrix of effect sizes for each variant in the genotype matrix.
        :param standardize_genotype: If True, standardize the genotype when computing the polygenic score.
        :param skip_na: If True, skip missing values when computing the polygenic score.

        :return: The polygenic score(s) (PGS) for each sample in the genotype matrix.

        """

        import dask.array as da

        mat = self.xr_mat.data

        chunked_beta = da.from_array(beta, chunks=mat.chunksize[1])

        if standardize_genotype:
            from .stats.transforms.genotype import standardize
            mat = standardize(mat)
            mat = da.nan_to_num(mat)
            pgs = da.dot(mat, chunked_beta).compute()
        else:
            if skip_na:
                pgs = da.dot(da.nan_to_num(mat), chunked_beta).compute()
            else:
                pgs = da.dot(self.xr_mat.fillna(self.maf).data, chunked_beta).compute()

        return pgs

    def perform_gwas(self, **gwa_kwargs):
        """
        A convenience method that calls specialized utility functions that perform
        genome-wide association testing of all variants against the phenotype.

        :return: A Summary statistics table containing the results of the association testing.
        """

        from .stats.gwa.utils import perform_gwa_xarray
        return perform_gwa_xarray(self, **gwa_kwargs)

    def compute_allele_frequency(self):
        """
        A convenience method that calls specialized utility functions that
        compute the allele frequency of each variant or SNP in the genotype matrix.
        """
        self.snp_table['MAF'] = (self.xr_mat.sum(axis=0) / (2. * self.n_per_snp)).compute().values

    def compute_sample_size_per_snp(self):
        """
        A convenience method that calls specialized utility functions that compute
        the sample size for each variant in the genotype matrix, accounting for
        potential missing values.
        """
        self.snp_table['N'] = self.xr_mat.shape[0] - self.xr_mat.isnull().sum(axis=0).compute().values

    def split_by_chromosome(self):
        """
        Split the genotype matrix by chromosome.
        :return: A dictionary of `xarrayGenotypeMatrix` objects, one for each chromosome.
        """
        split = super().split_by_chromosome()

        for c, gt in split.items():
            gt.xr_mat = self.xr_mat
            if len(split) > 1:
                gt.filter_snps(extract_snps=gt.snps)

        return split

    def split_by_variants(self, variant_group_dict):
        """
        Split the genotype matrix by variants into separate `xarrayGenotypeMatrix` objects
        based on the groups defined in `variant_group_dict`. The dictionary should have
        the group name as the key and the list of SNP rsIDs in that group as the value.

        :param variant_group_dict: A dictionary where the key is the group name and the value
        is a list of SNP rsIDs to group together.

        :return: A dictionary of `xarrayGenotypeMatrix` objects, one for each group.
        """

        split = super().split_by_variants(variant_group_dict)

        for g, gt in split.items():
            gt.xr_mat = self.xr_mat
            gt.filter_snps(extract_snps=gt.snps)

        return split


class bedReaderGenotypeMatrix(GenotypeMatrix):
    """
    NOTE: Still experimental.
    Requires more testing and fine-tuning.
    """

    def __init__(self,
                 sample_table=None,
                 snp_table=None,
                 bed_file=None,
                 temp_dir='temp',
                 bed_reader=None,
                 genome_build=None,
                 threads=1):

        super().__init__(sample_table=sample_table,
                         snp_table=snp_table,
                         temp_dir=temp_dir,
                         bed_file=bed_file,
                         genome_build=genome_build,
                         threads=threads)

        # The bed_reader object:
        self.bed_reader = bed_reader

    @classmethod
    def from_file(cls, file_path, temp_dir='temp', **kwargs):

        from bed_reader import open_bed

        try:
            bed_reader = open_bed(file_path)
        except Exception as e:
            raise e

        # Set the sample table:
        sample_table = pd.DataFrame({
            'FID': bed_reader.fid,
            'IID': bed_reader.iid,
            'fatherID': bed_reader.father,
            'motherID': bed_reader.mother,
            'sex': bed_reader.sex,
            'phenotype': bed_reader.pheno
        }).astype({
            'FID': str,
            'IID': str,
            'fatherID': str,
            'motherID': str,
            'sex': float,
            'phenotype': float
        })

        sample_table['phenotype'] = sample_table['phenotype'].replace({-9.: np.nan})
        sample_table = sample_table.reset_index()

        # Set the snp table:
        snp_table = pd.DataFrame({
            'CHR': bed_reader.chromosome,
            'SNP': bed_reader.sid,
            'cM': bed_reader.cm_position,
            'POS': bed_reader.bp_position,
            'A1': bed_reader.allele_1,
            'A2': bed_reader.allele_2
        }).astype({
            'CHR': int,
            'SNP': str,
            'cM': np.float32,
            'POS': np.int32,
            'A1': str,
            'A2': str
        })

        g_mat = cls(sample_table=SampleTable(sample_table),
                    snp_table=snp_table,
                    temp_dir=temp_dir,
                    bed_reader=bed_reader,
                    **kwargs)

        return g_mat

    def score(self, beta, standardize_genotype=False, skip_na=True):
        """
        Perform linear scoring on the genotype matrix.
        :param beta: A vector or matrix of effect sizes for each variant in the genotype matrix.
        :param standardize_genotype: If True, standardize the genotype when computing the polygenic score.
        :param skip_na: If True, skip missing values when computing the polygenic score.
        """

        if len(beta.shape) > 1:
            pgs = np.zeros((self.n, beta.shape[1]))
        else:
            pgs = np.zeros(self.n)

        if standardize_genotype:
            from .stats.transforms.genotype import standardize
            for (start, end), chunk in self._iter_col_chunks(return_slice=True):
                pgs += standardize(chunk).dot(beta[start:end])
        else:
            for (start, end), chunk in self._iter_col_chunks(return_slice=True):
                if skip_na:
                    chunk_pgs = np.nan_to_num(chunk).dot(beta[start:end])
                else:
                    chunk_pgs = np.where(np.isnan(chunk), self.maf[start:end], chunk).dot(beta[start:end])

                pgs += chunk_pgs

        return pgs

    def perform_gwas(self, **gwa_kwargs):
        """
        Perform genome-wide association testing of all variants against the phenotype.

        TODO: Implement this method...

        :param gwa_kwargs: Keyword arguments to pass to the GWA functions. Consult `stats.gwa.utils`
        """

        raise NotImplementedError

    def compute_allele_frequency(self):
        """
        Compute the allele frequency of each variant or SNP in the genotype matrix.
        """
        self.snp_table['MAF'] = (np.concatenate([np.nansum(bed_chunk, axis=0)
                                                 for bed_chunk in self._iter_col_chunks()]) / (2. * self.n_per_snp))

    def compute_sample_size_per_snp(self):
        """
        Compute the sample size for each variant in the genotype matrix, accounting for
        potential missing values.
        """

        self.snp_table['N'] = self.n - np.concatenate([np.sum(np.isnan(bed_chunk), axis=0)
                                                       for bed_chunk in self._iter_col_chunks()])

    def _iter_row_chunks(self, chunk_size='auto', return_slice=False):
        """
        Iterate over the genotype matrix by rows.

        :param chunk_size: The size of the chunk to read from the genotype matrix.
        :param return_slice: If True, return the slice of the genotype matrix corresponding to the chunk.

        :return: A generator that yields chunks of the genotype matrix.
        """
        if chunk_size == 'auto':
            matrix_size = self.estimate_memory_allocation()
            # By default, we allocate 128MB per chunk:
            chunk_size = int(self.n // (matrix_size // 128))

        for i in range(int(np.ceil(self.n / chunk_size))):
            start, end = int(i * chunk_size), min(int((i + 1) * chunk_size), self.n)
            chunk = self.bed_reader.read(np.s_[self.sample_index[start:end], self.snp_index],
                                         num_threads=self.threads)
            if return_slice:
                yield (start, end), chunk
            else:
                yield chunk

    def _iter_col_chunks(self, chunk_size='auto', return_slice=False):
        """
        Iterate over the genotype matrix by columns.

        :param chunk_size: The size of the chunk to read from the genotype matrix.
        :param return_slice: If True, return the slice of the genotype matrix corresponding to the chunk.

        :return: A generator that yields chunks of the genotype matrix.
        """

        if chunk_size == 'auto':
            matrix_size = self.estimate_memory_allocation()
            # By default, we allocate 128MB per chunk:
            chunk_size = int(self.m // (matrix_size // 128))

        for i in range(int(np.ceil(self.m / chunk_size))):
            start, end = int(i * chunk_size), min(int((i + 1) * chunk_size), self.m)
            chunk = self.bed_reader.read(np.s_[self.sample_index, self.snp_index[start:end]],
                                         num_threads=self.threads)
            if return_slice:
                yield (start, end), chunk
            else:
                yield chunk


class plinkBEDGenotypeMatrix(GenotypeMatrix):
    """
    A class that defines methods and interfaces for interacting with genotype matrices
    using `plink2` software. This class provides a convenient interface to perform various
    computations on genotype matrices stored in the plink BED format.

    This class inherits all the attributes of the `GenotypeMatrix` class.
    """

    def __init__(self,
                 sample_table=None,
                 snp_table=None,
                 temp_dir='temp',
                 bed_file=None,
                 genome_build=None,
                 threads=1):
        """
        Initialize a `plinkBEDGenotypeMatrix` object.

        :param sample_table: A table containing information about the samples in the genotype matrix.
        :param snp_table: A table containing information about the genetic variants in the genotype matrix.
        :param temp_dir: The directory where temporary files will be stored (if needed).
        :param bed_file: The path to the plink BED file containing the genotype matrix.
        :param genome_build: The genome build or assembly under which the SNP coordinates are defined.
        :param threads: The number of threads to use for parallel computations.
        """

        super().__init__(sample_table=sample_table,
                         snp_table=snp_table,
                         temp_dir=temp_dir,
                         bed_file=bed_file,
                         genome_build=genome_build,
                         threads=threads)

        from .parsers.plink_parsers import parse_fam_file, parse_bim_file

        if self.bed_file is not None:
            self.bed_file = self.bed_file.replace('.bed', '')

        if self.sample_table is None and self.bed_file:
            self.sample_table = SampleTable(parse_fam_file(self.bed_file))

        if self.snp_table is None and self.bed_file:
            self.snp_table = parse_bim_file(self.bed_file)
            self.snp_table['original_index'] = np.arange(len(self.snp_table))

    @classmethod
    def from_file(cls, file_path, temp_dir='temp', **kwargs):
        """
        A convenience method to create a `plinkBEDGenotypeMatrix` object by
         providing a path to a PLINK BED file.

        :param file_path: The path to the plink BED file.
        :param temp_dir: The directory where temporary files will be stored.
        :param kwargs: Additional keyword arguments.
        """

        p_gt = cls(bed_file=file_path, temp_dir=temp_dir, **kwargs)

        return p_gt

    def score(self, beta, standardize_genotype=False):
        """
        Perform linear scoring on the genotype matrix. This function takes a vector (or matrix) of
        effect sizes and returns the matrix-vector or matrix-matrix product of the genotype matrix
        multiplied by the effect sizes.

        This can be used for polygenic score calculation or projecting the genotype matrix.

        :param beta: A vector or matrix of effect sizes for each variant in the genotype matrix.
        :param standardize_genotype: If True, standardize the genotype when computing the polygenic score.

        :return: The polygenic score (PGS) for each sample in the genotype matrix.
        """

        from .stats.score.utils import score_plink2

        # Create a temporary directory where we store intermediate results:
        tmp_score_dir = tempfile.TemporaryDirectory(dir=self.temp_dir,
                                                    prefix=self.temp_dir_prefix + 'score_')

        plink_score = score_plink2(self,
                                   beta,
                                   standardize_genotype=standardize_genotype,
                                   temp_dir=tmp_score_dir.name)

        tmp_score_dir.cleanup()

        return plink_score

    def perform_gwas(self, **gwa_kwargs):
        """
        Perform genome-wide association testing of all variants against the phenotype.
        This method calls specialized functions that, in turn, call `plink2` to perform
        the association testing.

        :return: A Summary statistics table containing the results of the association testing.
        """

        from .stats.gwa.utils import perform_gwa_plink2

        # Create a temporary directory where we store intermediate results:
        tmp_gwas_dir = tempfile.TemporaryDirectory(dir=self.temp_dir,
                                                   prefix=self.temp_dir_prefix + 'gwas_')

        plink_gwa = perform_gwa_plink2(self, temp_dir=tmp_gwas_dir.name, **gwa_kwargs)

        tmp_gwas_dir.cleanup()

        return plink_gwa

    def compute_allele_frequency(self):
        """
        Compute the allele frequency of each variant or SNP in the genotype matrix.
        This method calls specialized functions that, in turn, call `plink2` to compute
        allele frequency.
        """

        from .stats.variant.utils import compute_allele_frequency_plink2

        # Create a temporary directory where we store intermediate results:
        tmp_freq_dir = tempfile.TemporaryDirectory(dir=self.temp_dir,
                                                   prefix=self.temp_dir_prefix + 'freq_')

        self.snp_table['MAF'] = compute_allele_frequency_plink2(self, temp_dir=tmp_freq_dir.name)

        tmp_freq_dir.cleanup()

    def compute_sample_size_per_snp(self):
        """
        Compute the sample size for each variant in the genotype matrix, accounting for
        potential missing values.

        This method calls specialized functions that, in turn, call `plink2` to compute sample
        size per variant.
        """

        from .stats.variant.utils import compute_sample_size_per_snp_plink2

        # Create a temporary directory where we store intermediate results:
        tmp_miss_dir = tempfile.TemporaryDirectory(dir=self.temp_dir,
                                                   prefix=self.temp_dir_prefix + 'miss_')

        self.snp_table['N'] = compute_sample_size_per_snp_plink2(self, temp_dir=tmp_miss_dir.name)

        tmp_miss_dir.cleanup()
