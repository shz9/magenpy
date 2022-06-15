
from typing import Union
import tempfile
import pandas as pd
import numpy as np
from .parsers.plink_parsers import parse_fam_file, parse_bim_file
from .SampleTable import SampleTable


class GenotypeMatrix(object):

    def __init__(self,
                 sample_table: Union[pd.DataFrame, SampleTable, None] = None,
                 snp_table: Union[pd.DataFrame, None] = None,
                 temp_dir: str = 'temp',
                 **kwargs):

        self.sample_table: Union[pd.DataFrame, SampleTable, None] = None
        self.snp_table: Union[pd.DataFrame, None] = snp_table

        if sample_table is not None:
            self.set_sample_table(sample_table)

        from .utils.system_utils import makedir
        makedir(temp_dir)
        self.temp_dir = temp_dir
        self.cleanup_dir_list = []  # Directories to clean up after execution.

    @classmethod
    def from_file(cls, file_path, temp_dir='temp'):
        """
        Read and parse the genotype matrix information from file.
        """
        raise NotImplementedError

    @property
    def shape(self):
        """
        The shape of the genotype matrix. Rows correspond to the
        number of samples and columns to the number of SNPs.
        """
        return self.n, self.m

    @property
    def n(self):
        """
        The sample size, see also `.sample_size()`
        """
        return self.sample_table.n

    @property
    def sample_size(self):
        """
        The sample size of the genotype matrix. See also `.n()`.
        """
        return self.n

    @property
    def samples(self):
        """
        Obtain a vector of sample IDs.
        """
        return self.sample_table.iid

    @property
    def m(self):
        """
        The number of SNPs, see also `n_snps`
        """
        if self.snp_table is not None:
            return len(self.snp_table)

    @property
    def n_snps(self):
        return self.m

    @property
    def chromosome(self):
        """
        If the genotype matrix is comprised of a single chromosome, return the chromosome number.
        """
        chrom = self.chromosomes
        if chrom is not None:
            if len(chrom) == 1:
                return chrom[0]

    @property
    def chromosomes(self):
        """
        Return the unique set of chromosomes comprising the genotype matrix.
        """
        chrom = self.get_snp_attribute('CHR')
        if chrom is not None:
            return np.unique(chrom)

    @property
    def snps(self):
        """
        Return the SNP IDs.
        """
        return self.get_snp_attribute('SNP')

    @property
    def bp_pos(self):
        """
        The position for the genetic variants in base pairs.
        """
        return self.get_snp_attribute('POS')

    @property
    def cm_pos(self):
        """
        The position for the genetic variants in Centi Morgan.
        """
        cm = self.get_snp_attribute('cM')
        if len(set(cm)) == 1:
            raise Exception("Genetic distance in centi Morgan (cM) is not "
                            "set in the genotype file!")
        return cm

    @property
    def a1(self):
        """
        Return the effect allele `A1`. See also `.alt_allele()`, `.effect_allele()`.
        """
        return self.get_snp_attribute('A1')

    @property
    def a2(self):
        """
        Return the reference allele `A2`. See also `.ref_allele()`.
        """
        return self.get_snp_attribute('A2')

    @property
    def ref_allele(self):
        """
        Return the reference allele `A2`. See also `.a2()`.
        """
        return self.a2

    @property
    def alt_allele(self):
        """
        Return the alternative (i.e. effect) allele `A1`. See also `.a1()`, `.effect_allele()`.
        """
        return self.a1

    @property
    def effect_allele(self):
        """
        Return the effect allele `A1`. See also `.a1()`, `.alt_allele()`.
        """
        return self.a1

    @property
    def n_per_snp(self):
        """
        Sample size per genetic variant (this accounts for missing values).
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
        Minor allele frequency
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
        The variance in minor allele frequency.
        """
        return 2. * self.maf * (1. - self.maf)

    def get_snp_table(self, col_subset=None):
        """
        Return the SNP table or a subset of its columns.
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

            return table[col_subset]

    def get_snp_attribute(self, attr):
        """
        A utility function to extract a given column from the SNP table.
        """
        if self.snp_table is not None:
            if attr in self.snp_table.columns:
                return self.snp_table[attr].values

    def compute_ld(self, estimator, output_dir, **ld_kwargs):
        """
        Compute the Linkage-Disequilibrium (LD) or SNP-by-SNP correlation matrix
        for the genotype matrix.

        :param estimator: The estimator for the LD matrix. We currently support
        4 different estimators: `sample`, `windowed`, `shrinkage`, and `block`.
        :param output_dir: The output directory where the Zarr array containing the
        entries of the LD matrix will be stored.
        :param ld_kwargs: keyword arguments for the various LD estimators. Consult
        the implementations of `WindowedLD`, `ShrinkageLD`, and `BlockLD` for details.
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

        # Create a temporary directory where we store intermediate results:
        tmp_ld_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='ld_')
        self.cleanup_dir_list.append(tmp_ld_dir)

        return ld_est.compute(output_dir, temp_dir=tmp_ld_dir.name)

    def set_sample_table(self, sample_table):
        """
        A convenience method set the sample table for genotype matrix.
        This may be useful for syncing sample tables across different Genotype matrices
        corresponding to different chromosomes or genomic regions.
        """

        if isinstance(sample_table, SampleTable):
            self.sample_table = sample_table
        elif isinstance(sample_table, pd.DataFrame):
            self.sample_table = SampleTable(sample_table)
        else:
            raise Exception("The sample table is invalid!")

    def filter_snps(self, extract_snps=None, extract_file=None):
        """
        Filter variants from the genotype matrix. User must specify
        either a list of variants to extract or the path to a file
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
        Filter variants by minimum minor allele frequency or allele count
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
        Drop variants with duplicated SNP IDs.
        """

        u_snps, counts = np.unique(self.snps, return_counts=True)
        if len(u_snps) < self.n_snps:
            # Keep only SNPs which occur once in the sequence:
            self.filter_snps(u_snps[counts == 1])

    def filter_samples(self, keep_samples=None, keep_file=None):
        """
        Filter samples from the genotype matrix. User must specify
        either a list of samples to keep or the path to a file
        with the list of samples to keep.

        :param keep_samples: A list (or array) of sample IDs to keep in the genotype matrix.
        :param keep_file: The path to a file with the list of samples to keep.
        """

        self.sample_table.filter_samples(keep_samples=keep_samples, keep_file=keep_file)

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

        :param gwa_kwargs: Keyword arguments to pass to the GWA functions. Consult stats.gwa.utils
        for relevant keyword arguments for each backend.
        """
        raise NotImplementedError

    def compute_allele_frequency(self):
        """
        Compute the allele frequency of each variant or SNP in the genotype matrix.
        """
        raise NotImplementedError

    def compute_sample_size_per_snp(self):
        """
        Compute the sample size for each variant in the genotype matrix, accounting for
        potential missing values.
        """
        raise NotImplementedError

    def split_by_chromosome(self):
        """
        Split the genotype matrix by chromosome, so that we would
        have a separate `GenotypeMatrix` objects for each chromosome.
        This method returns a dictionary where the key is the chromosome number
        and the value is an object of `GenotypeMatrix` for that chromosome.
        """

        chromosome = self.chromosome

        if chromosome:
            return {chromosome: self}
        else:
            chrom_tables = self.snp_table.groupby('CHR')
            return {
                c: self.__class__(sample_table=self.sample_table,
                                  snp_table=chrom_tables.get_group(c),
                                  temp_dir=self.temp_dir)
                for c in chrom_tables.groups
            }

    def cleanup(self):
        """
        Clean up all temporary files and directories
        """

        for tmpdir in self.cleanup_dir_list:
            try:
                tmpdir.cleanup()
            except FileNotFoundError:
                continue


class xarrayGenotypeMatrix(GenotypeMatrix):

    def __init__(self, sample_table=None, snp_table=None, temp_dir='temp', xr_mat=None):
        super().__init__(sample_table=sample_table, snp_table=snp_table, temp_dir=temp_dir)

        # xarray matrix object, as defined by pandas-plink:
        self.xr_mat = xr_mat

    @classmethod
    def from_file(cls, file_path, temp_dir='temp'):

        from pandas_plink import read_plink1_bin

        try:
            xr_gt = read_plink1_bin(file_path + ".bed", ref="a0", verbose=False)
        except ValueError:
            xr_gt = read_plink1_bin(file_path, ref="a0", verbose=False)
        except Exception as e:
            raise e

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

        # Set the snp table:
        snp_table = xr_gt.variant.coords.to_dataset().to_dataframe()
        snp_table.columns = ['CHR', 'SNP', 'cM', 'POS', 'A1', 'A2']
        snp_table.reset_index(inplace=True, drop=True)
        snp_table = snp_table.astype({
            'CHR': int,
            'SNP': str,
            'cM': float,
            'POS': np.int,
            'A1': str,
            'A2': str
        })

        # Set the index to be the SNP ID:
        xr_gt = xr_gt.set_index(variant='snp')

        g_mat = cls(sample_table=SampleTable(sample_table),
                    snp_table=snp_table,
                    temp_dir=temp_dir,
                    xr_mat=xr_gt)

        return g_mat

    def set_sample_table(self, sample_table):

        try:
            if len(sample_table) != self.n:
                update_mat = True
            else:
                update_mat = False
        except Exception:
            update_mat = False

        super(xarrayGenotypeMatrix, self).set_sample_table(sample_table)

        if update_mat:
            self.xr_mat = self.xr_mat.sel(sample=self.samples)

    def filter_snps(self, extract_snps=None, extract_file=None):

        super(xarrayGenotypeMatrix, self).filter_snps(extract_snps=extract_snps, extract_file=extract_file)
        self.xr_mat = self.xr_mat.sel(variant=self.snps)

    def filter_samples(self, keep_samples=None, keep_file=None):

        super(xarrayGenotypeMatrix, self).filter_samples(keep_samples=keep_samples, keep_file=keep_file)
        self.xr_mat = self.xr_mat.sel(sample=self.samples)

    def score(self, beta, standardize_genotype=False):

        if standardize_genotype:
            from .stats.transforms.genotype import standardize
            pgs = np.dot(standardize(self.xr_mat), beta)
        else:
            pgs = np.dot(self.xr_mat.fillna(self.maf), beta)

        return pgs

    def perform_gwas(self, **gwa_kwargs):

        from magenpy.stats.gwa.utils import perform_gwa_xarray
        return perform_gwa_xarray(self, **gwa_kwargs)

    def compute_allele_frequency(self):
        self.snp_table['MAF'] = (self.xr_mat.sum(axis=0) / (2. * self.n_per_snp)).compute().values

    def compute_sample_size_per_snp(self):
        self.snp_table['N'] = self.xr_mat.shape[0] - self.xr_mat.isnull().sum(axis=0).compute().values

    def split_by_chromosome(self):
        split = super(xarrayGenotypeMatrix, self).split_by_chromosome()

        for c, gt in split.items():
            if len(split) > 1:
                gt.xr_mat = gt.xr_mat.sel(variant=gt.snps)
            else:
                gt.xr_mat = self.xr_mat

        return split


class plinkBEDGenotypeMatrix(GenotypeMatrix):

    def __init__(self, sample_table=None, snp_table=None, temp_dir='temp', bed_file=None):
        super().__init__(sample_table=sample_table, snp_table=snp_table, temp_dir=temp_dir)

        self.bed_file = bed_file
        if self.bed_file is not None:
            self.bed_file = self.bed_file.replace('.bed', '')

        if self.sample_table is None and self.bed_file:
            self.sample_table = SampleTable(parse_fam_file(self.bed_file))

        if self.snp_table is None and self.bed_file:
            self.snp_table = parse_bim_file(self.bed_file)

    @classmethod
    def from_file(cls, file_path, temp_dir='temp'):

        p_gt = cls(bed_file=file_path, temp_dir=temp_dir)

        return p_gt

    def score(self, beta, standardize_genotype=False):

        from .stats.score.utils import score_plink2

        # Create a temporary directory where we store intermediate results:
        tmp_score_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='score_')
        self.cleanup_dir_list.append(tmp_score_dir)

        return score_plink2(self, beta, standardize_genotype=standardize_genotype, temp_dir=tmp_score_dir.name)

    def perform_gwas(self, **gwa_kwargs):

        from magenpy.stats.gwa.utils import perform_gwa_plink2

        # Create a temporary directory where we store intermediate results:
        tmp_gwas_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='gwas_')
        self.cleanup_dir_list.append(tmp_gwas_dir)

        return perform_gwa_plink2(self, temp_dir=tmp_gwas_dir.name, **gwa_kwargs)

    def compute_allele_frequency(self):

        from magenpy.stats.variant.utils import compute_allele_frequency_plink2

        # Create a temporary directory where we store intermediate results:
        tmp_freq_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='freq_')
        self.cleanup_dir_list.append(tmp_freq_dir)

        self.snp_table['MAF'] = compute_allele_frequency_plink2(self, temp_dir=tmp_freq_dir.name)

    def compute_sample_size_per_snp(self):
        from magenpy.stats.variant.utils import compute_sample_size_per_snp_plink2

        # Create a temporary directory where we store intermediate results:
        tmp_miss_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='miss_')
        self.cleanup_dir_list.append(tmp_miss_dir)

        self.snp_table['N'] = compute_sample_size_per_snp_plink2(self, temp_dir=tmp_miss_dir.name)

    def split_by_chromosome(self):

        split = super(plinkBEDGenotypeMatrix, self).split_by_chromosome()

        for c, gt in split.items():
            gt.bed_file = self.bed_file

        return split
