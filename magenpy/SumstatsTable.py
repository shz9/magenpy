
import pandas as pd
import warnings
import numpy as np
from magenpy.utils.compute_utils import intersect_arrays
from .parsers.sumstats_parsers import *


class SumstatsTable(object):

    def __init__(self, ss_table: pd.DataFrame):
        self.table: pd.DataFrame = ss_table

        assert all([col in self.table.columns for col in ('SNP', 'A1')])

    @property
    def shape(self):
        return self.table.shape

    @property
    def chromosome(self):
        chrom = self.chromosomes
        if chrom is not None:
            if len(chrom) == 1:
                return chrom[0]

    @property
    def chromosomes(self):
        if 'CHR' in self.table.columns:
            return self.table['CHR'].unique()

    @property
    def m(self):
        return self.n_snps

    @property
    def n_snps(self):
        return len(self.table)

    @property
    def snps(self):
        return self.table['SNP'].values

    @property
    def a1(self):
        return self.table['A1'].values

    @property
    def a2(self):
        return self.get_col('A2')

    @property
    def ref_allele(self):
        return self.a2

    @property
    def alt_allele(self):
        return self.a1

    @property
    def effect_allele(self):
        return self.a1

    @property
    def bp_pos(self):
        return self.get_col('POS')

    @property
    def maf(self):
        return self.get_col('MAF')

    @property
    def maf_var(self):
        return 2.*self.maf*(1. - self.maf)

    @property
    def n(self):
        return self.get_col('N')

    @property
    def n_per_snp(self):
        return self.get_col('N')

    @property
    def beta_hat(self):
        return self.get_col('BETA')

    @property
    def marginal_beta(self):
        return self.beta_hat

    @property
    def z_score(self):
        z = self.get_col('Z')
        if z is not None:
            return z
        else:
            beta = self.beta_hat
            se = self.se

            if beta is not None and se is not None:
                self.table['Z'] = beta / se
                return self.table['Z'].values
            else:
                raise Exception("Z-score statistic is not available!")

    @property
    def standard_error(self):
        return self.get_col('SE')

    @property
    def se(self):
        return self.standard_error

    @property
    def pval(self):
        p = self.get_col('PVAL')

        if p is not None:
            return p
        else:
            from scipy import stats
            self.table['PVAL'] = 2.*stats.norm.sf(np.abs(self.z_score))
            return self.table['PVAL'].values

    @property
    def p_value(self):
        return self.pval

    @property
    def log10_p_value(self):
        """
        Computes -log10(p_value).
        May be useful for Manhattan plots.
        """
        return -np.log10(self.pval)

    def match(self, reference_table, correct_flips=True):
        """
        Match the summary statistics table with a reference table,
        correcting for potential flips in the effect allele.

        :param reference_table: The SNP table to use as a reference. Must be a pandas
        table with at least two columns: SNP and A1 (the effect allele).
        :param correct_flips: If True, correct the direction of effect size
         estimates if the effect allele is reversed.
        """

        from magenpy.utils.model_utils import merge_snp_tables

        self.table = merge_snp_tables(ref_table=reference_table[['SNP', 'A1']],
                                      alt_table=self.table,
                                      how='inner',
                                      correct_flips=correct_flips)

    def filter_by_allele_frequency(self, min_maf=None, min_mac=None):
        """
        Filter variants by minimum minor allele frequency or allele count
        :param min_maf: Minimum minor allele frequency
        :param min_mac: Minimum minor allele count
        """

        if min_mac or min_maf:
            maf = self.maf
            n = self.n_per_snp
        else:
            return

        keep_flag = None

        if min_mac and n and maf:
            mac = (2*maf*n).astype(np.int64)
            keep_flag = (mac >= min_mac) & ((2*n - mac) >= min_mac)

        if min_maf and maf:

            maf_cond = (maf >= min_maf) & (1. - maf >= min_maf)
            if keep_flag is not None:
                keep_flag = keep_flag & maf_cond
            else:
                keep_flag = maf_cond

        if keep_flag is not None:
            self.filter_snps(extract_index=np.where(keep_flag)[0])

    def filter_snps(self, extract_snps=None, extract_file=None, extract_index=None):
        """
        Filter the summary statistics table to a subset of SNPs.
        :param extract_snps: A list or array of SNP IDs to keep.
        :param extract_file: A file containing the SNP IDs to keep.
        :param extract_index: A list or array of the indices of SNPs to retain.
        """

        assert extract_snps is not None or extract_file is not None or extract_index is not None

        if extract_file:
            from .parsers.misc_parsers import read_snp_filter_file
            extract_snps = read_snp_filter_file(extract_file)

        if extract_snps is not None:
            extract_index = intersect_arrays(self.snps, extract_snps, return_index=True)

        if extract_index is not None:
            self.table = self.table.iloc[extract_index, ].reset_index(drop=True)
        else:
            raise Exception("To filter a summary statistics table, you must provide "
                            "the list of SNPs, a file containing the list of SNPs, or a list of indices to retain.")

    def drop_duplicates(self):
        """
        Drop duplicated SNP rsIDs.
        """

        self.table = self.table.drop_duplicates(subset='SNP', keep=False)

    def get_col(self, col_name):
        """
        Returns a particular summary statistic or column from the summary statistics 
        table.
        :param col_name: The name of the column
        """
        if col_name in self.table.columns:
            return self.table[col_name].values

    def get_chisq_statistic(self):
        """
        Obtain the Chi-Squared statistic
        """
        chisq = self.get_col('CHISQ')

        if chisq is not None:
            return chisq
        else:
            z = self.z_score
            if z is not None:
                self.table['CHISQ'] = z**2
                return self.table['CHISQ'].values
            else:
                raise ValueError("Chi-Squared statistic is not available!")

    def get_snp_pseudo_corr(self):
        """
        Computes the pseudo-correlation coefficient (standardized beta) between the SNP and
        the phenotype (X_jTy / N) from GWAS summary statistics.
        Uses Equation 15 in Mak et al. 2017
        beta =  z_j / sqrt(n - 1 + z_j^2)
        Where z_j is the marginal GWAS Z-score
        """

        zsc = self.z_score
        n = self.n

        if zsc is not None:
            if n is not None:
                return zsc / (np.sqrt(n - 1 + zsc**2))
            else:
                raise Exception("Sample size is not available!")
        else:
            raise Exception("Z-scores are not available!")
        
    def get_yy_per_snp(self):
        """
        Computes the quantity (y'y)_j/n_j following SBayesR (Lloyd-Jones 2019) and Yang et al. (2012).
        (y'y)_j/n_j is the empirical variance for continuous phenotypes and may be estimated
        from GWAS summary statistics by re-arranging the equation for the
        squared standard error:

        SE(b_j)^2 = (Var(y) - Var(x_j)*b_j^2) / (Var(x)*n)

        Which gives the following estimate:

        (y'y)_j / n_j = (n_j - 2)*SE(b_j)^2 + b_j^2

        """
        
        b = self.beta_hat
        se = self.standard_error
        n = self.n

        if n is not None:
            if b is not None:
                if se is not None:
                    return (n - 2)*se**2 + b**2
                else:
                    raise Exception("Standard errors are not available!")
            else:
                raise Exception("Marginal betas are not available!")
        else:
            raise Exception("Sample size per SNP is not available!")

    def split_by_chromosome(self, snps_per_chrom=None):
        """
        Split the summary statistics table by chromosome, so that we would
        have a separate `SumstatsTable` table for each chromosome.
        :param snps_per_chrom: A dictionary where the keys are the chromosome number 
        and the value is an array or list of SNPs on that chromosome.
        """

        if 'CHR' in self.table.columns:
            chrom_tables = self.table.groupby('CHR')
            return {
                c: SumstatsTable(chrom_tables.get_group(c))
                for c in chrom_tables.groups
            }
        elif snps_per_chrom is not None:
            chrom_dict = {
                c: SumstatsTable(pd.DataFrame({'SNP': snps}).merge(self.table))
                for c, snps in snps_per_chrom.items()
            }

            for c, ss_tab in chrom_dict.items():
                ss_tab.table['CHR'] = c

            return chrom_dict
        else:
            raise Exception("To split the summary statistics table by chromosome, "
                            "you must provide the a dictionary mapping chromosome number "
                            "to an array of SNPs `snps_per_chrom`.")

    def get_table(self, col_subset=None):
        """
        Get the summary statistics table or a subset of it.
        :param col_subset: A list corresponding to a subset of columns to return.
        """

        col_subset = col_subset or ['CHR', 'SNP', 'POS', 'A1', 'A2', 'MAF',
                                    'N', 'BETA', 'Z', 'SE', 'PVAL']

        # Because some of the quantities that the user needs may be need to be
        # computed, we separate the column subset into those that are already
        # present in the table and those that are not (but can still be computed
        # from other summary statistics):

        present_cols = list(set(col_subset).intersection(set(self.table.columns)))
        non_present_cols = list(set(col_subset) - set(present_cols))

        if len(present_cols) > 0:
            table = self.table[present_cols].copy()
        else:
            table = pd.DataFrame({c: [] for c in non_present_cols})

        for col in non_present_cols:

            if col == 'Z':
                table['Z'] = self.z_score
            elif col == 'PVAL':
                table['PVAL'] = self.p_value
            elif col == 'LOG10_PVAL':
                table['LOG10_PVAL'] = self.log10_p_value
            elif col == 'CHISQ':
                table['CHISQ'] = self.get_chisq_statistic()
            elif col == 'MAF_VAR':
                table['MAF_VAR'] = self.maf_var
            elif col == 'STD_BETA':
                table['STD_BETA'] = self.get_snp_pseudo_corr()
            else:
                raise warnings.warn(f"Column '{col}' is not available in the summary statistics table!")

        return table[col_subset]

    def to_file(self, output_file, col_subset=None, **to_csv_kwargs):
        """
        Write the summary statistics table to file.
        :param output_file: The path to the file where to write the summary statistics.
        :param col_subset: A subset of the columns to write to file.
        """

        if 'sep' not in to_csv_kwargs and 'delimiter' not in to_csv_kwargs:
            to_csv_kwargs['sep'] = '\t'

        if 'index' not in to_csv_kwargs:
            to_csv_kwargs['index'] = False

        table = self.get_table(col_subset)
        table.to_csv(output_file, **to_csv_kwargs)

    @classmethod
    def from_file(cls, sumstats_file, sumstats_format=None, parser=None, **parse_kwargs):
        """
        Initialize a summary statistics table from file. The user must provide either
        the format for the summary statistics file or the parser object
        (see parsers.sumstats_parsers).
        :param sumstats_file: The path to the summary statistics file.
        :param sumstats_format: The format for the summary statistics file. Currently,
        we explicitly support the following three formats formats: magenpy, plink, COJO.
        :param parser: An instance of SumstatsParser parser, implements basic parsing/conversion
        functionalities.
        :param parse_kwargs: arguments for the pandas `read_csv` function, such as the delimiter.
        """
        assert sumstats_format or parser

        if parser is None:
            if sumstats_format == 'magenpy':
                parser = SumstatsParser()
            elif sumstats_format == 'plink':
                parser = plinkSumstatsParser()
            elif sumstats_format == 'COJO':
                parser = COJOSumstatsParser()
            elif sumstats_format == 'fastGWA':
                parser = fastGWASumstatsParser()
            else:
                raise KeyError(f"Parsers for summary statistics format {sumstats_format} are not implemented!")

        sumstats_table = parser.parse(sumstats_file, **parse_kwargs)
        return cls(sumstats_table)
