
import warnings
import pandas as pd
import numpy as np
from .utils.compute_utils import intersect_arrays


class SumstatsTable(object):
    """
    A wrapper class for representing the summary statistics obtained from
    Genome-wide Association Studies (GWAS). GWAS software tools publish their
    results in the form of summary statistics, which include the SNP rsIDs,
    the effect/reference alleles tested, the marginal effect sizes (BETA),
    the standard errors (SE), the Z-scores, the p-values, etc.

    This class provides a convenient way to access/manipulate/harmonize these summary statistics
    across various formats. Particularly, given the heterogeneity in summary statistics
    formats, this class provides a common interface to access these statistics
    in a consistent manner. The class also supports computing some derived statistics
    from the summary statistics, such as the pseudo-correlation between the SNP and the
    phenotype, the Chi-squared statistics, etc.

    :ivar table: A pandas DataFrame containing the summary statistics.
    """

    def __init__(self, ss_table: pd.DataFrame):
        """
        Initialize the summary statistics table.

        :param ss_table: A pandas DataFrame containing the summary statistics.

        !!! seealso "See Also"
            * [from_file][magenpy.SumstatsTable.SumstatsTable.from_file]
        """
        self.table: pd.DataFrame = ss_table

        # Check that the table contains some of the required columns (non-exhaustive):

        # Either has SNP or CHR+POS:
        assert 'SNP' in self.table.columns or all([col in self.table.columns for col in ('CHR', 'POS')])
        # Assert that the table has at least one of the alleles:
        assert any([col in self.table.columns for col in ('A1', 'A2')])
        # TODO: Add other assertions?

    @property
    def shape(self):
        """
        :return: The shape of the summary statistics table.
        """
        return self.table.shape

    def __len__(self):
        return len(self.table)

    @property
    def chromosome(self):
        """
        A convenience method to return the chromosome number if there is only
        one chromosome in the summary statistics.
        If multiple chromosomes are present, it returns None.

        :return: The chromosome number if there is only one chromosome in the summary statistics.
        """
        chrom = self.chromosomes
        if chrom is not None and len(chrom) == 1:
            return chrom[0]

    @property
    def chromosomes(self):
        """
        :return: The unique chromosomes in the summary statistics table.
        """
        if 'CHR' in self.table.columns:
            return sorted(self.table['CHR'].unique())

    @property
    def m(self):
        """
        !!! seealso "See Also"
            * [n_snps][magenpy.SumstatsTable.SumstatsTable.n_snps]

        :return: The number of variants in the summary statistics table.
        """
        return self.n_snps

    @property
    def identifier_cols(self):
        if 'SNP' in self.table.columns:
            return ['SNP']
        else:
            return ['CHR', 'POS']

    @property
    def n_snps(self):
        """
        !!! seealso "See Also"
            * [m][magenpy.SumstatsTable.SumstatsTable.m]

        :return: The number of variants in the summary statistics table.
        """
        return len(self.table)

    @property
    def snps(self):
        """
        :return: The rsIDs associated with each variant in the summary statistics table.
        """
        return self.table['SNP'].values

    @property
    def a1(self):
        """
        !!! seealso "See Also"
            * [effect_allele][magenpy.SumstatsTable.SumstatsTable.effect_allele]
            * [alt_allele][magenpy.SumstatsTable.SumstatsTable.alt_allele]

        :return: The alternative or effect allele for each variant in the summary statistics table.

        """
        return self.table['A1'].values

    @property
    def a2(self):
        """
        !!! seealso "See Also"
            * [ref_allele][magenpy.SumstatsTable.SumstatsTable.ref_allele]

        :return: The reference allele for each variant in the summary statistics table.
        """
        return self.get_col('A2')

    @property
    def ref_allele(self):
        """
        !!! seealso "See Also"
            * [a2][magenpy.SumstatsTable.SumstatsTable.a2]

        :return: The reference allele for each variant in the summary statistics table.
        """
        return self.a2

    @property
    def alt_allele(self):
        """
        !!! seealso "See Also"
            * [effect_allele][magenpy.SumstatsTable.SumstatsTable.effect_allele]
            * [a1][magenpy.SumstatsTable.SumstatsTable.a1]

        :return: The alternative or effect allele for each variant in the summary statistics table.
        """
        return self.a1

    @property
    def effect_allele(self):
        """
        !!! seealso "See Also"
            * [alt_allele][magenpy.SumstatsTable.SumstatsTable.alt_allele]
            * [a1][magenpy.SumstatsTable.SumstatsTable.a1]

        :return: The alternative or effect allele for each variant in the summary statistics table.
        """
        return self.a1

    @property
    def bp_pos(self):
        """
        :return: The base pair position for each variant in the summary statistics table.
        """
        return self.get_col('POS')

    @property
    def maf(self):
        """
        :return: The minor allele frequency for each variant in the summary statistics table.
        """
        return self.get_col('MAF')

    @property
    def maf_var(self):
        """
        :return: The variance of the minor allele frequency for each variant in the summary statistics table.
        """
        return 2.*self.maf*(1. - self.maf)

    @property
    def n(self):
        """
        !!! seealso "See Also"
            * [n_per_snp][magenpy.SumstatsTable.SumstatsTable.n_per_snp]

        :return: The sample size for the association test of each variant in the summary statistics table.
        """
        return self.get_col('N')

    @property
    def n_per_snp(self):
        """
        # TODO: Add a way to infer N from other sumstats if missing.

        !!! seealso "See Also"
            * [n][magenpy.SumstatsTable.SumstatsTable.n]

        :return: The sample size for the association test of each variant in the summary statistics table.
        """
        return self.get_col('N')

    @property
    def beta_hat(self):
        """
        !!! seealso "See Also"
            * [marginal_beta][magenpy.SumstatsTable.SumstatsTable.marginal_beta]

        :return: The marginal beta from the association test of each variant on the phenotype.
        """

        beta = self.get_col('BETA')

        if beta is None:
            odds_ratio = self.odds_ratio
            if odds_ratio is not None:
                self.table['BETA'] = np.log(odds_ratio)
                return self.table['BETA'].values
        else:
            return beta

    @property
    def marginal_beta(self):
        """
        !!! seealso "See Also"
            * [beta_hat][magenpy.SumstatsTable.SumstatsTable.beta_hat]

        :return: The marginal beta from the association test of each variant on the phenotype.
        """
        return self.beta_hat

    @property
    def odds_ratio(self):
        """
        :return: The odds ratio from the association test of each variant on case-control phenotypes.
        """
        return self.get_col('OR')

    @property
    def standardized_marginal_beta(self):
        """
        Get the marginal BETAs assuming that both the genotype matrix
        and the phenotype vector are standardized column-wise to have mean zero and variance 1.
        In some contexts, this is also known as the per-SNP correlation or
        pseudo-correlation with the phenotype.

        !!! seealso "See Also"
            * [get_snp_pseudo_corr][magenpy.SumstatsTable.SumstatsTable.get_snp_pseudo_corr]

        :return: The standardized marginal beta from the association test of each variant on the phenotype.
        """
        return self.get_snp_pseudo_corr()

    @property
    def z_score(self):
        """
        :return: The Z-score from the association test of each SNP on the phenotype.
        :raises KeyError: If the Z-score statistic is not available and could not be inferred from available data.
        """

        z = self.get_col('Z')
        if z is not None:
            return z
        else:
            beta = self.beta_hat
            se = self.se

            if beta is not None and se is not None:
                self.table['Z'] = beta / se
                return self.table['Z'].values

        raise KeyError("Z-score statistic is not available and could not be inferred from available data!")

    @property
    def standard_error(self):
        """
        !!! seealso "See Also"
            * [se][magenpy.SumstatsTable.SumstatsTable.se]

        :return: The standard error from the association test of each variant on the phenotype.

        """
        return self.get_col('SE')

    @property
    def se(self):
        """
        !!! seealso "See Also"
            * [standard_error][magenpy.SumstatsTable.SumstatsTable.standard_error]

        :return: The standard error from the association test of each variant on the phenotype.
        """
        return self.standard_error

    @property
    def pval(self):
        """
        !!! seealso "See Also"
            * [p_value][magenpy.SumstatsTable.SumstatsTable.p_value]

        :return: The p-value from the association test of each variant on the phenotype.
        """
        p = self.get_col('PVAL')

        if p is not None:
            return p
        else:
            from scipy import stats
            self.table['PVAL'] = 2.*stats.norm.sf(np.abs(self.z_score))
            return self.table['PVAL'].values

    @property
    def p_value(self):
        """
        !!! seealso "See Also"
            * [pval][magenpy.SumstatsTable.SumstatsTable.pval]

        :return: The p-value from the association test of each variant on the phenotype.
        """
        return self.pval

    @property
    def negative_log10_p_value(self):
        """
        :return: The negative log10 of the p-value (-log10(p_value)) of association
        test of each variant on the phenotype.
        """
        return -np.log10(self.pval)

    @property
    def effect_sign(self):
        """
        :return: The sign for the effect size (1 for positive effect, -1 for negative effect)
        of each genetic variant ib the phenotype.

        :raises KeyError: If the sign could not be inferred from available data.
        """

        signed_statistics = ['BETA', 'Z', 'OR']

        for ss in signed_statistics:
            ss_value = self.get_col(ss)
            if ss_value is not None:
                if ss == 'OR':
                    return np.sign(np.log(ss_value))
                else:
                    return np.sign(ss_value)

        raise KeyError("No signed statistic to extract the sign from!")

    def infer_a2(self, reference_table, allow_na=False):
        """
        Infer the reference allele A2 (if not present in the SumstatsTable)
        from a reference table. Make sure that the reference table contains the identifier information
        for each SNP, in addition to the reference allele A2 and the alternative (i.e. effect) allele A1.
        It is the user's responsibility to make sure that the reference table matches the summary
        statistics in terms of the specification of reference vs. alternative. They have to be consistent
        across the two tables.

        :param reference_table: A pandas table containing the following columns at least:
        SNP identifiers (`SNP` or `CHR` & `POS`) and allele information (`A1` & `A2`).
        :param allow_na: If True, allow the reference allele to be missing from the final result.
        """

        # Get the identifier columns for this table:
        id_cols = self.identifier_cols

        # Sanity checks:
        assert all([col in reference_table.columns for col in id_cols + ['A1', 'A2']])

        # Merge the summary statistics table with the reference table on unique ID:
        merged_table = self.table[id_cols + ['A1']].merge(
            reference_table[id_cols + ['A1', 'A2']],
            how='left',
            on=id_cols
        )
        # If `A1_x` agrees with `A1_y`, then `A2` is indeed the reference allele.
        # Otherwise, they are flipped and `A1_y` should be the reference allele:
        merged_table['A2'] = np.where(merged_table['A1_x'] == merged_table['A1_y'],
                                      merged_table['A2'],
                                      merged_table['A1_y'])

        # Check that the reference allele could be inferred for all SNPs:
        if not allow_na and merged_table['A2'].isna().any():
            raise ValueError("The reference allele could not be inferred for some SNPs!")
        else:
            self.table['A2'] = merged_table['A2']

    def infer_snp_id(self, reference_table, allow_na=False):
        """
        Infer the SNP ID (if not present in the SumstatsTable) from a reference table.
        Make sure that the reference table contains the SNP ID, chromosome ID, and position.

        :param reference_table: A pandas table containing the following columns at least:
        `SNP`, `CHR`, `POS`.
        :param allow_na: If True, allow the SNP ID to be missing from the final result.
        """

        # Merge the summary statistics table with the reference table:
        merged_table = self.table[['CHR', 'POS']].merge(reference_table[['SNP', 'CHR', 'POS']], how='left')

        # Check that the SNP ID could be inferred for all SNPs:
        if not allow_na and merged_table['SNP'].isna().any():
            raise ValueError("The SNP ID could not be inferred for some SNPs!")
        else:
            self.table['SNP'] = merged_table['SNP'].values

    def set_sample_size(self, n):
        """
        Set the sample size for each variant in the summary table.
        This can be useful when the overall sample size from the GWAS analysis is available,
        but not on a per-SNP basis.

        :param n: A scalar or array of sample sizes for each variant.
        """
        self.table['N'] = n

    def run_quality_control(self, reference_table=None):
        """
        Run quality control checks on the summary statistics table.
        TODO: Implement quality control checks following recommendations given by Prive et al.:
        https://doi.org/10.1016/j.xhgg.2022.100136
        Given user fine-control over which checks to run and which to skip.
        Maybe move parts of this implementation to a module in `stats` (TBD)
        """
        pass

    def match(self, reference_table, correct_flips=True):
        """
        Match the summary statistics table with a reference table,
        correcting for potential flips in the effect alleles.

        :param reference_table: The SNP table to use as a reference. Must be a pandas
        table with the following columns: SNP identifier (either `SNP` or `CHR` & `POS`) and allele information
        (`A1` & `A2`).
        :param correct_flips: If True, correct the direction of effect size
         estimates if the effect allele is reversed.
        """

        from .utils.model_utils import merge_snp_tables

        self.table = merge_snp_tables(ref_table=reference_table[self.identifier_cols + ['A1', 'A2']],
                                      alt_table=self.table,
                                      how='inner',
                                      correct_flips=correct_flips)

    def filter_by_allele_frequency(self, min_maf=None, min_mac=None):
        """
        Filter variants in the summary statistics table by minimum minor allele frequency or allele count
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
        Filter the summary statistics table to keep a subset of SNPs.
        :param extract_snps: A list or array of SNP IDs to keep.
        :param extract_file: A plink-style file containing the SNP IDs to keep.
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
                            "the list of SNPs, a file containing the list of SNPs, "
                            "or a list of indices to retain.")

    def drop_duplicates(self):
        """
        Drop variants with duplicated rsIDs from the summary statistics table.
        """

        self.table = self.table.drop_duplicates(subset=self.identifier_cols, keep=False)

    def get_col(self, col_name):
        """
        :param col_name: The name of the column to extract.

        :return: The column associated with `col_name` from summary statistics table.
        """
        if col_name in self.table.columns:
            return self.table[col_name].values

    def get_chisq_statistic(self):
        """
        :return: The Chi-Squared statistic from the association test of each variant on the phenotype.
        :raises KeyError: If the Chi-Squared statistic is not available and could not be inferred from available data.
        """
        chisq = self.get_col('CHISQ')

        if chisq is not None:
            return chisq
        else:
            z = self.z_score
            if z is not None:
                self.table['CHISQ'] = z**2
            else:
                p_val = self.p_value
                if p_val is not None:
                    from scipy.stats import chi2

                    self.table['CHISQ'] = chi2.ppf(1. - p_val, 1)
                else:
                    raise KeyError("Chi-Squared statistic is not available!")

        return self.table['CHISQ'].values

    def get_snp_pseudo_corr(self):
        """

        Computes the pseudo-correlation coefficient (standardized beta) between the SNP and
        the phenotype (X_jTy / N) from GWAS summary statistics.

        This method uses Equation 15 in Mak et al. 2017

            $$
            beta =  z_j / sqrt(n - 1 + z_j^2)
            $$

        Where `z_j` is the marginal GWAS Z-score.

        !!! seealso "See Also"
            * [standardized_marginal_beta][magenpy.SumstatsTable.SumstatsTable.standardized_marginal_beta]

        :return: The pseudo-correlation coefficient between the SNP and the phenotype.
        :raises KeyError: If the Z-scores are not available or the sample size is not available.

        """

        zsc = self.z_score
        n = self.n

        if zsc is not None:
            if n is not None:
                return zsc / (np.sqrt(n - 1 + zsc**2))
            else:
                raise KeyError("Sample size is not available!")
        else:
            raise KeyError("Z-scores are not available!")
        
    def get_yy_per_snp(self):
        """
        Computes the quantity (y'y)_j/n_j following SBayesR (Lloyd-Jones 2019) and Yang et al. (2012).

        (y'y)_j/n_j is defined as the empirical variance for continuous phenotypes and may be estimated
        from GWAS summary statistics by re-arranging the equation for the
        squared standard error:

            $$
            SE(b_j)^2 = (Var(y) - Var(x_j)*b_j^2) / (Var(x)*n)
            $$

        Which gives the following estimate:

            $$
            (y'y)_j / n_j = (n_j - 2)*SE(b_j)^2 + b_j^2
            $$

        :return: The quantity (y'y)_j/n_j for each SNP in the summary statistics table.
        :raises KeyError: If the marginal betas, standard errors or sample sizes are not available.

        """
        
        b = self.beta_hat
        se = self.standard_error
        n = self.n

        if n is not None:
            if b is not None:
                if se is not None:
                    return (n - 2)*se**2 + b**2
                else:
                    raise KeyError("Standard errors are not available!")
            else:
                raise KeyError("Marginal betas are not available!")
        else:
            raise KeyError("Sample size per SNP is not available!")

    def split_by_chromosome(self, snps_per_chrom=None):
        """
        Split the summary statistics table by chromosome, so that we would
        have a separate `SumstatsTable` object for each chromosome.
        :param snps_per_chrom: A dictionary where the keys are the chromosome number 
        and the value is an array or list of SNPs on that chromosome.

        :return: A dictionary where the keys are the chromosome number and the value is a `SumstatsTable` object.
        """

        if 'CHR' in self.table.columns:
            chrom_tables = self.table.groupby('CHR')
            return {
                c: SumstatsTable(chrom_tables.get_group(c).copy())
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

    def to_table(self, col_subset=None):
        """
        A convenience method to extract the summary statistics table or subsets of it.

        :param col_subset: A list corresponding to a subset of columns to return.

        :return: A pandas DataFrame containing the summary statistics with the requested column subset.
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
                table['NLOG10_PVAL'] = self.negative_log10_p_value
            elif col == 'CHISQ':
                table['CHISQ'] = self.get_chisq_statistic()
            elif col == 'MAF_VAR':
                table['MAF_VAR'] = self.maf_var
            elif col == 'STD_BETA':
                table['STD_BETA'] = self.get_snp_pseudo_corr()
            else:
                warnings.warn(f"Column '{col}' is not available in the summary statistics table!")

        return table[list(col_subset)]

    def to_file(self, output_file, col_subset=None, **to_csv_kwargs):
        """
        A convenience method to write the summary statistics table to file.

        TODO: Add a format argument to this method and allow the user to output summary statistics
        according to supported formats (e.g. COJO, plink, fastGWA, etc.).

        :param output_file: The path to the file where to write the summary statistics.
        :param col_subset: A subset of the columns to write to file.
        :param to_csv_kwargs: Keyword arguments to pass to pandas' `to_csv` method.

        """

        if 'sep' not in to_csv_kwargs and 'delimiter' not in to_csv_kwargs:
            to_csv_kwargs['sep'] = '\t'

        if 'index' not in to_csv_kwargs:
            to_csv_kwargs['index'] = False

        table = self.to_table(col_subset)
        table.to_csv(output_file, **to_csv_kwargs)

    @classmethod
    def from_file(cls, sumstats_file, sumstats_format=None, parser=None, **parse_kwargs):
        """
        Initialize a summary statistics table from file. The user must provide either
        the format for the summary statistics file or the parser object
        (see `parsers.sumstats_parsers`).

        :param sumstats_file: The path to the summary statistics file.
        :param sumstats_format: The format for the summary statistics file. Currently,
        we support the following summary statistics formats: `magenpy`, `plink1.9`, `plink` or `plink2`,
        `COJO`, `fastGWA`, `SAIGE`, `GWASCatalog` (also denoted as `GWAS-SSF` and `SSF`).
        :param parser: An instance of SumstatsParser parser, implements basic parsing/conversion
        functionalities.
        :param parse_kwargs: arguments for the pandas `read_csv` function, such as the delimiter.

        :return: A `SumstatsTable` object initialized from the summary statistics file.
        """
        assert sumstats_format is not None or parser is not None

        from .parsers.sumstats_parsers import (
            SumstatsParser, Plink1SSParser, Plink2SSParser, COJOSSParser,
            FastGWASSParser, SSFParser, SaigeSSParser
        )

        sumstats_format_l = sumstats_format.lower()

        if parser is None:
            if sumstats_format_l == 'magenpy':
                parser = SumstatsParser(None, **parse_kwargs)
            elif sumstats_format_l in ('plink', 'plink2'):
                parser = Plink2SSParser(None, **parse_kwargs)
            elif sumstats_format_l == 'plink1.9':
                parser = Plink1SSParser(None, **parse_kwargs)
            elif sumstats_format_l == 'cojo':
                parser = COJOSSParser(None, **parse_kwargs)
            elif sumstats_format_l == 'fastgwa':
                parser = FastGWASSParser(None, **parse_kwargs)
            elif sumstats_format_l in ('ssf', 'gwas-ssf', 'gwascatalog'):
                parser = SSFParser(None, **parse_kwargs)
            elif sumstats_format_l == 'saige':
                parser = SaigeSSParser(None, **parse_kwargs)
            else:
                raise KeyError(f"Parsers for summary statistics format {sumstats_format} are not implemented!")

        sumstats_table = parser.parse(sumstats_file)
        return cls(sumstats_table)
