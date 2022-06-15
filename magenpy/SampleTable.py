
from typing import Union
import numpy as np
import pandas as pd
from .parsers.plink_parsers import parse_fam_file


class SampleTable(object):

    def __init__(self, table: Union[pd.DataFrame, None] = None, phenotype_likelihood: Union[str, None] = None):

        self.table: Union[pd.DataFrame, None] = table

        assert phenotype_likelihood in (None, 'binomial', 'gaussian')

        self._phenotype_likelihood: Union[str, None] = phenotype_likelihood
        self._covariate_cols = None

        self.post_check_phenotype()

    @property
    def shape(self):
        return (self.n,)

    @property
    def n(self):
        return len(self.table)

    @property
    def sample_size(self):
        return self.n

    @property
    def iid(self):
        if self.table is not None:
            return self.table['IID'].values

    @property
    def fid(self):
        if self.table is not None:
            return self.table['FID'].values

    @property
    def phenotype(self):
        if self.table is not None:
            try:
                return self.table['phenotype'].values
            except KeyError:
                raise KeyError("The phenotype is not set!")

    @property
    def covariates(self):
        return self._covariate_cols

    @property
    def phenotype_likelihood(self):
        return self._phenotype_likelihood

    @classmethod
    def from_fam_file(cls, fam_file):
        s_tab = parse_fam_file(fam_file)
        return cls(table=s_tab)

    @classmethod
    def from_phenotype_file(cls, phenotype_file, filter_na=True, **read_csv_kwargs):
        s_tab = cls()
        s_tab.read_phenotype_file(phenotype_file, filter_na, **read_csv_kwargs)
        return s_tab

    @classmethod
    def from_covariate_file(cls, covar_file, **read_csv_kwargs):
        s_tab = cls()
        s_tab.read_covariates_file(covar_file, **read_csv_kwargs)
        return s_tab

    def read_phenotype_file(self, phenotype_file, drop_na=True, **read_csv_kwargs):
        """
        Read the phenotype file from disk. The expected format is Family ID (`FID`),
        Individual ID (`IID`) and the phenotype column `phenotype`. You may adjust
        the parsing configurations with keyword arguments that will be passed to `pandas.read_csv`.

        :param phenotype_file: The path to the phenotype file.
        :param drop_na: Drop samples whose phenotype value is missing (Default: True).
        :param read_csv_kwargs: keyword arguments to pass to the `read_csv` function of `pandas`.
        """

        if 'sep' not in read_csv_kwargs and 'delimiter' not in read_csv_kwargs:
            read_csv_kwargs['delim_whitespace'] = True

        if 'na_values' not in read_csv_kwargs:
            read_csv_kwargs['na_values'] = {'phenotype': [-9.]}

        pheno_table = pd.read_csv(phenotype_file, **read_csv_kwargs)
        pheno_table.columns = ['FID', 'IID', 'phenotype']

        if self.table is not None:
            pheno_table['FID'] = pheno_table['FID'].astype(type(self.fid[0]))
            pheno_table['IID'] = pheno_table['IID'].astype(type(self.iid[0]))

            self.table = self.table.merge(pheno_table)
        else:
            self.table = pheno_table

        if self.table['phenotype'].isnull().all():
            self.table.drop('phenotype', axis=1, inplace=True)
        elif drop_na:
            # Maybe using converters in the read_csv above?
            self.table = self.table.dropna(subset=['phenotype'])

        self.post_check_phenotype()

    def read_covariates_file(self, covar_file, **read_csv_kwargs):

        """
        Read the covariates file from the provided path. The expected format is Family ID (`FID`),
        Individual ID (`IID`) and the remaining columns are assumed to be covariates. You may adjust
        the parsing configurations with keyword arguments that will be passed to `pandas.read_csv`.

        :param covar_file: The path to the covariates file.
        :param read_csv_kwargs: keyword arguments to pass to the `read_csv` function of `pandas`.
        """

        if 'sep' not in read_csv_kwargs and 'delimiter' not in read_csv_kwargs:
            read_csv_kwargs['delim_whitespace'] = True

        covar_table = pd.read_csv(covar_file, **read_csv_kwargs)
        self._covariate_cols = covar_table.columns[2:]
        covar_table.columns = ['FID', 'IID'] + self._covariate_cols

        if self.table is not None:
            covar_table['FID'] = covar_table['FID'].astype(type(self.fid[0]))
            covar_table['IID'] = covar_table['IID'].astype(type(self.iid[0]))

            self.table = self.table.merge(covar_table)
        else:
            self.table = covar_table

    def post_check_phenotype(self):
        """
        Apply some simple heuristics to check the phenotype values
        provided by the user and infer the phenotype likelihood (if needed).
        """

        if 'phenotype' in self.table.columns:

            unique_vals = self.table['phenotype'].unique()

            if self.table['phenotype'].isnull().all() or unique_vals == [-9.]:
                self.table.drop('phenotype', axis=1, inplace=True)
            elif self.phenotype_likelihood in ('binomial', None):

                unique_vals = sorted(unique_vals)
                if unique_vals == [1, 2]:
                    # Plink coding for case/control
                    self.table['phenotype'] -= 1
                    self._phenotype_likelihood = 'binomial'
                elif unique_vals == [0, 1]:
                    self._phenotype_likelihood = 'binomial'
                else:
                    raise ValueError(f"Unknown values for binary traits: {unique_vals}. "
                                     f"The software only supports 0/1 or 1/2 coding for cases and controls.")

    def filter_samples(self, keep_samples=None, keep_file=None):
        """
        Filter samples from the samples table. User must specify
        either a list of samples to keep or the path to a file
        with the list of samples to keep.

        :param keep_samples: A list (or array) of sample IDs to keep.
        :param keep_file: The path to a file with the list of samples to keep.
        """

        assert keep_samples is not None or keep_file is not None

        if keep_samples is None:
            from .parsers.misc_parsers import read_sample_filter_file
            keep_samples = read_sample_filter_file(keep_file)

        self.table = self.table.merge(pd.DataFrame({'IID': keep_samples}))

    def get_table(self, col_subset=None):
        if col_subset is not None:
            return self.table[col_subset]
        else:
            return self.table

    def get_individual_table(self):
        return self.get_table(col_subset=['FID', 'IID'])

    def get_phenotype_table(self):
        try:
            return self.get_table(col_subset=['FID', 'IID', 'phenotype'])
        except KeyError:
            raise KeyError("The phenotype is not set!")

    def get_covariates_table(self, covar_subset=None):
        assert self._covariate_cols is not None

        if covar_subset is None:
            covar = self._covariate_cols
        else:
            covar = list(set(self._covariate_cols).intersection(set(covar_subset)))

        assert len(covar) >= 1

        return self.get_table(col_subset=['FID', 'IID'] + covar)

    def get_covariates(self, covar_subset=None):
        return self.get_covariates_table(covar_subset=covar_subset).iloc[:, 2:].values

    def set_phenotype(self, phenotype, phenotype_likelihood=None):

        self.table['phenotype'] = phenotype

        if phenotype_likelihood:
            self._phenotype_likelihood = phenotype_likelihood

    def to_file(self, output_file, col_subset=None, **to_csv_kwargs):
        """
        Write the sample table to file.
        :param output_file: The path to the file where to write the sample table.
        :param col_subset: A subset of the columns to write to file.
        """

        if 'sep' not in to_csv_kwargs and 'delimiter' not in to_csv_kwargs:
            to_csv_kwargs['sep'] = '\t'

        if 'index' not in to_csv_kwargs:
            to_csv_kwargs['index'] = False

        if col_subset is not None:
            table = self.table[col_subset]
        else:
            table = self.table

        table.to_csv(output_file, **to_csv_kwargs)

    def __len__(self):
        return self.n

    def __eq__(self, other):
        return np.array_equal(self.iid, other.iid)
