
from typing import Union
import numpy as np
import pandas as pd


class SampleTable(object):
    """
    A class to represent sample (individual) information and attributes in
    the context of a genotype matrix. The sample table is a wrapper around
    a `pandas.DataFrame` object that contains the sample information. The
    table provides methods to read and write sample information from/to
    disk, filter samples, perform checks/validation, and extract specific columns
    from the table.

    :ivar table: The sample table as a pandas `DataFrame`.
    :ivar _phenotype_likelihood: The likelihood of the phenotype values (if present).
    :ivar _covariate_cols: The names or IDs of covariates that are present in the sample table.

    """

    def __init__(self,
                 table: Union[pd.DataFrame, None] = None,
                 phenotype_likelihood: Union[str, None] = None):
        """
        Initialize the sample table object.
        :param table: A pandas DataFrame with the sample information.
        :param phenotype_likelihood: The likelihood of the phenotype values.
        """

        self.table: Union[pd.DataFrame, None] = table

        if self.table is not None and 'original_index' not in self.table.columns:
            self.table['original_index'] = np.arange(len(self.table))

        assert phenotype_likelihood in (None, 'binomial', 'gaussian', 'infer')

        self._phenotype_likelihood: Union[str, None] = phenotype_likelihood
        self._covariate_cols = None

        if self.table is not None:
            self.post_check_phenotype()

    @property
    def shape(self):
        """
        :return: The shape of the sample table (mainly sample size) as a tuple (n,).
        """
        return (self.n,)

    @property
    def n(self):
        """
        !!! seealso "See Also"
            * [sample_size][magenpy.SampleTable.SampleTable.sample_size]

        :return: The sample size (number of individuals) in the sample table.
        """
        return len(self.table)

    @property
    def sample_size(self):
        """
        !!! seealso "See Also"
            * [n][magenpy.SampleTable.SampleTable.n]

        :return: he sample size (number of individuals) in the sample table.
        """
        return self.n

    @property
    def iid(self):
        """
        :return: The individual ID of each individual in the sample table.
        """
        if self.table is not None:
            return self.table['IID'].values

    @property
    def fid(self):
        """
        :return: The family ID of each individual in the sample table.
        """
        if self.table is not None:
            return self.table['FID'].values

    @property
    def phenotype(self):
        """
        :return: The phenotype column from the sample table.
        :raises KeyError: If the phenotype is not set.
        """
        if self.table is not None:
            try:
                return self.table['phenotype'].values
            except KeyError:
                raise KeyError("The phenotype is not set!")

    @property
    def original_index(self):
        """
        :return: The original index of each individual in the sample table (before applying any filters).
        """
        if self.table is not None:
            return self.table['original_index'].values

    @property
    def covariates(self):
        """
        :return: The column names for the covariates stored in the sample table.
        """
        return self._covariate_cols

    @property
    def phenotype_likelihood(self):
        """
        :return: The phenotype likelihood family.
        """
        return self._phenotype_likelihood

    @classmethod
    def from_fam_file(cls, fam_file):
        """
        Initialize a sample table object from a path to PLINK FAM file.
        :param fam_file: The path to the FAM file.

        :return: A `SampleTable` object.
        """

        from .parsers.plink_parsers import parse_fam_file

        s_tab = parse_fam_file(fam_file)
        return cls(table=s_tab)

    @classmethod
    def from_phenotype_file(cls, phenotype_file, filter_na=True, **read_csv_kwargs):
        """
        Initialize a sample table from a phenotype file.
        :param phenotype_file: The path to the phenotype file.
        :param filter_na: Filter samples with missing phenotype values (Default: True).
        :param read_csv_kwargs: keyword arguments to pass to the `read_csv` function of `pandas`.

        :return: A `SampleTable` object.
        """
        s_tab = cls()
        s_tab.read_phenotype_file(phenotype_file, filter_na, **read_csv_kwargs)
        return s_tab

    @classmethod
    def from_covariate_file(cls, covar_file, **read_csv_kwargs):
        """
        Initialize a sample table from a file of covariates.
        :param covar_file: The path to the covariates file.
        :param read_csv_kwargs: keyword arguments to pass to the `read_csv` function of `pandas`.

        :return: A `SampleTable` object.
        """
        s_tab = cls()
        s_tab.read_covariates_file(covar_file, **read_csv_kwargs)
        return s_tab

    def read_phenotype_file(self, phenotype_file, drop_na=True, **read_csv_kwargs):
        """
        Read the phenotype file from disk. The expected format is Family ID (`FID`),
        Individual ID (`IID`) and the phenotype column `phenotype`. You may adjust
        the parsing configurations with keyword arguments that will be passed to `pandas.read_csv`.

        !!! warning "Warning"
            If a phenotype column is already present in the sample table, it will be replaced.
            The data structure currently does not support multiple phenotype columns.

        :param phenotype_file: The path to the phenotype file.
        :param drop_na: Drop samples whose phenotype value is missing (Default: True).
        :param read_csv_kwargs: keyword arguments to pass to the `read_csv` function of `pandas`.
        """

        if 'sep' not in read_csv_kwargs and 'delimiter' not in read_csv_kwargs:
            read_csv_kwargs['sep'] = r'\s+'

        if 'na_values' not in read_csv_kwargs:
            read_csv_kwargs['na_values'] = {'phenotype': [-9.]}

        if 'dtype' not in read_csv_kwargs:
            read_csv_kwargs['dtype'] = {'phenotype': float}

        from .utils.compute_utils import detect_header_keywords

        if all([col not in read_csv_kwargs for col in ('header', 'names')]):

            if detect_header_keywords(phenotype_file, ['FID', 'IID']):
                read_csv_kwargs['header'] = 0
            else:
                read_csv_kwargs['names'] = ['FID', 'IID', 'phenotype']

        pheno_table = pd.read_csv(phenotype_file, **read_csv_kwargs)

        if self.table is not None:
            pheno_table['FID'] = pheno_table['FID'].astype(type(self.fid[0]))
            pheno_table['IID'] = pheno_table['IID'].astype(type(self.iid[0]))

            # Drop the phenotype column if it already exists:
            if 'phenotype' in self.table.columns:
                self.table.drop(columns=['phenotype'], inplace=True)

            self.table = self.table.merge(pheno_table, on=['FID', 'IID'])
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

        !!! warning "Warning"
            If covariate columns are already present in the sample table, they will be replaced.
            The data structure currently does not support reading separate covariates files.

        :param covar_file: The path to the covariates file.
        :param read_csv_kwargs: keyword arguments to pass to the `read_csv` function of `pandas`.
        """

        if 'sep' not in read_csv_kwargs and 'delimiter' not in read_csv_kwargs:
            read_csv_kwargs['sep'] = r'\s+'

        from .utils.compute_utils import detect_header_keywords

        if 'header' not in read_csv_kwargs:

            if detect_header_keywords(covar_file, ['FID', 'IID']):
                read_csv_kwargs['header'] = 0
            else:
                read_csv_kwargs['header'] = None

        covar_table = pd.read_csv(covar_file, **read_csv_kwargs)

        if self._covariate_cols is not None:
            self.table.drop(columns=self._covariate_cols, inplace=True)

        if read_csv_kwargs['header'] is None:
            self._covariate_cols = np.array([f'covar_{i + 1}' for i in range(covar_table.shape[1] - 2)])
        else:
            self._covariate_cols = covar_table.columns[2:]

        covar_table.columns = ['FID', 'IID'] + list(self._covariate_cols)

        if self.table is not None:
            covar_table['FID'] = covar_table['FID'].astype(type(self.fid[0]))
            covar_table['IID'] = covar_table['IID'].astype(type(self.iid[0]))

            self.table = self.table.merge(covar_table, on=['FID', 'IID'])
        else:
            self.table = covar_table

    def post_check_phenotype(self):
        """
        Apply some simple heuristics to check the phenotype values
        provided by the user and infer the phenotype likelihood (if feasible).

        :raises ValueError: If the phenotype values could not be matched with the
        inferred phenotype likelihood.
        """

        if 'phenotype' in self.table.columns:

            unique_vals = self.table['phenotype'].unique()

            if self.table['phenotype'].isnull().all():
                self.table.drop('phenotype', axis=1, inplace=True)
            elif self._phenotype_likelihood != 'gaussian':

                if len(unique_vals) > 2:
                    self._phenotype_likelihood = 'gaussian'
                    return

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

        self.table = self.table.merge(pd.DataFrame({'IID': keep_samples},
                                                   dtype=type(self.iid[0])))

    def to_table(self, col_subset=None):
        """
        Get the sample table as a pandas DataFrame.

        :param col_subset: A subset of the columns to include in the table.
        :return: A pandas DataFrame with the sample information.
        """
        if col_subset is not None:
            return self.table[list(col_subset)]
        else:
            return self.table

    def get_individual_table(self):
        """
        :return: A table of individual IDs (FID, IID) present in the sample table.
        """
        return self.to_table(col_subset=['FID', 'IID'])

    def get_phenotype_table(self):
        """
        :return: A table of individual IDs and phenotype values (FID IID phenotype) in the sample table.
        """
        try:
            return self.to_table(col_subset=['FID', 'IID', 'phenotype'])
        except KeyError:
            raise KeyError("The phenotype is not set!")

    def get_covariates_table(self, covar_subset=None):
        """
        Get a table of covariates associated with each individual in the
        sample table. The table will be formatted as (FID, IID, covar1, covar2, ...).

        :param covar_subset: A subset of the covariate names or IDs to include in the table.
        :return: A pandas DataFrame with the covariate information.
        """
        assert self._covariate_cols is not None

        if covar_subset is None:
            covar = list(self._covariate_cols)
        else:
            covar = list(set(self._covariate_cols).intersection(set(covar_subset)))

        assert len(covar) >= 1

        return self.to_table(col_subset=['FID', 'IID'] + covar)

    def get_covariates_matrix(self, covar_subset=None):
        """
        Get the covariates associated with each individual in the sample table as a matrix.
        :param covar_subset: A subset of the covariate names or IDs to include in the matrix.

        :return: A numpy matrix with the covariates values for each individual.
        """
        return self.get_covariates_table(covar_subset=covar_subset).drop(['FID', 'IID'], axis=1).values

    def set_phenotype(self, phenotype, phenotype_likelihood=None):
        """
        Update the phenotype in the sample table using the provided values.
        :param phenotype: The new phenotype values, represented by a numpy array or Iterable.
        :param phenotype_likelihood: The likelihood of the phenotype values.
        """

        self.table['phenotype'] = phenotype

        if phenotype_likelihood is not None:
            self._phenotype_likelihood = phenotype_likelihood
        else:
            self.post_check_phenotype()

    def to_file(self, output_file, col_subset=None, **to_csv_kwargs):
        """
        Write the contents of the sample table to file.
        :param output_file: The path to the file where to write the sample table.
        :param col_subset: A subset of the columns to write to file.
        :param to_csv_kwargs: keyword arguments to pass to the `to_csv` function of `pandas`.
        """

        assert self.table is not None

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
