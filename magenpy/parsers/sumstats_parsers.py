
import pandas as pd


class SumstatsParser(object):
    """
    A generic GWAS summary statistics parser class.
    """

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        :param col_name_converter: A dictionary mapping column names
        in the original table to magenpy's column names for the various
        summary statistics.
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """
        self.col_name_converter = col_name_converter
        self.read_csv_kwargs = read_csv_kwargs

        # If the delimiter is not specified, assume whitespace by default:
        if 'sep' not in self.read_csv_kwargs and 'delimiter' not in self.read_csv_kwargs:
            self.read_csv_kwargs['delim_whitespace'] = True

    def parse(self, file_name, drop_na=True):
        """
        Parse a summary statistics file.
        :param file_name: The path to the summary statistics file.
        :param drop_na: Drop any entries with missing values.
        """

        df = pd.read_csv(file_name, **self.read_csv_kwargs)

        if drop_na:
            df = df.dropna()

        if self.col_name_converter is not None:
            df.rename(columns=self.col_name_converter, inplace=True)

        return df


class plinkSumstatsParser(SumstatsParser):
    """
    A parser for plink GWAS summary statistics files.
    """

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        :param col_name_converter: A dictionary mapping column names
        in the original table to magenpy's column names for the various
        summary statistics.
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """

        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}

        self.col_name_converter.update(
            {
                '#CHROM': 'CHR',
                'ID': 'SNP',
                'P': 'PVAL',
                'OBS_CT': 'N',
                'A1_FREQ': 'MAF',
                'T_STAT': 'Z',
                'Z_STAT': 'Z'
            }
        )

    def parse(self, file_name, drop_na=True):
        """
        Parse a summary statistics file.
        :param file_name: The path to the summary statistics file.
        :param drop_na: Drop any entries with missing values.
        """

        df = super().parse(file_name, drop_na=drop_na)

        try:
            df['A2'] = df.apply(lambda x: [x['ALT1'], x['REF']][x['A1'] == x['ALT1']], axis=1)
        except KeyError:
            print("Warning: the reference allele A2 could not be inferred "
                  "from the summary statistics file!")

        return df


class COJOSumstatsParser(SumstatsParser):
    """
    A parser for COJO GWAS summary statistics files.
    """

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        :param col_name_converter: A dictionary mapping column names
        in the original table to magenpy's column names for the various
        summary statistics.
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """
        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}

        self.col_name_converter.update(
            {
                'freq': 'MAF',
                'b': 'BETA',
                'se': 'SE',
                'p': 'PVAL'
            }
        )


class fastGWASumstatsParser(SumstatsParser):
    """
    A parser for fastGWA summary statistics files
    """

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        :param col_name_converter: A dictionary mapping column names
        in the original table to magenpy's column names for the various
        summary statistics.
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """
        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}

        self.col_name_converter.update(
            {
                'AF1': 'MAF',
                'P': 'PVAL'
            }
        )
