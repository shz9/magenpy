import warnings

import numpy as np
import pandas as pd

MAGENPY_SUMSTATS_COLUMNS = (
    "CHR",
    "SNP",
    "POS",
    "A1",
    "A2",
    "MAF",
    "N",
    "BETA",
    "Z",
    "SE",
    "PVAL",
)


DEFAULT_ESSENTIAL_COL_GROUPS = (
    (
        ("SNP", "A1"),
        ("CHR", "POS", "A1"),
    ),
    (
        ("BETA", "SE"),
        ("Z",),
        ("PVAL", "BETA"),
        ("PVAL", "OR"),
        ("PVAL", "Z"),
        ("CHISQ", "BETA"),
        ("CHISQ", "OR"),
        ("CHISQ", "Z"),
    ),
)


class SumstatsParser(object):
    """
    A wrapper class for parsing summary statistics files that are written by statistical genetics software
    for Genome-wide Association testing. A common challenge is the fact that different software tools
    output summary statistics in different formats and with different column names. Thus, this class
    provides a common interface for parsing summary statistics files from different software tools
    and aims to make this process as seamless as possible.

    The class is designed to be extensible, so that users can easily add new parsers for different software tools.

    !!! seealso "See Also"
        * [Plink2SSParser][magenpy.parsers.sumstats_parsers.Plink2SSParser]
        * [Plink1SSParser][magenpy.parsers.sumstats_parsers.Plink1SSParser]
        * [COJOSSParser][magenpy.parsers.sumstats_parsers.COJOSSParser]
        * [FastGWASSParser][magenpy.parsers.sumstats_parsers.FastGWASSParser]
        * [SSFParser][magenpy.parsers.sumstats_parsers.SSFParser]
        * [SaigeSSParser][magenpy.parsers.sumstats_parsers.SaigeSSParser]

    :ivar col_name_converter: A dictionary mapping column names in the original table to magenpy's column names.
    :ivar read_csv_kwargs: Keyword arguments to pass to pandas' `read_csv`.

    """

    format_name = "magenpy"
    aliases = ("magenpy",)
    standard_cols = MAGENPY_SUMSTATS_COLUMNS
    essential_col_groups = DEFAULT_ESSENTIAL_COL_GROUPS
    output_col_name_converter = {}

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        Initialize the summary statistics parser.

        :param col_name_converter: A dictionary/string mapping column names
        in the original table to magenpy's column names for the various
        summary statistics. If a string, it should be a comma-separated list of
        key-value pairs (e.g. 'rsid=SNP,pos=POS').
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """

        if isinstance(col_name_converter, str):
            self.col_name_converter = {
                k: v
                for entry in col_name_converter.split(",")
                for k, v in [entry.strip().split("=")]
                if len(entry.strip()) > 0
            }
        else:
            self.col_name_converter = col_name_converter

        self.read_csv_kwargs = read_csv_kwargs

        # If the delimiter is not specified, assume whitespace by default:
        if (
            "sep" not in self.read_csv_kwargs
            and "delimiter" not in self.read_csv_kwargs
        ):
            self.read_csv_kwargs["sep"] = r"\s+"

    @classmethod
    def get_standard_cols(cls):
        """
        :return: The standard magenpy columns for this summary statistics format.
        """
        return list(cls.standard_cols)

    @classmethod
    def get_essential_col_groups(cls):
        """
        Essential columns are encoded as groups of alternatives. Each group must
        be satisfied by at least one of its alternatives.

        :return: A tuple of essential column alternative groups.
        """
        return cls.essential_col_groups

    @classmethod
    def get_essential_cols(cls, columns):
        """
        Resolve the essential column groups against a concrete set of columns.

        :param columns: The available columns in a summary statistics table.

        :return: A list of essential columns present in the table.
        """
        columns = set(columns)
        essential_cols = []

        for alternatives in cls.essential_col_groups:
            for col_set in alternatives:
                if set(col_set).issubset(columns):
                    essential_cols.extend(
                        [c for c in col_set if c not in essential_cols]
                    )
                    break

        return essential_cols

    @classmethod
    def drop_na_from_essential_cols(cls, df):
        """
        Drop rows with missing values in columns that are essential for variant
        identity and effect-size interpretation.

        :param df: A pandas DataFrame containing summary statistics.

        :return: A DataFrame with missing essential values removed.
        """
        essential_cols = cls.get_essential_cols(df.columns)

        if len(essential_cols) > 0:
            return df.dropna(subset=essential_cols)

        warnings.warn(
            "Could not identify any essential summary statistics columns; "
            "no rows were dropped for missing values."
        )
        return df

    @classmethod
    def format_table(cls, df):
        """
        Format a magenpy summary statistics table for output.

        :param df: A pandas DataFrame with magenpy-standard column names.

        :return: A DataFrame with format-specific column names.
        """
        return df.rename(columns=cls.output_col_name_converter)

    def parse(self, file_name, drop_na=True):
        """
        Parse a summary statistics file.
        :param file_name: The path to the summary statistics file.
        :param drop_na: If True, drop any entries with missing values.

        :return: A pandas DataFrame containing the parsed summary statistics.
        """

        df = pd.read_csv(file_name, **self.read_csv_kwargs)

        if self.col_name_converter is not None:
            df.rename(columns=self.col_name_converter, inplace=True)

        try:
            df["POS"] = df["POS"].astype(np.int32)
        except KeyError:
            pass

        if drop_na:
            df = self.drop_na_from_essential_cols(df)

        return df


class Plink2SSParser(SumstatsParser):
    """
    A specialized class for parsing GWAS summary statistics files generated by `plink2`.

    !!! seealso "See Also"
        * [Plink1SSParser][magenpy.parsers.sumstats_parsers.Plink1SSParser]
        * [COJOSSParser][magenpy.parsers.sumstats_parsers.COJOSSParser]
        * [FastGWASSParser][magenpy.parsers.sumstats_parsers.FastGWASSParser]
        * [SSFParser][magenpy.parsers.sumstats_parsers.SSFParser]
        * [SaigeSSParser][magenpy.parsers.sumstats_parsers.SaigeSSParser]

    :ivar col_name_converter: A dictionary mapping column names in the original table to magenpy's column names.
    :ivar read_csv_kwargs: Keyword arguments to pass to pandas' `read_csv`.

    """

    format_name = "plink2"
    aliases = ("plink", "plink2")
    standard_cols = (
        "CHR",
        "POS",
        "SNP",
        "A1",
        "A2",
        "MAF",
        "N",
        "BETA",
        "SE",
        "Z",
        "PVAL",
    )
    output_col_name_converter = {
        "CHR": "#CHROM",
        "SNP": "ID",
        "MAF": "A1_FREQ",
        "N": "OBS_CT",
        "Z": "Z_STAT",
        "PVAL": "P",
    }

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """

        Initialize the `plink2` summary statistics parser.

        :param col_name_converter: A dictionary/string mapping column names
        in the original table to magenpy's column names for the various
        summary statistics. If a string, it should be a comma-separated list of
        key-value pairs (e.g. 'rsid=SNP,pos=POS').
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """

        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}

        self.col_name_converter.update(
            {
                "#CHROM": "CHR",
                "ID": "SNP",
                "P": "PVAL",
                "OBS_CT": "N",
                "A1_FREQ": "MAF",
                "T_STAT": "Z",
                "Z_STAT": "Z",
            }
        )

    def parse(self, file_name, drop_na=True):
        """
        Parse a summary statistics file.
        :param file_name: The path to the summary statistics file.
        :param drop_na: Drop any entries with missing values.

        :return: A pandas DataFrame containing the parsed summary statistics.
        """

        df = super().parse(file_name, drop_na=False)

        if "A2" not in df.columns:
            try:
                if "ALT1" in df.columns:
                    df["A2"] = np.where(df["A1"] == df["ALT1"], df["REF"], df["ALT1"])
                elif "ALT" in df.columns:
                    df["A2"] = np.where(df["A1"] == df["ALT"], df["REF"], df["ALT"])
                else:
                    warnings.warn(
                        "The reference allele A2 could not be inferred "
                        "from the summary statistics file!"
                    )
            except KeyError:
                warnings.warn(
                    "The reference allele A2 could not be inferred "
                    "from the summary statistics file! Some of the columns needed to infer "
                    "the A2 allele are missing or coded differently than what we expect."
                )

        if drop_na:
            df = self.drop_na_from_essential_cols(df)

        return df


class Plink1SSParser(SumstatsParser):
    """
    A specialized class for parsing GWAS summary statistics files generated by `plink1.9`.

    !!! seealso "See Also"
        * [Plink2SSParser][magenpy.parsers.sumstats_parsers.Plink2SSParser]
        * [COJOSSParser][magenpy.parsers.sumstats_parsers.COJOSSParser]
        * [FastGWASSParser][magenpy.parsers.sumstats_parsers.FastGWASSParser]
        * [SSFParser][magenpy.parsers.sumstats_parsers.SSFParser]
        * [SaigeSSParser][magenpy.parsers.sumstats_parsers.SaigeSSParser]

    :ivar col_name_converter: A dictionary mapping column names in the original table to magenpy's column names.
    :ivar read_csv_kwargs: Keyword arguments to pass to pandas' `read_csv`.

    """

    format_name = "plink1.9"
    aliases = ("plink1.9",)
    standard_cols = ("CHR", "SNP", "POS", "A1", "A2", "N", "BETA", "Z", "PVAL")
    output_col_name_converter = {"POS": "BP", "N": "NMISS", "Z": "STAT", "PVAL": "P"}

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        Initialize the `plink1.9` summary statistics parser.

        :param col_name_converter: A dictionary/string mapping column names
        in the original table to magenpy's column names for the various
        summary statistics. If a string, it should be a comma-separated list of
        key-value pairs (e.g. 'rsid=SNP,pos=POS').
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """

        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}

        self.col_name_converter.update(
            {"P": "PVAL", "NMISS": "N", "STAT": "Z", "BP": "POS"}
        )


class COJOSSParser(SumstatsParser):
    """
    A specialized class for parsing GWAS summary statistics files generated by the `COJO` software.

    !!! seealso "See Also"
        * [Plink2SSParser][magenpy.parsers.sumstats_parsers.Plink2SSParser]
        * [Plink1SSParser][magenpy.parsers.sumstats_parsers.Plink1SSParser]
        * [FastGWASSParser][magenpy.parsers.sumstats_parsers.FastGWASSParser]
        * [SSFParser][magenpy.parsers.sumstats_parsers.SSFParser]
        * [SaigeSSParser][magenpy.parsers.sumstats_parsers.SaigeSSParser]

    :ivar col_name_converter: A dictionary mapping column names in the original table to magenpy's column names.
    :ivar read_csv_kwargs: Keyword arguments to pass to pandas' `read_csv`.
    """

    format_name = "cojo"
    aliases = ("cojo",)
    standard_cols = ("SNP", "A1", "A2", "MAF", "BETA", "SE", "PVAL", "N")
    output_col_name_converter = {"MAF": "freq", "BETA": "b", "SE": "se", "PVAL": "p"}

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """

        Initialize the COJO summary statistics parser.

        :param col_name_converter: A dictionary/string mapping column names
        in the original table to magenpy's column names for the various
        summary statistics. If a string, it should be a comma-separated list of
        key-value pairs (e.g. 'rsid=SNP,pos=POS').
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """
        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}

        self.col_name_converter.update(
            {"freq": "MAF", "b": "BETA", "se": "SE", "p": "PVAL"}
        )


class FastGWASSParser(SumstatsParser):
    """
    A specialized class for parsing GWAS summary statistics files generated by the `FastGWA` software.

    !!! seealso "See Also"
        * [Plink2SSParser][magenpy.parsers.sumstats_parsers.Plink2SSParser]
        * [Plink1SSParser][magenpy.parsers.sumstats_parsers.Plink1SSParser]
        * [COJOSSParser][magenpy.parsers.sumstats_parsers.COJOSSParser]
        * [SSFParser][magenpy.parsers.sumstats_parsers.SSFParser]
        * [SaigeSSParser][magenpy.parsers.sumstats_parsers.SaigeSSParser]

    :ivar col_name_converter: A dictionary mapping column names in the original table to magenpy's column names.
    :ivar read_csv_kwargs: Keyword arguments to pass to pandas' `read_csv`.


    """

    format_name = "fastgwa"
    aliases = ("fastgwa",)
    standard_cols = ("CHR", "SNP", "POS", "A1", "A2", "MAF", "BETA", "SE", "PVAL", "N")
    output_col_name_converter = {"MAF": "AF1", "PVAL": "P"}

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        :param col_name_converter: A dictionary/string mapping column names
        in the original table to magenpy's column names for the various
        summary statistics. If a string, it should be a comma-separated list of
        key-value pairs (e.g. 'rsid=SNP,pos=POS').
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """
        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}

        self.col_name_converter.update({"AF1": "MAF", "P": "PVAL"})


class SSFParser(SumstatsParser):
    """
    A specialized class for parsing GWAS summary statistics that are formatted according
     to the standardized summary statistics format adopted by the GWAS Catalog. This format is
     sometimes denoted as `GWAS-SSF`.

    Reference and details:
    https://github.com/EBISPOT/gwas-summary-statistics-standard

    !!! seealso "See Also"
        * [Plink2SSParser][magenpy.parsers.sumstats_parsers.Plink2SSParser]
        * [Plink1SSParser][magenpy.parsers.sumstats_parsers.Plink1SSParser]
        * [COJOSSParser][magenpy.parsers.sumstats_parsers.COJOSSParser]
        * [FastGWASSParser][magenpy.parsers.sumstats_parsers.FastGWASSParser]
        * [SaigeSSParser][magenpy.parsers.sumstats_parsers.SaigeSSParser]

    :ivar col_name_converter: A dictionary mapping column names in the original table to magenpy's column names.
    :ivar read_csv_kwargs: Keyword arguments to pass to pandas' `read_csv`.

    """

    format_name = "ssf"
    aliases = ("ssf", "gwas-ssf", "gwascatalog")
    standard_cols = ("CHR", "POS", "SNP", "A1", "A2", "BETA", "SE", "MAF", "PVAL", "N")
    output_col_name_converter = {
        "CHR": "chromosome",
        "POS": "base_pair_location",
        "SNP": "rsid",
        "A1": "effect_allele",
        "A2": "other_allele",
        "BETA": "beta",
        "SE": "standard_error",
        "MAF": "effect_allele_frequency",
        "PVAL": "p_value",
        "N": "n",
    }

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """

        Initialize the standardized summary statistics parser.

        :param col_name_converter: A dictionary/string mapping column names
        in the original table to magenpy's column names for the various
        summary statistics. If a string, it should be a comma-separated list of
        key-value pairs (e.g. 'rsid=SNP,pos=POS').
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """

        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}

        self.col_name_converter.update(
            {
                "chromosome": "CHR",
                "base_pair_location": "POS",
                "rsid": "SNP",
                "effect_allele": "A1",
                "other_allele": "A2",
                "beta": "BETA",
                "standard_error": "SE",
                "effect_allele_frequency": "MAF",
                "p_value": "PVAL",
                "n": "N",
            }
        )


class SaigeSSParser(SumstatsParser):
    """
    A specialized class for parsing GWAS summary statistics files generated by the `SAIGE` software.
    Reference and details:
    https://saigegit.github.io/SAIGE-doc/docs/single_step2.html

    TODO: Ensure that the column names are correct across different trait types
    and the inference of the sample size is correct.

    !!! seealso "See Also"
        * [Plink2SSParser][magenpy.parsers.sumstats_parsers.Plink2SSParser]
        * [Plink1SSParser][magenpy.parsers.sumstats_parsers.Plink1SSParser]
        * [COJOSSParser][magenpy.parsers.sumstats_parsers.COJOSSParser]
        * [FastGWASSParser][magenpy.parsers.sumstats_parsers.FastGWASSParser]
        * [SSFParser][magenpy.parsers.sumstats_parsers.SSFParser]

    :ivar col_name_converter: A dictionary mapping column names in the original table to magenpy's column names.
    :ivar read_csv_kwargs: Keyword arguments to pass to pandas' `read_csv`.

    """

    format_name = "saige"
    aliases = ("saige",)
    standard_cols = ("SNP", "A1", "A2", "MAF", "MAC", "Z", "PVAL", "N")
    output_col_name_converter = {
        "SNP": "MarkerID",
        "A2": "Allele1",
        "A1": "Allele2",
        "MAF": "AF_Allele2",
        "MAC": "AC_Allele2",
        "Z": "Tstat",
        "PVAL": "p.value",
    }

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        Initialize the `SAIGE` summary statistics parser.

        :param col_name_converter: A dictionary/string mapping column names
        in the original table to magenpy's column names for the various
        summary statistics. If a string, it should be a comma-separated list of
        key-value pairs (e.g. 'rsid=SNP,pos=POS').
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """
        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}

        # NOTE: SAIGE considers Allele2 to be the effect allele, so
        # we switch their designation here:
        self.col_name_converter.update(
            {
                "MarkerID": "SNP",
                "Allele1": "A2",
                "Allele2": "A1",
                "AF_Allele2": "MAF",
                "AC_Allele2": "MAC",
                "Tstat": "Z",
                "p.value": "PVAL",
            }
        )

    def parse(self, file_name, drop_na=True):
        """
        Parse the summary statistics file.
        :param file_name: The path to the summary statistics file.
        :param drop_na: Drop any entries with missing values.

        :return: A pandas DataFrame containing the parsed summary statistics.
        """

        df = super().parse(file_name, drop_na=False)

        # Infer the sample size N
        df["N"] = df["MAC"] / (2.0 * df["MAF"])

        if drop_na:
            df = self.drop_na_from_essential_cols(df)

        return df


SUMSTATS_PARSER_CLASSES = (
    SumstatsParser,
    Plink2SSParser,
    Plink1SSParser,
    COJOSSParser,
    FastGWASSParser,
    SSFParser,
    SaigeSSParser,
)

SUMSTATS_PARSER_REGISTRY = {
    alias: parser_cls
    for parser_cls in SUMSTATS_PARSER_CLASSES
    for alias in parser_cls.aliases
}


def get_sumstats_parser(sumstats_format):
    """
    Get the parser class for a supported summary statistics format.

    :param sumstats_format: The summary statistics format name or alias.

    :return: A SumstatsParser subclass.
    """
    try:
        return SUMSTATS_PARSER_REGISTRY[sumstats_format.lower()]
    except KeyError:
        raise KeyError(
            f"Parsers for summary statistics format {sumstats_format} are not implemented!"
        )
