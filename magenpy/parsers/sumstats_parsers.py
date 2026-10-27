import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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


class SumstatsParser:
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
    column_dtypes = {
        "SNP": str,
        # Parse through NumPy's fast numeric path, then downcast to int32 (or
        # nullable Int32 when missing positions remain) in post-processing.
        "POS": np.float64,
        "A1": str,
        "A2": str,
        "MAF": np.float64,
        "MAC": np.float64,
        "N": np.float64,
        "BETA": np.float64,
        "STD_BETA": np.float64,
        "OR": np.float64,
        "Z": np.float64,
        "SE": np.float64,
        "PVAL": np.float64,
        "CHISQ": np.float64,
    }
    raw_column_dtypes = {}

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
            self.col_name_converter = (
                None if col_name_converter is None else dict(col_name_converter)
            )

        self.read_csv_kwargs = read_csv_kwargs

        # If the delimiter is not specified, assume whitespace by default:
        if (
            "sep" not in self.read_csv_kwargs
            and "delimiter" not in self.read_csv_kwargs
        ):
            self.read_csv_kwargs["sep"] = "\t"

    def _get_read_csv_kwargs(self):
        """Return read options with format-aware column dtypes.

        Dtypes supplied by the caller always take precedence. Chromosomes remain
        inferred because real files mix integers with labels such as X, Y, and MT.
        """

        kwargs = self.read_csv_kwargs.copy()
        user_dtype = kwargs.get("dtype")

        # A scalar dtype is an explicit instruction for every column and should
        # not be combined with the parser defaults.
        if user_dtype is not None and not isinstance(user_dtype, Mapping):
            return kwargs

        dtype = dict(self.column_dtypes)
        converters = self.col_name_converter or {}

        # Files may already use magenpy names, or may use raw names that are
        # renamed after reading. pandas ignores dtype entries absent from a file.
        for raw_column, output_column in converters.items():
            if output_column in self.column_dtypes:
                dtype[raw_column] = self.column_dtypes[output_column]
        dtype.update(self.raw_column_dtypes)

        # pandas converters take precedence over dtype. Remove those defaults to
        # avoid warnings and unnecessary dtype bookkeeping.
        for column in kwargs.get("converters") or {}:
            dtype.pop(column, None)

        if isinstance(user_dtype, Mapping):
            dtype.update(user_dtype)
        if dtype:
            kwargs["dtype"] = dtype

        return kwargs

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

        logger.warning(
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

    def post_process(self, df):
        """Apply format-specific transformations after reading and renaming."""

        return df

    @staticmethod
    def _standardize_position_dtype(df):
        """Store base-pair positions as int32, allowing missing nonessential POS."""

        if "POS" not in df.columns:
            return df

        position = df["POS"]
        if str(position.dtype) == "Int32":
            if not position.isna().any():
                df["POS"] = position.astype(np.int32)
            return df

        if pd.api.types.is_integer_dtype(position.dtype):
            if position.dtype != np.dtype(np.int32):
                df["POS"] = position.astype(np.int32)
            return df

        if not pd.api.types.is_numeric_dtype(position.dtype):
            position = pd.to_numeric(position, errors="raise")

        if position.isna().any():
            if str(position.dtype) != "Int32":
                df["POS"] = position.astype("Int32")
        elif position.dtype != np.dtype(np.int32):
            df["POS"] = position.astype(np.int32)

        return df

    def parse(self, file_name, drop_na=True):
        """
        Parse a summary statistics file.
        :param file_name: The path to the summary statistics file.
        :param drop_na: If True, drop any entries with missing values.

        :return: A pandas DataFrame containing the parsed summary statistics.
        """

        df = pd.read_csv(file_name, **self._get_read_csv_kwargs())

        if self.col_name_converter is not None:
            df.rename(columns=self.col_name_converter, inplace=True)

        df = self.post_process(df)

        if drop_na:
            df = self.drop_na_from_essential_cols(df)

        return self._standardize_position_dtype(df)


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
    raw_column_dtypes = {"REF": str, "ALT": str, "ALT1": str}

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

    def post_process(self, df):
        """
        Infer A2 from PLINK's REF/ALT fields when it is not emitted directly.

        :param df: Parsed and renamed PLINK2 summary statistics.
        :return: The processed DataFrame.
        """

        if "A2" not in df.columns:
            alt_column = "ALT1" if "ALT1" in df.columns else "ALT"
            if alt_column in df.columns and all(
                column in df.columns for column in ("A1", "REF")
            ):
                df["A2"] = np.where(
                    df["A1"] == df[alt_column],
                    df["REF"],
                    df[alt_column],
                )
            else:
                logger.warning(
                    "The reference allele A2 could not be inferred "
                    "from the summary statistics file! Some of the columns needed to infer "
                    "the A2 allele are missing or coded differently than what we expect."
                )

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

    def post_process(self, df):
        """
        Infer sample size from minor-allele count and frequency.

        :param df: Parsed and renamed SAIGE summary statistics.
        :return: The processed DataFrame.
        """

        df["N"] = df["MAC"] / (2.0 * df["MAF"])

        return df


class RegenieSSParser(SumstatsParser):
    """Parse REGENIE step-2 association results.

    REGENIE reports effects for ``ALLELE1`` and reports association p-values
    as ``-log10(p)`` in ``LOG10P``. These are converted to magenpy's ``A1``
    and ``PVAL`` conventions, respectively.
    """

    format_name = "regenie"
    aliases = ("regenie",)
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
        "CHISQ",
        "PVAL",
    )
    output_col_name_converter = {
        "CHR": "CHROM",
        "POS": "GENPOS",
        "SNP": "ID",
        "A1": "ALLELE1",
        "A2": "ALLELE0",
        "MAF": "A1FREQ",
        "PVAL": "LOG10P",
    }
    raw_column_dtypes = {
        "INFO": np.float64,
        "TEST": str,
        "LOG10P": np.float64,
    }

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        if "sep" not in read_csv_kwargs and "delimiter" not in read_csv_kwargs:
            read_csv_kwargs["sep"] = r"\s+"

        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}
        self.col_name_converter.update(
            {
                "CHROM": "CHR",
                "GENPOS": "POS",
                "ID": "SNP",
                "ALLELE0": "A2",
                "ALLELE1": "A1",
                "A1FREQ": "MAF",
            }
        )

    def post_process(self, df):
        if "LOG10P" in df.columns:
            df["PVAL"] = np.power(10.0, -df["LOG10P"].to_numpy())
        return df

    @classmethod
    def format_table(cls, df):
        formatted = super().format_table(df)
        if "LOG10P" in formatted.columns:
            with np.errstate(divide="ignore"):
                formatted["LOG10P"] = -np.log10(
                    formatted["LOG10P"].to_numpy()
                )
        return formatted


class LDAKSSParser(SumstatsParser):
    """Parse LDAK association and SumHer/MegaPRS summary statistics.

    LDAK accepts four statistic layouts: ``Z``; ``BETA`` and ``SE``;
    ``Stat`` and ``Direction``; or ``P`` and ``Direction``. The latter two
    layouts are normalized to a signed Z-score during parsing.
    """

    format_name = "ldak"
    aliases = ("ldak", "sumher", "megaprs")
    standard_cols = ("SNP", "A1", "A2", "Z", "N", "MAF")
    output_col_name_converter = {
        "SNP": "Predictor",
        "N": "n",
        "MAF": "A1Freq",
    }
    raw_column_dtypes = {
        "Stat": np.float64,
        "Direction": np.float64,
        "A1_mean": np.float64,
    }

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        if "sep" not in read_csv_kwargs and "delimiter" not in read_csv_kwargs:
            read_csv_kwargs["sep"] = r"\s+"

        super().__init__(col_name_converter, **read_csv_kwargs)

        self.col_name_converter = self.col_name_converter or {}
        self.col_name_converter.update(
            {
                "Predictor": "SNP",
                "Chromosome": "CHR",
                "Basepair": "POS",
                "n": "N",
                "A1Freq": "MAF",
                "Stat": "CHISQ",
                "P": "PVAL",
                "Wald_Stat": "Z",
                "Wald_P": "PVAL",
                "Effect": "BETA",
                "SD": "SE",
            }
        )

    def post_process(self, df):
        # LDAK .assoc files report the mean A1 allele count (0..2), whereas
        # SumHer files report A1 frequency directly (0..1).
        if "A1_mean" in df.columns:
            df["MAF"] = 0.5 * df["A1_mean"].to_numpy()

        if "Z" in df.columns or "Direction" not in df.columns:
            return df

        direction = np.sign(df["Direction"].to_numpy())
        if "CHISQ" in df.columns:
            df["Z"] = direction * np.sqrt(df["CHISQ"].to_numpy())
        elif "PVAL" in df.columns:
            from scipy.special import ndtri

            df["Z"] = direction * -ndtri(0.5 * df["PVAL"].to_numpy())

        return df


SUMSTATS_PARSER_CLASSES = (
    SumstatsParser,
    Plink2SSParser,
    Plink1SSParser,
    COJOSSParser,
    FastGWASSParser,
    SSFParser,
    SaigeSSParser,
    RegenieSSParser,
    LDAKSSParser,
)

SUMSTATS_PARSER_REGISTRY = {
    alias: parser_cls
    for parser_cls in SUMSTATS_PARSER_CLASSES
    for alias in parser_cls.aliases
}


def get_supported_sumstats_formats(include_aliases=False):
    """Return registered summary-statistics format names.

    :param include_aliases: Include every accepted alias instead of only each
        parser's canonical format name.
    """
    if include_aliases:
        return tuple(SUMSTATS_PARSER_REGISTRY)
    return tuple(parser_cls.format_name for parser_cls in SUMSTATS_PARSER_CLASSES)


def get_sumstats_parser(sumstats_format):
    """
    Get the parser class for a supported summary statistics format.

    :param sumstats_format: The summary statistics format name or alias.

    :return: A SumstatsParser subclass.
    """
    try:
        return SUMSTATS_PARSER_REGISTRY[sumstats_format.lower()]
    except (AttributeError, KeyError):
        supported = ", ".join(get_supported_sumstats_formats())
        raise KeyError(
            f"Parsers for summary statistics format {sumstats_format} are not "
            f"implemented. Supported formats: {supported}."
        ) from None
