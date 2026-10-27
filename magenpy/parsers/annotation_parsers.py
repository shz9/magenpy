import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnnotationMatrixParser:
    """
    A generic annotation matrix parser class.
    """

    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        :param col_name_converter: A dictionary mapping column names
        in the original table to magenpy's column names for the various
        SNP features in the annotation matrix.
        :param read_csv_kwargs: Keyword arguments to pass to pandas' `read_csv`.
        """

        self.col_name_converter = col_name_converter
        self.read_csv_kwargs = read_csv_kwargs

        # If the delimiter is not specified, assume whitespace by default:
        if (
            "sep" not in self.read_csv_kwargs
            and "delimiter" not in self.read_csv_kwargs
        ):
            self.read_csv_kwargs["sep"] = "\t"

    def _get_read_csv_kwargs(self):
        """Return read options with dtypes for stable variant metadata."""

        kwargs = self.read_csv_kwargs.copy()
        user_dtype = kwargs.get("dtype")
        if user_dtype is not None and not isinstance(user_dtype, Mapping):
            return kwargs

        dtype = {"SNP": str, "POS": np.float64}
        for raw_column, output_column in (self.col_name_converter or {}).items():
            if output_column in dtype:
                dtype[raw_column] = dtype[output_column]
        if isinstance(user_dtype, Mapping):
            dtype.update(user_dtype)
        for column in kwargs.get("converters") or {}:
            dtype.pop(column, None)
        kwargs["dtype"] = dtype
        return kwargs

    def parse(self, annotation_file, drop_na=True):
        """
        Parse the annotation matrix file
        :param annotation_file: The path to the annotation file.
        :param drop_na: Drop any entries with missing values.
        """

        try:
            df = pd.read_csv(annotation_file, **self._get_read_csv_kwargs())
        except Exception as e:
            logger.error(
                "Failed to parse the annotation file(s):\n%s",
                e.stderr.decode(),
            )
            raise e

        if drop_na:
            df = df.dropna()

        if self.col_name_converter is not None:
            df.rename(columns=self.col_name_converter, inplace=True)

        if "POS" in df:
            if df["POS"].isna().any():
                df["POS"] = df["POS"].astype("Int32")
            else:
                df["POS"] = df["POS"].astype(np.int32)

        df.sort_values(["CHR", "POS"], inplace=True)

        annotations = [ann for ann in df.columns if ann not in ("CHR", "SNP", "POS")]

        return df, annotations


class LDSCAnnotationMatrixParser(AnnotationMatrixParser):
    def __init__(self, col_name_converter=None, **read_csv_kwargs):
        """
        :param col_name_converter: A dictionary mapping column names
        in the original table to magenpy's column names for the various
        SNP features in the annotation matrix.
        :param read_csv_kwargs: Keyword arguments to pass to pandas' read_csv
        """

        super().__init__(col_name_converter, **read_csv_kwargs)
        self.col_name_converter = self.col_name_converter or {}
        self.col_name_converter.update({"BP": "POS"})

    def parse(self, annotation_file, drop_na=True):
        """
        Parse the annotation matrix file
        :param annotation_file: The path to the annotation file.
        :param drop_na: Drop any entries with missing values.
        """

        df, annotations = super().parse(annotation_file, drop_na=drop_na)

        df = df.drop(["CM", "base"], axis=1)
        annotations = [ann for ann in annotations if ann not in ("CM", "base")]

        return df, annotations


def parse_annotation_bed_file(annot_bed_file):
    """
    Parse an annotation bed file in the format specified by Ensemble:
    https://uswest.ensembl.org/info/website/upload/bed.html

    The file contains 3-12 columns, starting with Chromosome, start_coordinate, end_coordinate, etc.
    After reading the raw file, we let pandas infer whether the file has a header or not and we
    standardize the names of the first 3 columns and convert the chromosome column into an integer.

    :param annot_bed_file: The path to the annotation BED file.
    :type annot_bed_file: str
    """

    try:
        annot_bed = pd.read_csv(
            annot_bed_file,
            usecols=[0, 1, 2],
            sep="\t",
            names=["CHR", "Start", "End"],
            dtype={"CHR": str, "Start": np.int32, "End": np.int32},
        )
    except Exception as e:
        logger.error(
            "Failed to parse the BED file:\n%s",
            e.stderr.decode(),
        )
        raise e

    annot_bed["CHR"] = annot_bed["CHR"].str.replace("chr", "").astype(int)

    return annot_bed
