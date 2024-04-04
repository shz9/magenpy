import pandas as pd


class AnnotationMatrixParser(object):
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
        if 'sep' not in self.read_csv_kwargs and 'delimiter' not in self.read_csv_kwargs:
            self.read_csv_kwargs['sep'] = r'\s+'

    def parse(self, annotation_file, drop_na=True):
        """
        Parse the annotation matrix file
        :param annotation_file: The path to the annotation file.
        :param drop_na: Drop any entries with missing values.
        """

        try:
            df = pd.read_csv(annotation_file, **self.read_csv_kwargs)
        except Exception as e:
            raise e

        if drop_na:
            df = df.dropna()

        if self.col_name_converter is not None:
            df.rename(columns=self.col_name_converter, inplace=True)

        df.sort_values(['CHR', 'POS'], inplace=True)

        annotations = [ann for ann in df.columns if ann not in ('CHR', 'SNP', 'POS')]

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
        self.col_name_converter.update(
            {
                'BP': 'POS'
            }
        )

    def parse(self, annotation_file, drop_na=True):
        """
        Parse the annotation matrix file
        :param annotation_file: The path to the annotation file.
        :param drop_na: Drop any entries with missing values.
        """

        df, annotations = super().parse(annotation_file, drop_na=drop_na)

        df = df.drop(['CM', 'base'], axis=1)
        annotations = [ann for ann in annotations if ann not in ('CM', 'base')]

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
        annot_bed = pd.read_csv(annot_bed_file, usecols=[0, 1, 2],
                                sep=r'\s+',
                                names=['CHR', 'Start', 'End'])
    except Exception as e:
        raise e

    annot_bed['CHR'] = annot_bed['CHR'].str.replace('chr', '').astype(int)

    return annot_bed
