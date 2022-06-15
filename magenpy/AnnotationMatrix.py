import numpy as np
import pandas as pd
from magenpy.utils.compute_utils import intersect_arrays


class AnnotationMatrix(object):

    def __init__(self):

        self._annot_matrix = None
        self._snps = None
        self._chromosome = None
        self._annotations = None
        self._binary_annotations = None

    @classmethod
    def from_file(cls, annot_file, **read_csv_kwargs):
        """
        Takes an LDSC-style annotation file `annot_file` and reads the table
        into a pandas dataframe.
        NOTE: Assumes that SNPs are sorted by BP within a chromosome.

        :param annot_file: LDSC-style annotation file, containing columns for
        the chromosome (CHR), SNP ID (SNP), Base pair position (BP), position in centi Morgans (CM),
        and the annotations to load into the model.

        :param read_csv_kwargs: Keyword arguments to pass to pandas `read_csv`.
        """

        annot_mat = cls()

        # If the delimiter is not specified, assume whitespace by default:
        if 'sep' not in read_csv_kwargs and 'delimiter' not in read_csv_kwargs:
            read_csv_kwargs['delim_whitespace'] = True

        try:
            annot_df = pd.read_csv(annot_file, **read_csv_kwargs)
        except Exception as e:
            raise e

        annot_mat._snps = annot_df['SNP'].values
        annot_mat._chromosome = annot_df['CHR'][0]

        annot_df = annot_df.drop(['CHR', 'SNP', 'BP', 'CM', 'base'], axis=1)

        annot_mat._annotations = np.array(annot_df.columns)
        annot_mat._binary_annotations = np.array([c for c in annot_df.columns if len(annot_df[c].unique()) == 2])

        annot_mat._annot_matrix = annot_df.values

        return annot_mat

    @property
    def shape(self):
        return self.n_snps, self.n_annotations

    @property
    def n_snps(self):
        return len(self.snps)

    @property
    def chromosome(self):
        return self._chromosome

    @property
    def snps(self):
        return self._snps

    @property
    def n_annotations(self):
        return len(self.annotations)

    @property
    def binary_annotations(self):
        return self._binary_annotations

    @property
    def annotations(self):
        return self._annotations

    def annotation_index(self, annot):
        return list(self.annotations).index(annot)

    def values(self, add_intercept=False):
        """
        Returns the annotation matrix.
        :param add_intercept: Adds a base annotation corresponding to the intercept.
        """
        if add_intercept:
            return np.hstack([np.ones((self._annot_matrix.shape[0], 1)), self._annot_matrix])
        else:
            return self._annot_matrix

    def filter_snps(self, extract_snps=None, extract_file=None):
        """
        Filter variants from the annotation matrix. User must specify
        either a list of variants to extract or the path to a file
        with the list of variants to extract.

        :param extract_snps: A list (or array) of SNP IDs to keep in the annotation matrix.
        :param extract_file: The path to a file with the list of variants to extract.
        """

        assert extract_snps is not None or extract_file is not None

        if extract_file is not None:
            from .parsers.misc_parsers import read_snp_filter_file
            extract_snps = read_snp_filter_file(extract_file)

        arr_idx = intersect_arrays(self._snps, extract_snps, return_index=True)

        self._snps = self._snps[arr_idx]
        self._annot_matrix = self._annot_matrix[arr_idx, :]

    def filter_annotations(self, keep_annotations):
        """
        Filter the list of annotations in the matrix.
        :param keep_annotations: A list or vector of annotations to keep.
        """

        ann_idx = intersect_arrays(self._annotations, keep_annotations, return_index=True)

        self._annotations = self._annotations[ann_idx]
        self._annot_matrix = self._annot_matrix[:, ann_idx]

        self._binary_annotations = intersect_arrays(self._annotations, self._binary_annotations)

    def get_binary_annotation_index(self, bin_annot):
        """
        Get the indices of all SNPs that belong to binary annotation `bin_annot`
        """
        assert bin_annot in self._binary_annotations
        return np.where(self._annot_matrix[:, self.annotation_index(bin_annot)] == 1)[0]
