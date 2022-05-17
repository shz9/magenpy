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
    def from_file(cls, annot_file, sep="\t"):
        """
        Takes an LDSC-style annotation file `annot_file`.
        NOTE: Assumes that SNPs are sorted by BP within a chromosome.
        """

        annot_mat = cls()

        try:
            annot_df = pd.read_csv(annot_file, sep=sep)
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
        if add_intercept:
            return np.hstack([np.ones((self._annot_matrix.shape[0], 1)), self._annot_matrix])
        else:
            return self._annot_matrix

    def filter_snps(self, keep_snps):

        arr_idx = intersect_arrays(self._snps, keep_snps, return_index=True)

        self._snps = self._snps[arr_idx]
        self._annot_matrix = self._annot_matrix[arr_idx, :]

    def filter_annotations(self, keep_annotations):

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
