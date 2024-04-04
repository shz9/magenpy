import numpy as np


class AnnotationMatrix(object):
    """
    A wrapper class for handling annotation matrices, which are essentially tables of
    features for each variant in the genome. These features include information such as
    whether the variant is in coding regions, enhancers, etc. It can also include continuous
    features derived from experimental assays or other sources.

    The purpose of this class is to present a unified and consistent interface for handling
    annotations across different tools and applications. It should be able to read and write
    annotation matrices in different formats, filter annotations, and perform basic operations
    on the annotation matrix. It should also allow users to define new custom annotations
    that can be used for downstream statistical genetics applications.

    :ivar table: A pandas dataframe containing the annotation information.
    :ivar _annotations: A list or array of column namees to consider as annotations. If not provided,
    will be inferred heuristically, though we recommend that the user specify this information.
    """

    def __init__(self, annotation_table=None, annotations=None):
        """
        Initialize an AnnotationMatrix object

        :param annotation_table: A pandas dataframe containing the annotation information.
        :param annotations: A list of array of columns to consider as annotations. If not provided, will be
        inferred heuristically, though we recommend that the user specify this information.
        """

        self.table = annotation_table
        self._annotations = annotations

        if self.table is not None:
            if self._annotations is None:
                self._annotations = [ann for ann in self.table.columns if ann not in ('CHR', 'SNP', 'POS')]
                if len(self._annotations) < 1:
                    self._annotations = None

    @classmethod
    def from_file(cls, annot_file, annot_format='magenpy', annot_parser=None,
                  **parse_kwargs):
        """
        Initialize an AnnotationMatrix object from a file.

        :param annot_file: The path to the annotation file.
        :param annot_format: The format of the annotation file. For now, we mainly support
        annotation files in the `magenpy` and `ldsc` formats.
        :param annot_parser: An `AnnotationMatrixParser` derived object, which can be tailored to
        specific annotation formats that the user has.
        :param parse_kwargs: arguments for the pandas `read_csv` function, such as the delimiter.

        :return: An instance of the `AnnotationMatrix` class.
        """

        from .parsers.annotation_parsers import AnnotationMatrixParser, LDSCAnnotationMatrixParser

        if annot_parser is None:
            if annot_format == 'magenpy':
                annot_parser = AnnotationMatrixParser(None, **parse_kwargs)
            elif annot_format == 'ldsc':
                annot_parser = LDSCAnnotationMatrixParser(None, **parse_kwargs)
            else:
                raise KeyError(f"Annotation matrix format {annot_format} not recognized!")

        annot_table, annotations = annot_parser.parse(annot_file)

        annot_mat = cls(annotation_table=annot_table, annotations=annotations)

        return annot_mat

    @property
    def shape(self):
        """
        :return: The dimensions of the annotation matrix (number of variants x number of annotations).
        """
        return self.n_snps, self.n_annotations

    @property
    def n_snps(self):
        """
        :return: The number of variants in the annotation matrix.
        """
        return len(self.table)

    @property
    def chromosome(self):
        """
        A convenience method to get the chromosome if there is only one chromosome in the annotation matrix.

        :return: The chromosome number if there is only one chromosome in the annotation matrix. Otherwise, None.
        """
        chrom = self.chromosomes
        if chrom is not None:
            if len(chrom) == 1:
                return chrom[0]

    @property
    def chromosomes(self):
        """
        :return: The list of unique chromosomes in the annotation matrix.
        """
        if 'CHR' in self.table.columns:
            return self.table['CHR'].unique()

    @property
    def snps(self):
        """
        :return: The list of SNP rsIDs in the annotation matrix.
        """
        return self.table['SNP'].values

    @property
    def n_annotations(self):
        """
        :return: The number of annotations in the annotation matrix.
        """
        if self.annotations is None:
            return 0
        else:
            return len(self.annotations)

    @property
    def binary_annotations(self):
        """
        :return: A list of binary (0/1) annotations in the annotation matrix.
        """
        assert self.annotations is not None
        return np.array([c for c in self.annotations
                         if len(self.table[c].unique()) == 2])

    @property
    def annotations(self):
        """
        :return: The list of annotation names or IDs in the annotation matrix.
        """
        return self._annotations

    def values(self, add_intercept=False):
        """
        :param add_intercept: Adds a base annotation corresponding to the intercept.

        :return: The annotation matrix as a numpy matrix.
        :raises KeyError: If no annotations are defined in the table.
        """

        if self.annotations is None:
            raise KeyError("No annotations are defined in this table!")
        annot_mat = self.table[self.annotations].values
        if add_intercept:
            return np.hstack([np.ones((annot_mat.shape[0], 1)), annot_mat])
        else:
            return annot_mat

    def filter_snps(self, extract_snps=None, extract_file=None):
        """
        Filter variants from the annotation matrix. User must specify
        either a list of variants to extract or the path to a file
        with the list of variants to extract.

        :param extract_snps: A list or array of SNP IDs to keep in the annotation matrix.
        :param extract_file: The path to a file with the list of variants to extract.
        """

        assert extract_snps is not None or extract_file is not None

        if extract_file is not None:
            from .parsers.misc_parsers import read_snp_filter_file
            extract_snps = read_snp_filter_file(extract_file)

        from .utils.compute_utils import intersect_arrays

        arr_idx = intersect_arrays(self.snps, extract_snps, return_index=True)

        self.table = self.table.iloc[arr_idx, :].reset_index()

    def filter_annotations(self, keep_annotations):
        """
        Filter the list of annotations in the matrix.
        :param keep_annotations: A list or array of annotations to keep.
        """

        if self.annotations is None:
            return

        self._annotations = [annot for annot in self._annotations if annot in keep_annotations]
        self.table = self.table[['CHR', 'SNP', 'POS'] + self._annotations]

    def add_annotation(self, annot_vec, annotation_name):
        """
        Add an annotation vector or list to the AnnotationMatrix object.

        :param annot_vec: A vector/list/Series containing the annotation information for each SNP in the
        AnnotationMatrix. For now, it's the responsibility of the user to make sure that the annotation
        list or vector are sorted properly.
        :param annotation_name: The name of the annotation to create. Make sure the name is not already
        in the matrix!
        """

        if self.annotations is not None:
            assert annotation_name not in self.annotations
        assert len(annot_vec) == self.n_snps

        self.table[annotation_name] = annot_vec

        if self.annotations is None:
            self._annotations = [annotation_name]
        else:
            self._annotations = list(self._annotations) + [annotation_name]

    def add_annotation_from_bed(self, bed_file, annotation_name):
        """
        Add an annotation to the AnnotationMatrix from a BED file that lists
        the range of coordinates associated with that annotation (e.g. coding regions, enhancers, etc.).
        The BED file has to adhere to the format specified by,
        https://uswest.ensembl.org/info/website/upload/bed.html
        with the first three columns being:

        CHR StartCoordinate EndCoordinate ...

        !!! note
            This implementation is quite slow at the moment. May need to find more efficient
            ways to do the merge over list of ranges.

        :param bed_file: The path to the BED file containing the annotation coordinates.
        :param annotation_name: The name of the annotation to create. Make sure the name is not already
        in the matrix!

        :raises AssertionError: If the annotation name is already in the matrix.
        """

        from .parsers.annotation_parsers import parse_annotation_bed_file

        if self.annotations is not None:
            assert annotation_name not in self.annotations

        bed_df = parse_annotation_bed_file(bed_file)
        # Group the BED annotation file by chromosome:
        range_groups = bed_df.groupby('CHR').groups

        def annotation_overlap(row):
            """
            This function takes a row from the annotation matrix table
            and returns True if and only if the BP position for the
            SNP is within the range specified by the annotation BED file.
            """
            try:
                chr_range = bed_df.iloc[range_groups[row['CHR']], :]
            except KeyError:
                return False

            check = (chr_range.Start <= row['POS']) & (chr_range.End >= row['POS'])
            return int(np.any(check))

        self.table[annotation_name] = self.table.apply(annotation_overlap, axis=1)

        if self.annotations is None:
            self._annotations = [annotation_name]
        else:
            self._annotations = list(self._annotations) + [annotation_name]

    def get_binary_annotation_index(self, bin_annot):
        """
        :param bin_annot: The name of the binary annotation for which to fetch the relevant variants.
        :return: The indices of all variants that belong to binary annotation `bin_annot`
        """
        assert bin_annot in self.binary_annotations
        return np.where(self.table[bin_annot] == 1)[0]

    def split_by_chromosome(self):
        """
        Split the annotation matrix by chromosome.

        :return: A dictionary of `AnnotationMatrix` objects, where the keys are the chromosome numbers.
        """

        if 'CHR' in self.table.columns:
            chrom_tables = self.table.groupby('CHR')
            return {
                c: AnnotationMatrix(annotation_table=chrom_tables.get_group(c),
                                    annotations=self.annotations)
                for c in chrom_tables.groups
            }
        else:
            raise KeyError("Chromosome information is not available in the annotation table!")

    def to_file(self, output_path, col_subset=None, compress=True, **to_csv_kwargs):
        """
        A convenience method to write the annotation matrix to a file.

        :param output_path: The path and prefix to the file where to write the annotation matrix.
        :param col_subset: A subset of the columns to write to file.
        :param compress: Whether to compress the output file (default: True).
        :param to_csv_kwargs: Key-word arguments to the pandas csv writer.
        """

        if 'sep' not in to_csv_kwargs and 'delimiter' not in to_csv_kwargs:
            to_csv_kwargs['sep'] = '\t'

        if 'index' not in to_csv_kwargs:
            to_csv_kwargs['index'] = False

        if col_subset is not None:
            table = self.table[col_subset]
        else:
            table = self.table

        file_name = output_path + '.annot'
        if compress:
            file_name += '.gz'

        table.to_csv(file_name, **to_csv_kwargs)
