import numpy as np


class SampleLD(object):
    """
    A basic wrapper class to facilitate computing Linkage-Disequilibrium (LD) matrices.

    Linkage-Disequilibrium (LD) is a measure of the SNP-by-SNP pairwise correlation between
    genetic variants in a population. LD tends to decay with genomic distance, and the rate
    of decay is influenced by many factors. Therefore, LD matrices are often diagonally-dominant.

    This class `SampleLD` provides a basic interface to compute sample correlation coefficient between
     all variants defined in a genotype matrix. The resulting LD matrix is a square and dense matrix.

     For sparse LD matrices, consider using the `WindowedLD`, `ShrinkageLD` or `BlockLD` estimators instead.

     !!! seealso "See Also"
        * [WindowedLD][magenpy.stats.ld.estimator.WindowedLD]
        * [ShrinkageLD][magenpy.stats.ld.estimator.ShrinkageLD]
        * [BlockLD][magenpy.stats.ld.estimator.BlockLD]

     :ivar genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix` or its children.

    """

    def __init__(self, genotype_matrix):
        """
        Initialize the LD estimator with a genotype matrix.
        :param genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
        """

        self.genotype_matrix = genotype_matrix

        # Ensure that the genotype matrix has data for a single chromosome only:
        if self.genotype_matrix.chromosome is None:
            raise Exception("We do not support computing inter-chromosomal LD matrices! "
                            "You may need to split the genotype matrix by chromosome. "
                            "See GenotypeMatrix.split_by_chromosome().")

    def compute_ld_boundaries(self):
        """
         Compute the Linkage-Disequilibrium (LD) boundaries. LD boundaries define the window
         for which we compute the correlation coefficient between the focal SNP and all other SNPs in
         the genome. Typically, this window is local, since the LD decays exponentially with
         genomic distance.

         The LD boundaries are a 2xM matrix, where M is the number of SNPs on the chromosome.
         The first row contains the start position for the window and the second row contains
         the end position.

         For the sample LD matrix, we simply take the entire square matrix as our window,
         so the start position is 0 and end position is M for all SNPs.

         :return: A 2xM matrix of LD boundaries.
        """
        m = self.genotype_matrix.n_snps
        return np.array((np.zeros(m), np.ones(m)*m)).astype(np.int64)

    def compute(self,
                output_dir,
                temp_dir='temp',
                overwrite=True,
                delete_original=True,
                dtype='int16',
                compressor_name='lz4',
                compression_level=5):
        """
        A utility method to compute the LD matrix and store in Zarr array format.
        The computes the LD matrix and stores it in Zarr array format, set its attributes,
        and performs simple validation at the end.

        :param output_dir: The path where to store the resulting LD matrix.
        :param temp_dir: A temporary directory to store intermediate files and results.
        :param overwrite: If True, overwrite any existing LD matrices in `temp_dir` and `output_dir`.
        :param delete_original: If True, deletes dense or intermediate LD matrices generated along the way.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compressor to use for the LD matrix.
        :param compression_level: The compression level to use for the LD matrix (1-9).

        :return: An instance of `LDMatrix` containing the computed LD matrix.

        """

        from .utils import compute_ld_xarray, compute_ld_plink1p9
        from ...GenotypeMatrix import xarrayGenotypeMatrix, plinkBEDGenotypeMatrix

        assert str(dtype) in ('float32', 'float64', 'int8', 'int16')

        ld_boundaries = self.compute_ld_boundaries()

        if isinstance(self.genotype_matrix, xarrayGenotypeMatrix):
            ld_mat = compute_ld_xarray(self.genotype_matrix,
                                       ld_boundaries,
                                       output_dir,
                                       temp_dir=temp_dir,
                                       overwrite=overwrite,
                                       delete_original=delete_original,
                                       dtype=dtype,
                                       compressor_name=compressor_name,
                                       compression_level=compression_level)
        elif isinstance(self.genotype_matrix, plinkBEDGenotypeMatrix):
            ld_mat = compute_ld_plink1p9(self.genotype_matrix,
                                         ld_boundaries,
                                         output_dir,
                                         temp_dir=temp_dir,
                                         overwrite=overwrite,
                                         dtype=dtype,
                                         compressor_name=compressor_name,
                                         compression_level=compression_level)
        else:
            raise NotImplementedError

        # Add attributes to the LDMatrix object:
        ld_mat.set_store_attr('Chromosome', int(self.genotype_matrix.chromosome))
        ld_mat.set_store_attr('Sample size', int(self.genotype_matrix.sample_size))
        ld_mat.set_store_attr('LD estimator', 'sample')

        if self.genotype_matrix.genome_build is not None:
            ld_mat.set_store_attr('Genome build', self.genotype_matrix.genome_build)

        ld_mat.set_metadata('snps', self.genotype_matrix.snps, overwrite=overwrite)
        ld_mat.set_metadata('bp', self.genotype_matrix.bp_pos, overwrite=overwrite)
        ld_mat.set_metadata('maf', self.genotype_matrix.maf, overwrite=overwrite)
        ld_mat.set_metadata('a1', self.genotype_matrix.a1, overwrite=overwrite)
        ld_mat.set_metadata('a2', self.genotype_matrix.a2, overwrite=overwrite)

        try:
            ld_mat.set_metadata('cm', self.genotype_matrix.cm_pos, overwrite=overwrite)
        except KeyError:
            pass

        ld_mat.set_metadata('ldscore', ld_mat.compute_ld_scores(), overwrite=overwrite)

        if ld_mat.validate_ld_matrix():
            return ld_mat


class WindowedLD(SampleLD):
    """
    A wrapper class to facilitate computing windowed Linkage-Disequilibrium (LD) matrices.
    Windowed LD matrices only record pairwise correlations between variants that are within a certain
    distance of each other along the chromosome. This is useful for reducing the memory requirements
    and noise in the LD matrix.

    The `WindowedLD` estimator supports a variety of ways for defining the window size:

    * `window_size`: The number of neighboring SNPs to consider on each side when computing LD.
    * `kb_window_size`: The maximum distance in kilobases to consider when computing LD.
    * `cm_window_size`: The maximum distance in centi Morgan to consider when computing LD.

    The LD boundaries computed here are the intersection of the windows defined by the window size around
    each SNP (`window_size`), the window size in kilobases (`kb_window_size`), and the window size in centi Morgan
    (`cm_window_size`).

    !!! seealso "See Also"
        * [WindowedLD][magenpy.stats.ld.estimator.ShrinkageLD]
        * [BlockLD][magenpy.stats.ld.estimator.BlockLD]

    :ivar genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
    :ivar window_size: The number of neighboring SNPs to consider on each side when computing LD.
    :ivar kb_window_size: The maximum distance in kilobases to consider when computing LD.
    :ivar cm_window_size: The maximum distance in centi Morgan to consider when computing LD.

    """

    def __init__(self,
                 genotype_matrix,
                 window_size=None,
                 kb_window_size=None,
                 cm_window_size=None):
        """

        Initialize the windowed LD estimator with a genotype matrix and window size parameters.

        :param genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
        :param window_size: The number of neighboring SNPs to consider on each side when computing LD.
        :param kb_window_size: The maximum distance in kilobases to consider when computing LD.
        :param cm_window_size: The maximum distance in centi Morgan to consider when computing LD.
        """

        super().__init__(genotype_matrix=genotype_matrix)

        assert not all([w is None for w in (window_size, kb_window_size, cm_window_size)])

        self.window_size = window_size
        self.kb_window_size = kb_window_size
        self.cm_window_size = cm_window_size

    def compute_ld_boundaries(self):
        """
         Compute the windowed Linkage-Disequilibrium (LD) boundaries.
         The LD boundaries computed here are the intersection of the windows defined by
         the window size around each SNP (`window_size`), the window size in kilobases (`kb_window_size`),
         and the window size in centi Morgan (`cm_window_size`).

         :return: A 2xM matrix of LD boundaries.
        """

        bounds = []

        m = self.genotype_matrix.n_snps
        indices = np.arange(m)

        if self.window_size is not None:
            bounds.append(
                np.clip(np.array(
                    [indices - self.window_size,
                     indices + self.window_size
                     ]
                ),  a_min=0, a_max=m)
            )

        from .c_utils import find_windowed_ld_boundaries

        if self.kb_window_size is not None:
            bounds.append(
                find_windowed_ld_boundaries(.001*self.genotype_matrix.bp_pos,
                                            self.kb_window_size)
            )

        if self.cm_window_size is not None:
            bounds.append(
                find_windowed_ld_boundaries(self.genotype_matrix.cm_pos,
                                            self.cm_window_size)
            )

        if len(bounds) == 1:
            return bounds[0]
        else:
            return np.array([
                np.maximum.reduce([b[0, :] for b in bounds]),
                np.minimum.reduce([b[1, :] for b in bounds])
            ])

    def compute(self,
                output_dir,
                temp_dir='temp',
                overwrite=True,
                delete_original=True,
                dtype='int16',
                compressor_name='lz4',
                compression_level=5):
        """

        Compute the windowed LD matrix and store in Zarr array format.

        :param output_dir: The path where to store the resulting LD matrix.
        :param temp_dir: A temporary directory to store intermediate files and results.
        :param overwrite: If True, overwrite any existing LD matrices in `temp_dir` and `output_dir`.
        :param delete_original: If True, deletes dense or intermediate LD matrices generated along the way.
        :param dtype: The data type for the entries of the LD matrix.
        :param compressor_name: The name of the compressor to use for the LD matrix.
        :param compression_level: The compression level to use for the LD matrix (1-9).

        :return: An instance of `LDMatrix` containing the computed LD matrix.
        """

        ld_mat = super().compute(output_dir,
                                 temp_dir,
                                 overwrite=overwrite,
                                 delete_original=delete_original,
                                 dtype=dtype,
                                 compressor_name=compressor_name,
                                 compression_level=compression_level)

        ld_mat.set_store_attr('LD estimator', 'windowed')

        w_properties = {}
        if self.window_size is not None:
            w_properties['Window size'] = self.window_size

        if self.kb_window_size is not None:
            w_properties['Window size (kb)'] = self.kb_window_size

        if self.cm_window_size is not None:
            w_properties['Window size (cM)'] = self.cm_window_size

        ld_mat.set_store_attr('Estimator properties', w_properties)

        return ld_mat


class ShrinkageLD(SampleLD):
    """
    A wrapper class to facilitate computing shrinkage-based Linkage-Disequilibrium (LD) matrices.
    Shrinkage LD matrices are a way to reduce noise in the LD matrix by shrinking the off-diagonal pairwise
    correlation coefficients towards zero. This is useful for reducing the noise in the LD matrix and
    improving the quality of downstream analyses.

    The shrinkage estimator implemented uses the shrinking procedure derived in:

    Wen X, Stephens M. USING LINEAR PREDICTORS TO IMPUTE ALLELE FREQUENCIES FROM SUMMARY OR POOLED GENOTYPE DATA.
    Ann Appl Stat. 2010 Sep;4(3):1158-1182. doi: 10.1214/10-aoas338. PMID: 21479081; PMCID: PMC3072818.

    Computing the shrinkage intensity requires specifying the effective population size (Ne) and the sample size
    used to infer the genetic map. In addition, it requires specifying a threshold below which the LD is set to zero.

    !!! seealso "See Also"
        * [WindowedLD][magenpy.stats.ld.estimator.WindowedLD]
        * [BlockLD][magenpy.stats.ld.estimator.BlockLD]

    :ivar genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
    :ivar genetic_map_ne: The effective population size (Ne) from which the genetic map is derived.
    :ivar genetic_map_sample_size: The sample size used to infer the genetic map.
    :ivar threshold: The shrinkage cutoff below which the LD is set to zero.

    """

    def __init__(self,
                 genotype_matrix,
                 genetic_map_ne,
                 genetic_map_sample_size,
                 threshold=1e-3):
        """

        Initialize the shrinkage LD estimator with a genotype matrix and shrinkage parameters.

        :param genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
        :param genetic_map_ne: The effective population size (Ne) from which the genetic map is derived.
        :param genetic_map_sample_size: The sample size used to infer the genetic map.
        :param threshold: The shrinkage cutoff below which the LD is set to zero.
        """

        super().__init__(genotype_matrix=genotype_matrix)

        self.genetic_map_ne = genetic_map_ne
        self.genetic_map_sample_size = genetic_map_sample_size
        self.threshold = threshold

    def compute_ld_boundaries(self):
        """
        Compute the shrinkage-based Linkage-Disequilibrium (LD) boundaries.

        :return: A 2xM matrix of LD boundaries.
        """

        from .c_utils import find_shrinkage_ld_boundaries
        return find_shrinkage_ld_boundaries(self.genotype_matrix.cm_pos,
                                            self.genetic_map_ne,
                                            self.genetic_map_sample_size,
                                            self.threshold)

    def compute(self,
                output_dir,
                temp_dir='temp',
                overwrite=True,
                delete_original=True,
                dtype='int16',
                compressor_name='lz4',
                compression_level=5,
                chunk_size=1000):
        """

        TODO: Add a mechanism to either automatically adjust the shrinkage threshold depending on the
        float precision (dtype) or purge trailing zero entries that got quantized to zero. For example,
        if we select a shrinkage threshold of 1e-3 with (int8), then we will have a lot of
        trailing zeros stored in the resulting LD matrix. It's better if we got rid of those zeros to
        minimize storage requirements and computation time.

        !!! note
            LD Scores are computed before applying shrinkage.

        :param output_dir: The path where to store the resulting LD matrix.
        :param temp_dir: A temporary directory to store intermediate files and results.
        :param overwrite: If True, overwrite any existing LD matrices in `temp_dir` and `output_dir`.
        :param delete_original: If True, deletes dense or intermediate LD matrices generated along the way.
        :param dtype: The data type for the entries of the LD matrix.
        :param compressor_name: The name of the compressor to use for the LD matrix.
        :param compression_level: The compression level to use for the LD matrix (1-9).
        :param chunk_size: An optional parameter that sets the maximum number of rows processed simultaneously.
        The smaller the `chunk_size`, the less memory requirements needed for the shrinkage step.

        :return: An instance of `LDMatrix` containing the computed LD matrix.

        """

        ld_mat = super().compute(output_dir,
                                 temp_dir,
                                 overwrite=overwrite,
                                 delete_original=delete_original,
                                 dtype=dtype,
                                 compressor_name=compressor_name,
                                 compression_level=compression_level)

        from .utils import shrink_ld_matrix

        ld_mat = shrink_ld_matrix(ld_mat,
                                  self.genotype_matrix.cm_pos,
                                  self.genotype_matrix.maf_var,
                                  self.genetic_map_ne,
                                  self.genetic_map_sample_size,
                                  self.threshold,
                                  chunk_size=chunk_size)

        ld_mat.set_store_attr('LD estimator', 'shrinkage')

        ld_mat.set_store_attr('Estimator properties', {
                    'Genetic map Ne': self.genetic_map_ne,
                    'Genetic map sample size': self.genetic_map_sample_size,
                    'Threshold': self.threshold
                })

        return ld_mat


class BlockLD(SampleLD):
    """
    A wrapper class to facilitate computing block-based Linkage-Disequilibrium (LD) matrices.
    Block-based LD matrices are a way to reduce the memory requirements of the LD matrix by
    computing the pairwise correlation coefficients only between SNPs that are within the same LD block.

    LD blocks can be inferred by external software tools, such as `LDetect` of Berisa and Pickrell (2016):

    Berisa T, Pickrell JK. Approximately independent linkage disequilibrium blocks in human populations.
    Bioinformatics. 2016 Jan 15;32(2):283-5. doi: 10.1093/bioinformatics/btv546.
    Epub 2015 Sep 22. PMID: 26395773; PMCID: PMC4731402.

    The `BlockLD` estimator requires the LD blocks to be provided as input. The LD blocks are a Bx2 matrix
    where B is the number of blocks and the columns are the start and end of each block, respectively.

    !!! seealso "See Also"
        * [WindowedLD][magenpy.stats.ld.estimator.WindowedLD]
        * [ShrinkageLD][magenpy.stats.ld.estimator.ShrinkageLD]

    :ivar genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
    :ivar ld_blocks: The LD blocks, a Bx2 matrix where B is the number of blocks and the columns are
    the start and end of each block, respectively.

    """

    def __init__(self,
                 genotype_matrix,
                 ld_blocks=None,
                 ld_blocks_file=None):
        """
        Initialize the block-based LD estimator with a genotype matrix and LD blocks.

        :param genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
        :param ld_blocks: The LD blocks, a Bx2 matrix where B is the number of blocks and the
        columns are the start and end of each block, respectively.
        :param ld_blocks_file: The path to the LD blocks file
        """

        assert ld_blocks_file is not None or ld_blocks is not None

        super().__init__(genotype_matrix=genotype_matrix)

        if ld_blocks is None:
            from ...parsers.misc_parsers import parse_ld_block_data
            self.ld_blocks = parse_ld_block_data(ld_blocks_file)[self.genotype_matrix.chromosome]

    def compute_ld_boundaries(self):
        """
        Compute the per-SNP Linkage-Disequilibrium (LD) boundaries for the block-based estimator.

        :return: A 2xM matrix of LD boundaries.
        """

        from .c_utils import find_ld_block_boundaries
        return find_ld_block_boundaries(self.genotype_matrix.bp_pos, self.ld_blocks)

    def compute(self,
                output_dir,
                temp_dir='temp',
                overwrite=True,
                delete_original=True,
                dtype='int16',
                compressor_name='lz4',
                compression_level=5):
        """

        Compute the block-based LD matrix and store in Zarr array format.

        :param output_dir: The path where to store the resulting LD matrix.
        :param temp_dir: A temporary directory to store intermediate files and results.
        :param overwrite: If True, overwrite any existing LD matrices in `temp_dir` and `output_dir`.
        :param delete_original: If True, deletes dense or intermediate LD matrices generated along the way.
        :param dtype: The data type for the entries of the LD matrix.
        :param compressor_name: The name of the compressor to use for the LD matrix.
        :param compression_level: The compression level to use for the LD matrix (1-9).

        :return: An instance of `LDMatrix` containing the computed LD matrix.
        """

        ld_mat = super().compute(output_dir,
                                 temp_dir,
                                 overwrite=overwrite,
                                 delete_original=delete_original,
                                 dtype=dtype,
                                 compressor_name=compressor_name,
                                 compression_level=compression_level)

        ld_mat.set_store_attr('LD estimator', 'block')

        ld_mat.set_store_attr('Estimator properties', {
            'LD blocks': self.ld_blocks.tolist()
        })

        return ld_mat
