import numpy as np
import pandas as pd
import os.path as osp
import tempfile
from ...LDMatrix import LDMatrix
from ...utils.system_utils import makedir


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
     :ivar ld_boundaries: The LD boundaries for each variant in the LD matrix.
     :ivar temp_dir: A temporary directory to store intermediate files and results.
     :ivar temp_dir_prefix: A prefix for the temporary directory.

    """

    estimator_id = 'sample'

    def __init__(self, genotype_matrix):
        """
        Initialize the LD estimator with a genotype matrix.
        :param genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
        """

        self.genotype_matrix = genotype_matrix
        self.ld_boundaries = None

        self.temp_dir = self.genotype_matrix.temp_dir
        self.temp_dir_prefix = self.genotype_matrix.temp_dir_prefix + 'ld_'

        # Ensure that the genotype matrix has data for a single chromosome only:
        if self.genotype_matrix.chromosome is None:
            raise Exception("`magenpy` does not support computing inter-chromosomal LD matrices! "
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

        if self.ld_boundaries is None:

            m = self.genotype_matrix.n_snps
            self.ld_boundaries = np.array((np.zeros(m),
                                           np.ones(m)*m)).astype(np.int64)

        return self.ld_boundaries

    def compute_plink_window_thresholds(self, ld_boundaries=None):
        """
        Computes the LD window thresholds to pass to plink1.9 for computing LD matrices.
        Unfortunately, plink1.9 sets some default values for the window size and it
        is important to set all the thresholds to obtain the desired shape for the
        LD matrix.

        :param ld_boundaries: The LD boundaries for which to compute the thresholds. If not passed,
        we compute the LD boundaries using the `compute_ld_boundaries` method.

        :return: A dictionary containing the window size thresholds for plink1.9.

        """

        if ld_boundaries is None:
            ld_boundaries = self.compute_ld_boundaries()

        threshold_dict = {}

        # (1) Determine maximum window size (Maximum number of neighbors on each side):
        try:
            threshold_dict['window_size'] = getattr(self, "window_size")
            assert threshold_dict['window_size'] is not None
        except (AttributeError, AssertionError):
            threshold_dict['window_size'] = np.abs(ld_boundaries -
                                                   np.arange(ld_boundaries.shape[1])).max()

        # (2) Determine the maximum window size in kilobases + Centi Morgan (if available):

        positional_bounds = np.array([ld_boundaries[0, :], ld_boundaries[1, :] - 1])

        try:
            threshold_dict['kb_window_size'] = getattr(self, "kb_window_size")
            assert threshold_dict['kb_window_size'] is not None
        except (AttributeError, AssertionError):
            kb_pos = .001 * self.genotype_matrix.bp_pos
            kb_bounds = kb_pos[positional_bounds]
            threshold_dict['kb_window_size'] = np.abs(kb_bounds - kb_pos).max()

        # (3) centi Morgan:
        try:
            threshold_dict['cm_window_size'] = getattr(self, "cm_window_size")
            assert threshold_dict['cm_window_size'] is not None
        except (AttributeError, AssertionError):
            try:
                # Checks if cm_pos is available in the genotype matrix:
                cm_pos = self.genotype_matrix.cm_pos
                cm_bounds = self.genotype_matrix.cm_pos[positional_bounds]
                threshold_dict['cm_window_size'] = np.abs(cm_bounds - cm_pos).max()
            except KeyError:
                del threshold_dict['cm_window_size']

        if self.estimator_id == 'sample':
            # If we're using the sample estimator, then expand the plink thresholds slightly
            # to avoid the program dropping LD between variants at the extreme ends:
            for key, val in threshold_dict.items():
                threshold_dict[key] = val*1.05
                if isinstance(val, int):
                    threshold_dict[key] = int(threshold_dict[key])

        return threshold_dict

    def compute(self,
                output_dir,
                overwrite=True,
                delete_original=True,
                dtype='int16',
                compressor_name='zstd',
                compression_level=7,
                compute_spectral_properties=False) -> LDMatrix:
        """
        A utility method to compute the LD matrix and store in Zarr array format.
        The computes the LD matrix and stores it in Zarr array format, set its attributes,
        and performs simple validation at the end.

        :param output_dir: The path where to store the resulting LD matrix.
        :param overwrite: If True, overwrite any existing LD matrices in `temp_dir` and `output_dir`.
        :param delete_original: If True, deletes dense or intermediate LD matrices generated along the way.
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compressor to use for the LD matrix.
        :param compression_level: The compression level to use for the LD matrix (1-9).
        :param compute_spectral_properties: If True, compute and store information about the eigenvalues of
        the LD matrix.

        :return: An instance of `LDMatrix` containing the computed LD matrix.

        """

        from .utils import compute_ld_xarray, compute_ld_plink1p9
        from ...GenotypeMatrix import xarrayGenotypeMatrix, plinkBEDGenotypeMatrix

        assert str(dtype) in ('float32', 'float64', 'int8', 'int16')

        # Create the temporary directory using tempfile:
        temp_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix=self.temp_dir_prefix)

        ld_boundaries = self.compute_ld_boundaries()

        if isinstance(self.genotype_matrix, xarrayGenotypeMatrix):
            ld_mat = compute_ld_xarray(self.genotype_matrix,
                                       ld_boundaries,
                                       output_dir,
                                       temp_dir=temp_dir.name,
                                       overwrite=overwrite,
                                       delete_original=delete_original,
                                       dtype=dtype,
                                       compressor_name=compressor_name,
                                       compression_level=compression_level)
        elif isinstance(self.genotype_matrix, plinkBEDGenotypeMatrix):

            # Compute the window size thresholds to pass to plink 1.9:
            window_size_thersh = self.compute_plink_window_thresholds(ld_boundaries)

            ld_mat = compute_ld_plink1p9(self.genotype_matrix,
                                         ld_boundaries,
                                         output_dir,
                                         window_size_thersh,
                                         trim_boundaries=self.estimator_id not in ('sample', 'windowed'),
                                         temp_dir=temp_dir.name,
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

        if compute_spectral_properties:

            extreme_eigs = ld_mat.estimate_extremal_eigenvalues()

            ld_mat.set_store_attr('Spectral properties', {
                'Extremal': extreme_eigs
            })

        if ld_mat.validate_ld_matrix():
            temp_dir.cleanup()
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
    :ivar ld_boundaries: The LD boundaries for each variant in the LD matrix.
    :ivar window_size: The number of neighboring SNPs to consider on each side when computing LD.
    :ivar kb_window_size: The maximum distance in kilobases to consider when computing LD.
    :ivar cm_window_size: The maximum distance in centi Morgan to consider when computing LD.

    """

    estimator_id = 'windowed'

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

        if self.ld_boundaries is None:

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
                self.ld_boundaries = bounds[0]
            else:
                self.ld_boundaries = np.array([
                    np.maximum.reduce([b[0, :] for b in bounds]),
                    np.minimum.reduce([b[1, :] for b in bounds])
                ])

        return self.ld_boundaries

    def compute(self,
                output_dir,
                overwrite=True,
                delete_original=True,
                dtype='int16',
                compressor_name='zstd',
                compression_level=7,
                compute_spectral_properties=False) -> LDMatrix:
        """

        Compute the windowed LD matrix and store in Zarr array format.

        :param output_dir: The path where to store the resulting LD matrix.
        :param overwrite: If True, overwrite any existing LD matrices in `temp_dir` and `output_dir`.
        :param delete_original: If True, deletes dense or intermediate LD matrices generated along the way.
        :param dtype: The data type for the entries of the LD matrix.
        :param compressor_name: The name of the compressor to use for the LD matrix.
        :param compression_level: The compression level to use for the LD matrix (1-9).
        :param compute_spectral_properties: If True, compute and store information about the eigenvalues of
        the LD matrix.

        :return: An instance of `LDMatrix` encapsulating the computed LD matrix, its attributes, and metadata.
        """

        ld_mat = super().compute(output_dir,
                                 overwrite=overwrite,
                                 delete_original=delete_original,
                                 compute_spectral_properties=compute_spectral_properties,
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

        if compute_spectral_properties:

            if 'Spectral properties' in ld_mat.list_store_attributes():
                spectral_prop = ld_mat.get_store_attr('Spectral properties')
            else:
                spectral_prop = {
                    'Extremal': ld_mat.estimate_extremal_eigenvalues()
                }

            # Estimate extremal eigenvalues within blocks:
            # To quantify the impact of sparsification, we increase the window sizes here by 20%:

            if self.window_size is not None:
                eig_window_size = int(1.2*self.window_size)
            else:
                eig_window_size = None

            if self.kb_window_size is not None:
                eig_kb_window_size = 1.2*self.kb_window_size
            else:
                eig_kb_window_size = None

            if self.cm_window_size is not None:
                eig_cm_window_size = 1.2*self.cm_window_size
            else:
                eig_cm_window_size = None

            eigs, block_bounds = ld_mat.estimate_extremal_eigenvalues(
                block_size=eig_window_size,
                block_size_kb=eig_kb_window_size,
                block_size_cm=eig_cm_window_size,
                return_block_boundaries=True
            )

            spectral_prop['Eigenvalues per block'] = {**eigs, **block_bounds}

            # Estimate minimum eigenvalues while excluding long-range LD regions (if they're present):
            n_snps_before = ld_mat.n_snps
            ld_mat.filter_long_range_ld_regions()
            n_snps_after = ld_mat.n_snps

            if n_snps_after < n_snps_before:
                spectral_prop['Extremal (excluding LRLD)'] = ld_mat.estimate_extremal_eigenvalues()

            # Update or set the spectral properties attribute:
            ld_mat.set_store_attr('Spectral properties', spectral_prop)

            # Reset the mask:
            ld_mat.reset_mask()

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

    !!! note
        The threshold may be adjusted depending on the requested storage data type.

    !!! seealso "See Also"
        * [WindowedLD][magenpy.stats.ld.estimator.WindowedLD]
        * [BlockLD][magenpy.stats.ld.estimator.BlockLD]

    :ivar genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
    :ivar ld_boundaries: The LD boundaries for each variant in the LD matrix.
    :ivar genetic_map_ne: The effective population size (Ne) from which the genetic map is derived.
    :ivar genetic_map_sample_size: The sample size used to infer the genetic map.
    :ivar threshold: The shrinkage cutoff below which the LD is set to zero.

    """

    estimator_id = 'shrinkage'

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

        if self.ld_boundaries is None:
            from .c_utils import find_shrinkage_ld_boundaries
            self.ld_boundaries = find_shrinkage_ld_boundaries(self.genotype_matrix.cm_pos,
                                                              self.genetic_map_ne,
                                                              self.genetic_map_sample_size,
                                                              self.threshold)

        return self.ld_boundaries

    def compute(self,
                output_dir,
                overwrite=True,
                delete_original=True,
                dtype='int16',
                compressor_name='zstd',
                compression_level=7,
                compute_spectral_properties=False,
                chunk_size=1000) -> LDMatrix:
        """

        !!! note
            The threshold is adjusted depending on the requested storage data type.

        !!! note
            LD Scores are computed before applying shrinkage.

        :param output_dir: The path where to store the resulting LD matrix.
        :param overwrite: If True, overwrite any existing LD matrices in `temp_dir` and `output_dir`.
        :param delete_original: If True, deletes dense or intermediate LD matrices generated along the way.
        :param dtype: The data type for the entries of the LD matrix.
        :param compressor_name: The name of the compressor to use for the LD matrix.
        :param compression_level: The compression level to use for the LD matrix (1-9).
        :param compute_spectral_properties: If True, compute and store information about the eigenvalues of
        the LD matrix.
        :param chunk_size: An optional parameter that sets the maximum number of rows processed simultaneously.
        The smaller the `chunk_size`, the less memory requirements needed for the shrinkage step.

        :return: An instance of `LDMatrix` encapsulating the computed LD matrix, its attributes, and metadata.

        """

        # Adjust the threshold depending on the requested storage data type:
        if np.issubdtype(dtype, np.integer):
            threshold = max(self.threshold, 1./np.iinfo(dtype).max)
        else:
            threshold = self.threshold

        ld_mat = super().compute(output_dir,
                                 overwrite=overwrite,
                                 delete_original=delete_original,
                                 compute_spectral_properties=False,  # Compute after shrinkage if requested
                                 dtype=dtype,
                                 compressor_name=compressor_name,
                                 compression_level=compression_level)

        from .utils import shrink_ld_matrix

        ld_mat = shrink_ld_matrix(ld_mat,
                                  self.genotype_matrix.cm_pos,
                                  self.genotype_matrix.maf_var,
                                  self.genetic_map_ne,
                                  self.genetic_map_sample_size,
                                  threshold,
                                  chunk_size=chunk_size)

        ld_mat.set_store_attr('LD estimator', 'shrinkage')

        ld_mat.set_store_attr('Estimator properties', {
                    'Genetic map Ne': self.genetic_map_ne,
                    'Genetic map sample size': self.genetic_map_sample_size,
                    'Threshold': threshold
                })

        if compute_spectral_properties:

            spectral_prop = {
                'Extremal': ld_mat.estimate_extremal_eigenvalues()
            }

            cm_ld_bounds_start = self.genotype_matrix.cm_pos[self.ld_boundaries[0, :]]
            cm_ld_bounds_end = self.genotype_matrix.cm_pos[self.ld_boundaries[1, :] - 1]

            median_dist_cm = np.median(cm_ld_bounds_end - cm_ld_bounds_start)

            eigs, block_bounds = ld_mat.estimate_extremal_eigenvalues(block_size_cm=median_dist_cm,
                                                                      return_block_boundaries=True)
            spectral_prop['Eigenvalues per block'] = {**eigs, **block_bounds}

            ld_mat.set_store_attr('Spectral properties', spectral_prop)

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
    :ivar ld_boundaries: The LD boundaries for each variant in the LD matrix.
    :ivar ld_blocks: The LD blocks, a Bx2 matrix where B is the number of blocks and the columns are
    the start and end of each block, respectively.

    """

    estimator_id = 'block'

    def __init__(self,
                 genotype_matrix,
                 ld_blocks=None,
                 ld_blocks_file=None):
        """
        Initialize the block-based LD estimator with a genotype matrix and LD blocks.

        :param genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
        :param ld_blocks: The LD blocks, a Bx2 matrix where B is the number of blocks and the
        columns are the start and end of each block in units of base pair, respectively.
        :param ld_blocks_file: The path to the LD blocks file
        """

        assert ld_blocks_file is not None or ld_blocks is not None

        super().__init__(genotype_matrix=genotype_matrix)

        if ld_blocks is None:
            from ...parsers.misc_parsers import parse_ld_block_data
            self.ld_blocks = parse_ld_block_data(ld_blocks_file)[self.genotype_matrix.chromosome]
        else:
            self.ld_blocks = ld_blocks

        from ...utils.model_utils import map_variants_to_genomic_blocks

        # Map variants to their associated genomic blocks:
        variants_to_blocks = map_variants_to_genomic_blocks(
            pd.DataFrame({'SNP': self.genotype_matrix.snps,
                          'bp_pos': self.genotype_matrix.bp_pos}),
            pd.DataFrame(self.ld_blocks, columns=['block_start', 'block_end']).assign(
                group=np.arange(len(self.ld_blocks))),
            variant_pos_col='bp_pos',
            filter_unmatched=True
        )

        # Split the genotype matrix by the blocks:
        split_geno_matrices = self.genotype_matrix.split_by_variants(variants_to_blocks)

        self._block_estimators = {
            i: SampleLD(geno_matrix) for i, geno_matrix in split_geno_matrices.items()
        }

    def compute_ld_boundaries(self):
        """
        Compute the per-SNP Linkage-Disequilibrium (LD) boundaries for the block-based estimator.

        :return: A 2xM matrix of LD boundaries.
        """

        if self.ld_boundaries is None:
            from .c_utils import find_ld_block_boundaries
            self.ld_boundaries = find_ld_block_boundaries(self.genotype_matrix.bp_pos, self.ld_blocks)

        return self.ld_boundaries

    def compute(self,
                output_dir,
                overwrite=True,
                delete_original=True,
                dtype='int16',
                compressor_name='zstd',
                compression_level=7,
                compute_spectral_properties=False) -> LDMatrix:
        """

        Compute the block-based LD matrix and store in Zarr array format.

        :param output_dir: The path where to store the resulting LD matrix.
        :param overwrite: If True, overwrite any existing LD matrices in `temp_dir` and `output_dir`.
        :param delete_original: If True, deletes dense or intermediate LD matrices generated along the way.
        :param compute_spectral_properties: If True, compute and store information about the eigenvalues of
        the LD matrix.
        :param dtype: The data type for the entries of the LD matrix.
        :param compressor_name: The name of the compressor to use for the LD matrix.
        :param compression_level: The compression level to use for the LD matrix (1-9).

        :return: An instance of `LDMatrix` encapsulating the computed LD matrix, its attributes, and metadata.
        """

        ld_mats = []

        # TODO: Parallelize this loop

        temp_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix=self.temp_dir_prefix)

        for i, block_estimator in self._block_estimators.items():

            block_output_dir = osp.join(temp_dir.name,
                                        f'chr_{self.genotype_matrix.chromosome}_block_{i}/')

            makedir(block_output_dir)

            ld_mats.append(
                block_estimator.compute(
                    block_output_dir,
                    overwrite=overwrite,
                    delete_original=delete_original,
                    dtype=dtype,
                    compressor_name=compressor_name,
                    compression_level=compression_level,
                    compute_spectral_properties=compute_spectral_properties
                )
            )

        # If the user requested computing the spectral properties, we need to obtain
        # the minimum eigenvalue from the blocks:
        if compute_spectral_properties:

            extremal_eigs = pd.DataFrame([ld.get_store_attr('Spectral properties')['Extremal']
                                          for ld in ld_mats])
            blocks = np.insert(np.cumsum([ld.stored_n_snps for ld in ld_mats]), 0, 0)
            block_starts = blocks[:-1]
            block_ends = blocks[1:]

        # Combine the LD matrices for the individual blocks into a single LD matrix:
        from .utils import combine_ld_matrices

        output_dir = osp.join(output_dir, f'chr_{self.genotype_matrix.chromosome}/')

        ld_mat = combine_ld_matrices(ld_mats,
                                     output_dir,
                                     overwrite=overwrite,
                                     delete_original=delete_original)

        # Populate the attributes of the LDMatrix object:
        ld_mat.set_store_attr('LD estimator', 'block')
        ld_mat.set_store_attr('Estimator properties', {
            'LD blocks': self.ld_blocks.tolist()
        })

        ld_mat.set_store_attr('Chromosome', int(self.genotype_matrix.chromosome))
        ld_mat.set_store_attr('Sample size', int(self.genotype_matrix.sample_size))

        if self.genotype_matrix.genome_build is not None:
            ld_mat.set_store_attr('Genome build', self.genotype_matrix.genome_build)

        if compute_spectral_properties:

            spectral_prop = {
                'Extremal': {
                    'min': extremal_eigs['min'].min(),
                    'max': extremal_eigs['max'].max()
                },
                'Eigenvalues per block': {
                    'min': list(extremal_eigs['min']),
                    'max': list(extremal_eigs['max']),
                    'block_start': list(block_starts),
                    'block_end': list(block_ends)
                }
            }

            ld_mat.set_store_attr('Spectral properties', spectral_prop)

        if ld_mat.validate_ld_matrix():
            temp_dir.cleanup()
            return ld_mat

