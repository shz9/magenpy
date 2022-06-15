import numpy as np
from magenpy import LDMatrix


class SampleLD(object):
    """
    Compute the sample correlation (LD) matrix between
    pairs of variants along a given chromosome.
    """

    def __init__(self, genotype_matrix):
        """
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
        """
        m = self.genotype_matrix.n_snps
        return np.array((np.zeros(m), np.ones(m)*m)).astype(np.int64)

    def compute(self, output_dir, temp_dir='temp'):
        """
        A utility method to compute the LD matrix and store in Zarr array format.
        The computes the LD matrix and stores it in Zarr array format, set its attributes,
        and performs simple validation at the end.

        :param output_dir: The path where to store the resulting LD matrix.
        :param temp_dir: A temporary directory to store intermediate files and results.
        """

        from .utils import compute_ld_xarray, compute_ld_plink1p9, _validate_ld_matrix
        from magenpy.GenotypeMatrix import xarrayGenotypeMatrix, plinkBEDGenotypeMatrix

        ld_boundaries = self.compute_ld_boundaries()

        if isinstance(self.genotype_matrix, xarrayGenotypeMatrix):
            zarr_ld_mat = compute_ld_xarray(self.genotype_matrix,
                                            ld_boundaries,
                                            output_dir,
                                            temp_dir=temp_dir)
        elif isinstance(self.genotype_matrix, plinkBEDGenotypeMatrix):
            zarr_ld_mat = compute_ld_plink1p9(self.genotype_matrix,
                                              ld_boundaries,
                                              output_dir,
                                              temp_dir=temp_dir)
        else:
            raise NotImplementedError

        zarr_ld_mat.attrs['Chromosome'] = int(self.genotype_matrix.chromosome)
        zarr_ld_mat.attrs['Sample size'] = int(self.genotype_matrix.sample_size)

        zarr_ld_mat.attrs['SNP'] = list(self.genotype_matrix.snps)
        zarr_ld_mat.attrs['BP'] = list(map(int, self.genotype_matrix.bp_pos))
        zarr_ld_mat.attrs['cM'] = list(map(float, self.genotype_matrix.cm_pos))
        zarr_ld_mat.attrs['MAF'] = list(map(float, self.genotype_matrix.maf))
        zarr_ld_mat.attrs['A1'] = list(self.genotype_matrix.a1)

        zarr_ld_mat.attrs['LD estimator'] = 'sample'
        zarr_ld_mat.attrs['LD boundaries'] = ld_boundaries.tolist()

        ld_mat = LDMatrix(zarr_ld_mat)
        ld_mat.set_store_attr('LDScore', ld_mat.compute_ld_scores().tolist())

        if _validate_ld_matrix(ld_mat):
            return ld_mat


class WindowedLD(SampleLD):
    """
    Compute the sample correlation matrix, but only in
    pre-specified windows for each variant.
    """

    def __init__(self,
                 genotype_matrix,
                 window_size=None,
                 kb_window_size=None,
                 cm_window_size=None):
        """
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
        """

        bounds = []

        m = self.genotype_matrix.n_snps
        indices = np.arange(m)

        if self.window_size:
            bounds.append(
                np.clip(np.array(
                    [indices - self.window_size,
                     indices + self.window_size
                     ]
                ),  a_min=0, a_max=m)
            )

        from .c_utils import find_windowed_ld_boundaries

        if self.kb_window_size:
            bounds.append(
                find_windowed_ld_boundaries(.001*self.genotype_matrix.bp_pos,
                                            self.kb_window_size)
            )

        if self.cm_window_size:
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

    def compute(self, output_dir, temp_dir='temp'):
        """
        :param output_dir: The path where to store the resulting LD matrix.
        :param temp_dir: A temporary directory to store intermediate files and results.
        """

        ld_mat = super(WindowedLD, self).compute(output_dir,
                                                 temp_dir)
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

    def __init__(self,
                 genotype_matrix,
                 genetic_map_ne,
                 genetic_map_sample_size,
                 threshold=1e-5):
        """
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
        Find the LD boundaries based on the shrinkage operator.
        """

        from .c_utils import find_shrinkage_ld_boundaries
        return find_shrinkage_ld_boundaries(self.genotype_matrix.cm_pos,
                                            self.genetic_map_ne,
                                            self.genetic_map_sample_size,
                                            self.threshold)

    def compute(self, output_dir, temp_dir='temp'):
        """
        :param output_dir: The path where to store the resulting LD matrix.
        :param temp_dir: A temporary directory to store intermediate files and results.
        """

        ld_mat = super(ShrinkageLD, self).compute(output_dir,
                                                  temp_dir)

        from .utils import shrink_ld_matrix

        ld_mat = LDMatrix(shrink_ld_matrix(ld_mat.z_array,
                                           self.genotype_matrix.cm_pos,
                                           self.genetic_map_ne,
                                           self.genetic_map_sample_size,
                                           self.threshold,
                                           ld_boundaries=ld_mat.ld_boundaries))

        ld_mat.set_store_attr('LD estimator', 'shrinkage')

        ld_mat.set_store_attr('Estimator properties', {
                    'Genetic map Ne': self.genetic_map_ne,
                    'Genetic map sample size': self.genetic_map_sample_size,
                    'Threshold': self.threshold
                })

        return ld_mat


class BlockLD(SampleLD):

    def __init__(self,
                 genotype_matrix,
                 ld_blocks=None,
                 ld_blocks_file=None):
        """
        :param genotype_matrix: The genotype matrix, an instance of `GenotypeMatrix`.
        :param ld_blocks: The LD blocks, a Bx2 matrix where B is the number of blocks and the
        columns are the start and end of each block, respectively.
        :param ld_blocks_file: The path to the LD blocks file
        """

        assert ld_blocks_file is not None or ld_blocks is not None

        super().__init__(genotype_matrix=genotype_matrix)

        if ld_blocks is None:
            from magenpy.parsers.misc_parsers import parse_ld_block_data
            self.ld_blocks = parse_ld_block_data(ld_blocks_file)[self.genotype_matrix.chromosome]

    def compute_ld_boundaries(self):
        """
        Find the per-SNP ld boundaries, given the provided LD blocks.
        """

        from .c_utils import find_ld_block_boundaries
        return find_ld_block_boundaries(self.genotype_matrix.bp_pos, self.ld_blocks)

    def compute(self, output_dir, temp_dir='temp'):
        """
        :param output_dir: The path where to store the resulting LD matrix.
        :param temp_dir: A temporary directory to store intermediate files and results.
        """

        ld_mat = super(BlockLD, self).compute(output_dir,
                                              temp_dir)

        ld_mat.set_store_attr('LD estimator', 'block')

        ld_mat.set_store_attr('Estimator properties', {
            'LD blocks': self.ld_blocks.tolist()
        })

        return ld_mat
