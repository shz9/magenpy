"""
Author: Shadi Zabad
Date: December 2020
"""

import os.path as osp
import tempfile
from tqdm import tqdm

import dask.array as da
from pandas_plink import read_plink1_bin
from itertools import zip_longest

from scipy import stats

from .LDWrapper import LDWrapper
from .c_utils import find_windowed_ld_boundaries, find_shrinkage_ld_boundaries
from .utils import *


class GWASDataLoader(object):

    def __init__(self,
                 bed_files=None,
                 standardize_genotype=True,
                 phenotype_likelihood='gaussian',
                 phenotype_file=None,
                 phenotype_header=None,
                 phenotype_col=2,
                 phenotype_id=None,
                 standardize_phenotype=True,
                 sumstats_file=None,
                 keep_individuals=None,
                 keep_snps=None,
                 annotation_files=None,
                 train_samples=None,
                 train_idx=None,
                 test_samples=None,
                 test_idx=None,
                 genmap_Ne=None,
                 genmap_sample_size=None,
                 shrinkage_cutoff=1e-3,
                 compute_ld=False,
                 ld_store_files=None,
                 ld_block_files=None,
                 ld_subset_samples=None,
                 ld_subset_idx=None,
                 ld_estimator='windowed',
                 window_unit='cM',
                 cm_window_cutoff=3.,
                 window_size_cutoff=2000,
                 use_plink=False,
                 batch_size=200,
                 temp_dir='temp',
                 output_dir='output',
                 verbose=True,
                 n_threads=1):

        # ------- General options -------
        self.verbose = verbose
        self.n_threads = n_threads

        makedir([temp_dir, output_dir])

        self.use_plink = use_plink
        self.bed_files = None
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.cleanup_dir_list = []  # Directories to clean up after execution.
        self.batch_size = batch_size

        # ------- General parameters -------

        self.standardize_phenotype = standardize_phenotype
        self.phenotype_likelihood = phenotype_likelihood
        self.phenotype_id = None  # Name or ID of the phenotype
        self.C = None  # Number of annotations

        # ------- LD computation options -------
        self.ld_estimator = ld_estimator

        assert self.ld_estimator in ('windowed', 'sample', 'shrinkage')

        # For the shrinkage estimator of the LD matrix:
        self.genmap_Ne = genmap_Ne
        self.genmap_sample_size = genmap_sample_size
        self.shrinkage_cutoff = shrinkage_cutoff

        if self.ld_estimator == 'shrinkage':
            assert self.genmap_Ne is not None
            assert self.genmap_sample_size is not None

        # For the windowed estimator of the LD matrix:
        self.window_unit = window_unit
        self.cm_window_cutoff = cm_window_cutoff
        self.window_size_cutoff = window_size_cutoff

        # ------- Filter data -------
        try:
            self.keep_individuals = read_individual_filter_file(keep_individuals)
        except ValueError:
            self.keep_individuals = None

        try:
            self.keep_snps = read_snp_filter_file(keep_snps)
        except ValueError:
            self.keep_snps = None

        # ------- Genotype data -------

        self.standardize_genotype = standardize_genotype
        self.genotypes = None
        self._snps = None
        self.n_per_snp = None  # Sample size per SNP
        self._a1 = None  # Minor allele
        self._a2 = None  # Major allele
        self.maf = None  # Minor allele frequency
        self.yy = None
        self.snp_var = None
        self._iid = None
        self.annotations = None

        # ------- LD-related data -------
        self.ld_boundaries = None
        self.ld = None

        # ------- Phenotype data -------
        self.phenotypes = None

        # ------- Summary statistics data -------

        self.beta_hats = None
        self.z_scores = None
        self.se = None
        self.p_values = None

        # ------- Read data files -------

        self.read_genotypes(bed_files, ld_block_files)
        self.read_annotations(annotation_files)

        # ------- Compute LD matrices -------

        if ld_store_files is not None:
            self.read_ld(ld_store_files)
        elif compute_ld:
            self.ld_subset_idx = None
            self.set_ld_subset_samples(ld_subset_idx, ld_subset_samples)
            self.compute_ld()

        # ------- Train/test assignment -------

        self.train_idx = None
        self.test_idx = None

        if self.genotypes is not None:
            self.set_training_samples(train_idx, train_samples)
            self.set_testing_samples(test_idx, test_samples)

        # ------- Read phenotype/sumstats files -------
        self.read_phenotypes(phenotype_file, phenotype_id=phenotype_id,
                             header=phenotype_header, phenotype_col=phenotype_col,
                             standardize=self.standardize_phenotype)
        self.read_summary_stats(sumstats_file)

        # ------- Harmonize data sources -------

        if ld_store_files is not None or sumstats_file is not None:
            self.harmonize_data()

    @property
    def sample_size(self):
        if self.train_idx is None:
            return self.N
        else:
            return len(self.train_idx)

    @property
    def N(self, agg='max'):
        """
        The number of samples
        :param agg: Aggregation (max, mean, or None)
        :return:
        """

        if agg == 'max':
            if self.genotypes is not None:
                return len(self._iid)
            else:
                return max([nps.max() for nps in self.n_per_snp.values()])
        else:
            if self.n_per_snp is None:
                self.compute_n_per_snp()

            if agg is None:
                return self.n_per_snp
            elif agg == 'mean':
                return np.mean([nps.mean() for nps in self.n_per_snp.values()])

    @property
    def M(self):
        return sum([len(s) for s in self.snps.values()])

    @property
    def snps(self):
        return self._snps

    @property
    def ref_alleles(self):
        return self._a2

    @property
    def alt_alleles(self):
        return self._a1

    @property
    def shapes(self):
        return {c: len(snps) for c, snps in self.snps.items()}

    @property
    def chromosomes(self):
        return list(self.shapes.keys())

    def set_training_samples(self, train_idx=None, train_samples=None):

        if train_samples is None and train_idx is None:
            self.train_idx = np.arange(self.N)
        elif train_idx is None:
            self.train_idx = self.sample_ids_to_index(train_samples)
        else:
            self.train_idx = train_idx

    def set_testing_samples(self, test_idx=None, test_samples=None):
        if test_samples is None and test_idx is None:
            self.test_idx = np.arange(self.N)
        elif test_idx is None:
            self.test_idx = self.sample_ids_to_index(test_samples)
        else:
            self.test_idx = test_idx

    def set_ld_subset_samples(self, ld_sample_idx=None, ld_samples=None):
        if ld_samples is None and ld_sample_idx is None:
            self.ld_subset_idx = np.arange(self.N)
        elif ld_sample_idx is None:
            self.ld_subset_idx = self.sample_ids_to_index(ld_samples)
        else:
            self.ld_subset_idx = ld_sample_idx

    def sample_ids_to_index(self, ids):
        return np.where(np.isin(self._iid, ids))[0]

    def sample_index_to_ids(self, idx):
        return self._iid[idx]

    def filter_snps(self, keep_snps, chrom=None):
        """
        :param keep_snps:
        :param chrom:
        :return:
        """

        if chrom is None:
            snp_dict = self._snps
        else:
            snp_dict = {chrom: self._snps[chrom]}

        for c, snps in snp_dict.items():

            if np.array_equal(snps, keep_snps):
                continue

            common_idx = intersect_arrays(snps, keep_snps, return_index=True)

            if self.genotypes is not None:
                self.genotypes[c] = self.genotypes[c].isel(variant=common_idx)

            if self.n_per_snp is not None:
                self.n_per_snp[c] = self.n_per_snp[c][common_idx]

            self._snps[c] = self._snps[c][common_idx]
            self._a1[c] = self._a1[c][common_idx]
            self._a2[c] = self._a2[c][common_idx]

    def filter_samples(self, keep_samples):

        common_samples = intersect_arrays(self._iid, keep_samples)

        for c, gt in self.genotypes.items():
            self.genotypes[c] = gt.sel(sample=common_samples)
            self._iid = self.genotypes[c].sample.values

    def read_annotations(self, annot_files):
        """
        Read the annotation files
        :return:
        """

        if annot_files is None:
            return

        if self.verbose:
            print("> Reading annotation files...")

        if not iterable(annot_files):
            files_to_read = [annot_files]
        else:
            files_to_read = annot_files

        assert len(files_to_read) == len(self.genotypes)

        self.annotations = []

        for i, annot_file in enumerate(files_to_read):

            try:
                annot_df = pd.read_csv(annot_file, sep="\t")
            except Exception as e:
                self.annotations = None
                raise e

            annot_df = annot_df.set_index('SNP')
            annot_df = annot_df.drop(['CHR', 'BP', 'CM', 'base'], axis=1)
            annot_df = annot_df.loc[self.genotypes[i].variant.snp]

            if i == 0:
                self.C = len(annot_df.columns)

            self.annotations.append(da.array(annot_df.values))

    def read_genotypes(self, bed_files, ld_block_files):
        """
        Read the genotype files
        :return:
        """

        if bed_files is None:
            return

        if self.verbose:
            print("> Reading genotype files...")

        if not iterable(bed_files):
            bed_files = get_filenames(bed_files, extension='.bed')

        if not iterable(ld_block_files):
            ld_block_files = [ld_block_files]

        self._snps = {}
        self._a1 = {}
        self._a2 = {}
        self.genotypes = {}
        self.bed_files = {}

        for i, (bfile, ldb_file) in tqdm(enumerate(zip_longest(bed_files, ld_block_files))):

            # Read plink file:
            try:
                gt_ac = read_plink1_bin(bfile + ".bed", ref="a0", verbose=False)
            except ValueError:
                gt_ac = read_plink1_bin(bfile, ref="a0", verbose=False)
            except Exception as e:
                self.genotypes = None
                self._iid = None
                raise e

            gt_ac = gt_ac.set_index(variant='snp')

            # Filter individuals:
            if self.keep_individuals is not None:
                common_samples = intersect_arrays(gt_ac.sample.values, self.keep_individuals)
                gt_ac = gt_ac.sel(sample=common_samples)

            # Filter SNPs:
            if self.keep_snps is not None:
                common_snps = intersect_arrays(gt_ac.variant.values, self.keep_snps)
                gt_ac = gt_ac.sel(variant=common_snps)

            # Obtain information about current chromosome:
            chr_id = int(gt_ac.chrom.values[0])

            # Add filename to the bedfiles dictionary:
            self.bed_files[chr_id] = bfile
            # Keep track of the SNPs:
            self._snps[chr_id] = gt_ac.variant.values
            self._a1[chr_id] = gt_ac.variant.a1.values
            self._a2[chr_id] = gt_ac.variant.a0.values

            if i == 0:
                self._iid = gt_ac.sample.values

            self.genotypes[chr_id] = gt_ac

            # TODO: Fully implement this functionality!
            # If an LD block file is provided, then read it,
            # match snps with their corresponding blocks,
            # and create a list of snp coordinates in each block:
            if ldb_file is not None:

                # Read LD block file:
                ldb_df = pd.read_csv(ldb_file, delim_whitespace=True)

                # Create a SNP dataframe with BP position:
                snp_df = pd.DataFrame({'pos': gt_ac.pos.values})

                # Assign each SNP its appropriate block ID
                snp_df['block_id'] = snp_df['pos'].apply(
                    lambda pos: ldb_df.loc[(pos >= ldb_df['start']) &
                                           (pos < ldb_df['stop'])].index[0])

                ld_blocks = []

                for b_idx in range(len(ldb_df)):
                    ld_blocks.append(
                        da.array(snp_df.loc[snp_df['block_id'] == b_idx].index.tolist())
                    )

                self.genotypes[i]['LD Blocks'] = ld_blocks

    def read_phenotypes(self, phenotype_file, header=None,
                        phenotype_col=2, standardize=True,
                        phenotype_id=None,
                        filter_na=True):

        if phenotype_file is None:
            return

        if self.verbose:
            print("> Reading phenotype files...")

        try:
            phe = pd.read_csv(phenotype_file, sep="\s+", header=header)
            phe = phe.iloc[:, [0, 1, phenotype_col]]
            phe.columns = ['FID', 'IID', 'phenotype']
            phe['IID'] = phe['IID'].astype(type(self._iid[0]))
        except Exception as e:
            raise e

        phe = pd.DataFrame({'IID': self._iid}).merge(phe)

        # Filter individuals with missing phenotypes:
        # TODO: Add functionality to filter on other values (e.g. -9)
        if filter_na:
            phe = phe.dropna(subset=['phenotype'])
            self.filter_samples(phe['IID'].values)

        self.phenotypes = phe['phenotype'].values

        if standardize:
            self.phenotypes -= self.phenotypes.mean()
            self.phenotypes /= self.phenotypes.std()

        if phenotype_id is None:
            self.phenotype_id = str(np.random.randint(1, 1000))
        else:
            self.phenotype_id = phenotype_id

    def read_summary_stats(self, sumstats_file, ss_format='pystatgen'):
        """
        TODO: implement parsers for summary statistics
        """

        if sumstats_file is None:
            return

        ss = pd.read_csv(sumstats_file, sep="\s+")

        if ss_format == 'LDSC':
            # Useful here: https://www.biostars.org/p/319584/
            pass
        elif ss_format == 'SBayesR':
            pass
        elif ss_format == 'plink':
            ss['SNP'] = ss['ID']
            ss['PVAL'] = ss['P']
            ss['Z'] = ss['BETA'] / ss['SE']

        self.n_per_snp = {}
        self.beta_hats = {}
        self.z_scores = {}
        self.se = {}
        self.p_values = {}

        if self.snps is None:

            self._snps = {}
            self._a1 = {}
            self._a2 = {}

            for c in ss['CHR'].unique():

                m_ss = ss.loc[ss['CHR'] == c].sort_values('POS')

                self._snps[c] = m_ss['SNP'].values
                self._a1[c] = m_ss['A1'].values
                self._a2[c] = m_ss['A2'].values
                # Populate the sumstats fields:
                self.n_per_snp[c] = m_ss['N'].values
                self.beta_hats[c] = m_ss['BETA'].values
                self.z_scores[c] = m_ss['Z'].values
                self.se[c] = m_ss['SE'].values
                self.p_values[c] = m_ss['PVAL'].values

        else:
            for c, snps in self.snps.items():
                m_ss = pd.DataFrame({'SNP': snps}).merge(ss)

                if len(m_ss) > 1:
                    # Populate the sumstats fields:
                    self.n_per_snp[c] = m_ss['N'].values
                    self.beta_hats[c] = m_ss['BETA'].values
                    self.z_scores[c] = m_ss['Z'].values
                    self.se[c] = m_ss['SE'].values
                    self.p_values[c] = m_ss['PVAL'].values

                    if len(snps) != len(m_ss):
                        self.filter_snps(m_ss['SNP'], chrom=c)

    def read_ld(self, ld_store_files):

        """
        :param ld_store_files:
        :return:
        """

        if self.verbose:
            print("> Reading LD matrices...")

        if not iterable(ld_store_files):
            ld_store_files = get_filenames(ld_store_files, extension='.zarr')

        self.ld = {}

        for f in ld_store_files:

            z = LDWrapper(zarr.open(f))
            self.ld[z.chromosome] = z

    def load_ld(self):
        if self.ld is not None:
            for ld in self.ld.values():
                ld.load()

    def release_ld(self):
        if self.ld is not None:
            for ld in self.ld.values():
                ld.release()

    def compute_ld_boundaries(self, recompute=False):

        self.ld_boundaries = {}

        if self.verbose:
            if recompute:
                print("> Re-computing LD boundaries...")
            else:
                print("> Computing LD boundaries...")

        if recompute:
            # If recomputing from existing LD matrices:
            shapes = self.shapes

            for c, ld in self.ld.items():

                common_idx = intersect_arrays(ld.snps, self.snps[c], return_index=True)
                M = shapes[c]
                estimator = ld.ld_estimator
                est_properties = ld.estimator_properties

                if estimator == 'sample':
                    self.ld_boundaries[c] = np.array((np.zeros(shapes[c]),
                                                      np.ones(shapes[c])*shapes[c])).astype(np.int64)
                elif estimator == 'windowed':
                    if est_properties['Window units'] == 'cM':
                        self.ld_boundaries[c] = find_windowed_ld_boundaries(ld.position()[common_idx],
                                                                            est_properties['Window cutoff'],
                                                                            self.n_threads)
                    else:
                        idx = np.arange(M)
                        self.ld_boundaries[c] = np.array((idx - est_properties['Window cutoff'],
                                                          idx + est_properties['Window cutoff'])).astype(np.int64)
                        self.ld_boundaries[c] = np.clip(self.ld_boundaries[c],
                                                        0, M)
                else:
                    self.ld_boundaries[c] = find_shrinkage_ld_boundaries(ld.position()[common_idx],
                                                                         est_properties['Genetic map Ne'],
                                                                         est_properties['Genetic map sample size'],
                                                                         est_properties['Cutoff'],
                                                                         self.n_threads)

        else:

            for c, gt in tqdm(self.genotypes.items()):

                _, M = gt.shape

                if self.ld_estimator == 'sample':

                    self.ld_boundaries[c] = np.array((np.zeros(M), np.ones(M)*M)).astype(np.int64)

                elif self.ld_estimator == 'windowed':
                    if self.window_unit == 'cM':
                        cm_dist = gt.cm.values
                        if cm_dist.any():
                            self.ld_boundaries[c] = find_windowed_ld_boundaries(cm_dist,
                                                                                self.cm_window_cutoff,
                                                                                self.n_threads)
                        else:
                            raise Exception("cM information for SNPs is missing."
                                            "Make sure to populate it with a reference genetic map "
                                            "or use a pre-specified window size around each SNP.")
                    else:

                        idx = np.arange(M)
                        self.ld_boundaries[c] = np.array((idx - self.window_size_cutoff,
                                                          idx + self.window_size_cutoff)).astype(np.int64)
                        self.ld_boundaries[c] = np.clip(self.ld_boundaries[c],
                                                        0, M)
                elif self.ld_estimator == 'shrinkage':
                    cm_dist = gt.cm.values
                    if cm_dist.any():
                        self.ld_boundaries[c] = find_shrinkage_ld_boundaries(cm_dist,
                                                                             self.genmap_Ne,
                                                                             self.genmap_sample_size,
                                                                             self.shrinkage_cutoff,
                                                                             self.n_threads)
                    else:
                        raise Exception("cM information for SNPs is missing."
                                        "Make sure to populate it with a reference genetic map "
                                        "or use a different LD estimator.")

        return self.ld_boundaries

    def compute_ld(self):

        if self.maf is None:
            self.compute_allele_frequency()

        if self.ld_boundaries is None:
            self.compute_ld_boundaries()

        if self.verbose:
            print("> Computing LD matrices...")

        tmp_ld_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='ld_')
        self.cleanup_dir_list.append(tmp_ld_dir)

        self.ld = {}

        for c, g_data in tqdm(self.genotypes.items()):

            tmp_ld_store = osp.join(tmp_ld_dir.name, 'chr_' + str(c))
            fin_ld_store = osp.join(self.output_dir, 'ld', 'chr_' + str(c))

            g_mat = g_data[self.ld_subset_idx, :]

            # Re-chunk the array
            g_mat = g_mat.chunk((min(1024, g_mat.shape[0]),
                                 min(1024, g_mat.shape[1])))

            g_mat = standardize_genotype_matrix(g_mat)
            ld_mat = (da.dot(g_mat.T, g_mat) / self.N).astype(np.float64)
            ld_mat.to_zarr(tmp_ld_store, overwrite=True)

            z_ld_mat = zarr.open(tmp_ld_store)
            z_ld_mat = rechunk_zarr(z_ld_mat,
                                    ld_mat.rechunk({0: 'auto', 1: None}).chunksize,
                                    tmp_ld_store + '_rechunked',
                                    tmp_ld_store + '_intermediate')

            # Add LD matrix properties:
            z_ld_mat.attrs['Chromosome'] = c
            z_ld_mat.attrs['Sample size'] = g_mat.shape[0]
            z_ld_mat.attrs['SNP'] = list(g_mat.variant.values)
            z_ld_mat.attrs['LD estimator'] = self.ld_estimator
            z_ld_mat.attrs['LD boundaries'] = self.ld_boundaries[c].tolist()

            ld_estimator_properties = None

            if self.ld_estimator == 'sample':
                final_sample_store = zarr.open(fin_ld_store)
                zarr.convenience.copy_store(z_ld_mat, final_sample_store, if_exists='replace')
                z_ld_mat = final_sample_store

            if self.ld_estimator == 'shrinkage':

                z_ld_mat = shrink_ld_matrix(z_ld_mat,
                                            g_data.cm.values,
                                            self.genmap_Ne,
                                            self.genmap_sample_size,
                                            self.shrinkage_cutoff)

                ld_estimator_properties = {
                    'Genetic map Ne': self.genmap_Ne,
                    'Genetic map sample size': self.genmap_sample_size,
                    'Cutoff': self.shrinkage_cutoff
                }

            elif self.ld_estimator == 'windowed':
                z_ld_mat = sparsify_chunked_matrix(z_ld_mat, self.ld_boundaries[c])

                ld_estimator_properties = {
                    'Window units': self.window_unit,
                    'Window cutoff': [self.window_size_cutoff, self.cm_window_cutoff][self.window_unit == 'cM']
                }

            if self.ld_estimator in ('shrinkage', 'windowed'):
                z_ld_mat = zarr_array_to_ragged(z_ld_mat,
                                                fin_ld_store,
                                                bounds=self.ld_boundaries[c],
                                                delete_original=True)

            # Add detailed LD matrix properties:
            z_ld_mat.attrs['BP'] = list(map(int, g_mat.variant.pos.values))
            z_ld_mat.attrs['cM'] = list(map(float, g_mat.variant.cm.values))
            z_ld_mat.attrs['MAF'] = list(map(float, self.maf[c]))

            if ld_estimator_properties is not None:
                z_ld_mat.attrs['Estimator properties'] = ld_estimator_properties

            self.ld[c] = LDWrapper(z_ld_mat)

    def get_ld_matrices(self):
        return self.ld

    def get_ld_boundaries(self):
        if self.ld is None:
            return None

        return {c: ld.ld_boundaries for c, ld in self.ld.items()}

    def realign_ld(self):
        """
        This method realigns a pre-computed LD matrix with the
        current genotype matrix and/or summary statistics.
        :return:
        """

        if self.ld is None:
            raise Exception("No pre-computed LD matrices are provided.")

        self.compute_ld_boundaries(recompute=True)

        if self.verbose:
            print("> Aligning pre-computed LD matrices with summary statistics and/or genotypes.")

        ld_tmpdir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='ld_')
        self.cleanup_dir_list.append(ld_tmpdir)

        for c, snps in self.snps.items():
            ld_snps = self.ld[c].snps
            if not np.array_equal(snps, ld_snps):
                self.ld[c] = LDWrapper(
                    zarr_array_to_ragged(self.ld[c].zarr_array,
                                         dir_store=osp.join(ld_tmpdir.name, f'chr_{c}'),
                                         keep_snps=snps,
                                         bounds=self.ld_boundaries[c])
                )

    def harmonize_data(self):
        """
        This method ensures that all the data sources (reference genotype,
        LD matrices, summary statistics) are aligned.
        :return:
        """

        if self.verbose:
            print("> Harmonizing data...")

        update_ld = False

        for c, snps in self.snps.items():
            # Harmonize SNPs in LD store and genotype matrix:
            if self.ld is not None:
                ld_snps = self.ld[c].snps

                if not np.array_equal(snps, ld_snps):
                    self.filter_snps(ld_snps, chrom=c)
                    update_ld = True

        if update_ld:
            self.realign_ld()

    def predict_plink(self, betas=None):
        """
        Perform linear scoring using PLINK2
        :param betas:
        :return:
        """

        if betas is None:
            if self.beta_hats is None:
                raise Exception("Neither betas nor beta hats are provided or set."
                                " Please provide betas to perform prediction.")
            else:
                betas = self.beta_hats

        if self.verbose:
            print("> Generating polygenic scores using PLINK...")

        pgs = np.zeros(shape=self.N)

        # Create a temporary directory for the score files:
        score_tmpdir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='score_')

        # Create the samples file:
        keep_file = osp.join(score_tmpdir.name, 'samples.keep')
        keep_table = self.to_individual_table()
        keep_table.to_csv(keep_file, index=False, header=False, sep="\t")

        for c, beta in betas.items():

            eff_file = osp.join(score_tmpdir.name, f'chr_{c}.txt')
            df = pd.DataFrame({'SNP': self.snps[c], 'A2': self.ref_alleles[c], 'BETA': betas[c]})

            try:
                df.to_csv(eff_file, index=False, sep="\t")

                cmd = [
                    "plink2",
                    f"--bfile {self.bed_files[c].replace('.bed', '')}",
                    f"--keep {keep_file}",
                    f"--score {eff_file} 1 2 3 header-read cols=+scoresums variance-standardize",
                    f"--out {eff_file.replace('.txt', '')}"
                ]
                run_shell_script(" ".join(cmd))

                chr_pgs = pd.read_csv(eff_file.replace('.txt', '.sscore'), sep='\s+',
                                      names=['FID', 'IID', 'NMISS_ALLELE_CT',
                                             'NAMED_ALLELE_DOSAGE_SUM', 'BETA_AVG', 'BETA_SUM'])
                # To ensure PRS is ordered correctly, we merge:
                str_dtype = {'FID': str, 'IID': str}
                chr_pgs = keep_table.astype(str_dtype).merge(chr_pgs.astype(str_dtype))

                pgs += chr_pgs['BETA_SUM'].values

            except Exception as e:
                raise e

        score_tmpdir.cleanup()
        return pgs

    def predict(self, betas=None):

        if betas is None:
            if self.beta_hats is None:
                raise Exception("Neither betas nor beta hats are provided or set."
                                " Please provide betas to perform prediction.")
            else:
                betas = self.beta_hats

        if self.verbose:
            print("> Generating polygenic scores...")

        pgs = np.zeros(shape=self.N)

        for c, gt in self.genotypes.items():

            if self.standardize_genotype:
                pgs += da.dot(standardize_genotype_matrix(gt), betas[c]).compute()
            else:
                pgs += da.dot(gt, betas[c]).compute()

        return pgs

    def perform_gwas_plink(self):

        if self.verbose:
            print("> Performing GWAS using PLINK...")

        # Create a temporary directory for the gwas files:
        gwas_tmpdir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='gwas_')

        # Output the phenotype file:
        phe_fname = osp.join(gwas_tmpdir.name, "pheno.txt")
        phe_table = self.to_phenotype_table()
        phe_table.to_csv(phe_fname, sep="\t", index=False, header=False)

        plink_reg_type = ['linear', 'logistic'][self.phenotype_likelihood == 'binomial']

        self.n_per_snp = {}
        self.beta_hats = {}
        self.se = {}
        self.z_scores = {}
        self.p_values = {}

        for c, bf in self.bed_files.items():

            plink_output = osp.join(gwas_tmpdir.name, f"chr_{c}")

            cmd = [
                "plink2",
                f"--bfile {bf.replace('.bed', '')}",
                f"--{plink_reg_type} hide-covar allow-no-covars",
                f"--pheno {phe_fname}",
                f"--out {plink_output}"
            ]

            if self.standardize_phenotype:
                cmd.append('--covar-variance-standardize')

            run_shell_script(" ".join(cmd))
            res = pd.read_csv(plink_output + f".PHENO1.glm.{plink_reg_type}", sep="\s+")
            # Merge to make sure that summary statistics are in order:
            res = pd.DataFrame({'ID': self.snps[c]}).merge(res)

            self.n_per_snp[c] = res['OBS_CT'].values
            self.beta_hats[c] = res['BETA'].values
            self.se[c] = res['SE'].values
            self.z_scores[c] = self.beta_hats[c] / self.se[c]
            self.p_values[c] = res['P'].values

        gwas_tmpdir.cleanup()

    def perform_gwas(self):

        if self.verbose:
            print("> Performing GWAS...")

        if self.use_plink:
            self.perform_gwas_plink()
        else:

            if self.n_per_snp is None:
                self.compute_n_per_snp()

            self.compute_beta_hats()
            self.compute_standard_errors()
            self.compute_z_scores()
            self.compute_p_values()

    def compute_allele_frequency(self):

        if self.n_per_snp is None:
            self.compute_n_per_snp()

        if self.verbose:
            print(">> Computing allele frequencies...")

        self.maf = {}
        for c, gt in self.genotypes.items():
            self.maf[c] = (gt.sum(axis=0) / (2. * self.n_per_snp[c])).compute().values

        return self.maf

    def compute_allele_frequency_variance(self):

        if self.maf is None:
            self.compute_allele_frequency()

        if self.verbose:
            print(">> Computing allele frequency variance for each SNP...")

        self.snp_var = {}
        for c, gt in self.genotypes.items():
            self.snp_var[c] = 2.*self.maf[c]*(1. - self.maf[c])

        return self.snp_var

    def compute_n_per_snp(self):
        self.n_per_snp = {}

        if self.verbose:
            print(">> Computing effective sample size per SNP...")

        for c, gt in self.genotypes.items():
            self.n_per_snp[c] = gt.shape[0] - gt.isnull().sum(axis=0).compute().values

        return self.n_per_snp

    def compute_yy_per_snp(self):
        """
        Computes (yTy)j following SBayesR and Yang et al. (2012)
        :return:
        """

        if self.verbose:
            print(">> Computing yTy per SNP...")

        self.yy = {c: (self.n_per_snp[c] - 2)*self.se[c]**2 + b_hat**2
                   for c, b_hat in self.beta_hats.items()}
        return self.yy

    def compute_beta_hats(self):

        self.beta_hats = {}

        if self.verbose:
            print(">> Computing beta hats...")

        for c, gt in self.genotypes.items():

            if self.standardize_genotype:
                numer = da.dot(standardize_genotype_matrix(gt).T, self.phenotypes)
                denom = self.n_per_snp[c]
            else:
                numer = da.dot(gt.T, self.phenotypes)
                denom = self.n_per_snp[c] * gt.var(axis=0)

            self.beta_hats[c] = (numer / denom).compute()

        return self.beta_hats

    def compute_standard_errors(self):

        if self.verbose:
            print(">> Computing standard errors...")

        self.se = {}

        sigma_y = np.var(self.phenotypes)

        for c, gt in self.genotypes.items():

            if self.standardize_genotype:
                xtx = self.n_per_snp[c]
            else:
                xtx = self.n_per_snp[c]*gt.var(axis=0).compute()

            self.se[c] = np.sqrt(sigma_y/xtx)

        return self.se

    def compute_z_scores(self):

        if self.verbose:
            print(">> Computing z-scores...")

        self.z_scores = {c: b_hat / self.se[c]
                         for c, b_hat in self.beta_hats.items()}
        return self.z_scores

    def compute_p_values(self, log10=False):

        if self.verbose:
            print(">> Computing p-values...")

        self.p_values = {c: 2.*stats.norm.sf(abs(z_sc))
                         for c, z_sc in self.z_scores.items()}

        if log10:
            self.p_values = {c: np.log10(pval) for c, pval in self.p_values.items()}

        return self.p_values

    def to_individual_table(self):

        if self.genotypes is None:
            raise Exception("Individual data is not provided!")

        genotype_data = next(iter(self.genotypes.values()))

        return pd.DataFrame({
            'FID': genotype_data.sample.fid.values,
            'IID': genotype_data.sample.iid.values
        })

    def to_phenotype_table(self):

        if self.phenotypes is None:
            print("Warning: Phenotypes are not set! Exporting NaNs")

        pheno_df = self.to_individual_table()
        pheno_df['phenotype'] = self.phenotypes

        return pheno_df

    def to_sumstats_table(self, per_chromosome=False):

        ss_tables = {}

        if self.genotypes is None:
            raise Exception('Cannot generate summary statistics without genotype data.')

        if self.maf is None:
            self.compute_allele_frequency()

        for c, gt in self.genotypes.items():
            ss_df = pd.DataFrame({
                'CHR': gt.chrom.values,
                'POS': gt.pos.values,
                'SNP': self.snps[c],
                'A1': self.alt_alleles[c],
                'A2': self.ref_alleles[c],
                'MAF': self.maf[c],
                'N': self.n_per_snp[c],
                'BETA': self.beta_hats[c],
                'Z': self.z_scores[c],
                'SE': self.se[c],
                'PVAL': self.p_values[c]
            })

            ss_tables[c] = ss_df

        if per_chromosome:
            return ss_tables
        else:
            return pd.concat(list(ss_tables.values()))

    def cleanup(self):
        """
        Clean up all temporary files and directories
        :return:
        """
        if self.verbose:
            print("> Cleaning up workspace.")

        for tmpdir in self.cleanup_dir_list:
            tmpdir.cleanup()

