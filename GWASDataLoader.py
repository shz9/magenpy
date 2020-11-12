
import copy
import numpy as np
import pandas as pd
import dask.array as da

from pandas_plink import read_plink1_bin
from itertools import zip_longest

from scipy import sparse, stats
from sksparse.cholmod import cholesky as sparse_cholesky
from sksparse.cholmod import CholmodNotPositiveDefiniteError

from .utils import iterable


class GWASDataLoader(object):

    def __init__(self, bed_files,
                 ld_block_files=None,
                 annotation_files=None,
                 phenotype_file=None,
                 sumstats_file=None,
                 keep_individuals=None,
                 keep_snps=None,
                 train_samples=None,
                 train_idx=None,
                 test_samples=None,
                 test_idx=None,
                 ld_subset_samples=None,
                 ld_subset_idx=None,
                 regularize_ld=True,
                 max_cm_dist=1.,
                 lam=0.1,
                 batch_size=200,
                 phenotype_id=None,
                 output_dir=None,
                 verbose=True):

        # ------- General options -------
        self.verbose = verbose

        if not iterable(bed_files):
            bed_files = [bed_files]

        self.bed_files = bed_files
        self.output_dir = output_dir
        self.batch_size = batch_size

        # ------- General parameters -------

        self.phenotype_id = None  # Name or ID of the phenotype
        self.N = None  # Number of individuals
        self.M = None  # Total number of SNPs
        self.C = None  # Number of annotations

        # ------- LD computation options -------
        self.regularize_ld = regularize_ld
        self.max_cm_dist = max_cm_dist
        self.lam = lam

        # ------- Filter data -------
        self.keep_individuals = self.read_filter_files(keep_individuals)
        self.keep_snps = self.read_filter_files(keep_snps)

        # ------- Genotype data -------

        self.genotypes = None
        self.sample_ids = None
        self.annotations = None

        # ------- LD-related data -------
        self.ld = None
        self.ld_cholesky_factors = None

        # ------- Phenotype data -------
        self.phenotypes = None

        # ------- Summary statistics data -------

        self.beta_hats = None
        self.z_scores = None
        self.p_values = None

        # ------- Read data files -------
        self.read_genotypes(bed_files, ld_block_files)
        self.read_annotations(annotation_files)
        self.read_phenotypes(phenotype_file, phenotype_id=phenotype_id)
        self.read_summary_stats(sumstats_file)

        # ------- Compute LD matrices -------

        self.ld_subset_idx = None
        self.set_ld_subset_samples(ld_subset_idx, ld_subset_samples)
        self.compute_ld()

        # ------- Train/test assignment -------

        self.train_idx = None
        self.test_idx = None

        self.set_training_samples(train_idx, train_samples)
        self.set_testing_samples(test_idx, test_samples)

    @staticmethod
    def read_filter_files(file):

        if file is None:
            return

        try:
            keep_list = pd.read_csv(file, sep="\t").values[:, 0]
        except Exception as e:
            raise e

        return keep_list

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
        return np.where(np.isin(self.sample_ids, ids))[0]

    def sample_index_to_ids(self, idx):
        return self.sample_ids[idx]

    def compute_summary_statistics(self):

        self.get_beta_hat()
        self.get_z_scores()
        self.get_p_values()

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
            annot_df = annot_df.loc[self.genotypes[i]['G'].variant.snp]

            if i == 0:
                self.C = len(annot_df.columns)

            self.annotations.append(da.array(annot_df.values))

    def read_genotypes(self, genotype_files, ld_block_files, normalize=True):
        """
        Read the genotype files
        :return:
        """

        if self.verbose:
            print("> Reading genotype files...")

        if not iterable(ld_block_files):
            ld_block_files = [ld_block_files]

        self.M = 0
        self.genotypes = {}

        for i, (bfile, ldb_file) in enumerate(zip_longest(genotype_files, ld_block_files)):

            # Read plink file:
            try:
                gt_ac = read_plink1_bin(bfile + ".bed", verbose=False)
            except ValueError:
                gt_ac = read_plink1_bin(bfile, verbose=False)
            except Exception as e:
                self.genotypes = None
                self.sample_ids = None
                raise e

            # Filter individuals:
            if self.keep_individuals is not None:
                gt_ac = gt_ac[pd.Series(gt_ac.sample).isin(self.keep_individuals), :]

            # Filter SNPs:
            if self.keep_snps is not None:
                gt_ac = gt_ac[:, pd.Series(gt_ac.variant.snp).isin(self.keep_snps)]

            maf = gt_ac.sum(axis=0) / (2. * gt_ac.shape[0])
            maf = np.round(np.where(maf > .5, 1. - maf, maf), 6)
            gt_ac = gt_ac.assign_coords({"MAF": ("variant", maf)})

            # Normalize genotype matrix:
            if normalize:
                gt_ac = (gt_ac - gt_ac.mean(axis=0)) / gt_ac.std(axis=0)

            # Obtain information about current chromosome:
            chr_id, (chr_n, chr_p) = gt_ac.chrom.values[0], gt_ac.shape

            if i == 0:
                self.N = chr_n
                self.sample_ids = gt_ac.sample.values

            self.M += chr_p

            self.genotypes[i] = {
                'CHR': chr_id,
                'G': gt_ac
            }

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

    def read_phenotypes(self, phenotype_file, normalize=True, phenotype_id=None):

        if phenotype_file is None:
            return

        if self.verbose:
            print("> Reading phenotype files...")

        try:
            phe = pd.read_csv(phenotype_file)
            phe.set_index('ID', inplace=True)
        except Exception as e:
            raise e

        self.phenotypes = phe.loc[self.sample_ids, 0].values

        if normalize:
            self.phenotypes -= self.phenotypes.mean()
            self.phenotypes /= self.phenotypes.std()

        if phenotype_id is None:
            self.phenotype_id = str(np.random.randint(1, 1000))
        else:
            self.phenotype_id = phenotype_id

        self.compute_summary_statistics()

    def read_summary_stats(self, sumstats_file):
        """
        TODO: implement reading functions for summary statistics
        """
        pass

    def compute_ld(self):

        self.ld = {}

        if self.verbose:
            print("> Computing LD matrices...")

        for i, g_data in self.genotypes.items():

            if 'LD Blocks' in g_data:
                g_matrices = [g_data['G'][self.ld_subset_idx, snp_block] for snp_block in g_data['LD Blocks']]

            else:
                g_matrices = [g_data['G'][self.ld_subset_idx, :]]

            self.ld[i] = []

            for g_mat in g_matrices:

                ld_mat = da.dot(g_mat.T, g_mat) / self.N + da.diag(da.ones(g_mat.shape[1]))*self.lam

                if self.regularize_ld:
                    cm_dist = g_mat.cm.values
                    dist_mat = da.absolute(da.subtract.outer(cm_dist, cm_dist))

                    dist_mat[dist_mat > self.max_cm_dist] = 0.
                    dist_mat[dist_mat > 0.] = 1.
                    dist_mat = dist_mat + da.diag(da.ones(dist_mat.shape[0]))

                    ld_mat *= dist_mat

                ld_mat = sparse.csc_matrix(ld_mat)
                self.ld[i].append(ld_mat)

    def compute_cholesky_factors(self):

        self.ld_cholesky_factors = {}

        for i, ld_matrices in self.ld.items():

            self.ld_cholesky_factors[i] = []

            for j, ld_mat in enumerate(ld_matrices):
                try:
                    self.ld_cholesky_factors[i].append(sparse_cholesky(ld_mat).L())
                except CholmodNotPositiveDefiniteError:
                    self.ld_cholesky_factors = None
                    raise Exception(f"The LD matrix in chromosome {i} in block {j} is not positive definite. "
                                    f"Consider increasing the regularization parameter from `lam={self.lam}`.")

    def update_ld_regularization(self, new_lam):

        for i, ld_matrices in self.ld.items():

            for j, ld_mat in enumerate(ld_matrices):
                self.ld[i][j] += sparse.diags(np.ones(ld_mat.shape[1])) * (new_lam - self.lam)

        self.lam = new_lam

    def get_beta_hat(self):

        self.beta_hats = {i: pd.Series((da.dot(gt['G'][self.train_idx, :].T,
                                               self.phenotypes[self.train_idx]) / self.N).compute(),
                                       index=gt['G'].variant.snp.values)
                          for i, gt in self.genotypes.items()}
        return self.beta_hats

    def get_z_scores(self):
        self.z_scores = {i: b_hat * np.sqrt(self.N)
                         for i, b_hat in self.beta_hats.items()}
        return self.z_scores

    def get_p_values(self, log10=False):
        self.p_values = {i: pd.Series(2.*stats.norm.sf(abs(z_sc)),
                                      index=z_sc.index)
                         for i, z_sc in self.z_scores.items()}

        if log10:
            self.p_values = {i: np.log10(pval) for i, pval in self.p_values.items()}

        return self.p_values

    def iter_sumstats(self, w_annots=False):

        assert self.beta_hats is not None

        for gidx in range(len(self.beta_hats)):
            bh = self.beta_hats[gidx]
            for bidx in range(0, len(bh), self.batch_size):
                if w_annots and self.annotations is not None:
                    yield (self.annotations[gidx][bidx:bidx + self.batch_size, :].compute(),
                           bh[bidx:bidx + self.batch_size].compute())
                else:
                    yield bh[bidx:min(bidx + self.batch_size, len(bh))].compute()

    def to_sumstats_table(self, per_chromosome=False):

        ss_tables = []

        beta_hat = self.get_beta_hat()
        z_score = self.get_z_scores()
        pval = self.get_p_values()

        for k, v in self.genotypes.items():
            ss_df = pd.DataFrame({
                'CHR': v['G'].chrom.values,
                'SNP': v['G'].snp.values,
                'BETA': beta_hat[k],
                'Z': z_score[k],
                'PVAL': pval[k],
                'MAF': v['G'].MAF.values,
                'A1': v['G'].a0.values,
                'A2': v['G'].a1.values
            })

            ss_df['N'] = len(self.train_idx)
            ss_df['SE'] = 1./np.sqrt(ss_df['N'])

            ss_tables.append(ss_df)

        if per_chromosome:
            return ss_tables
        else:
            return pd.concat(ss_tables)
