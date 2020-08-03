import numpy as np
import pandas as pd
import dask.array as da
from pandas_plink import read_plink1_bin
from itertools import zip_longest

from scipy import sparse
from sksparse.cholmod import cholesky as sparse_cholesky
from sksparse.cholmod import CholmodNotPositiveDefiniteError

from utils import iterable


class GWASDataLoader(object):

    def __init__(self, bed_files,
                 ld_block_files=None,
                 annotation_files=None,
                 phenotype_file=None,
                 sumstats_file=None,
                 keep_individuals=None,
                 keep_snps=None,
                 regularize_ld=True,
                 max_cm_dist=1.,
                 lam=0.1,
                 batch_size=200,
                 output_dir=None):

        self.output_dir = output_dir
        self.batch_size = batch_size

        self.N = None  # Number of individuals
        self.M = None  # Total number of SNPs
        self.C = None  # Number of annotations

        # ------- LD regularization options -------
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

        self.ld = None
        self.ld_cholesky_factors = None

        # ------- Phenotype data -------
        self.phenotypes = None

        # ------- Summary statistics data -------

        self.beta_hats = None
        self.z_scores = None

        # ------- Read data files -------
        self.read_genotypes(bed_files, ld_block_files)
        self.read_annotations(annotation_files)
        self.read_phenotypes(phenotype_file)
        self.read_summary_stats(sumstats_file)
        self.compute_ld()

    @staticmethod
    def read_filter_files(file):

        if file is None:
            return

        try:
            keep_list = pd.read_csv(file, sep="\t").values[:, 0]
        except Exception as e:
            raise e

        return keep_list

    def read_annotations(self, annot_files):
        """
        Read the annotation files
        :return:
        """

        if annot_files is None:
            return

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

        print("> Reading genotype files...")

        if not iterable(genotype_files):
            genotype_files = [genotype_files]

        if not iterable(ld_block_files):
            ld_block_files = [ld_block_files]

        self.M = 0
        self.genotypes = {}

        for i, (bfile, ldb_file) in enumerate(zip_longest(genotype_files, ld_block_files)):

            # Read plink file:
            try:
                gt_ac = read_plink1_bin(bfile + ".bed")
            except ValueError:
                gt_ac = read_plink1_bin(bfile)
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

    def read_phenotypes(self, phenotype_file, normalize=True):

        if phenotype_file is None:
            return

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

        self.get_beta_hat()
        self.get_z_scores()

    def read_summary_stats(self, sumstats_file):
        """
        TODO: implement reading functions for summary statistics
        """
        pass

    def compute_ld(self):

        self.ld = {}

        print("> Computing LD matrices...")

        for i, g_data in self.genotypes.items():

            if 'LD Blocks' in g_data:
                g_matrices = [g_data['G'][:, snp_block] for snp_block in g_data['LD Blocks']]

            else:
                g_matrices = [g_data['G']]

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
        self.beta_hats = {i: da.dot(gt['G'].T, self.phenotypes) / self.N
                          for i, gt in self.genotypes.items()}
        return self.beta_hats

    def get_z_scores(self):
        self.z_scores = {i: b_hat * self.N / np.sqrt(self.N)
                         for i, b_hat in self.beta_hats.items()}
        return self.z_scores

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

    def iter_individual_data(self):
        pass


class GWASSimulator(GWASDataLoader):

    def __init__(self, bed_files, h2g=0.2, pi=0.1, **kwargs):

        super().__init__(bed_files, **kwargs)

        self.h2g = h2g
        self.pi = pi

        self.annotation_weights = None

        self.betas = None
        self.pis = None

    def simulate_genotypes(self, n):

        for i, ld_fac_list in self.ld_cholesky_factors.items():

            for j, ld_fac in enumerate(ld_fac_list):

                _, p = ld_fac.shape

                ng_mat = da.array(ld_fac.dot(da.random.normal(size=(n, p)).T).T)
                ng_mat -= da.mean(ng_mat, axis=0)
                ng_mat /= da.std(ng_mat, axis=0)

                if j > 0:
                    self.genotypes[i]['G'] = da.concatenate([self.genotypes[i]['G'], ng_mat], axis=1)
                else:
                    self.genotypes[i]['G'] = ng_mat

        self.N = n

    def simulate_pi(self):

        self.pis = {}

        for i, g_data in self.genotypes.items():
            _, p = g_data['G'].shape
            self.pis[i] = da.random.binomial(1, self.pi, size=p)

        return self.pis

    def simulate_betas(self):

        self.betas = {}

        for i, g_data in self.genotypes.items():

            _, p = g_data['G'].shape

            if self.annotation_weights is not None:
                std_beta = da.sqrt(da.absolute(da.dot(self.annotations[i], self.annotation_weights)))
            else:
                std_beta = 1.

            betas = da.random.normal(loc=0.0, scale=std_beta, size=p)*self.pis[i]

            self.betas[i] = betas

    def simulate_annotation_weights(self):
        if self.C is not None:
            self.annotation_weights = da.random.normal(scale=1./self.M, size=self.C)

    def simulate_phenotypes(self):

        g_comp = da.zeros(shape=self.N)

        for chrom_id in self.genotypes:
            g_comp += da.dot(self.genotypes[chrom_id]['G'], self.betas[chrom_id])

        g_var = np.var(g_comp, ddof=1)
        e_var = g_var * ((1.0 / self.h2g) - 1.0)

        e = da.random.normal(0, np.sqrt(e_var), self.N)

        y = g_comp + e
        y -= y.mean()
        y /= y.std()

        self.phenotypes = y

        return self.phenotypes

    def simulate(self, n=None, reset_beta=False):

        if n is not None:
            if self.ld_cholesky_factors is None:
                self.compute_cholesky_factors()
            self.simulate_genotypes(n)

        if self.betas is None or reset_beta:
            self.simulate_pi()
            self.simulate_annotation_weights()
            self.simulate_betas()

        self.simulate_phenotypes()
        self.get_beta_hat()
        self.get_z_scores()
