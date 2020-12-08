"""
Author: Shadi Zabad
Date: December 2020
"""

import os
import numpy as np
import pandas as pd
import dask.array as da
import zarr
from tqdm import tqdm

from pandas_plink import read_plink1_bin
from itertools import zip_longest

from scipy import stats

from .c_utils import find_ld_boundaries
from .utils import iterable, sparsify_chunked_matrix


class GWASDataLoader(object):

    def __init__(self, bed_files,
                 standardize_genotype=True,
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
                 compute_ld=False,
                 ld_store_files=None,
                 ld_block_files=None,
                 ld_subset_samples=None,
                 ld_subset_idx=None,
                 regularize_ld=True,
                 regularize_dist='cM',
                 max_cm_dist=1.,
                 n_snps_wind=2000,
                 lam=0.0,
                 batch_size=200,
                 temp_dir='temp',
                 output_dir=None,
                 verbose=True,
                 n_threads=1):

        # ------- General options -------
        self.verbose = verbose
        self.n_threads = n_threads

        if not iterable(bed_files):
            bed_files = [bed_files]

        self.bed_files = bed_files
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.batch_size = batch_size

        # ------- General parameters -------

        self.phenotype_id = None  # Name or ID of the phenotype
        self.N = None  # Number of individuals
        self.M = None  # Total number of SNPs
        self.C = None  # Number of annotations

        # ------- LD computation options -------
        self.regularize_ld = regularize_ld
        self.regularize_dist = regularize_dist
        self.max_cm_dist = max_cm_dist
        self.n_snps_wind = n_snps_wind
        self.lam = lam

        # ------- Filter data -------
        self.keep_individuals = self.read_filter_files(keep_individuals)
        self.keep_snps = self.read_filter_files(keep_snps)

        # ------- Genotype data -------

        self.standardized_genotype = standardize_genotype
        self.genotypes = None
        self.snp_var = None
        self.sample_ids = None
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

        self.read_genotypes(bed_files, ld_block_files, standardize=standardize_genotype)
        self.read_annotations(annotation_files)

        # ------- Compute LD matrices -------

        if compute_ld or ld_store_files is not None:
            self.ld_subset_idx = None
            self.set_ld_subset_samples(ld_subset_idx, ld_subset_samples)
            self.get_ld_matrices(ld_store_files)

        # ------- Train/test assignment -------

        self.train_idx = None
        self.test_idx = None

        self.set_training_samples(train_idx, train_samples)
        self.set_testing_samples(test_idx, test_samples)

        # ------- Read phenotype/sumstats files -------
        self.read_phenotypes(phenotype_file, phenotype_id=phenotype_id,
                             header=phenotype_header, phenotype_col=phenotype_col,
                             standardize=standardize_phenotype)
        self.read_summary_stats(sumstats_file)

    @property
    def sample_size(self):
        if self.train_idx is None:
            return self.N
        else:
            return len(self.train_idx)

    @property
    def genotype_index(self):
        if self.genotypes is None:
            return None
        else:
            return self.genotypes.keys()

    @property
    def chromosomes(self):
        if self.genotypes is None:
            return None
        else:
            return [g['CHR'] for g in self.genotypes.values()]

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

        self.get_beta_hats()
        self.get_standard_errors()
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

    def read_genotypes(self, genotype_files, ld_block_files, standardize=True):
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

        for i, (bfile, ldb_file) in tqdm(enumerate(zip_longest(genotype_files, ld_block_files)),
                                         disable=not self.verbose):

            # Read plink file:
            try:
                gt_ac = read_plink1_bin(bfile + ".bed", verbose=False)
            except ValueError:
                gt_ac = read_plink1_bin(bfile, verbose=False)
            except Exception as e:
                self.genotypes = None
                self.sample_ids = None
                raise e

            # plink-pandas assumes A1 is reference allele by default
            # In our case, A0 is reference allele, so we reverse status:
            gt_ac.data = np.abs(gt_ac.values - 2).astype(np.int64)

            # Filter individuals:
            if self.keep_individuals is not None:
                gt_ac = gt_ac[pd.Series(gt_ac.sample).isin(self.keep_individuals), :]

            # Filter SNPs:
            if self.keep_snps is not None:
                gt_ac = gt_ac[:, pd.Series(gt_ac.variant.snp).isin(self.keep_snps)]

            maf = gt_ac.sum(axis=0) / (2. * gt_ac.shape[0])
            maf = np.round(np.where(maf > .5, 1. - maf, maf), 6)
            gt_ac = gt_ac.assign_coords({"MAF": ("variant", maf)})

            # Standardize genotype matrix:
            if standardize:
                gt_ac = (gt_ac - gt_ac.mean(axis=0)) / gt_ac.std(axis=0)
                self.standardized_genotype = standardize

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

    def read_phenotypes(self, phenotype_file, header=None,
                        phenotype_col=2, standardize=True, phenotype_id=None):

        if phenotype_file is None:
            return

        if self.verbose:
            print("> Reading phenotype files...")

        try:
            phe = pd.read_csv(phenotype_file, sep="\s+", header=header)
            phe.columns = ['FID', 'IID'] + list(phe.columns[2:])
            phe['IID'] = phe['IID'].astype(type(self.sample_ids[0]))
            phe.set_index('IID', inplace=True)
        except Exception as e:
            raise e

        self.phenotypes = phe.loc[self.sample_ids, :].iloc[:, phenotype_col - 1].values

        if len(self.phenotypes) < len(self.sample_ids):
            raise Exception(f"Some samples do not have recorded phenotypes!"
                            f"Check the phenotype file: {phenotype_file}")

        if standardize:
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

    def get_ld_boundaries(self):

        if self.ld_boundaries is None:

            if self.verbose:
                print("> Computing LD boundaries...")

            self.ld_boundaries = {}

            for c, gt in tqdm(self.genotypes.items(), disable=not self.verbose):

                if self.regularize_dist == 'cM':
                    cm_dist = gt['G'].cm.values
                    if cm_dist.any():
                        self.ld_boundaries[c] = find_ld_boundaries(cm_dist,
                                                                   self.max_cm_dist,
                                                                   self.n_threads)
                    else:
                        raise Exception("cM information for SNPs is missing."
                                        "Make sure to populate it with a reference genetic map")
                else:

                    idx = np.arange(gt['G'].shape[1])
                    self.ld_boundaries[c] = np.array((idx - self.n_snps_wind,
                                                      idx + self.n_snps_wind))
                    self.ld_boundaries[c] = np.clip(self.ld_boundaries[c],
                                                    0, gt['G'].shape[1])

        return self.ld_boundaries

    def compute_ld(self):

        self.ld = {}

        if self.regularize_ld:
            self.get_ld_boundaries()

        if self.verbose:
            print("> Computing LD matrices...")

        for c, g_data in tqdm(self.genotypes.items(), disable=not self.verbose):

            ld_store = os.path.join(self.temp_dir, 'ld', str(c))

            g_mat = g_data['G'][self.ld_subset_idx, :]

            ld_mat = da.dot(g_mat.T, g_mat) / self.N
            ld_mat = ld_mat.rechunk({0: 'auto', 1: None})
            ld_mat.to_zarr(ld_store)

            ld_mat = zarr.open(ld_store)

            if self.regularize_ld:
                ld_mat = sparsify_chunked_matrix(ld_mat, self.ld_boundaries[c])

            self.ld[c] = ld_mat

    def get_ld_matrices(self, ld_store_files=None):

        if ld_store_files is None:
            self.compute_ld()
        else:

            if self.verbose:
                print("> Reading LD matrices...")

            if not iterable(ld_store_files):
                ld_store_files = [ld_store_files]

            self.ld = {}

            for c, f in zip(self.genotype_index, ld_store_files):
                self.ld[c] = zarr.open(f)

    def get_snp_variances(self):

        if self.snp_var is None:

            self.snp_var = {
                c: 2.*gt['G'].MAF.values*(1. - gt['G'].MAF.values)
                for c, gt in self.genotypes.items()
            }

        return self.snp_var

    def get_beta_hats(self):

        self.beta_hats = {c: pd.Series((da.dot(gt['G'][self.train_idx, :].T,
                                               self.phenotypes[self.train_idx]) / self.N).compute(),
                                       index=gt['G'].variant.snp.values)
                          for c, gt in self.genotypes.items()}

        return self.beta_hats

    def get_standard_errors(self):

        self.se = {}

        for c, gt in self.genotypes.items():
            # maf = gt['G'].MAF.values
            # 2.*maf*(1. - maf)*self.sample_size
            self.se[c] = 1./np.sqrt(self.sample_size)

        return self.se

    def get_z_scores(self):
        self.z_scores = {i: b_hat / self.se[i]
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

    def to_phenotype_table(self):

        if self.phenotypes is None:
            raise Exception("Phenotypes are not set and cannot be exported!")

        genotype_data = next(iter(self.genotypes.values()))['G']

        return pd.DataFrame({
            'FID': genotype_data.sample.fid.values,
            'IID': genotype_data.sample.iid.values,
            'Phenotype': self.phenotypes
        })

    def to_sumstats_table(self, per_chromosome=False):

        ss_tables = []

        beta_hat = self.get_beta_hats()
        z_score = self.get_z_scores()
        pval = self.get_p_values()
        se = self.get_standard_errors()

        for k, v in self.genotypes.items():
            ss_df = pd.DataFrame({
                'CHR': v['G'].chrom.values,
                'POS': v['G'].pos.values,
                'SNP': v['G'].snp.values,
                'BETA': beta_hat[k],
                'Z': z_score[k],
                'SE': se[k],
                'PVAL': pval[k],
                'MAF': v['G'].MAF.values,
                'A1': v['G'].a0.values,
                'A2': v['G'].a1.values
            })

            ss_df['N'] = len(self.train_idx)

            ss_tables.append(ss_df)

        if per_chromosome:
            return ss_tables
        else:
            return pd.concat(ss_tables)
