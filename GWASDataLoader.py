"""
Author: Shadi Zabad
Date: December 2020
"""

from tqdm import tqdm

from pandas_plink import read_plink1_bin
from itertools import zip_longest

from scipy import stats

from .LDWrapper import LDWrapper
from .c_utils import find_windowed_ld_boundaries, find_shrinkage_ld_boundaries
from .utils import *


class GWASDataLoader(object):

    def __init__(self, bed_files,
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
                 sparse_ld=True,
                 compute_ld=True,
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
                 output_dir=None,
                 verbose=True,
                 n_threads=1):

        # ------- General options -------
        self.verbose = verbose
        self.n_threads = n_threads

        if not iterable(bed_files):
            bed_files = get_filenames(bed_files, extension='.bed')

        makedir(temp_dir)

        self.use_plink = use_plink
        self.bed_files = None
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.batch_size = batch_size

        # ------- General parameters -------

        self.phenotype_likelihood = phenotype_likelihood
        self.phenotype_id = None  # Name or ID of the phenotype
        self.C = None  # Number of annotations

        # ------- LD computation options -------
        self.ld_estimator = ld_estimator
        self.sparse_ld = sparse_ld

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

        self.ld_subset_idx = None
        self.set_ld_subset_samples(ld_subset_idx, ld_subset_samples)

        if ld_store_files is not None:
            self.read_ld(ld_store_files)
        elif compute_ld:
            self.compute_ld()

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

        # ------- Harmonize data sources -------

        if ld_store_files is not None or sumstats_file is not None:
            self.harmonize_data()

    @classmethod
    def from_sumstats(cls, sum_stats_file, ld_store_files):

        cls.read_summary_stats(sum_stats_file)
        cls.get_ld_matrices(ld_store_files)

        return cls

    @property
    def sample_size(self):
        if self.train_idx is None:
            return self.N
        else:
            return len(self.train_idx)

    @property
    def N(self):
        return len(self.sample_ids)

    @property
    def M(self):
        return sum([len(s) for s in self.snps.values()])

    @property
    def snps(self):
        return {c: gt['G'].variant.values
                for c, gt in self.genotypes.items()}

    @property
    def genotype_index(self):
        if self.genotypes is None:
            return None
        else:
            return self.genotypes.keys()

    @property
    def shapes(self):
        return {c: gt['G'].shape[1] for c, gt in self.genotypes.items()}

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
            keep_list = pd.read_csv(file, sep="\t", header=None).values[:, 0]
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

    def filter_snps(self, keep_snps):
        """
        SNP list to keep
        :param snp_list:
        :return:
        """

        for c, gt in self.genotypes.items():
            common_snps = pd.DataFrame({'SNP': gt['G'].variant.values}).merge(
                pd.DataFrame({'SNP': keep_snps})
            )['SNP'].values
            self.genotypes[c]['G'] = gt['G'].sel(variant=common_snps)

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

        self.genotypes = {}
        self.bed_files = {}

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

            # TODO: Figure out a more efficient way to update gt_ac and select reference allele
            # plink-pandas assumes A1 is reference allele by default
            # In our case, A0 is reference allele, so we reverse status:
            gt_ac = abs(gt_ac - 2)

            gt_ac = gt_ac.set_index(variant='snp')

            # Filter individuals:
            if self.keep_individuals is not None:
                common_samples = pd.DataFrame({'Sample': gt_ac.sample.values}).merge(
                    pd.DataFrame({'Sample': self.keep_individuals})
                )['Sample'].values
                gt_ac = gt_ac.sel(sample=common_samples)

            # Filter SNPs:
            if self.keep_snps is not None:
                common_snps = pd.DataFrame({'SNP': gt_ac.variant.values}).merge(
                    pd.DataFrame({'SNP': self.keep_snps})
                )['SNP'].values
                gt_ac = gt_ac.sel(variant=common_snps)

            maf = gt_ac.sum(axis=0) / (2. * gt_ac.shape[0])
            maf = np.round(np.where(maf > .5, 1. - maf, maf), 6)
            gt_ac = gt_ac.assign_coords({"MAF": ("variant", maf)})

            # Standardize genotype matrix:
            if standardize:
                gt_ac = (gt_ac - gt_ac.mean(axis=0)) / gt_ac.std(axis=0)
                self.standardized_genotype = standardize

            # Obtain information about current chromosome:
            chr_id, (chr_n, chr_p) = int(gt_ac.chrom.values[0]), gt_ac.shape

            # Add filename to the bedfiles dictionary:
            self.bed_files[chr_id] = bfile

            if i == 0:
                self.sample_ids = gt_ac.sample.values

            # TODO: Harmonize the code given the updated keys (using chrom_id now).
            self.genotypes[chr_id] = {
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

        self.perform_gwas()

    def read_summary_stats(self, sumstats_file, ss_type='VEM'):
        """
        TODO: implement reading functions for summary statistics
        """

        if sumstats_file is None:
            return

        ss = pd.read_csv(sumstats_file, sep="\s+")

        if ss_type == 'LDSC':
            # Useful here: https://www.biostars.org/p/319584/
            pass
        elif ss_type == 'sbayesr':
            pass

        self.beta_hats = {}
        self.z_scores = {}
        self.se = {}
        self.p_values = {}

        for c, snp in self.snps.items():
            m_ss = pd.DataFrame({'SNP': snp}).reset_index().merge(ss)

            # Populate the sumstats fields:
            self.beta_hats[c] = pd.Series(m_ss['BETA'], index=m_ss['SNP'].values)
            self.z_scores[c] = pd.Series(m_ss['Z'], index=m_ss['SNP'].values)
            self.se[c] = pd.Series(m_ss['SE'], index=m_ss['SNP'].values)
            self.p_values[c] = pd.Series(m_ss['PVAL'], index=m_ss['SNP'].values)

    def read_ld(self, ld_store_files=None):

        """
        :param ld_store_files:
        :return:
        """

        if ld_store_files is None:
            self.compute_ld()
        else:

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

    def compute_ld_boundaries(self):

        self.ld_boundaries = {}

        if self.verbose:
            print("> Computing LD boundaries...")

        for c, gt in tqdm(self.genotypes.items(), disable=not self.verbose):

            _, M = gt['G'].shape

            if self.ld_estimator == 'sample':

                self.ld_boundaries[c] = np.array((np.zeros(M), np.ones(M)*M)).astype(np.int64)

            elif self.ld_estimator == 'windowed':
                if self.window_unit == 'cM':
                    cm_dist = gt['G'].cm.values
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
                cm_dist = gt['G'].cm.values
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

        self.compute_ld_boundaries()

        self.ld = {}

        if self.verbose:
            print("> Computing LD matrices...")

        for c, g_data in tqdm(self.genotypes.items(), disable=not self.verbose):

            tmp_ld_store = os.path.join(self.temp_dir, 'tmp_ld', 'chr_' + str(c))
            fin_ld_store = os.path.join(self.temp_dir, 'ld', 'chr_' + str(c))

            g_mat = g_data['G'][self.ld_subset_idx, :]

            # Chunk the array along the SNP-axis:
            g_mat = g_mat.chunk((None, min(5000, g_mat.shape[1])))

            ld_mat = (da.dot(g_mat.T, g_mat) / self.N).astype(np.float64)
            ld_mat.to_zarr(tmp_ld_store, overwrite=True)

            z_ld_mat = zarr.open(tmp_ld_store)
            z_ld_mat = rechunk_zarr(z_ld_mat,
                                    ld_mat.rechunk({0: 'auto', 1: None}).chunksize,
                                    fin_ld_store)

            z_ld_mat.attrs['Chromosome'] = c
            z_ld_mat.attrs['SNPs'] = list(g_mat.variant.values)
            z_ld_mat.attrs['LD Estimator'] = self.ld_estimator
            z_ld_mat.attrs['LD Boundaries'] = self.ld_boundaries[c].tolist()

            if self.ld_estimator == 'shrinkage':

                z_ld_mat = shrink_ld_matrix(z_ld_mat, g_data['G'].cm.values,
                                            self.genmap_Ne,
                                            self.genmap_sample_size,
                                            self.shrinkage_cutoff)

                z_ld_mat.attrs['Ne'] = self.genmap_Ne
                z_ld_mat.attrs['Sample size'] = self.genmap_sample_size
                z_ld_mat.attrs['Cutoff'] = self.shrinkage_cutoff

            elif self.ld_estimator == 'windowed':
                z_ld_mat = sparsify_chunked_matrix(z_ld_mat, self.ld_boundaries[c])

                z_ld_mat.attrs['Window units'] = self.window_unit

                if self.window_unit == 'cM':
                    z_ld_mat.attrs['Window Cutoff'] = self.cm_window_cutoff
                else:
                    z_ld_mat.attrs['Window Cutoff'] = self.window_size_cutoff

            if self.sparse_ld and self.ld_estimator in ('shrinkage', 'windowed'):
                z_ld_mat = zarr_to_ragged(z_ld_mat, bounds=self.ld_boundaries[c])

            self.ld[c] = LDWrapper(z_ld_mat)

    def get_ld_matrices(self):
        return self.ld

    def get_ld_boundaries(self):
        if self.ld is None:
            return None

        return {c: np.array(ld.ld_boundaries) for c, ld in self.ld.items()}

    def transform_ld_matrices(self, recompute_boundaries=False):

        if recompute_boundaries:
            self.compute_ld_boundaries()

        for c, snps in self.snps.items():
            print(self.ld[c].path)
            ld_snps = self.ld[c].snps
            if len(snps) != len(ld_snps) or any(snps != ld_snps):
                self.ld[c] = zarr_to_ragged(self.ld[c], keep_snps=snps, bounds=self.ld_boundaries[c])
            elif self.sparse_ld:
                self.ld[c] = zarr_to_ragged(self.ld[c], bounds=self.ld_boundaries[c])

    def get_snp_variances(self):

        if self.snp_var is None:

            self.snp_var = {
                c: 2.*gt['G'].MAF.values*(1. - gt['G'].MAF.values)
                for c, gt in self.genotypes.items()
            }

        return self.snp_var

    def harmonize_data(self):
        """
        This method ensures that all the data sources (reference genotype,
        LD matrices, summary statistics) are aligned.
        :return:
        """

        if self.verbose:
            print("> Harmonizing data...")

        sum_stats = [ss for ss in [self.beta_hats, self.se,
                                   self.z_scores, self.p_values]
                     if ss is not None]

        update_ld = False

        for c, snps in self.snps.items():

            # Harmonize with SNPs in summary statistics
            if len(sum_stats) > 0:

                snps = pd.DataFrame({'SNP': snps}).merge(
                    pd.DataFrame({'SNP': sum_stats[0][c].index})
                )['SNP'].values

            # Harmonize SNPs in LD store and genotype matrix:
            if self.ld is not None:
                ld_snps = self.ld[c].snps

                snps = pd.DataFrame({'SNP': snps}).merge(
                    pd.DataFrame({'SNP': ld_snps})
                )['SNP'].values

                if len(ld_snps) != len(snps) or any(snps != ld_snps):
                    update_ld = True

            # Filter the genotype matrix to the common subset of SNPs:
            self.genotypes[c]['G'] = self.genotypes[c]['G'].sel(variant=snps)

            for ss in sum_stats:
                ss[c] = ss[c][snps]

        self.transform_ld_matrices(recompute_boundaries=update_ld)

    def perform_gwas_plink(self):

        phe_fname = os.path.join(self.temp_dir, "tmp_pheno.txt")

        phe_table = self.to_phenotype_table()
        phe_table.to_csv(phe_fname, sep="\t", index=False, header=False)

        plink_reg_type = ['linear', 'logistic'][self.phenotype_likelihood == 'binomial']

        self.beta_hats = {}
        self.se = {}
        self.z_scores = {}
        self.p_values = {}

        for c, bf in self.bed_files.items():

            plink_output = os.path.join(self.temp_dir, f"tmp_pheno_{c}")

            cmd = ["plink",
                   f"--bfile {bf.replace('.bed', '')}",
                   f"--{plink_reg_type}",
                   "--hide-covar",
                   "--allow-no-sex",
                   f"--pheno {phe_fname}",
                   f"--out {plink_output}"
            ]

            run_shell_script(cmd)
            res = pd.read_csv(plink_output + f".assoc.{plink_reg_type}", sep="\s+")

            print(res)

            self.beta_hats[c] = pd.Series(res['BETA'], index=res['SNP'])
            self.z_scores[c] = pd.Series(res['STAT'], index=res['SNP'])
            self.se[c] = self.beta_hats[c] / self.z_scores[c]
            self.p_values[c] = pd.Series(res['P'], index=res['SNP'])

            #delete_temp_files(plink_output)

        delete_temp_files(phe_fname)

    def perform_gwas(self):

        if self.use_plink:
            self.perform_gwas_plink()
        else:
            self.compute_beta_hats()
            self.compute_standard_errors()
            self.compute_z_scores()
            self.compute_p_values()

    def compute_beta_hats(self):

        self.beta_hats = {c: pd.Series((da.dot(gt['G'][self.train_idx, :].T,
                                               self.phenotypes[self.train_idx]) / self.N).compute(),
                                       index=gt['G'].variant.values)
                          for c, gt in self.genotypes.items()}

        return self.beta_hats

    def compute_yy_per_snp(self):
        """
        Computes (yTy)j following SBayesR and Yang et al. (2012)
        :return:
        """
        self.yy = {c: (self.N - 2)*self.se[c]**2 + b_hat**2
                   for c, b_hat in self.beta_hats.items()}
        return self.yy

    def compute_standard_errors(self):

        self.se = {}

        for c, gt in self.genotypes.items():
            # maf = gt['G'].MAF.values
            # 2.*maf*(1. - maf)*self.sample_size
            self.se[c] = 1./np.sqrt(self.sample_size)

        return self.se

    def compute_z_scores(self):
        self.z_scores = {c: b_hat / self.se[c]
                         for c, b_hat in self.beta_hats.items()}
        return self.z_scores

    def compute_p_values(self, log10=False):
        self.p_values = {c: pd.Series(2.*stats.norm.sf(abs(z_sc)),
                                      index=z_sc.index)
                         for c, z_sc in self.z_scores.items()}

        if log10:
            self.p_values = {c: np.log10(pval) for c, pval in self.p_values.items()}

        return self.p_values

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

        for k, v in self.genotypes.items():
            ss_df = pd.DataFrame({
                'CHR': v['G'].chrom.values,
                'POS': v['G'].pos.values,
                'SNP': v['G'].variant.values,
                'A1': v['G'].a0.values,
                'A2': v['G'].a1.values,
                'MAF': v['G'].MAF.values,
                'BETA': self.beta_hats[k],
                'Z': self.z_scores[k],
                'SE': self.se[k],
                'PVAL': self.p_values[k]
            })

            ss_df['N'] = len(self.train_idx)

            ss_tables.append(ss_df)

        if per_chromosome:
            return ss_tables
        else:
            return pd.concat(ss_tables)
