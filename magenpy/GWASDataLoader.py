"""
Author: Shadi Zabad
Date: December 2020
"""

import configparser
import os.path as osp
import tempfile
from tqdm import tqdm

from pandas_plink import read_plink1_bin

import dask.array as da
import pandas as pd
import numpy as np
from scipy import stats
import zarr

from magenpy.AnnotationMatrix import AnnotationMatrix
from magenpy.LDMatrix import LDMatrix

from magenpy.parsers.plink_parsers import parse_fam_file, parse_bim_file
from magenpy.parsers.misc_parsers import read_snp_filter_file, read_individual_filter_file, parse_ld_block_data

from magenpy.utils.c_utils import (find_windowed_ld_boundaries,
                                   find_shrinkage_ld_boundaries,
                                   find_ld_block_boundaries)
from magenpy.utils.ld_utils import (_validate_ld_matrix,
                                    from_plink_ld_bin_to_zarr,
                                    from_plink_ld_table_to_zarr_chunked,
                                    shrink_ld_matrix,
                                    zarr_array_to_ragged,
                                    rechunk_zarr,
                                    move_ld_store)

from magenpy.utils.model_utils import standardize_genotype_matrix, merge_snp_tables
from magenpy.utils.compute_utils import intersect_arrays, iterable
from magenpy.utils.system_utils import makedir, get_filenames, run_shell_script, is_cmd_tool


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
                 sumstats_files=None,
                 sumstats_format='magenpy',
                 keep_individuals=None,
                 keep_snps=None,
                 min_maf=None,
                 min_mac=1,
                 remove_duplicated=True,
                 annotation_files=None,
                 genmap_Ne=None,
                 genmap_sample_size=None,
                 shrinkage_cutoff=1e-5,
                 compute_ld=False,
                 ld_store_files=None,
                 ld_block_files=None,
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

        # Access the configuration file:
        config = configparser.ConfigParser()
        config.read(osp.join(osp.dirname(__file__), 'config/paths.ini'))

        try:
            self.config = config['USER']
        except KeyError:
            self.config = config['DEFAULT']

        if self.use_plink:
            if not is_cmd_tool(self.config.get('plink2_path')):
                raise Exception("To use `plink` as a backend, make sure that the path for the "
                                "plink2 executable is configured properly.")

        # ------- General parameters -------

        self.standardize_phenotype = standardize_phenotype
        self.phenotype_likelihood = phenotype_likelihood
        self.phenotype_id = None  # Name or ID of the phenotype

        # ------- LD computation options -------
        self.ld_estimator = ld_estimator

        assert self.ld_estimator in ('block', 'windowed', 'sample', 'shrinkage')

        # For the block estimator of the LD matrix:
        self.ld_blocks = None

        if self.ld_estimator == 'block':
            assert ld_block_files is not None
            self.ld_blocks = parse_ld_block_data(ld_block_files)

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
        self._bp_pos = None  # SNP position in BP
        self._cm_pos = None  # SNP position in cM
        self.n_per_snp = None  # Sample size per SNP
        self._a1 = None  # Minor allele
        self._a2 = None  # Major allele
        self.maf = None  # Minor allele frequency

        self._fid = None  # Family IDs
        self._iid = None  # Individual IDs
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

        self.read_genotypes(bed_files)
        self.read_annotations(annotation_files)

        # TODO: Figure out optimal checks and placement of SNP filters
        if bed_files is not None:
            self.filter_by_allele_frequency(min_maf=min_maf, min_mac=min_mac)

        # ------- Compute LD matrices -------

        if ld_store_files is not None:
            self.read_ld(ld_store_files)
        elif compute_ld:
            self.compute_ld()

        # ------- Read phenotype/sumstats files -------
        self.read_phenotypes(phenotype_file, phenotype_id=phenotype_id,
                             header=phenotype_header, phenotype_col=phenotype_col,
                             standardize=self.standardize_phenotype)
        self.read_summary_stats(sumstats_files, sumstats_format)

        # ------- Harmonize data sources -------

        if bed_files is None:
            self.filter_by_allele_frequency(min_maf=min_maf, min_mac=min_mac)

        if self.genotypes is None and remove_duplicated:
            self.filter_duplicated_snps()

        if ld_store_files is not None or sumstats_files is not None:
            self.harmonize_data()

    @classmethod
    def from_table(cls, table):
        """
        Initialize a GDL object from table.
        :param table: A pandas dataframe with at leat 4 column defined: `CHR`, `SNP`, `A1`, `POS`
        Other column names that will be parsed from this table are:
        A2, MAF, N
        """

        assert all([col in table.columns for col in ('CHR', 'SNP', 'A1', 'POS')])

        gdl = cls()

        gdl._snps = {}
        gdl._bp_pos = {}
        gdl._a1 = {}

        for c in table['CHR'].unique():
            chrom_table = table.loc[table['CHR'] == c].sort_values('POS')

            gdl._snps[c] = chrom_table['SNP'].values
            gdl._a1[c] = chrom_table['A1'].values
            gdl._bp_pos[c] = chrom_table['POS'].values

            if 'A2' in chrom_table.columns:
                if gdl._a2 is None:
                    gdl._a2 = {}
                gdl._a2[c] = chrom_table['A2'].values

            if 'MAF' in chrom_table.columns:
                if gdl.maf is None:
                    gdl.maf = {}
                gdl.maf[c] = chrom_table['MAF'].values

            if 'N' in chrom_table.columns:
                if gdl.n_per_snp is None:
                    gdl.n_per_snp = {}
                gdl.n_per_snp[c] = chrom_table['N'].values

        return gdl

    @property
    def n_annotations(self):
        assert self.annotations is not None
        return self.annotations[self.chromosomes[0]].n_annotations

    @property
    def sample_size(self):
        return self.N

    @property
    def N(self, agg='max'):
        """
        The number of samples
        :param agg: Aggregation (max, mean, or None)
        """

        if agg == 'max':
            if self._iid is not None:
                return len(self._iid)
            else:
                if self.n_per_snp is None:
                    return None
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
        return sum(self.shapes.values())

    @property
    def snps(self):
        return self._snps

    @property
    def bp_pos(self):
        return self._bp_pos

    @property
    def cm_pos(self):
        return self._cm_pos

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

    def sample_ids_to_index(self, ids):
        return np.where(np.isin(self._iid, ids))[0]

    def sample_index_to_ids(self, idx):
        return self._iid[idx]

    def filter_snps(self, keep_snps, chrom=None):
        """
        :param keep_snps:
        :param chrom:
        """

        if chrom is None:
            snp_dict = self._snps
        else:
            snp_dict = {chrom: self._snps[chrom]}

        for c, snps in snp_dict.items():

            if np.array_equal(snps, keep_snps):
                continue

            common_idx = intersect_arrays(snps, keep_snps, return_index=True)

            # SNP vectors that must exist in all GDL objects:
            self._snps[c] = self._snps[c][common_idx]
            self._a1[c] = self._a1[c][common_idx]

            # Optional SNP vectors/matrices:
            if self._bp_pos is not None and c in self._bp_pos:
                self._bp_pos[c] = self._bp_pos[c][common_idx]

            if self._cm_pos is not None and c in self._cm_pos:
                self._cm_pos[c] = self._cm_pos[c][common_idx]

            if self._a2 is not None and c in self._a2:
                self._a2[c] = self._a2[c][common_idx]

            if self.genotypes is not None and c in self.genotypes:
                self.genotypes[c] = self.genotypes[c].isel(variant=common_idx)

            if self.n_per_snp is not None and c in self.n_per_snp:
                self.n_per_snp[c] = self.n_per_snp[c][common_idx]

            if self.maf is not None and c in self.maf:
                self.maf[c] = self.maf[c][common_idx]

            if self.beta_hats is not None and c in self.beta_hats:
                self.beta_hats[c] = self.beta_hats[c][common_idx]

            if self.se is not None and c in self.se:
                self.se[c] = self.se[c][common_idx]

            if self.z_scores is not None and c in self.z_scores:
                self.z_scores[c] = self.z_scores[c][common_idx]

            if self.p_values is not None and c in self.p_values:
                self.p_values[c] = self.p_values[c][common_idx]

            # Filter the annotation table as well:
            if self.annotations is not None and c in self.annotations:
                self.annotations[c].filter_snps(self._snps[c])

    def filter_by_allele_frequency(self, min_maf=None, min_mac=1):
        """
        Filter SNPs by minimum allele frequency or allele count
        :param min_maf: Minimum allele frequency
        :param min_mac: Minimum allele count (1 by default)
        """

        cond_dict = {}

        if min_mac is not None or min_maf is not None:
            if self.maf is None:
                self.compute_allele_frequency()
            if self.n_per_snp is None:
                self.compute_n_per_snp()

        if min_mac is not None:
            for c, maf in self.maf.items():
                mac = (2*maf*self.n_per_snp[c]).astype(np.int64)
                cond_dict[c] = (mac >= min_mac) & ((2*self.n_per_snp[c] - mac) >= min_mac)

        if min_maf is not None:

            for c, maf in self.maf.items():
                maf_cond = (maf >= min_maf) & (1. - maf >= min_maf)
                if c in cond_dict:
                    cond_dict[c] = cond_dict[c] & maf_cond
                else:
                    cond_dict[c] = maf_cond

        if len(cond_dict) > 0:

            filt_count = 0

            for c, snps in tqdm(self.snps.items(),
                                total=len(self.chromosomes),
                                desc="Filtering SNPs by allele frequency/count",
                                disable=not self.verbose):
                keep_snps = snps[cond_dict[c]]
                if len(keep_snps) != len(snps):
                    filt_count += len(snps) - len(keep_snps)
                    self.filter_snps(keep_snps, chrom=c)

            if filt_count > 0:
                if self.verbose:
                    print(f"> Filtered {filt_count} SNPs due to MAC/MAF thresholds.")

    def filter_duplicated_snps(self):
        """
        This method filters all duplicated SNPs.
        TODO: Add options to keep at least one of the duplicated snps.
        :return:
        """

        for c, snps in self.snps.items():
            u_snps, counts = np.unique(snps, return_counts=True)
            if len(u_snps) < len(snps):
                # Keep only SNPs which occur once in the sequence:
                self.filter_snps(u_snps[counts == 1], chrom=c)

    def filter_samples(self, keep_samples):

        common_samples = intersect_arrays(self._iid, keep_samples, return_index=True)

        for c in self.chromosomes:
            if self.genotypes is not None and c in self.genotypes:
                self.genotypes[c] = self.genotypes[c].isel(sample=common_samples)

            self._fid = self._fid[common_samples]
            self._iid = self._iid[common_samples]

    def read_annotations(self, annot_files):
        """
        Read the annotation files
        """

        if annot_files is None:
            return

        if not iterable(annot_files):
            annot_files = [annot_files]

        self.annotations = {}

        for annot_file in tqdm(annot_files,
                               total=len(annot_files),
                               desc="Reading annotation files",
                               disable=not self.verbose):
            annot_mat = AnnotationMatrix.from_file(annot_file)
            annot_mat.filter_snps(self.snps[annot_mat.chromosome])
            self.annotations[annot_mat.chromosome] = annot_mat

    def read_genotypes_plink(self, bed_files):
        """
        This is an alternative to `.read_genotypes` that doesn't attempt to
        parse to process the genotype matrix and instead focuses on loading
        the individual and SNP data and preparing it for downstream tasks.
        """

        if bed_files is None:
            return

        if not iterable(bed_files):
            bed_files = get_filenames(bed_files, extension='.bed')

        self._snps = {}
        self._a1 = {}
        self._a2 = {}
        self._cm_pos = {}
        self._bp_pos = {}
        self.bed_files = {}

        for i, bfile in tqdm(enumerate(bed_files),
                             total=len(bed_files),
                             desc="Reading genotype files",
                             disable=not self.verbose):

            # Read plink file:
            try:
                bim_df = parse_bim_file(bfile)
                if i == 0:
                    fam_df = parse_fam_file(bfile)
            except Exception as e:
                self.genotypes = None
                self._fid = None
                self._iid = None
                raise e

            # Filter individuals:
            if self.keep_individuals is not None and i == 0:
                common_samples = intersect_arrays(fam_df.IID.values, self.keep_individuals, return_index=True)
                fam_df = fam_df.iloc[common_samples, ]

            # Filter SNPs:
            if self.keep_snps is not None:
                common_snps = intersect_arrays(bim_df.SNP.values, self.keep_snps, return_index=True)
                bim_df = bim_df.iloc[common_snps, ]

            # Obtain information about current chromosome:
            chr_id = int(bim_df.CHR.values[0])

            # Add filename to the bedfiles dictionary:
            self.bed_files[chr_id] = bfile
            # Keep track of the SNPs:
            self._snps[chr_id] = bim_df.SNP.values
            self._a1[chr_id] = bim_df.A1.values
            self._a2[chr_id] = bim_df.A2.values
            self._bp_pos[chr_id] = bim_df.POS.values
            self._cm_pos[chr_id] = bim_df.cM.values

            if i == 0:
                self._fid = fam_df.FID.values
                self._iid = fam_df.IID.values

    def read_genotypes(self, bed_files):
        """
        Read the genotype files
        """

        if self.use_plink:
            return self.read_genotypes_plink(bed_files)

        if bed_files is None:
            return

        if not iterable(bed_files):
            bed_files = get_filenames(bed_files, extension='.bed')

        self._snps = {}
        self._a1 = {}
        self._a2 = {}
        self._cm_pos = {}
        self._bp_pos = {}
        self.genotypes = {}
        self.bed_files = {}

        for i, bfile in tqdm(enumerate(bed_files),
                             total=len(bed_files),
                             desc="Reading genotype files",
                             disable=not self.verbose):

            # Read plink file:
            try:
                gt_ac = read_plink1_bin(bfile + ".bed", ref="a0", verbose=False)
            except ValueError:
                gt_ac = read_plink1_bin(bfile, ref="a0", verbose=False)
            except Exception as e:
                self.genotypes = None
                self._fid = None
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
            self._a1[chr_id] = gt_ac.variant.a0.values
            self._a2[chr_id] = gt_ac.variant.a1.values
            self._bp_pos[chr_id] = gt_ac.variant.pos.values
            self._cm_pos[chr_id] = gt_ac.variant.cm.values

            if i == 0:
                self._fid = gt_ac.fid.values
                self._iid = gt_ac.iid.values

            self.genotypes[chr_id] = gt_ac

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

        if self.phenotype_likelihood == 'binomial':
            unique_vals = sorted(phe['phenotype'].unique())
            if unique_vals == [1, 2]:
                # Plink coding for case/control
                phe['phenotype'] -= 1
            elif unique_vals != [0, 1]:
                raise ValueError(f"Unknown values for binary traits: {unique_vals}")

        self.phenotypes = phe['phenotype'].values

        if standardize and self.phenotype_likelihood == 'gaussian':
            self.phenotypes -= self.phenotypes.mean()
            self.phenotypes /= self.phenotypes.std()

        if phenotype_id is None:
            self.phenotype_id = str(np.random.randint(1, 1000))
        else:
            self.phenotype_id = phenotype_id

    def read_summary_stats(self, sumstats_files, sumstats_format='magenpy'):
        """
        TODO: implement parsers for summary statistics
        TODO: Move these parsers to `parsers.py`
        """

        if sumstats_files is None:
            return

        if not iterable(sumstats_files):
            sumstats_files = get_filenames(sumstats_files)

        ss = []

        print("> Reading GWAS summary statistics...")

        for ssf in sumstats_files:
            ss_df = pd.read_csv(ssf, delim_whitespace=True)
            # Drop missing values:
            ss_df = ss_df.dropna()
            ss.append(ss_df)

        ss = pd.concat(ss)

        # ------------- Standardize inputs -------------
        # TODO: Move this part to parsers.py
        if sumstats_format == 'LDSC':
            # Useful here: https://www.biostars.org/p/319584/
            pass
        elif sumstats_format == 'SBayesR':
            pass
        elif sumstats_format == 'plink':
            ss.rename(columns={
                '#CHROM': 'CHR',
                'ID': 'SNP',
                'P': 'PVAL',
                'OBS_CT': 'N',
                'A1_FREQ': 'MAF'
            }, inplace=True)
            ss['A2'] = ss.apply(lambda x: [x['ALT1'], x['REF']][x['A1'] == x['ALT1']], axis=1)
            ss['Z'] = ss['BETA'] / ss['SE']

        # -------------------------------------------------

        # If SNP list is not set, initialize it using the sumstats table:
        if self.snps is None:

            # Check that the sumstats table has the following columns:
            assert all([col in ss.columns for col in ('CHR', 'POS', 'SNP', 'A1')])

            self._snps = {}
            self._a1 = {}
            self._bp_pos = {}

            for c in ss['CHR'].unique():

                m_ss = ss.loc[ss['CHR'] == c].sort_values('POS')

                self._snps[c] = m_ss['SNP'].values
                self._a1[c] = m_ss['A1'].values
                self._bp_pos[c] = m_ss['POS'].values

        # -------------------------------------------------
        # Prepare the fields for the sumstats provided in the table:

        if 'A1' in ss.columns:
            update_a1 = True
        else:
            update_a1 = False

        if 'POS' in ss.columns:
            update_pos = True
        else:
            update_pos = False

        if 'A2' in ss.columns:
            update_a2 = True
            self._a2 = {}
        else:
            update_a2 = False

        if 'MAF' in ss.columns:
            self.maf = {}
            update_maf = True
        else:
            update_maf = False

        if 'N' in ss.columns:
            self.n_per_snp = {}
            update_n = True
        else:
            update_n = False

        if 'BETA' in ss.columns:
            self.beta_hats = {}
            update_beta = True
        else:
            update_beta = False

        if 'Z' in ss.columns:
            self.z_scores = {}
            update_z = True
        else:
            update_z = False

        if 'SE' in ss.columns:
            self.se = {}
            update_se = True
        else:
            update_se = False

        if 'PVAL' in ss.columns:
            self.p_values = {}
            update_pval = True
        else:
            update_pval = False

        for c, snps in self.snps.items():
            m_ss = merge_snp_tables(pd.DataFrame({'SNP': snps, 'A1': self._a1[c]}), ss)

            if len(m_ss) > 1:

                # Filter the SNP list first!
                if len(snps) != len(m_ss):
                    self.filter_snps(m_ss['SNP'], chrom=c)

                # Populate the sumstats fields:
                if update_a1:
                    self._a1[c] = m_ss['A1'].values
                if update_pos:
                    self._bp_pos[c] = m_ss['POS'].values
                if update_a2:
                    self._a2[c] = m_ss['A2'].values
                if update_maf:
                    self.maf[c] = m_ss['MAF'].values
                if update_n:
                    self.n_per_snp[c] = m_ss['N'].values
                if update_beta:
                    self.beta_hats[c] = m_ss['BETA'].values
                if update_z:
                    self.z_scores[c] = m_ss['Z'].values
                if update_se:
                    self.se[c] = m_ss['SE'].values
                if update_pval:
                    self.p_values[c] = m_ss['PVAL'].values

        print(f"> Read summary statistics data for {self.M} SNPs.")

    def read_ld(self, ld_store_files):
        """
        :param ld_store_files:
        """

        if self.verbose:
            print("> Reading LD matrices...")

        if not iterable(ld_store_files):
            ld_store_files = get_filenames(ld_store_files, extension='.zarr')

        self.ld = {}

        if self._snps is None:
            init_snps = True
            self._snps = {}
            self._a1 = {}
            self._bp_pos = {}
            self.maf = {}
        else:
            init_snps = False

        for f in ld_store_files:
            z = LDMatrix.from_path(f)
            self.ld[z.chromosome] = z
            # If the SNP list is not set,
            # initialize it with the SNP list from the LD store:
            if init_snps:
                self._snps[z.chromosome] = z.snps
                self._a1[z.chromosome] = z.a1
                self._bp_pos[z.chromosome] = z.bp_position
                self.maf[z.chromosome] = z.maf

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

        if recompute:
            # If recomputing from existing LD matrices:
            shapes = self.shapes

            for c, ld in tqdm(self.ld.items(), total=len(self.chromosomes),
                              desc='Recomputing LD boundaries',
                              disable=not self.verbose):

                common_idx = intersect_arrays(ld.snps, self.snps[c], return_index=True)
                M = shapes[c]
                estimator = ld.ld_estimator
                est_properties = ld.estimator_properties

                if estimator == 'sample':
                    self.ld_boundaries[c] = np.array((np.zeros(M), np.ones(M)*M)).astype(np.int64)
                elif estimator == 'block':
                    self.ld_boundaries[c] = find_ld_block_boundaries(ld.bp_position[common_idx],
                                                                     np.array(est_properties['LD blocks'], dtype=int))
                elif estimator == 'windowed':
                    if est_properties['Window units'] == 'cM':
                        self.ld_boundaries[c] = find_windowed_ld_boundaries(ld.cm_position[common_idx],
                                                                            est_properties['Window cutoff'])
                    else:
                        idx = np.arange(M)
                        self.ld_boundaries[c] = np.array((idx - est_properties['Window cutoff'],
                                                          idx + est_properties['Window cutoff'])).astype(np.int64)
                        self.ld_boundaries[c] = np.clip(self.ld_boundaries[c], 0, M)
                else:
                    self.ld_boundaries[c] = find_shrinkage_ld_boundaries(ld.cm_position[common_idx],
                                                                         est_properties['Genetic map Ne'],
                                                                         est_properties['Genetic map sample size'],
                                                                         est_properties['Cutoff'])

        else:

            for c, M in tqdm(self.shapes.items(),
                             total=len(self.chromosomes),
                             desc="Computing LD boundaries",
                             disable=not self.verbose):

                if self.ld_estimator == 'sample':

                    self.ld_boundaries[c] = np.array((np.zeros(M), np.ones(M)*M)).astype(np.int64)

                elif self.ld_estimator == 'block':

                    if self._bp_pos and c in self._bp_pos:
                        self.ld_boundaries[c] = find_ld_block_boundaries(self._bp_pos[c].astype(int),
                                                                         self.ld_blocks[c])
                    else:
                        raise Exception("SNP position in BP is missing!")

                elif self.ld_estimator == 'windowed':
                    if self.window_unit == 'cM':
                        if self._cm_pos and c in self._cm_pos:
                            self.ld_boundaries[c] = find_windowed_ld_boundaries(self._cm_pos[c],
                                                                                self.cm_window_cutoff)
                        else:
                            raise Exception("cM information for SNPs is missing. "
                                            "Make sure to populate it with a reference genetic map "
                                            "or use a pre-specified window size around each SNP.")
                    else:

                        idx = np.arange(M)
                        self.ld_boundaries[c] = np.array((idx - self.window_size_cutoff,
                                                          idx + self.window_size_cutoff)).astype(np.int64)
                        self.ld_boundaries[c] = np.clip(self.ld_boundaries[c],
                                                        0, M)
                elif self.ld_estimator == 'shrinkage':
                    if self._cm_pos and c in self._cm_pos:
                        self.ld_boundaries[c] = find_shrinkage_ld_boundaries(self._cm_pos[c],
                                                                             self.genmap_Ne,
                                                                             self.genmap_sample_size,
                                                                             self.shrinkage_cutoff)
                    else:
                        raise Exception("cM information for SNPs is missing. "
                                        "Make sure to populate it with a reference genetic map "
                                        "or use a different LD estimator.")

        return self.ld_boundaries

    def compute_ld_plink(self):
        """
        Compute the Linkage-Disequilibrium (LD) matrix between SNPs using plink1.9
        """

        if not is_cmd_tool(self.config.get('plink1.9_path')):
            raise Exception("To use `plink` as a backend for LD calculation, "
                            "make sure that the path for the plink1.9 executable is configured properly.")

        if self.maf is None:
            self.compute_allele_frequency()

        if self.ld_boundaries is None:
            self.compute_ld_boundaries()

        tmp_ld_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='ld_')
        self.cleanup_dir_list.append(tmp_ld_dir)

        # Create the samples file:
        keep_file = osp.join(tmp_ld_dir.name, 'samples.keep')
        keep_table = self.to_individual_table()
        keep_table.to_csv(keep_file, index=False, header=False, sep="\t")

        self.ld = {}

        for c, b_file in tqdm(self.bed_files.items(),
                              total=len(self.chromosomes),
                              desc="Computing LD matrices using PLINK",
                              disable=not self.verbose):

            snp_keepfile = osp.join(tmp_ld_dir.name, f"chr_{c}.keep")
            pd.DataFrame({'SNP': self.snps[c]}).to_csv(snp_keepfile,
                                                       index=False, header=False)

            plink_output = osp.join(tmp_ld_dir.name, f"chr_{c}")

            cmd = [
                self.config.get('plink1.9_path'),
                f"--bfile {b_file.replace('.bed', '')}",
                f"--keep {keep_file}",
                f"--extract {snp_keepfile}",
                "--keep-allele-order",
                f"--out {plink_output}",
                f"--threads {self.n_threads}"
            ]

            # For the block and shrinkage estimators, ask plink to compute
            # LD between focal SNP and max(window_size) around it.
            # Then, once we have a square matrix out of that, we can apply
            # a per-SNP filter:

            max_window_size = (self.ld_boundaries[c][1, :] - self.ld_boundaries[c][0, :]).max() + 1
            max_kb = round(.001*(self.bp_pos[c].max() - self.bp_pos[c].min()))

            if self.ld_estimator in ('shrinkage', 'block'):
                cmd.append("--r gz")
                cmd.append(f"--ld-window {max_window_size} "
                           f"--ld-window-kb {max_kb}")
            elif self.ld_estimator == 'windowed':
                cmd.append("--r gz")
                cmd.append(f"--ld-window {len(self.snps[c]) + 1} "
                           f"--ld-window-kb {max_kb} "
                           f"--ld-window-cm {self.cm_window_cutoff}")
            else:
                cmd.append("--r bin")
                cmd.append(f"--ld-window {len(self.snps[c]) + 1} "
                           f"--ld-window-kb {max_kb} ")

            run_shell_script(" ".join(cmd))

            # Convert from PLINK LD files to Zarr:
            fin_ld_store = osp.join(self.output_dir, 'ld', 'chr_' + str(c))

            if self.ld_estimator == 'sample':
                z_ld_mat = from_plink_ld_bin_to_zarr(f"{plink_output}.ld.bin",
                                                     fin_ld_store,
                                                     self.ld_boundaries[c])
            else:
                z_ld_mat = from_plink_ld_table_to_zarr_chunked(f"{plink_output}.ld.gz",
                                                               fin_ld_store,
                                                               self.ld_boundaries[c],
                                                               self.snps[c])

            # Add LD matrix properties:
            z_ld_mat.attrs['Chromosome'] = c
            z_ld_mat.attrs['Sample size'] = self.sample_size
            z_ld_mat.attrs['SNP'] = list(self.snps[c])
            z_ld_mat.attrs['LD estimator'] = self.ld_estimator
            z_ld_mat.attrs['LD boundaries'] = self.ld_boundaries[c].tolist()

            ld_estimator_properties = None

            if self.ld_estimator == 'shrinkage':

                z_ld_mat = shrink_ld_matrix(z_ld_mat,
                                            self.cm_pos[c],
                                            self.genmap_Ne,
                                            self.genmap_sample_size,
                                            self.shrinkage_cutoff,
                                            ld_boundaries=self.ld_boundaries[c])

                ld_estimator_properties = {
                    'Genetic map Ne': self.genmap_Ne,
                    'Genetic map sample size': self.genmap_sample_size,
                    'Cutoff': self.shrinkage_cutoff
                }

            elif self.ld_estimator == 'windowed':
                ld_estimator_properties = {
                    'Window units': self.window_unit,
                    'Window cutoff': [self.window_size_cutoff, self.cm_window_cutoff][self.window_unit == 'cM']
                }

            elif self.ld_estimator == 'block':
                ld_estimator_properties = {
                    'LD blocks': self.ld_blocks[c].tolist()
                }

            # Add detailed LD matrix properties:
            z_ld_mat.attrs['BP'] = list(map(int, self.bp_pos[c]))
            z_ld_mat.attrs['cM'] = list(map(float, self.cm_pos[c]))
            z_ld_mat.attrs['MAF'] = list(map(float, self.maf[c]))
            z_ld_mat.attrs['A1'] = list(self._a1[c])

            if ld_estimator_properties is not None:
                z_ld_mat.attrs['Estimator properties'] = ld_estimator_properties

            self.ld[c] = LDMatrix(z_ld_mat)
            self.ld[c].set_store_attr('LDScore', self.ld[c].compute_ld_scores().tolist())
            _validate_ld_matrix(self.ld[c])

    def compute_ld(self):
        """
        Compute the Linkage-Disequilibrium (LD) matrix between SNPs.
        This function only considers correlations between SNPs on the same chromosome.
        The function involves computing X'X and then applying transformations to it,
        according to the estimator that the user specifies.
        """

        if self.use_plink:
            self.compute_ld_plink()
            return

        if self.maf is None:
            self.compute_allele_frequency()

        if self.ld_boundaries is None:
            self.compute_ld_boundaries()

        tmp_ld_dir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='ld_')
        self.cleanup_dir_list.append(tmp_ld_dir)

        self.ld = {}

        for c, g_data in tqdm(self.genotypes.items(),
                              total=len(self.chromosomes),
                              desc="Computing LD matrices",
                              disable=not self.verbose):

            tmp_ld_store = osp.join(tmp_ld_dir.name, 'chr_' + str(c))
            fin_ld_store = osp.join(self.output_dir, 'ld', 'chr_' + str(c))

            # Re-chunk the array
            g_data = g_data.chunk((min(1024, g_data.shape[0]),
                                   min(1024, g_data.shape[1])))

            # Standardize the genotype matrix and fill missing data with zeros:
            g_mat = standardize_genotype_matrix(g_data).fillna(0.)

            # Compute the LD matrix:
            ld_mat = (da.dot(g_mat.T, g_mat) / self.N).astype(np.float64)
            ld_mat.to_zarr(tmp_ld_store, overwrite=True)

            z_ld_mat = zarr.open(tmp_ld_store)
            z_ld_mat = rechunk_zarr(z_ld_mat,
                                    ld_mat.rechunk({0: 'auto', 1: None}).chunksize,
                                    tmp_ld_store + '_rechunked',
                                    tmp_ld_store + '_intermediate')

            # Add LD matrix properties:
            z_ld_mat.attrs['Chromosome'] = c
            z_ld_mat.attrs['Sample size'] = self.sample_size
            z_ld_mat.attrs['SNP'] = list(self.snps[c])
            z_ld_mat.attrs['LD estimator'] = self.ld_estimator
            z_ld_mat.attrs['LD boundaries'] = self.ld_boundaries[c].tolist()

            ld_estimator_properties = None

            if self.ld_estimator == 'sample':
                z_ld_mat = move_ld_store(z_ld_mat, fin_ld_store)

            if self.ld_estimator == 'shrinkage':

                z_ld_mat = shrink_ld_matrix(z_ld_mat,
                                            self.cm_pos[c],
                                            self.genmap_Ne,
                                            self.genmap_sample_size,
                                            self.shrinkage_cutoff)

                ld_estimator_properties = {
                    'Genetic map Ne': self.genmap_Ne,
                    'Genetic map sample size': self.genmap_sample_size,
                    'Cutoff': self.shrinkage_cutoff
                }

            elif self.ld_estimator == 'windowed':

                ld_estimator_properties = {
                    'Window units': self.window_unit,
                    'Window cutoff': [self.window_size_cutoff, self.cm_window_cutoff][self.window_unit == 'cM']
                }

            elif self.ld_estimator == 'block':

                ld_estimator_properties = {
                    'LD blocks': self.ld_blocks[c].tolist()
                }

            if self.ld_estimator in ('block', 'shrinkage', 'windowed'):
                z_ld_mat = zarr_array_to_ragged(z_ld_mat,
                                                fin_ld_store,
                                                bounds=self.ld_boundaries[c],
                                                delete_original=True)

            # Add detailed LD matrix properties:
            z_ld_mat.attrs['BP'] = list(map(int, self.bp_pos[c]))
            z_ld_mat.attrs['cM'] = list(map(float, self.cm_pos[c]))
            z_ld_mat.attrs['MAF'] = list(map(float, self.maf[c]))
            z_ld_mat.attrs['A1'] = list(self._a1[c])

            if ld_estimator_properties is not None:
                z_ld_mat.attrs['Estimator properties'] = ld_estimator_properties

            self.ld[c] = LDMatrix(z_ld_mat)
            self.ld[c].set_store_attr('LDScore', self.ld[c].compute_ld_scores().tolist())
            _validate_ld_matrix(self.ld[c])

    def get_ld_matrices(self):
        return self.ld

    def get_ld_boundaries(self):
        if self.ld is None:
            return None

        return {c: ld.get_masked_boundaries() for c, ld in self.ld.items()}

    def realign_ld(self):
        """
        This method realigns a pre-computed LD matrix with the
        current genotype matrix and/or summary statistics.
        """

        if self.ld is None:
            raise Exception("No pre-computed LD matrices are provided.")

        self.compute_ld_boundaries(recompute=True)

        ld_tmpdir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='ld_')
        self.cleanup_dir_list.append(ld_tmpdir)

        for c, snps in tqdm(self.snps.items(), total=len(self.chromosomes),
                            desc="Matching LD matrices with sumstats/genotypes",
                            disable=not self.verbose):
            ld_snps = self.ld[c].snps
            if not np.array_equal(snps, ld_snps):
                self.ld[c] = LDMatrix(
                    zarr_array_to_ragged(self.ld[c].z_array,
                                         dir_store=osp.join(ld_tmpdir.name, f'chr_{c}'),
                                         keep_snps=snps,
                                         bounds=self.ld_boundaries[c])
                )

    def harmonize_data(self):
        """
        This method ensures that all the data sources (reference genotype,
        LD matrices, summary statistics) are aligned.
        """

        if self.verbose:
            print("> Harmonizing data...")

        update_ld = False
        sumstats_tables = self.to_snp_table(per_chromosome=True, col_subset=['SNP', 'A1', 'MAF', 'BETA', 'Z'])

        for c, snps in self.snps.items():
            # Harmonize SNPs in LD store and summary statistics/genotype matrix:
            if self.ld is not None:

                self.ld[c].set_mask(None)

                ld_snps = self.ld[c].to_snp_table(col_subset=['SNP', 'A1'])
                matched_snps = merge_snp_tables(ld_snps, sumstats_tables[c])

                # If the SNP list doesn't align with the matched SNPs,
                # then filter the SNP list
                if len(snps) != len(matched_snps):
                    self.filter_snps(matched_snps['SNP'].values, chrom=c)

                if len(matched_snps) != len(ld_snps):

                    # If the percentage of SNPs that will need to be excluded from the
                    # LD matrix exceeds 30% (and greater than 5000), then copy and update the matrix.
                    # Otherwise, introduce a mask that ensures those SNPs are excluded from
                    # downstream tasks.
                    #
                    # NOTE: This behavior is deprecated for now...
                    # We simply apply a mask to the LD matrix, and depending on the size
                    # unmasked elements, downstream tasks can decide whether or not to load
                    # the matrix to memory.
                    #
                    # To be revisited...

                    #n_miss = len(ld_snps) - len(matched_snps)
                    #if float(n_miss) / len(ld_snps) > .3 and n_miss > 5000:
                    #    update_ld = True
                    #else:
                    remain_index = intersect_arrays(ld_snps['SNP'].values,
                                                    matched_snps['SNP'].values,
                                                    return_index=True)
                    mask = np.zeros(len(ld_snps))
                    mask[remain_index] = 1
                    self.ld[c].set_mask(mask.astype(bool))

                flip_01 = matched_snps['flip'].values
                num_flips = flip_01.sum()

                if num_flips > 0:
                    print(f"> Detected {num_flips} SNPs with strand flipping. Correcting summary statistics...")

                    # Correct strand information:
                    self._a1[c] = matched_snps['A1'].values

                    # Correct MAF:
                    if self.maf is not None:
                        self.maf[c] = matched_snps['MAF'].values

                    # Correct BETA:
                    if self.beta_hats is not None:
                        self.beta_hats[c] = matched_snps['BETA'].values

                    # Correct Z-score:
                    if self.z_scores is not None:
                        self.z_scores[c] = matched_snps['Z'].values

        if update_ld:
            self.realign_ld()

    def score_plink(self, betas=None):
        """
        Perform linear scoring using PLINK2
        :param betas:
        """

        if betas is None:
            if self.beta_hats is None:
                raise Exception("Neither betas nor beta hats are provided or set."
                                " Please provide betas to perform prediction.")
            else:
                betas = {c: b for c, b in self.beta_hats.items()}

        # Initialize the PGS object with zeros
        # The construction here accounts for multiple betas per SNP

        try:
            betas_shape = betas[next(iter(betas))].shape[1]
            if betas_shape == 1:
                raise IndexError
            score_col_nums = f"--score-col-nums 3-{3 + betas_shape - 1}"
        except IndexError:
            betas_shape = 1
            for c, b in betas.items():
                betas[c] = b.reshape(-1, 1)
            score_col_nums = f"--score-col-nums 3"

        pgs = np.zeros(shape=(self.N, betas_shape))

        # Create a temporary directory for the score files:
        score_tmpdir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='score_')
        self.cleanup_dir_list.append(score_tmpdir)

        # Create the samples file:
        keep_file = osp.join(score_tmpdir.name, 'samples.keep')
        keep_table = self.to_individual_table()
        keep_table.to_csv(keep_file, index=False, header=False, sep="\t")

        for c, beta in tqdm(betas.items(), total=len(betas),
                            desc="Generating polygenic scores using PLINK",
                            disable=not self.verbose):

            eff_file = osp.join(score_tmpdir.name, f'chr_{c}.txt')
            df = pd.DataFrame({'SNP': self.snps[c], 'A1': self.alt_alleles[c]})
            for i in range(betas_shape):
                df['BETA' + str(i)] = betas[c][:, i]

            df = df.loc[df[['BETA' + str(i) for i in range(betas_shape)]].sum(axis=1) != 0]

            try:
                df.to_csv(eff_file, index=False, sep="\t")

                cmd = [
                    self.config.get('plink2_path'),
                    f"--bfile {self.bed_files[c].replace('.bed', '')}",
                    f"--keep {keep_file}",
                    f"--score {eff_file} 1 2 header-read cols=+scoresums variance-standardize",
                    score_col_nums,
                    f"--out {eff_file.replace('.txt', '')}",
                    f"--threads {self.n_threads}"
                ]

                try:
                    run_shell_script(" ".join(cmd))
                    if not osp.isfile(eff_file.replace('.txt', '.sscore')):
                        raise FileNotFoundError
                except Exception as e:
                    raise Exception("plink polygenic scoring failed to run!\nDeployed command:" +
                                    " ".join(cmd))

                dtypes = {'FID': str, 'IID': str}
                for i in range(betas_shape):
                    dtypes.update({'PRS' + str(i): np.float64})

                chr_pgs = pd.read_csv(eff_file.replace('.txt', '.sscore'), delim_whitespace=True,
                                      names=['FID', 'IID'] + ['PRS' + str(i) for i in range(betas_shape)],
                                      skiprows=1,
                                      usecols=[0, 1] + [4 + betas_shape + i for i in range(betas_shape)],
                                      dtype=dtypes)
                chr_pgs = keep_table.astype({'FID': str, 'IID': str}).merge(chr_pgs)

                pgs += chr_pgs[['PRS' + str(i) for i in range(betas_shape)]].values

            except Exception as e:
                raise e

        if betas_shape == 1:
            pgs = pgs.flatten()

        return pgs

    def score(self, betas=None):

        if self.use_plink:
            return self.score_plink(betas)

        if betas is None:
            if self.beta_hats is None:
                raise Exception("Neither betas nor beta hats are provided or set."
                                " Please provide betas to perform prediction.")
            else:
                betas = {c: b for c, b in self.beta_hats.items()}

        if not self.standardize_genotype and self.maf is None:
            self.compute_allele_frequency()

        try:
            betas_shape = betas[next(iter(betas))].shape[1]
        except IndexError:
            betas_shape = 1
            for c, b in betas.items():
                betas[c] = b.reshape(-1, 1)

        pgs = np.zeros(shape=(self.N, betas_shape))

        for c, gt in tqdm(self.genotypes.items(), total=len(self.chromosomes),
                          desc="Generating polygenic scores",
                          disable=not self.verbose):

            if self.standardize_genotype:
                pgs += np.dot(standardize_genotype_matrix(gt).fillna(0.), betas[c])
            else:
                pgs += np.dot(gt.fillna(self.maf[c]), betas[c])

        if betas_shape == 1:
            pgs = pgs.flatten()

        return pgs

    def predict(self, betas=None):

        pgs = self.score(betas)

        if self.phenotype_likelihood == 'binomial':
            # apply sigmoid function:
            # TODO: Check this (maybe convert to probit?)
            pgs = 1./(1. + np.exp(-pgs))

        return pgs

    def perform_gwas_plink(self):
        """
        Perform GWAS using PLINK
        """

        # Create a temporary directory for the gwas files:
        gwas_tmpdir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='gwas_')
        self.cleanup_dir_list.append(gwas_tmpdir)

        # Output the phenotype file:
        phe_fname = osp.join(gwas_tmpdir.name, "pheno.txt")
        phe_table = self.to_phenotype_table()
        if self.phenotype_likelihood == 'binomial':
            phe_table['phenotype'] += 1
        phe_table.to_csv(phe_fname, sep="\t", index=False, header=False)

        plink_reg_type = ['linear', 'logistic'][self.phenotype_likelihood == 'binomial']

        self.n_per_snp = {}
        self.maf = {}
        self.beta_hats = {}
        self.se = {}
        self.z_scores = {}
        self.p_values = {}

        for c, bf in tqdm(self.bed_files.items(),
                          total=len(self.chromosomes),
                          desc="Performing GWAS using PLINK",
                          disable=not self.verbose):

            # Output a keep file for SNPs:
            snp_keepfile = osp.join(gwas_tmpdir.name, f"chr_{c}.keep")
            pd.DataFrame({'SNP': self.snps[c]}).to_csv(snp_keepfile,
                                                       index=False, header=False)

            plink_output = osp.join(gwas_tmpdir.name, f"chr_{c}")

            cmd = [
                self.config.get('plink2_path'),
                f"--bfile {bf.replace('.bed', '')}",
                f"--extract {snp_keepfile}",
                f"--{plink_reg_type} hide-covar cols=chrom,pos,alt1,ref,a1freq,nobs,beta,se,tz,p",
                f"--pheno {phe_fname}",
                f"--out {plink_output}",
                f"--threads {self.n_threads}"
            ]

            if self.standardize_phenotype:
                cmd.append('--variance-standardize')

            run_shell_script(" ".join(cmd))

            output_fname = plink_output + f".PHENO1.glm.{plink_reg_type}"

            if not osp.isfile(output_fname):
                if plink_reg_type == 'logistic' and osp.isfile(output_fname + ".hybrid"):
                    output_fname += ".hybrid"
                else:
                    raise FileNotFoundError

            res = pd.read_csv(output_fname, delim_whitespace=True)
            res.rename(columns={
                '#CHROM': 'CHR',
                'ID': 'SNP',
                'P': 'PVAL',
                'OBS_CT': 'N',
                'A1_FREQ': 'MAF'
            }, inplace=True)

            # TODO: Filter NaN values that may arise from PLINK.

            # Merge to make sure that summary statistics are in order:
            res = merge_snp_tables(pd.DataFrame({'SNP': self.snps[c], 'A1': self._a1[c]}), res)

            if len(res) != len(self.snps[c]):
                raise ValueError("Length of GWAS table does not match number of SNPs.")

            self.n_per_snp[c] = res['N'].values
            self.maf[c] = res['MAF'].values
            self.beta_hats[c] = res['BETA'].values
            self.se[c] = res['SE'].values
            self.z_scores[c] = self.beta_hats[c] / self.se[c]
            self.p_values[c] = res['PVAL'].values

    def perform_gwas(self):
        """
        Peform GWAS using closed form solutions.
        (Only applicable to quantitative traits)
        """

        if self.use_plink:
            self.perform_gwas_plink()
        else:

            if self.phenotype_likelihood == 'binomial':
                raise Exception("Software does not support GWAS with case/control phenotypes. Use plink instead.")

            if self.n_per_snp is None:
                self.compute_allele_frequency()

            for c in tqdm(self.chromosomes, desc="Performing GWAS", disable=not self.verbose):
                self.verbose = False
                self.compute_beta_hats(chrom=c)
                self.compute_standard_errors(chrom=c)
                self.compute_z_scores(chrom=c)
                self.compute_p_values(chrom=c)
                self.verbose = True

    def estimate_snp_heritability(self, per_chromosome=False):
        """
        Provides an estimate of SNP heritability from summary statistics using
        a simplified version of the LD Score Regression framework.
        E[X_j^2] = h_g^2*l_j + int
        Where the response is the Chi-Squared statistic for SNP j
        and the variable is its LD score.

        NOTE: For now, we constrain the slope to 1.

        TODO: Maybe move into its own module?

        :param per_chromosome: Estimate heritability per chromosome
        """

        if self.ld is None or self.z_scores is None:
            raise Exception("Estimating SNP heritability requires z-scores and LD matrices!")

        chr_ldsc = {}
        chr_xi_sq = {}

        for c, ldm in tqdm(self.ld.items(),
                           total=len(self.chromosomes),
                           desc="Estimating SNP-heritability",
                           disable=not self.verbose):

            chr_ldsc[c] = ldm.ld_score
            chr_xi_sq[c] = self.z_scores[c]**2

        if per_chromosome:
            chr_h2g = {}
            for c in chr_ldsc:
                # h2g, int, _, _, _ = stats.linregress(chr_ldsc[c], chr_xi_sq[c])
                # chr_h2g[c] = h2g
                chr_h2g[c] = (chr_xi_sq[c].mean() - 1.)*len(chr_ldsc[c]) / (chr_ldsc[c].mean()*self.N)

            return chr_h2g
        else:
            concat_ldsc = np.concatenate(list(chr_ldsc.values()))
            concat_xi_sq = np.concatenate(list(chr_xi_sq.values()))
            # h2g, int, _, _, _ = stats.linregress(concat_ldsc, concat_xi_sq)
            return (concat_xi_sq.mean() - 1.)*len(concat_ldsc) / (concat_ldsc.mean()*self.N)

    def compute_allele_frequency_plink(self):

        # Create a temporary directory for the allele frequency files:
        freq_tmpdir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='freq_')
        self.cleanup_dir_list.append(freq_tmpdir)

        # Create the samples file:
        keep_file = osp.join(freq_tmpdir.name, 'samples.keep')
        keep_table = self.to_individual_table()
        keep_table.to_csv(keep_file, index=False, header=False, sep="\t")

        self.maf = {}
        for c, bf in tqdm(self.bed_files.items(),
                          total=len(self.chromosomes),
                          desc="Computing allele frequencies using PLINK",
                          disable=not self.verbose):

            snp_keepfile = osp.join(freq_tmpdir.name, f"chr_{c}.keep")
            pd.DataFrame({'SNP': self.snps[c]}).to_csv(snp_keepfile,
                                                       index=False, header=False)

            plink_output = osp.join(freq_tmpdir.name, f"chr_{c}")

            cmd = [
                self.config.get('plink2_path'),
                f"--bfile {bf.replace('.bed', '')}",
                f"--keep {keep_file}",
                f"--extract {snp_keepfile}",
                f"--freq",
                f"--out {plink_output}",
                f"--threads {self.n_threads}"
            ]

            run_shell_script(" ".join(cmd))

            freq_df = pd.read_csv(plink_output + ".afreq", delim_whitespace=True)
            freq_df.rename(columns={'ID': 'SNP', 'ALT': 'A1', 'ALT_FREQS': 'MAF'}, inplace=True)
            merged_df = merge_snp_tables(pd.DataFrame({'SNP': self.snps[c], 'A1': self._a1[c]}), freq_df)

            if len(merged_df) != len(self.snps[c]):
                raise ValueError("Length of allele frequency table does not match number of SNPs.")

            self.maf[c] = merged_df['MAF'].values

        return self.maf

    def compute_allele_frequency(self):

        if self.use_plink:
            return self.compute_allele_frequency_plink()

        if self.n_per_snp is None:
            self.compute_n_per_snp()

        self.maf = {}
        for c, gt in tqdm(self.genotypes.items(),
                          total=len(self.chromosomes),
                          desc="Computing allele frequencies",
                          disable=not self.verbose):
            self.maf[c] = (gt.sum(axis=0) / (2. * self.n_per_snp[c])).compute().values

        return self.maf

    def compute_allele_frequency_variance(self):

        if self.maf is None:
            self.compute_allele_frequency()

        maf_var = {}

        for c, maf in tqdm(self.maf.items(),
                           total=len(self.chromosomes),
                           desc="Computing allele frequency variance",
                           disable=not self.verbose):
            maf_var[c] = 2.*maf*(1. - maf)

        return maf_var

    def compute_n_per_snp_plink(self):

        # Create a temporary directory for missingness count:
        miss_tmpdir = tempfile.TemporaryDirectory(dir=self.temp_dir, prefix='miss_')
        self.cleanup_dir_list.append(miss_tmpdir)

        # Create the samples file:
        keep_file = osp.join(miss_tmpdir.name, 'samples.keep')
        keep_table = self.to_individual_table()
        keep_table.to_csv(keep_file, index=False, header=False, sep="\t")

        self.n_per_snp = {}

        for c, bf in tqdm(self.bed_files.items(),
                          total=len(self.chromosomes),
                          desc="Computing effective sample size per SNP using PLINK",
                          disable=not self.verbose):

            snp_keepfile = osp.join(miss_tmpdir.name, f"chr_{c}.keep")
            pd.DataFrame({'SNP': self.snps[c]}).to_csv(snp_keepfile,
                                                       index=False, header=False)

            plink_output = osp.join(miss_tmpdir.name, f"chr_{c}")

            cmd = [
                self.config.get('plink2_path'),
                f"--bfile {bf.replace('.bed', '')}",
                f"--keep {keep_file}",
                f"--extract {snp_keepfile}",
                f"--missing variant-only",
                f"--out {plink_output}",
                f"--threads {self.n_threads}"
            ]

            run_shell_script(" ".join(cmd))

            miss_df = pd.read_csv(plink_output + ".vmiss", delim_whitespace=True)
            miss_df = pd.DataFrame({'ID': self.snps[c]}).merge(miss_df)

            if len(miss_df) != len(self.snps[c]):
                raise ValueError("Length of missingness table does not match number of SNPs.")

            self.n_per_snp[c] = (miss_df['OBS_CT'] - miss_df['MISSING_CT']).values

        return self.n_per_snp

    def compute_n_per_snp(self):

        if self.use_plink:
            return self.compute_n_per_snp_plink()

        self.n_per_snp = {}

        for c, gt in tqdm(self.genotypes.items(), total=len(self.chromosomes),
                          desc="Computing effective sample size per SNP",
                          disable=not self.verbose):
            self.n_per_snp[c] = gt.shape[0] - gt.isnull().sum(axis=0).compute().values

        return self.n_per_snp

    def compute_snp_pseudo_corr(self):
        """
        Computes the pseudo-correlation coefficient (standardized beta) between the SNP and
        the phenotype (X_jTy / N) from GWAS summary statistics.
        Uses Equation 15 in Mak et al. 2017
        beta =  z_j / sqrt(n - 1 + z_j^2)
        Where z_j is the marginal GWAS Z-score
        """

        if self.z_scores is None:
            raise Exception("Z-scores are not set!")
        if self.n_per_snp is None:
            raise Exception("Sample size is not set!")

        snp_corr = {}
        for c, zsc in tqdm(self.z_scores.items(),
                           total=len(self.chromosomes),
                           desc="Computing SNP-wise correlations",
                           disable=not self.verbose):
            # z_j / sqrt(n - 1 + z_j^2)
            snp_corr[c] = zsc / (np.sqrt(self.n_per_snp[c] - 1 + zsc**2))

        return snp_corr

    def compute_yy_per_snp(self):
        """
        Computes the quantity (y'y)_j/n_j following SBayesR (Lloyd-Jones 2019) and Yang et al. (2012).
        (y'y)_j/n_j is the empirical variance for continuous phenotypes and may be estimated
        from GWAS summary statistics by re-arranging the equation for the
        squared standard error:

        SE(b_j)^2 = (Var(y) - Var(x_j)*b_j^2) / (Var(x)*n)

        Which gives the following estimate:

        (y'y)_j / n_j = (n_j - 2)*SE(b_j)^2 + b_j^2

        TODO: Verify the derivation and logic here, ensure it's consistent.

        """

        if self.beta_hats is None:
            raise Exception("Betas are not set!")
        if self.n_per_snp is None:
            raise Exception("Sample size is not set!")
        if self.se is None:
            raise Exception("Standard errors are not set!")

        yy = {}

        for c, b_hat in tqdm(self.beta_hats.items(),
                             total=len(self.chromosomes),
                             desc="Computing SNP-wise yTy",
                             disable=not self.verbose):
            yy[c] = (self.n_per_snp[c] - 2)*self.se[c]**2 + b_hat**2

        return yy

    def compute_beta_hats(self, chrom=None):

        if self.phenotypes is None or self.genotypes is None:
            raise Exception("Genotype and phenotype data are needed to compute betas!")

        if self.maf is None:
            self.compute_allele_frequency()

        if chrom is None:
            self.beta_hats = {}
            chroms = self.chromosomes
        else:
            if chrom not in self.chromosomes:
                raise KeyError("Chromosome is not valid!")

            if self.beta_hats is None:
                self.beta_hats = {}

            chroms = [chrom]

        for c in tqdm(chroms, desc="Computing beta hats", disable=not self.verbose):

            if self.standardize_genotype:
                numer = np.dot(standardize_genotype_matrix(self.genotypes[c]).T, self.phenotypes)
                denom = self.n_per_snp[c]
            else:
                numer = np.dot(self.genotypes[c].fillna(self.maf[c]).T, self.phenotypes)
                denom = self.n_per_snp[c] * self.genotypes[c].var(axis=0).compute()

            self.beta_hats[c] = numer / denom

        return self.beta_hats

    def compute_standard_errors(self, chrom=None):

        if self.phenotypes is None or self.genotypes is None:
            raise Exception("Genotype and phenotype data are needed to compute standard errors!")

        if self.n_per_snp is None:
            self.compute_n_per_snp()

        if chrom is None:
            self.se = {}
            chroms = self.chromosomes
        else:
            if chrom not in self.chromosomes:
                raise KeyError("Chromosome is not valid!")

            if self.se is None:
                self.se = {}

            chroms = [chrom]

        sigma_y = np.var(self.phenotypes)  # phenotypic variance

        for c in tqdm(chroms, desc="Computing standard errors", disable=not self.verbose):

            if self.standardize_genotype:
                xtx = self.n_per_snp[c]
            else:
                xtx = self.n_per_snp[c]*self.genotypes[c].var(axis=0).compute()

            self.se[c] = np.sqrt(sigma_y/xtx)

        return self.se

    def compute_z_scores(self, chrom=None):

        if self.beta_hats is None or self.se is None:
            raise Exception("beta hats and standard errors are needed to compute z-scores!")

        if chrom is None:
            self.z_scores = {}
            chroms = self.chromosomes
        else:
            if chrom not in self.chromosomes:
                raise KeyError("Chromosome is not valid!")

            if self.z_scores is None:
                self.z_scores = {}

            chroms = [chrom]

        for c in tqdm(chroms, desc="Computing z-scores", disable=not self.verbose):
            self.z_scores[c] = self.beta_hats[c] / self.se[c]

        return self.z_scores

    def compute_p_values(self, chrom=None, log10=False):

        if self.z_scores is None:
            raise Exception("Z-scores are needed to compute p-values!")

        if chrom is None:
            self.p_values = {}
            chroms = self.chromosomes
        else:
            if chrom not in self.chromosomes:
                raise KeyError("Chromosome is not valid!")

            if self.p_values is None:
                self.p_values = {}

            chroms = [chrom]

        for c in tqdm(chroms, desc="Computing p-values", disable=not self.verbose):
            self.p_values[c] = 2.*stats.norm.sf(abs(self.z_scores[c]))
            if log10:
                self.p_values[c] = np.log10(self.p_values[c])

        return self.p_values

    def to_individual_table(self):

        if self._iid is None:
            raise Exception("Individual data is not provided!")

        return pd.DataFrame({
            'FID': self._fid,
            'IID': self._iid
        })

    def to_phenotype_table(self):

        if self.phenotypes is None:
            print("Warning: Phenotypes are not set! Exporting NaNs")

        pheno_df = self.to_individual_table()
        pheno_df['phenotype'] = self.phenotypes

        return pheno_df

    def to_snp_table(self, per_chromosome=False, col_subset=None):

        if col_subset is None:
            col_subset = ['CHR', 'SNP', 'POS', 'A1', 'A2', 'MAF', 'N', 'BETA', 'Z', 'SE', 'PVAL']

        snp_tables = {}

        for c in self.chromosomes:

            ss_df = pd.DataFrame({'SNP': self.snps[c], 'A1': self.alt_alleles[c]})

            for col in col_subset:
                if col == 'CHR':
                    ss_df['CHR'] = c
                if col == 'POS' and self.bp_pos is not None:
                    ss_df['POS'] = self.bp_pos[c]
                if col == 'A2' and self.ref_alleles is not None:
                    ss_df['A2'] = self.ref_alleles[c]
                if col == 'MAF' and self.maf is not None:
                    ss_df['MAF'] = self.maf[c]
                if col == 'N' and self.n_per_snp is not None:
                    ss_df['N'] = self.n_per_snp[c]
                if col == 'BETA' and self.beta_hats is not None:
                    ss_df['BETA'] = self.beta_hats[c]
                if col == 'Z' and self.z_scores is not None:
                    ss_df['Z'] = self.z_scores[c]
                if col == 'SE' and self.se is not None:
                    ss_df['SE'] = self.se[c]
                if col == 'PVAL' and self.p_values is not None:
                    ss_df['PVAL'] = self.p_values[c]

            snp_tables[c] = ss_df[list(col_subset)]

        if per_chromosome:
            return snp_tables
        else:
            return pd.concat(list(snp_tables.values()))

    def cleanup(self):
        """
        Clean up all temporary files and directories
        """
        if self.verbose:
            print("> Cleaning up workspace.")

        for tmpdir in self.cleanup_dir_list:
            try:
                tmpdir.cleanup()
            except FileNotFoundError:
                continue
