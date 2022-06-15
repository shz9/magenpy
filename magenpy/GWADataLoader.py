"""
Author: Shadi Zabad
Date: December 2020
"""

from typing import Union, Dict
import tempfile

import pandas as pd
import numpy as np

from .GenotypeMatrix import GenotypeMatrix, xarrayGenotypeMatrix, plinkBEDGenotypeMatrix
from .SampleTable import SampleTable
from .SumstatsTable import SumstatsTable
from .AnnotationMatrix import AnnotationMatrix
from .LDMatrix import LDMatrix

from .utils.compute_utils import iterable
from .utils.system_utils import makedir, get_filenames


class GWADataLoader(object):

    def __init__(self,
                 bed_files=None,
                 phenotype_file=None,
                 covariates_file=None,
                 keep_samples=None,
                 keep_file=None,
                 extract_snps=None,
                 extract_file=None,
                 min_maf=None,
                 min_mac=1,
                 drop_duplicated=True,
                 phenotype_likelihood='gaussian',
                 sumstats_files=None,
                 sumstats_format='magenpy',
                 ld_store_files=None,
                 annotation_files=None,
                 backend='xarray',
                 temp_dir='temp',
                 output_dir='output',
                 verbose=True,
                 n_threads=1):

        # ------- Sanity checks -------

        assert backend in ('xarray', 'plink')
        assert phenotype_likelihood in ('gaussian', 'binomial')

        # ------- General options -------

        self.backend = backend

        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.cleanup_dir_list = []  # Directories to clean up after execution.

        makedir([temp_dir, output_dir])

        self.verbose = verbose
        self.n_threads = n_threads

        # ------- General parameters -------

        self.phenotype_likelihood: str = phenotype_likelihood

        self.genotype: Union[Dict[int, GenotypeMatrix], None] = None
        self.sample_table: Union[SampleTable, None] = None
        self.ld: Union[Dict[int, LDMatrix], None] = None
        self.sumstats_table: Union[Dict[int, SumstatsTable], None] = None
        self.annotation: Union[Dict[int, AnnotationMatrix], None] = None

        # ------- Read data files -------

        self.read_genotypes(bed_files,
                            min_maf=min_maf,
                            min_mac=min_mac,
                            drop_duplicated=drop_duplicated)
        self.read_phenotype(phenotype_file)
        self.read_covariates(covariates_file)
        self.read_ld(ld_store_files)
        self.read_annotations(annotation_files)
        self.read_summary_statistics(sumstats_files,
                                     sumstats_format,
                                     drop_duplicated=drop_duplicated)

        # ------- Filter samples or SNPs -------

        if extract_snps is not None or extract_file is not None:
            self.filter_snps(extract_snps=extract_snps, extract_file=extract_file)

        if keep_samples is not None or keep_file is not None:
            self.filter_samples(keep_samples=keep_samples, keep_file=keep_file)

        # ------- Harmonize data sources -------

        self.harmonize_data()

    @property
    def sample_size(self, agg=np.max):
        """
        The number of samples.

        :param agg: A vectorized function to aggregate the sample-size per SNP (if available).
        By default, we take the maximum sample size per SNP e.g. `np.max`, but you may also
        take the mean by passing `np.mean` instead.
        """
        if self.sample_table:
            return self.sample_table.n
        elif self.sumstats_table:
            return agg([agg(ss.n_per_snp) for ss in self.sumstats_table.values()])
        else:
            raise Exception("Information about the sample size is not available!")

    @property
    def n(self, agg=np.max):
        """
        The number of samples. See also `.sample_size()`.

        :param agg: A vectorized function to aggregate the sample-size per SNP (if available).
        By default, we take the maximum sample size per SNP e.g. `np.max`, but you may also
        take the mean by, e.g. passing `np.mean` instead.
        """

        return self.sample_size(agg=agg)

    @property
    def m(self):
        """
        The number of variants. See also `.n_snps`
        """
        return sum(self.shapes.values())

    @property
    def n_snps(self):
        """
        The number of variants. See also `.m`
        """
        return self.m

    @property
    def shapes(self):
        """
        Return a dictionary where the key is the chromosome number and the value is
        the number of variants on that chromosome.
        """
        if self.genotype:
            return {c: g.shape[1] for c, g in self.genotype.items()}
        elif self.sumstats_table:
            return {c: s.shape[0] for c, s in self.sumstats_table.items()}
        elif self.ld:
            return {c: l.shape[0] for c, l in self.ld.items()}
        elif self.annotation:
            return {c: a.shape[0] for c, a in self.annotation.items()}
        else:
            raise Exception("GWADataLoader is not properly initialized!")

    @property
    def chromosomes(self):
        """
        Return the list of chromosomes that were loaded to `GWADataLoader`.
        """
        return list(self.shapes.keys())

    @property
    def n_annotations(self):
        """
        Return the number of annotations included in the annotation matrices.
        """
        assert self.annotation
        return self.annotation[self.chromosomes[0]].n_annotations

    def filter_snps(self, extract_snps=None, extract_file=None, chromosome=None):
        """
        Filter the SNP set from all the GWADataLoader objects.
        :param extract_snps: A list or vector of SNP IDs to keep.
        :param extract_file: A path to a plink-style file with SNP IDs to keep.
        :param chromosome: Chromosome number. If specified, applies the filter on that chromosome only.
        """

        if extract_snps is None and extract_file is None:
            return

        if chromosome:
            chroms = [chromosome]
        else:
            chroms = self.chromosomes

        if not extract_snps:
            from .parsers.misc_parsers import read_snp_filter_file
            extract_snps = read_snp_filter_file(extract_file)

        for c in chroms:

            # Filter the genotype matrix:
            if self.genotype and c in self.genotype:
                self.genotype[c].filter_snps(extract_snps=extract_snps)

            # Filter the summary statistics table:
            if self.sumstats_table and c in self.sumstats_table:
                self.sumstats_table[c].filter_snps(extract_snps=extract_snps)

            if self.ld and c in self.ld:
                self.ld[c].filter_snps(extract_snps=extract_snps)

            # Filter the annotation matrix:
            if self.annotation and c in self.annotation:
                self.annotation[c].filter_snps(extract_snps=extract_snps)

    def filter_samples(self, keep_samples=None, keep_file=None):
        """
        Filter samples from the samples table. User must specify
        either a list of samples to keep or the path to a file
        with the list of samples to keep.

        :param keep_samples: A list (or array) of sample IDs to keep.
        :param keep_file: The path to a file with the list of samples to keep.
        """

        self.sample_table.filter_samples(keep_samples=keep_samples, keep_file=keep_file)
        self.sync_sample_tables()

    def read_annotations(self, annot_files):
        """
        Read the annotation matrix from file. Annotations are a set of features associated
        with each SNP and are generally represented in table format. Here, we support tables
        of the form developed and standardized by the LDSC software.
        Consult the documentation for `AnnotationMatrix` for more details.

        :param annot_files: The path to the annotation file(s). You may use a wildcard here to
        read files for multiple chromosomes.
        """

        if annot_files is None:
            return

        # Find all the relevant files in the path passed by the user:
        if not iterable(annot_files):
            annot_files = get_filenames(annot_files, extension='.annot')

        if self.verbose:
            print("> Reading annotation files...")

        self.annotation = {}

        for annot_file in annot_files:
            annot_mat = AnnotationMatrix.from_file(annot_file)
            self.annotation[annot_mat.chromosome] = annot_mat

    def read_genotypes(self,
                       bed_files,
                       keep_samples=None,
                       keep_file=None,
                       extract_snps=None,
                       extract_file=None,
                       min_maf=None,
                       min_mac=1,
                       drop_duplicated=True):
        """
        Read the genotype matrix and/or associated metadata from plink's BED file format.
        Consult the documentation for `GenotypeMatrix` for more details.

        :param bed_files: The path to the BED file(s). You may use a wildcard here to read files for multiple
        chromosomes.
        :param keep_samples: A vector or list of sample IDs to keep when filtering the genotype matrix.
        :param keep_file: A path to a plink-style file containing sample IDs to keep.
        :param extract_snps: A vector or list of SNP IDs to keep when filtering the genotype matrix.
        :param extract_file: A path to a plink-style file containing SNP IDs to keep.
        :param min_maf: The minimum minor allele frequency cutoff.
        :param min_mac: The minimum minor allele count cutoff.
        :param drop_duplicated: If True, drop SNPs with duplicated rsID.
        """

        if bed_files is None:
            return

        # Find all the relevant files in the path passed by the user:
        if not iterable(bed_files):
            bed_files = get_filenames(bed_files, extension='.bed')

        # Depending on the backend, select the `GenotypeMatrix` class:
        if self.backend == 'xarray':
            gmat_class = xarrayGenotypeMatrix
        else:
            gmat_class = plinkBEDGenotypeMatrix

        if self.verbose:
            print("> Reading genotype files...")

        self.genotype = {}

        for bfile in bed_files:
            # Read BED file and update the genotypes dictionary:
            self.genotype.update(gmat_class.from_file(bfile, temp_dir=self.temp_dir).split_by_chromosome())

        # After reading the genotype matrices, apply some standard filters:
        for i, (c, g) in enumerate(self.genotype.items()):

            # Filter the genotype matrix to keep certain subsample:
            if keep_samples or keep_file:
                g.filter_samples(keep_samples=keep_samples, keep_file=keep_file)

            # Filter the genotype matrix to keep certain SNPs
            if extract_snps or extract_file:
                g.filter_snps(extract_snps=extract_snps, extract_file=extract_file)

            # Drop duplicated SNP IDs
            if drop_duplicated:
                g.drop_duplicated_snps()

            # Filter SNPs by minor allele frequency and/or count:
            g.filter_by_allele_frequency(min_maf=min_maf, min_mac=min_mac)

            if i == 0:
                self.sample_table = g.sample_table

    def read_phenotype(self, phenotype_file, drop_na=True, **read_csv_kwargs):
        """
        Read the phenotype file and integrate it with the sample tables and genotype matrices.

        :param phenotype_file: The path to the phenotype file
        (Default: tab-separated file with `FID IID phenotype` columns).
        :param drop_na: Drop samples with missing phenotype information.
        :param read_csv_kwargs: keyword arguments for the `read_csv` function of `pandas`.
        """

        if phenotype_file is None:
            return

        if self.verbose:
            print("> Reading phenotype file...")

        assert self.sample_table

        self.sample_table.read_phenotype_file(phenotype_file, drop_na=drop_na, **read_csv_kwargs)
        self.sync_sample_tables()

    def set_phenotype(self, new_phenotype, phenotype_likelihood=None):
        """
        A convenience method to update the phenotype column for the samples.
        :param new_phenotype: A vector or list of phenotype values.
        :param phenotype_likelihood: The phenotype likelihood (e.g. `binomial`, `gaussian`). Optional.
        """

        self.sample_table.set_phenotype(new_phenotype,
                                        phenotype_likelihood=phenotype_likelihood or self.phenotype_likelihood)
        self.sync_sample_tables()

    def read_covariates(self, covariates_file, **read_csv_kwargs):
        """
        Read the covariates file and integrate it with the sample tables and genotype matrices.

        :param covariates_file: The path to the covariates file
        (Default: tab-separated file starting with the `FID IID ...` columns and followed by the covariate columns).
        :param read_csv_kwargs: keyword arguments for the `read_csv` function of `pandas`.
        """

        if covariates_file is None:
            return

        if self.verbose:
            print("> Reading covariates file...")

        assert self.sample_table

        self.sample_table.read_covariates_file(covariates_file, **read_csv_kwargs)
        self.sync_sample_tables()

    def read_summary_statistics(self,
                                sumstats_path,
                                sumstats_format='magenpy',
                                parser=None,
                                drop_duplicated=True,
                                **parse_kwargs):
        """
        Read GWAS summary statistics file(s) and parse them to `SumstatsTable` objects.

        :param sumstats_path: The path to the summary statistics file(s). The path may be a wildcard.
        :param sumstats_format: The format for the summary statistics. Currently supports the following
         formats: `plink`, `magenpy`, `fastGWA`, `COJO`.
        :param parser: If the summary statistics file does not follow any of the formats above, you can create
        your own parser by inheriting from the base `SumstatsParser` class and passing it here as an argument.
        :param drop_duplicated: Drop SNPs with duplicated rsIDs.
        :param parse_kwargs: keyword arguments for the parser. These are mainly parameters that will be passed to
        `pandas.read_csv` function, such as the delimiter, header information, etc.
        """

        if sumstats_path is None:
            return

        if not iterable(sumstats_path):
            sumstats_files = get_filenames(sumstats_path)

            from .utils.system_utils import valid_url
            if len(sumstats_files) < 1 and valid_url(sumstats_path):
                sumstats_files = [sumstats_path]
        else:
            sumstats_files = sumstats_path

        if self.verbose:
            print("> Reading summary statistics files...")

        self.sumstats_table = {}

        for f in sumstats_files:

            ss_tab = SumstatsTable.from_file(f, sumstats_format=sumstats_format, parser=parser, **parse_kwargs)

            if drop_duplicated:
                ss_tab.drop_duplicates()

            if 'CHR' in ss_tab.table.columns:
                self.sumstats_table.update(ss_tab.split_by_chromosome())
            else:
                if self.genotype:
                    ref_table = {c: g.snps for c, g in self.genotype.items()}
                elif self.ld:
                    ref_table = {c: ld.snps for c, ld in self.ld.items()}
                else:
                    raise Exception("Cannot index summary statistics tables without chromosome information!")

                self.sumstats_table.update(ss_tab.split_by_chromosome(snps_per_chrom=ref_table))

    def read_ld(self, ld_store_files):
        """
        Read the LD matrix files stored on-disk in Zarr array format.
        :param ld_store_files: The path to the LD matrices. This may be a wildcard to accommodate reading data
        for multiple chromosomes.
        """

        if ld_store_files is None:
            return

        if self.verbose:
            print("> Reading LD matrices...")

        if not iterable(ld_store_files):
            ld_store_files = get_filenames(ld_store_files, extension='.zarray')

        self.ld = {}

        for f in ld_store_files:
            z = LDMatrix.from_path(f)
            self.ld[z.chromosome] = z

    def load_ld(self):
        """
        A utility function to load the LD matrices to memory from on-disk storage.
        """
        if self.ld is not None:
            for ld in self.ld.values():
                ld.load()

    def release_ld(self):
        """
        A utility function to release LD matrices from memory.
        """
        if self.ld is not None:
            for ld in self.ld.values():
                ld.release()

    def compute_ld(self, estimator, output_dir, **ld_kwargs):
        """
        Compute the Linkage-Disequilibrium (LD) matrix or SNP-by-SNP Pearson
        correlation matrix between genetic variants. This function only considers correlations
        between SNPs on the same chromosome. This is a utility function that calls the
        `.compute_ld()` method of the `GenotypeMatrix` objects associated with
        GWADataLoader.

        :param estimator: The estimator for the LD matrix. We currently support
        4 different estimators: `sample`, `windowed`, `shrinkage`, and `block`.
        :param output_dir: The output directory where the Zarr array containing the
        entries of the LD matrix will be stored.
        :param ld_kwargs: keyword arguments for the various LD estimators. Consult
        the implementations of `WindowedLD`, `ShrinkageLD`, and `BlockLD` for details.
        """

        self.ld = {c: g.compute_ld(estimator, output_dir, **ld_kwargs)
                   for c, g in self.genotype.items()}

    def get_ld_matrices(self):
        return self.ld

    def get_ld_boundaries(self):
        if self.ld is None:
            return None

        return {c: ld.get_masked_boundaries() for c, ld in self.ld.items()}

    def harmonize_data(self):
        """
        This method ensures that all the data sources (reference genotype,
        LD matrices, summary statistics, annotations) are all aligned in terms of the
        set of variants that they operate on as well as the designation of the effect allele for
        each variant.
        """

        # If less than two data sources are present, skip harmonization...
        if sum([self.genotype is not None, self.sumstats_table is not None,
                self.ld is not None, self.annotation is not None]) < 2:
            return

        if self.verbose:
            print("> Harmonizing data...")

        for c in self.chromosomes:

            data_source = []

            if self.genotype and c in self.genotype:
                data_source.append(self.genotype[c])
            if self.sumstats_table and c in self.sumstats_table:
                data_source.append(self.sumstats_table[c])
            if self.ld and c in self.ld:
                data_source.append(self.ld[c])
            if self.annotation and c in self.annotation:
                data_source.append(self.annotation[c])

            if len(data_source) > 1:

                # Find the set of SNPs that are shared across all data sources:
                merged_snps = np.array(list(set.intersection(*[set(ds.snps) for ds in data_source])))

                # If necessary, filter the data sources to only have those SNPs:
                for ds in data_source:
                    if len(ds.snps) != len(merged_snps):
                        ds.filter_snps(extract_snps=merged_snps)

                # Check for flips in effect allele in summary statistics data:
                if self.sumstats_table is not None:
                    if self.genotype is not None:
                        self.sumstats_table[c].match(self.genotype[c].get_snp_table(col_subset=['SNP', 'A1']))
                    elif self.ld is not None:
                        self.sumstats_table[c].match(self.ld[c].to_snp_table(col_subset=['SNP', 'A1']))

    def perform_gwas(self, **gwa_kwargs):
        """
        Perform genome-wide association testing of all variants against the phenotype.
        This is a utility function that calls the `.perform_gwas()` method of the
        `GenotypeMatrix` objects associated with GWADataLoader.

        :param gwa_kwargs: Keyword arguments to pass to the GWA functions. Consult stats.gwa.utils
        for relevant keyword arguments for each backend.
        """

        self.sumstats_table = {c: g.perform_gwas(**gwa_kwargs) for c, g in self.genotype.items()}

    def score(self, beta=None):
        """
        Perform linear scoring, i.e. multiply the genotype matrix by the vector of effect sizes, `beta`.

        :param beta: A dictionary where the keys are the chromosome numbers and the
        values are a vector of effect sizes for each variant on that chromosome. If the
        betas are not provided, we use the marginal betas by default (if those are available).
        """

        if not beta:
            try:
                beta = {c: s.marginal_beta for c, s in self.sumstats_table.items()}
            except Exception:
                raise Exception("To perform linear scoring, you must a provide effect size estimates (BETA)!")

        pgs = None

        for c, g in self.genotype.items():
            if pgs is None:
                pgs = g.score(beta[c])
            else:
                pgs += g.score(beta[c])

        return pgs

    def predict(self, beta=None):
        """
        Predict the phenotype for the genotyped samples using the provided effect size
        estimates `beta`. For quantitative traits, this is equivalent to performing
        linear scoring. For binary phenotypes, we transform the output using probit link function.

        :param beta: A dictionary where the keys are the chromosome numbers and the
        values are a vector of effect sizes for each variant on that chromosome. If the
        betas are not provided, we use the marginal betas by default (if those are available).
        """

        # Perform linear scoring:
        pgs = self.score(beta)

        if self.phenotype_likelihood == 'binomial':
            # Apply probit link function:
            from scipy.stats import norm
            pgs = norm.cdf(pgs)

        return pgs

    def to_individual_table(self):
        """
        Get a plink-style dataframe of individual IDs, in the form of
        Family ID (FID) and Individual ID (IID).
        """

        return self.sample_table.get_individual_table()

    def to_phenotype_table(self):
        """
        Get a plink-style dataframe with each individual's Family ID (FID),
        Individual ID (IID), and phenotype value.
        """

        return self.sample_table.get_phenotype_table()

    def to_snp_table(self, col_subset=None, per_chromosome=False):
        """
        Return a dataframe of SNP information for all variants
        across different chromosomes.

        :param col_subset: The subset of columns to obtain.
        :param per_chromosome: If True, returns a dictionary where the key
        is the chromosome number and the value is the SNP table per
        chromosome.
        """

        snp_tables = {}

        for c in self.chromosomes:
            if self.sumstats_table:
                snp_tables[c] = self.sumstats_table[c].get_table(col_subset=col_subset)
            elif self.genotype:
                snp_tables[c] = self.genotype[c].get_snp_table(col_subset=col_subset)
            elif self.ld:
                snp_tables[c] = self.ld[c].to_snp_table(col_subset=col_subset)
            else:
                raise Exception("GWADataLoader is not properly initialized!")

        if per_chromosome:
            return snp_tables
        else:
            return pd.concat(list(snp_tables.values()))

    def to_summary_statistics_table(self, col_subset=None, per_chromosome=False):
        """
        Return a dataframe of the GWAS summary statistics for all variants
        across different chromosomes.

        :param col_subset: The subset of columns (or summary statistics) to obtain.
        :param per_chromosome: If True, returns a dictionary where the key
        is the chromosome number and the value is the summary statistics table per
        chromosome.
        """

        assert self.sumstats_table

        snp_tables = {}

        for c in self.chromosomes:
            snp_tables[c] = self.sumstats_table[c].get_table(col_subset=col_subset)

        if per_chromosome:
            return snp_tables
        else:
            return pd.concat(list(snp_tables.values()))

    def sync_sample_tables(self):
        """
        A utility method to sync the sample tables of the
        `GenotypeMatrix` objects with the sample table under
        the `GWADataLoader` object. This is especially important
        when setting new phenotypes (from the simulators) or reading
        covariates files, etc.
        """

        for c, g in self.genotype.items():
            g.set_sample_table(self.sample_table)

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

        # Clean up the temporary files associated with the genotype matrices:
        if self.genotype:
            for g in self.genotype.values():
                g.cleanup()
