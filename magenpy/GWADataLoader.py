import copy

# Set up the logger:
import logging
from typing import Dict, Union

import numpy as np
import pandas as pd

from .AnnotationMatrix import AnnotationMatrix
from .GenotypeMatrix import *
from .LDMatrix import LDMatrix
from .SampleTable import SampleTable
from .SumstatsTable import SumstatsTable
from .utils.compute_utils import iterable
from .utils.model_utils import build_snp_harmonization_plan, match_chromosomes
from .utils.system_utils import get_filenames, makedir

logger = logging.getLogger(__name__)


def tqdm(*args, **kwargs):
    """Import tqdm only when a progress-producing operation is started."""

    from tqdm import tqdm as _tqdm

    return _tqdm(*args, **kwargs)


class GWADataLoader(object):
    """
    A class to load and manage multiple data sources for genetic association studies.
    This class is designed to handle genotype matrices, summary statistics, LD matrices,
    and annotation matrices. It also provides functionalities to filter samples and/or SNPs,
    harmonize data sources, and compute LD matrices. This is all done in order to facilitate
    downstream statistical genetics analyses that require multiple data sources to be aligned
    and harmonized. The use cases include:

    * Summary statistics-based PRS computation
    * Summary statistics-based heritability estimation.
    * Complex trait simulation.
    * Performing Genome-wide association tests.

    :ivar genotype: A dictionary of `GenotypeMatrix` objects, where the key is the chromosome number.
    :ivar sample_table: A `SampleTable` object containing the sample information.
    :ivar phenotype_likelihood: The likelihood of the phenotype (e.g. `gaussian`, `binomial`).
    :ivar ld: A dictionary of `LDMatrix` objects, where the key is the chromosome number.
    :ivar sumstats_table: A dictionary of `SumstatsTable` objects, where the key is the chromosome number.
    :ivar annotation: A dictionary of `AnnotationMatrix` objects, where the key is the chromosome number.
    :ivar backend: The backend used for genotype-backed computations. Currently, supports
    `magenpy`, `bed-reader`, `plink`, and `xarray`.
    :ivar temp_dir: The temporary directory where we store intermediate files (if necessary).
    """

    def __init__(
        self,
        bed_files=None,
        phenotype_file=None,
        covariates_file=None,
        keep_samples=None,
        keep_file=None,
        extract_snps=None,
        extract_file=None,
        min_maf=None,
        min_mac=None,
        drop_duplicated=True,
        phenotype_likelihood="gaussian",
        sumstats_files=None,
        sumstats_format="magenpy",
        ld_store_files=None,
        annotation_files=None,
        annotation_format="magenpy",
        backend="magenpy",
        temp_dir="temp",
        genome_build=None,
        threads=1,
    ):
        """
        Initialize the `GWADataLoader` object with the data sources required for
        downstream statistical genetics analyses.

        :param bed_files: The path to the BED file(s). You may use a wildcard here to read files for multiple
        chromosomes.
        :param phenotype_file: The path to the phenotype file.
        (Default: tab-separated file with `FID IID phenotype` columns).
        :param covariates_file: The path to the covariates file.
        (Default: tab-separated file starting with the `FID IID ...` columns and followed by the covariate columns).
        :param keep_samples: A vector or list of sample IDs to keep when filtering the genotype matrix.
        :param keep_file: A path to a plink-style keep file to select a subset of individuals.
        :param extract_snps: A vector or list of SNP IDs to keep when filtering the genotype matrix.
        :param extract_file: A path to a plink-style extract file to select a subset of SNPs.
        :param min_maf: The minimum minor allele frequency cutoff.
        :param min_mac: The minimum minor allele count cutoff.
        :param drop_duplicated: If True, drop SNPs with duplicated rsID.
        :param phenotype_likelihood: The likelihood of the phenotype (e.g. `gaussian`, `binomial`).
        :param sumstats_files: The path to the summary statistics file(s). The path may be a wildcard.
        :param sumstats_format: The format for the summary statistics. Currently, supports the following
        formats: `plink1.9`, `plink2`, `magenpy`, `fastGWA`, `COJO`, `SAIGE`, or `GWASCatalog` for the standard
        summary statistics format (also known as `ssf` or `gwas-ssf`).
        :param ld_store_files: The path to the LD matrices. This may be a wildcard to accommodate reading data
        for multiple chromosomes.
        :param annotation_files: The path to the annotation file(s). The path may contain a wildcard.
        :param annotation_format: The format for the summary statistics. Currently, supports the following
        formats: `magenpy`, `ldsc`.
        :param backend: The backend used for computations with the genotype matrix. Currently, supports
        `magenpy`, `bed-reader`, `plink`, and `xarray`.
        :param temp_dir: The temporary directory where to store intermediate files.
        :param genome_build: The genome build or assembly under which the SNP coordinates are defined.
        :param threads: The number of threads to use for computations.
        """

        # ------- Sanity checks -------

        assert backend in ("xarray", "plink", "bed-reader", "magenpy")
        assert phenotype_likelihood in ("gaussian", "binomial")

        # ------- General options -------

        self.backend = backend

        self.temp_dir = temp_dir
        self.cleanup_dir_list = []  # Directories to clean up after execution.

        makedir(temp_dir)

        self.threads = threads

        # ------- General parameters -------

        self.phenotype_likelihood: str = phenotype_likelihood

        self.genotype: Union[Dict[int, GenotypeMatrix], None] = None
        self.sample_table: Union[SampleTable, None] = None
        self.ld: Union[Dict[int, LDMatrix], None] = None
        self.sumstats_table: Union[Dict[int, SumstatsTable], None] = None
        self.annotation: Union[Dict[int, AnnotationMatrix], None] = None

        # ------- Read data files -------

        self.read_genotypes(
            bed_files,
            min_maf=min_maf,
            min_mac=min_mac,
            drop_duplicated=drop_duplicated,
            genome_build=genome_build,
        )
        self.read_phenotype(phenotype_file)
        self.read_covariates(covariates_file)
        self.read_ld(ld_store_files)
        self.read_annotations(annotation_files, annot_format=annotation_format)
        self.read_summary_statistics(
            sumstats_files, sumstats_format, drop_duplicated=drop_duplicated
        )

        # ------- Filter samples or SNPs -------

        if extract_snps is not None or extract_file is not None:
            self.filter_snps(extract_snps=extract_snps, extract_file=extract_file)

        if keep_samples is not None or keep_file is not None:
            self.filter_samples(keep_samples=keep_samples, keep_file=keep_file)

        # ------- Harmonize data sources -------

        self.harmonize_data()

    @property
    def samples(self):
        """
        :return: The list of samples retained in the sample table.
        """
        if self.sample_table is not None:
            return self.sample_table.iid

    @property
    def sample_size(self):
        """

        !!! seealso "See Also"
            * [n][magenpy.GWADataLoader.GWADataLoader.n]

        :return: The number of samples in the genotype matrix.

        """
        if self.sample_table is not None:
            return self.sample_table.n
        elif self.sumstats_table is not None:
            return np.max([np.max(ss.n_per_snp) for ss in self.sumstats_table.values()])
        else:
            raise ValueError("Information about the sample size is not available!")

    @property
    def n(self):
        """
        !!! seealso "See Also"
            * [sample_size][magenpy.GWADataLoader.GWADataLoader.sample_size]

        :return: The number of samples in the genotype matrix.
        """

        return self.sample_size

    @property
    def snps(self):
        """
        :return: The list of SNP rsIDs retained in each chromosome.
        :rtype: dict
        """
        if self.genotype is not None:
            return {c: g.snps for c, g in self.genotype.items()}
        elif self.sumstats_table is not None:
            return {c: s.snps for c, s in self.sumstats_table.items()}
        elif self.ld is not None:
            return {c: l.snps for c, l in self.ld.items()}
        elif self.annotation is not None:
            return {c: a.snps for c, a in self.annotation.items()}
        else:
            raise ValueError("GWADataLoader instance is not properly initialized!")

    @property
    def m(self):
        """
        !!! seealso "See Also"
            * [n_snps][magenpy.GWADataLoader.GWADataLoader.n_snps]

        :return: The number of variants in the harmonized data sources.
        """
        return sum(self.shapes.values())

    @property
    def n_snps(self):
        """
        !!! seealso "See Also"
            * [m][magenpy.GWADataLoader.GWADataLoader.m]

        :return: The number of variants in the harmonized data sources.
        """
        return self.m

    @property
    def shapes(self) -> Dict[int, int | None]:
        """
        :return: A dictionary where the key is the chromosome number and the value is
        the number of variants on that chromosome.
        """
        if self.genotype is not None:
            return {c: g.shape[1] for c, g in self.genotype.items()}
        elif self.sumstats_table is not None:
            return {c: s.shape[0] for c, s in self.sumstats_table.items()}
        elif self.ld is not None:
            return {c: l.n_snps for c, l in self.ld.items()}
        elif self.annotation is not None:
            return {c: a.shape[0] for c, a in self.annotation.items()}
        else:
            raise ValueError("GWADataLoader instance is not properly initialized!")

    @property
    def chromosomes(self):
        """
        :return: The list of chromosomes that were loaded to `GWADataLoader`.
        """
        return sorted(list(self.shapes.keys()))

    @property
    def n_annotations(self):
        """
        :return: The number of annotations included in the annotation matrices.
        """
        if self.annotation is not None:
            return self.annotation[self.chromosomes[0]].n_annotations

    def filter_snps(self, extract_snps=None, extract_file=None, chromosome=None):
        """
        Filter the SNP set from all the GWADataLoader objects.
        :param extract_snps: A list or array of SNP rsIDs to keep.
        :param extract_file: A path to a plink-style file with SNP rsIDs to keep.
        :param chromosome: Chromosome number. If specified, applies the filter to that chromosome only.
        """

        if extract_snps is None and extract_file is None:
            return

        if chromosome is not None:
            chroms = [chromosome]
        else:
            chroms = self.chromosomes

        if extract_snps is None:
            from .parsers.misc_parsers import read_snp_filter_file

            extract_snps = read_snp_filter_file(extract_file)

        for c in chroms:
            # Filter the genotype matrix:
            if self.genotype is not None and c in self.genotype:
                self.genotype[c].filter_snps(extract_snps=extract_snps)

                # If no SNPs remain in the genotype matrix for that chromosome, then remove it:
                if self.genotype[c].shape[1] < 1:
                    del self.genotype[c]

            # Filter the summary statistics table:
            if self.sumstats_table is not None and c in self.sumstats_table:
                self.sumstats_table[c].filter_snps(extract_snps=extract_snps)

                # If no SNPs remain in the summary statistics table for that chromosome, then remove it:
                if self.sumstats_table[c].shape[0] < 1:
                    del self.sumstats_table[c]

            if self.ld is not None and c in self.ld:
                self.ld[c].filter_snps(extract_snps=extract_snps)

                # If no SNPs remain in the summary statistics table for that chromosome, then remove it:
                if self.ld[c].n_snps < 1:
                    del self.ld[c]

            # Filter the annotation matrix:
            if self.annotation is not None and c in self.annotation:
                self.annotation[c].filter_snps(extract_snps=extract_snps)

                if self.annotation[c].shape[0] < 1:
                    del self.annotation[c]

    def filter_samples(self, keep_samples=None, keep_file=None):
        """
        Filter samples from the samples table. User must specify
        either a list of samples to keep or the path to a file
        with the list of samples to keep.

        :param keep_samples: A list or array of sample IDs to keep.
        :param keep_file: The path to a file with the list of samples to keep.
        """

        self.sample_table.filter_samples(keep_samples=keep_samples, keep_file=keep_file)
        self.sync_sample_tables()

    def read_annotations(
        self, annot_path, annot_format="magenpy", parser=None, **parse_kwargs
    ):
        """
        Read the annotation matrix from file. Annotations are a set of features associated
        with each SNP and are generally represented in table format.
        Consult the documentation for `AnnotationMatrix` for more details.

        :param annot_path: The path to the annotation file(s). The path may contain a wildcard.
        :param annot_format: The format for the summary statistics. Currently, supports the following
         formats: `magenpy`, `ldsc`.
        :param parser: If the annotation file does not follow any of the formats above, you can create
        your own parser by inheriting from the base `AnnotationMatrixParser` class and passing it here as an argument.
        :param parse_kwargs: keyword arguments for the parser. These are mainly parameters that will be passed to
        `pandas.read_csv` function, such as the delimiter, header information, etc.
        """

        if annot_path is None:
            return

        # Find all the relevant files in the path passed by the user:
        if not iterable(annot_path):
            annot_files = get_filenames(annot_path, extension=".annot")
        else:
            annot_files = annot_path

        if len(annot_files) < 1:
            logger.warning(f"No annotation files were found at: {annot_path}")
            return

        logger.info("> Reading annotation file...")

        self.annotation = {}

        for annot_file in tqdm(
            annot_files, total=len(annot_files), desc="Reading annotation files"
        ):
            annot_mat = AnnotationMatrix.from_file(
                annot_file,
                annot_format=annot_format,
                annot_parser=parser,
                **parse_kwargs,
            )
            self.annotation[annot_mat.chromosome] = annot_mat

    def read_genotypes(
        self,
        bed_paths,
        keep_samples=None,
        keep_file=None,
        extract_snps=None,
        extract_file=None,
        min_maf=None,
        min_mac=1,
        drop_duplicated=True,
        genome_build=None,
    ):
        """
        Read the genotype matrix and/or associated metadata from plink's BED file format.
        Consult the documentation for `GenotypeMatrix` for more details.

        :param bed_paths: The path to the BED file(s). You may use a wildcard here to read files for multiple
        chromosomes.
        :param keep_samples: A vector or list of sample IDs to keep when filtering the genotype matrix.
        :param keep_file: A path to a plink-style file containing sample IDs to keep.
        :param extract_snps: A vector or list of SNP IDs to keep when filtering the genotype matrix.
        :param extract_file: A path to a plink-style file containing SNP IDs to keep.
        :param min_maf: The minimum minor allele frequency cutoff.
        :param min_mac: The minimum minor allele count cutoff.
        :param drop_duplicated: If True, drop SNPs with duplicated rsID.
        :param genome_build: The genome build or assembly under which the SNP coordinates are defined.
        """

        if bed_paths is None:
            return

        # Find all the relevant files in the path passed by the user:
        if not iterable(bed_paths):
            bed_files = get_filenames(bed_paths, extension=".bed")
        else:
            bed_files = bed_paths

        if len(bed_files) < 1:
            logger.warning(f"No BED files were found at: {bed_paths}")
            return

        # Depending on the backend, select the `GenotypeMatrix` class:
        if self.backend == "magenpy":
            gmat_class = MagenpyGenotypeMatrix
        elif self.backend == "xarray":
            gmat_class = xarrayGenotypeMatrix
        elif self.backend == "bed-reader":
            gmat_class = bedReaderGenotypeMatrix
        else:
            gmat_class = plinkBEDGenotypeMatrix

        logger.info("> Reading genotype metadata...")

        self.genotype = {}

        for bfile in tqdm(
            bed_files, total=len(bed_files), desc="Reading genotype metadata"
        ):
            # Read BED file and update the genotypes dictionary:
            self.genotype.update(
                gmat_class.from_file(
                    bfile,
                    temp_dir=self.temp_dir,
                    threads=self.threads,
                    genome_build=genome_build,
                ).split_by_chromosome()
            )

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
        (Default: tab-separated file with `FID IID phenotype` columns). If different, supply
        details as additional arguments to this function.
        :param drop_na: Drop samples with missing phenotype information.
        :param read_csv_kwargs: keyword arguments for the `read_csv` function of `pandas`.
        """

        if phenotype_file is None:
            return

        logger.info("> Reading phenotype file...")

        assert self.sample_table is not None

        self.sample_table.read_phenotype_file(
            phenotype_file, drop_na=drop_na, **read_csv_kwargs
        )
        self.sync_sample_tables()

    def set_phenotype(self, new_phenotype, phenotype_likelihood=None):
        """
        A convenience method to update the phenotype column for the samples.
        :param new_phenotype: A vector or list of phenotype values.
        :param phenotype_likelihood: The phenotype likelihood (e.g. `binomial`, `gaussian`). Optional.
        """

        self.sample_table.set_phenotype(
            new_phenotype,
            phenotype_likelihood=phenotype_likelihood or self.phenotype_likelihood,
        )
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

        logger.info("> Reading covariates file...")

        assert self.sample_table is not None

        self.sample_table.read_covariates_file(covariates_file, **read_csv_kwargs)
        self.sync_sample_tables()

    def read_summary_statistics(
        self,
        sumstats_path,
        sumstats_format="magenpy",
        parser=None,
        drop_duplicated=True,
        **parse_kwargs,
    ):
        """
        Read GWAS summary statistics file(s) and parse them to `SumstatsTable` objects.

        :param sumstats_path: The path to the summary statistics file(s). The path may be a wildcard.
        :param sumstats_format: The format for the summary statistics. Currently supports the following
         formats: `plink1.9`, `plink2`, `magenpy`, `fastGWA`, `COJO`, `SAIGE`, or `GWASCatalog` for the standard
         summary statistics format (also known as `ssf` or `gwas-ssf`).
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

        if len(sumstats_files) < 1:
            logger.warning(
                f"No summary statistics files were found at: {sumstats_path}"
            )
            return

        logger.info("> Reading summary statistics...")

        self.sumstats_table = {}

        for f in tqdm(
            sumstats_files, total=len(sumstats_files), desc="Reading summary statistics"
        ):
            ss_tab = SumstatsTable.from_file(
                f, sumstats_format=sumstats_format, parser=parser, **parse_kwargs
            )

            if drop_duplicated:
                ss_tab.drop_duplicates()

            if "CHR" in ss_tab.table.columns:
                chromosome_values = ss_tab.table["CHR"]
                chromosomes = chromosome_values.dropna().unique()
                if len(chromosomes) == 1 and not chromosome_values.isna().any():
                    # The parsed table is already chromosome-specific and is not
                    # used after this point, so avoid groupby and a full copy.
                    self.sumstats_table[chromosomes[0]] = ss_tab
                else:
                    self.sumstats_table.update(ss_tab.split_by_chromosome())
            else:
                if self.genotype is not None:
                    ref_table = {c: g.snps for c, g in self.genotype.items()}
                elif self.ld is not None:
                    ref_table = {c: ld.snps for c, ld in self.ld.items()}
                else:
                    raise ValueError(
                        "Cannot index summary statistics tables without chromosome information!"
                    )

                self.sumstats_table.update(
                    ss_tab.split_by_chromosome(snps_per_chrom=ref_table)
                )

        # If SNP information is not present in the sumstats tables, try to impute it
        # using other reference tables:

        missing_snp = any(
            "SNP" not in ss.table.columns for ss in self.sumstats_table.values()
        )

        if missing_snp and (self.genotype is not None or self.ld is not None):
            ref_table = self.to_snp_table(
                col_subset=["CHR", "POS", "SNP"], per_chromosome=True
            )

            for c, ss in self.sumstats_table.items():
                if "SNP" not in ss.table.columns and c in ref_table:
                    ss.infer_snp_id(ref_table[c], allow_na=True)

    def read_ld(self, ld_store_paths):
        """
        Read the LD matrix files stored on-disk in Zarr array format.
        :param ld_store_paths: The path to the LD matrices. This may be a wildcard to accommodate reading data
        for multiple chromosomes.
        """

        if ld_store_paths is None:
            return

        if not iterable(ld_store_paths):
            if "s3://" in ld_store_paths:
                from .utils.system_utils import glob_s3_path

                ld_store_files = glob_s3_path(ld_store_paths)
            elif "hf://" in ld_store_paths:
                from .utils.system_utils import glob_hf_path

                ld_store_files = glob_hf_path(ld_store_paths)
            elif ".zip" in ld_store_paths:
                ld_store_files = get_filenames(ld_store_paths, extension=".zip")
            else:
                ld_store_files = get_filenames(ld_store_paths, extension=".zgroup")
        else:
            ld_store_files = ld_store_paths

        if len(ld_store_files) < 1:
            logger.warning(f"No LD matrix files were found at: {ld_store_paths}")
            return

        logger.info("> Reading LD metadata...")

        self.ld = {}

        for f in tqdm(
            ld_store_files, total=len(ld_store_files), desc="Reading LD metadata"
        ):
            z = LDMatrix.from_path(f)
            self.ld[z.chromosome] = z

    def load_ld(self):
        """
        A utility method to load the LD matrices to memory from on-disk storage.
        """
        if self.ld is not None:
            for ld in self.ld.values():
                ld.load()

    def release_ld(self):
        """
        A utility function to release the LD matrices from memory.
        """
        if self.ld is not None:
            for ld in self.ld.values():
                ld.release()

    def compute_ld(
        self,
        estimator,
        output_dir,
        dtype="int16",
        compressor_name="zstd",
        compression_level=7,
        compute_spectral_properties=False,
        store_type=None,
        **ld_kwargs,
    ):
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
        :param dtype: The data type for the entries of the LD matrix (supported data types are float32, float64
        and integer quantized data types int8 and int16).
        :param compressor_name: The name of the compression algorithm to use for the LD matrix.
        :param compression_level: The compression level to use for the entries of the LD matrix (1-9).
        :param compute_spectral_properties: If True, compute the spectral properties of the LD matrix.
        :param store_type: Optional Zarr store type. One of None, 'directory', or 'zip'.
        :param ld_kwargs: keyword arguments for the various LD estimators. Consult
        the implementations of `WindowedLD`, `ShrinkageLD`, and `BlockLD` for details.
        """

        if self.genotype is None:
            raise ValueError("Cannot compute LD without genotype data.")

        logger.info("> Computing LD matrix...")

        self.ld = {
            c: g.compute_ld(
                estimator,
                output_dir,
                dtype=dtype,
                compressor_name=compressor_name,
                compression_level=compression_level,
                compute_spectral_properties=compute_spectral_properties,
                store_type=store_type,
                **ld_kwargs,
            )
            for c, g in tqdm(
                sorted(self.genotype.items(), key=lambda x: x[0]),
                total=len(self.genotype),
                desc="Computing LD matrices",
            )
        }

    def get_ld_matrices(self):
        """
        :return: A dictionary containing the chromosome ID as key and corresponding LD matrices
        as value.
        """
        return self.ld

    def harmonize_data(self):
        """
        This method ensures that the data sources (reference genotype,
        LD matrices, summary statistics, annotations) are all aligned in terms of the
        set of variants that they operate on as well as the designation of the effect allele for
        each variant.

        Variant metadata is materialized once per source and chromosome. A single
        harmonization plan then supplies positional indexers for every source, so
        filtering and allele compatibility do not trigger repeated metadata reads.
        When LD is present, its stored order defines the canonical variant order.

        !!! note
            This method is called automatically during the initialization of the `GWADataLoader` object.
            However, if you read or manipulate the data sources after initialization,
            you may need to call this method again to ensure that the data sources remain aligned.

        !!! warning
            Harmonization for now depends on having SNP rsID be present in all the resources. Hopefully
            this requirement will be relaxed in the future.

        """

        named_data_sources = {
            "genotype": self.genotype,
            "sumstats": self.sumstats_table,
            "ld": self.ld,
            "annotation": self.annotation,
        }
        initialized_data_sources = {
            name: source for name, source in named_data_sources.items()
            if source is not None
        }

        # If less than two data sources are present, skip harmonization...
        if len(initialized_data_sources) < 2:
            return

        # Get the chromosomes information from all the data sources:
        chromosomes = list(
            set.union(*[set(ds.keys()) for ds in initialized_data_sources.values()])
        )

        logger.info("> Harmonizing data...")

        for c in tqdm(chromosomes, total=len(chromosomes), desc="Harmonizing data"):
            # Which initialized data sources have information for chromosome `c`
            miss_chroms = [c not in ds for ds in initialized_data_sources.values()]

            if sum(miss_chroms) > 0:
                # If the chromosome data only exists for some data sources but not others, remove the chromosome
                # from all data source.
                # Is this the best way to handle the missingness? Should we just post a warning?
                logger.debug(
                    f"Chromosome {c} is missing in some data sources. "
                    f"Removing it from all data sources."
                )
                for ds in initialized_data_sources.values():
                    if c in ds:
                        del ds[c]

            else:
                snp_ids = {}
                alleles = {}
                ld_original_positions = None
                ld_current_mask = None

                if self.genotype is not None:
                    genotype = self.genotype[c]
                    snp_ids["genotype"] = genotype.snps
                    alleles["genotype"] = (genotype.a1, genotype.a2)

                if self.sumstats_table is not None:
                    sumstats = self.sumstats_table[c]
                    snp_ids["sumstats"] = sumstats.snps
                    if self.genotype is not None or self.ld is not None:
                        if not all(
                            column in sumstats.table.columns
                            for column in ("A1", "A2")
                        ):
                            raise ValueError(
                                "To harmonize summary statistics, both `A1` and "
                                "`A2` must be present."
                            )
                        alleles["sumstats"] = (sumstats.a1, sumstats.a2)

                if self.ld is not None:
                    ld = self.ld[c]
                    stored_snps = ld.get_metadata("snps", apply_mask=False)
                    ld_current_mask = ld.get_mask()
                    if ld_current_mask is None:
                        ld_original_positions = np.arange(len(stored_snps), dtype=np.intp)
                        snp_ids["ld"] = stored_snps
                    else:
                        ld_original_positions = np.flatnonzero(ld_current_mask)
                        snp_ids["ld"] = stored_snps[ld_original_positions]

                    # LD alleles are only needed when LD is the allele reference.
                    if self.genotype is None and self.sumstats_table is not None:
                        stored_a1 = ld.get_metadata("a1", apply_mask=False)
                        stored_a2 = ld.get_metadata("a2", apply_mask=False)
                        alleles["ld"] = (
                            stored_a1[ld_original_positions],
                            stored_a2[ld_original_positions],
                        )

                if self.annotation is not None:
                    snp_ids["annotation"] = self.annotation[c].snps

                # An LD matrix cannot be reordered without permuting its rows and
                # columns, so its stored order is canonical whenever it is present.
                # Otherwise retain genotype, summary-statistic, then annotation order.
                reference = next(
                    source for source in ("ld", "genotype", "sumstats", "annotation")
                    if source in snp_ids
                )

                allele_reference = None
                allele_target = None
                if "sumstats" in snp_ids:
                    # TODO: When genotype and LD are both present, reconcile the
                    # genotype A1/A2 orientation with LD as well. This requires
                    # deciding how allele flips propagate to genotype dosages and
                    # LD correlations; for now, as in the legacy implementation,
                    # only summary statistics are allele-harmonized.
                    if "genotype" in snp_ids:
                        allele_reference = "genotype"
                    elif "ld" in snp_ids:
                        allele_reference = "ld"
                    if allele_reference is not None:
                        allele_target = "sumstats"

                plan = build_snp_harmonization_plan(
                    snp_ids,
                    reference=reference,
                    alleles=alleles,
                    allele_reference=allele_reference,
                    allele_target=allele_target,
                )
                indexers = plan["indexers"]

                if self.genotype is not None:
                    indexer = indexers["genotype"]
                    if not np.array_equal(indexer, np.arange(len(genotype.snp_table))):
                        genotype.snp_table = (
                            genotype.snp_table.iloc[indexer]
                            .copy()
                            .reset_index(drop=True)
                        )

                if self.annotation is not None:
                    annotation = self.annotation[c]
                    indexer = indexers["annotation"]
                    if not np.array_equal(indexer, np.arange(len(annotation.table))):
                        annotation.table = (
                            annotation.table.iloc[indexer]
                            .copy()
                            .reset_index(drop=True)
                        )

                if self.sumstats_table is not None:
                    indexer = indexers["sumstats"]
                    sumstats.table = (
                        sumstats.table.iloc[indexer]
                        .copy()
                        .reset_index(drop=True)
                    )

                    if allele_reference is not None:
                        flip = plan["flip"].astype(int)
                        if np.any(flip):
                            multiplier = 1. - 2. * flip
                            for statistic in ("BETA", "STD_BETA", "Z"):
                                if statistic in sumstats.table:
                                    sumstats.table[statistic] = (
                                        multiplier * sumstats.table[statistic]
                                    )
                            if "MAF" in sumstats.table:
                                sumstats.table["MAF"] = np.abs(
                                    flip - sumstats.table["MAF"]
                                )

                        sumstats.table["A1"] = plan["a1"]
                        sumstats.table["A2"] = plan["a2"]

                if self.ld is not None:
                    final_ld_positions = ld_original_positions[indexers["ld"]]
                    final_mask = np.zeros(ld.stored_n_snps, dtype=bool)
                    final_mask[final_ld_positions] = True

                    if final_mask.all() and ld_current_mask is None:
                        continue
                    if ld_current_mask is None or not np.array_equal(
                        final_mask, ld_current_mask
                    ):
                        ld.set_mask(final_mask)

    def perform_gwas(self, **gwa_kwargs):
        """
        Perform genome-wide association testing of all variants against the phenotype.
        This is a utility function that calls the `.perform_gwas()` method of the
        `GenotypeMatrix` objects associated with GWADataLoader.

        :param gwa_kwargs: Keyword arguments to pass to the GWA functions. Consult stats.gwa.utils
        for relevant keyword arguments for each backend.
        """

        logger.info("> Performing GWAS...")

        self.sumstats_table = {
            c: g.perform_gwas(**gwa_kwargs)
            for c, g in tqdm(
                sorted(self.genotype.items(), key=lambda x: x[0]),
                total=len(self.genotype),
                desc="Performing GWAS",
            )
        }

    def score(self, beta=None, standardize_genotype=False):
        """
        Perform linear scoring, i.e. multiply the genotype matrix by the vector of effect sizes, `beta`.

        :param beta: A dictionary where the keys are the chromosome numbers and the
        values are a vector of effect sizes for each variant on that chromosome. If the
        betas are not provided, we use the marginal betas by default (if those are available).
        :param standardize_genotype: If True, standardize the genotype matrix before scoring.
        """

        if beta is None:
            try:
                beta = {
                    c: s.marginal_beta or s.get_snp_pseudo_corr()
                    for c, s in self.sumstats_table.items()
                }
            except Exception:
                raise ValueError(
                    "To perform linear scoring, you must "
                    "provide effect size estimates (BETA)!"
                )

        # Here, we have a very ugly way of accounting for
        # the fact that the chromosomes may be coded differently between the genotype
        # and the beta dictionary. Maybe we can find a better solution in the future.
        common_chr_g, common_chr_b = match_chromosomes(
            self.genotype.keys(), beta.keys(), return_both=True
        )

        if len(common_chr_g) < 1:
            raise ValueError(
                "No common chromosomes found between "
                "the genotype and the effect size estimates!"
            )

        logger.info("> Generating polygenic scores...")

        pgs = None

        for c_g, c_b in tqdm(
            zip(common_chr_g, common_chr_b),
            total=len(common_chr_g),
            desc="Generating polygenic scores",
        ):
            if pgs is None:
                pgs = self.genotype[c_g].score(
                    beta[c_b], standardize_genotype=standardize_genotype
                )
            else:
                pgs += self.genotype[c_g].score(
                    beta[c_b], standardize_genotype=standardize_genotype
                )

        # If we only have a single set of betas, flatten the PGS vector:
        if len(pgs.shape) > 1 and pgs.shape[1] == 1:
            pgs = pgs.flatten()

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

        if self.phenotype_likelihood == "binomial":
            # Apply probit link function:
            from scipy.stats import norm

            pgs = norm.cdf(pgs)

        return pgs

    def to_individual_table(self):
        """
        :return: A plink-style dataframe of individual IDs, in the form of
        Family ID (FID) and Individual ID (IID).
        """
        assert self.sample_table is not None

        return self.sample_table.get_individual_table()

    def to_phenotype_table(self):
        """
        :return: A plink-style dataframe with each individual's Family ID (FID),
        Individual ID (IID), and phenotype value.
        """

        assert self.sample_table is not None

        return self.sample_table.get_phenotype_table()

    def to_snp_table(self, col_subset=None, per_chromosome=False, resource="auto"):
        """
        Get a dataframe of SNP data for all variants
        across different chromosomes.

        :param col_subset: The subset of columns to obtain.
        :param per_chromosome: If True, returns a dictionary where the key
        is the chromosome number and the value is the SNP table per
        chromosome.
        :param resource: The data source to extract the SNP table from. By default, the method
        will try to extract the SNP table from the genotype matrix. If the genotype matrix is not
        available, then it will try to extract the SNP information from the LD matrix or the summary
        statistics table. Possible values: `auto`, `genotype`, `ld`, `sumstats`.

        :return: A dataframe (or dictionary of dataframes) of SNP data.
        """

        # Sanity checks:
        assert resource in ("auto", "genotype", "ld", "sumstats")

        if resource != "auto":
            if resource == "genotype" and self.genotype is None:
                raise ValueError("Genotype matrix is not available!")
            if resource == "ld" and self.ld is None:
                raise ValueError("LD matrix is not available!")
            if resource == "sumstats" and self.sumstats_table is None:
                raise ValueError("Summary statistics table is not available!")
        else:
            if all(ds is None for ds in (self.genotype, self.ld, self.sumstats_table)):
                raise ValueError("No data sources available to extract SNP data from!")

        # Extract the SNP data:

        snp_tables = {}

        if resource in ("auto", "genotype") and self.genotype is not None:
            for c in self.chromosomes:
                snp_tables[c] = self.genotype[c].get_snp_table(col_subset=col_subset)
        elif resource in ("auto", "ld") and self.ld is not None:
            for c in self.chromosomes:
                snp_tables[c] = self.ld[c].to_snp_table(col_subset=col_subset)
        else:
            return self.to_summary_statistics_table(
                col_subset=col_subset, per_chromosome=per_chromosome
            )

        if per_chromosome:
            return snp_tables
        else:
            return pd.concat(list(snp_tables.values()))

    def to_summary_statistics_table(self, col_subset=None, per_chromosome=False):
        """
        Get a dataframe of the GWAS summary statistics for all variants
        across different chromosomes.

        :param col_subset: The subset of columns (or summary statistics) to obtain.
        :param per_chromosome: If True, returns a dictionary where the key
        is the chromosome number and the value is the summary statistics table per
        chromosome.

        :return: A dataframe (or dictionary of dataframes) of summary statistics.
        """

        assert self.sumstats_table is not None

        snp_tables = {}

        for c in self.chromosomes:
            snp_tables[c] = self.sumstats_table[c].to_table(col_subset=col_subset)

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

    def split_by_chromosome(self):
        """
        A utility method to split a GWADataLoader object by chromosome ID, such that
        we would have one `GWADataLoader` object per chromosome. The method returns a dictionary
        where the key is the chromosome number and the value is the `GWADataLoader` object corresponding
        to that chromosome only.
        """

        if len(self.chromosomes) == 1:
            return {self.chromosomes[0]: self}

        else:
            split_dict = {}

            for c in self.chromosomes:
                split_dict[c] = copy.copy(self)

                if self.genotype is not None and c in self.genotype:
                    split_dict[c].genotype = {c: self.genotype[c]}
                if self.sumstats_table is not None and c in self.sumstats_table:
                    split_dict[c].sumstats_table = {c: self.sumstats_table[c]}
                if self.ld is not None and c in self.ld:
                    split_dict[c].ld = {c: self.ld[c]}
                if self.annotation is not None and c in self.annotation:
                    split_dict[c].annotation = {c: self.annotation[c]}

            return split_dict

    def split_by_samples(self, proportions=None, groups=None, keep_original=True):
        """
        Split the `GWADataLoader` object by samples, if genotype or sample data
        is available. The user must provide a list or proportion of samples in each split,
        and the method will return a list of `GWADataLoader` objects with only the samples
        designated for each split. This may be a useful utility for training/testing split or some
        other downstream tasks.

        :param proportions: A list with the proportion of samples in each split. Must add to 1.
        :param groups: A list of lists containing the sample IDs in each split.
        :param keep_original: If True, keep the original `GWADataLoader` object and do not
        transform it in the splitting process.
        """

        if self.sample_table is None:
            raise ValueError("The sample table is not set!")

        if groups is None:
            if proportions is None:
                raise ValueError(
                    "To split a `GWADataloader` object by samples, the user must provide either the list "
                    "or proportion of individuals in each split."
                )
            else:
                # Assign each sample to a different split randomly by drawing from a multinomial:
                random_split = np.random.multinomial(
                    1, proportions, size=self.sample_size
                ).astype(bool)
                # Extract the individuals in each group from the multinomial sample:
                groups = [
                    self.samples[random_split[:, i]]
                    for i in range(random_split.shape[1])
                ]

        gdls = []
        for i, g in enumerate(groups):
            if len(g) < 1:
                raise ValueError(
                    f"Group {i} is empty! Please ensure that all splits have at least one sample."
                )

            if (i + 1) == len(groups) and not keep_original:
                new_gdl = self
            else:
                new_gdl = copy.deepcopy(self)

            new_gdl.filter_samples(keep_samples=g)

            gdls.append(new_gdl)

        return gdls

    def align_with(self, other_gdls, axis="SNP", how="inner"):
        """
        Align the `GWADataLoader` object with other GDL objects to have the same
        set of SNPs or samples. This utility method is meant to enable the user to
        align multiple data sources for downstream analyses.

        :param other_gdls: A `GWADataLoader` or list of `GWADataLoader` objects.
        :param axis: The axis on which to perform the alignment (can be `sample` for aligning individuals or
        `SNP` for aligning variants across the datasets).
        :param how: The type of join to perform across the datasets. For now, we support an inner join sort
        of operation.

        !!! warning
            Experimental for now, would like to add more features here in the near future.

        """

        if isinstance(other_gdls, GWADataLoader):
            other_gdls = [other_gdls]

        assert all([isinstance(gdl, GWADataLoader) for gdl in other_gdls])

        if axis == "SNP":
            # Ensure that all the GDLs have the same set of SNPs.
            # This may be useful if the goal is to select a common set of variants
            # that are shared across different datasets.
            for c in self.chromosomes:
                common_snps = set(self.snps[c])
                for gdl in other_gdls:
                    common_snps = common_snps.intersection(set(gdl.snps[c]))

                common_snps = np.array(list(common_snps))

                for gdl in other_gdls:
                    gdl.filter_snps(extract_snps=common_snps, chromosome=c)

                self.filter_snps(extract_snps=common_snps, chromosome=c)

        elif axis == "sample":
            # Ensure that all the GDLs have the same set of samples.
            # This may be useful when different GDLs have different covariates, phenotypes,
            # or other information pertaining to the individuals.

            common_samples = set(self.samples)

            for gdl in other_gdls:
                common_samples = common_samples.intersection(set(gdl.samples))

            common_samples = np.array(list(common_samples))

            for gdl in other_gdls:
                gdl.filter_samples(keep_samples=common_samples)

            self.filter_samples(keep_samples=common_samples)

        else:
            raise KeyError("Alignment axis can only be either 'SNP' or 'sample'!")

    def cleanup(self):
        """
        Clean up all temporary files and directories
        """

        logger.info("> Cleaning up workspace.")

        for tmpdir in self.cleanup_dir_list:
            try:
                tmpdir.cleanup()
            except FileNotFoundError:
                continue

        # Clean up the temporary files associated with the genotype matrices:
        if self.genotype is not None:
            for g in self.genotype.values():
                g.cleanup()

        # Release the LD data from memory:
        self.release_ld()

    def summary(self):
        """Return a table summarizing the loader and its available data sources.

        The summary only inspects metadata already held by the loader. In
        particular, it does not load LD matrix entries or concatenate variant
        tables.

        :return: A `pandas.DataFrame` with the main properties of this loader.
        """

        def source_description(source):
            if source is None:
                return "Not loaded"
            n_chromosomes = len(source)
            suffix = "chromosome" if n_chromosomes == 1 else "chromosomes"
            return f"Loaded ({n_chromosomes} {suffix})"

        # All initialized sources should be harmonized, so one non-empty source
        # is sufficient to determine the common chromosome and variant counts.
        variant_counts = {}
        if self.genotype:
            variant_counts = {c: g.n_snps for c, g in self.genotype.items()}
        elif self.sumstats_table:
            variant_counts = {
                c: sumstats.n_snps for c, sumstats in self.sumstats_table.items()
            }
        elif self.ld:
            variant_counts = {c: ld.n_snps for c, ld in self.ld.items()}
        elif self.annotation:
            variant_counts = {
                c: annotation.shape[0]
                for c, annotation in self.annotation.items()
            }

        chromosomes = list(variant_counts)
        try:
            chromosomes = sorted(chromosomes)
        except TypeError:
            # Mixed chromosome encodings (for example, 22 and "X") are still
            # valid display values even though Python cannot order them directly.
            chromosomes = sorted(chromosomes, key=str)

        if self.sample_table is not None:
            sample_size = self.sample_table.n
        else:
            sample_sizes = []
            if self.sumstats_table:
                for sumstats in self.sumstats_table.values():
                    if "N" in sumstats.table.columns and len(sumstats.table):
                        values = pd.to_numeric(
                            sumstats.table["N"], errors="coerce"
                        ).to_numpy(copy=False)
                        finite_values = values[np.isfinite(values)]
                        if finite_values.size:
                            sample_sizes.append(finite_values.max())
            sample_size = max(sample_sizes) if sample_sizes else "Not available"

        has_phenotype = (
            self.sample_table is not None
            and self.sample_table.table is not None
            and "phenotype" in self.sample_table.table.columns
        )
        n_covariates = 0
        if self.sample_table is not None and self.sample_table.covariates is not None:
            n_covariates = len(self.sample_table.covariates)

        genome_builds = set()
        if self.genotype:
            genome_builds.update(
                g.genome_build for g in self.genotype.values()
                if g.genome_build is not None
            )
        if self.ld:
            genome_builds.update(
                ld.genome_build for ld in self.ld.values()
                if ld.genome_build is not None
            )
        genome_build = (
            ", ".join(sorted(map(str, genome_builds)))
            if genome_builds
            else "Not specified"
        )

        ld_description = source_description(self.ld)
        if self.ld is not None:
            n_loaded = sum(ld.in_memory for ld in self.ld.values())
            ld_description += f"; {n_loaded}/{len(self.ld)} in memory"

        annotation_description = source_description(self.annotation)
        if self.annotation:
            annotation_counts = {
                annotation.n_annotations for annotation in self.annotation.values()
            }
            if len(annotation_counts) == 1:
                annotation_description += f"; {annotation_counts.pop()} annotations"

        return pd.DataFrame(
            [
                {"GWADataLoader property": "Backend", "Value": self.backend},
                {"GWADataLoader property": "Sample size", "Value": sample_size},
                {
                    "GWADataLoader property": "Variants",
                    "Value": sum(variant_counts.values()),
                },
                {
                    "GWADataLoader property": "Chromosomes",
                    "Value": ", ".join(map(str, chromosomes)) or "None",
                },
                {"GWADataLoader property": "Genome build", "Value": genome_build},
                {
                    "GWADataLoader property": "Genotype",
                    "Value": source_description(self.genotype),
                },
                {
                    "GWADataLoader property": "Summary statistics",
                    "Value": source_description(self.sumstats_table),
                },
                {"GWADataLoader property": "LD matrices", "Value": ld_description},
                {
                    "GWADataLoader property": "Annotations",
                    "Value": annotation_description,
                },
                {
                    "GWADataLoader property": "Phenotype",
                    "Value": "Available" if has_phenotype else "Not available",
                },
                {"GWADataLoader property": "Covariates", "Value": n_covariates},
            ]
        ).set_index("GWADataLoader property")

    def __repr__(self):
        """:return: A summary of the `GWADataLoader` object as a string."""

        return self.summary().to_string()

    def _repr_html_(self):
        """:return: A summary of the `GWADataLoader` object as an HTML table."""

        table = self.summary().to_html(
            header=False,
            classes=("dataframe", "magenpy-summary"),
            border=0,
        )
        style = """
<style>
.magenpy-summary {
  border-collapse: separate; border-spacing: 0; border-radius: 5px;
  box-shadow: 0 2px 3px rgba(0,0,0,0.1); margin: 20px 0;
  table-layout: fixed; width: 50%; overflow: hidden;
}
.magenpy-summary th, .magenpy-summary td {
  padding: 10px 15px; text-align: left; white-space: normal;
  overflow-wrap: anywhere; border-bottom: 1px solid #eaeaea;
}
.magenpy-summary thead th {
  background-color: #3b5f9e; color: white; font-weight: bold;
}
.magenpy-summary tbody th { width: 35%; background-color: #f8f9fa; }
.magenpy-summary tr:nth-of-type(odd) td { background-color: #f8f9fa; }
.magenpy-summary tr:hover th, .magenpy-summary tr:hover td {
  background-color: #e8f0fe;
}
</style>
"""
        return style + table
