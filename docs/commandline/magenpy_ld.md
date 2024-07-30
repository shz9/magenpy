Compute Linkage-Disequilibrium (LD) matrices (`magenpy_ld`)
---

The `magenpy_ld` script is used to compute Linkage-Disequilibrium (LD) matrices, which record the 
pairwise SNP-by-SNP correlations from a sample of genotype data stored in `plink`'s BED format. The script 
offers an interface to compute LD matrices by simply specifying the path to the genotype files, the type of LD 
estimator to use, the subset of variants or samples to keep, and the output directory.

A full listing of the options available for the `magenpy_ld` script can be found by running the 
following command in your terminal:

```bash
magenpy_ld -h
```

Which outputs the following help message:

```bash

        **********************************************                            
         _ __ ___   __ _  __ _  ___ _ __  _ __  _   _ 
        | '_ ` _ \ / _` |/ _` |/ _ \ '_ \| '_ \| | | |
        | | | | | | (_| | (_| |  __/ | | | |_) | |_| |
        |_| |_| |_|\__,_|\__, |\___|_| |_| .__/ \__, |
                         |___/           |_|    |___/
        Modeling and Analysis of Genetics data in python
        Version: 0.1.4 | Release date: June 2024
        Author: Shadi Zabad, McGill University
        **********************************************
        < Compute LD matrix and store in Zarr format >

usage: magenpy_ld [-h] [--estimator {shrinkage,block,windowed,sample}] --bfile BED_FILE [--keep KEEP_FILE] [--extract EXTRACT_FILE]
                  [--backend {xarray,plink}] [--temp-dir TEMP_DIR] --output-dir OUTPUT_DIR [--min-maf MIN_MAF] [--min-mac MIN_MAC]
                  [--genome-build GENOME_BUILD] [--metadata METADATA] [--storage-dtype {float64,float32,int16,int8}]
                  [--compute-spectral-properties] [--compressor {lz4,zlib,zstd,gzip}] [--compression-level COMPRESSION_LEVEL]
                  [--ld-window LD_WINDOW] [--ld-window-kb LD_WINDOW_KB] [--ld-window-cm LD_WINDOW_CM] [--ld-blocks LD_BLOCKS]
                  [--genmap-Ne GENMAP_NE] [--genmap-sample-size GENMAP_SS] [--shrinkage-cutoff SHRINK_CUTOFF]

Commandline arguments for LD matrix computation and storage

options:
  -h, --help            show this help message and exit
  --estimator {shrinkage,block,windowed,sample}
                        The LD estimator (windowed, shrinkage, block, sample)
  --bfile BED_FILE      The path to a plink BED file.
  --keep KEEP_FILE      A plink-style keep file to select a subset of individuals to compute the LD matrices.
  --extract EXTRACT_FILE
                        A plink-style extract file to select a subset of SNPs to compute the LD matrix for.
  --backend {xarray,plink}
                        The backend software used to compute the Linkage-Disequilibrium between variants.
  --temp-dir TEMP_DIR   The temporary directory where we store intermediate files.
  --output-dir OUTPUT_DIR
                        The output directory where the Zarr formatted LD matrices will be stored.
  --min-maf MIN_MAF     The minimum minor allele frequency for variants included in the LD matrix.
  --min-mac MIN_MAC     The minimum minor allele count for variants included in the LD matrix.
  --genome-build GENOME_BUILD
                        The genome build for the genotype data (recommend storing as metadata).
  --metadata METADATA   A comma-separated string with metadata keys and values. This is used to store information about the genotype data
                        from which the LD matrix was computed, such as the biobank/samples, cohort characteristics (e.g. ancestry), etc.
                        Keys and values should be separated by =, such that inputs are in the form of:--metadata
                        Biobank=UKB,Ancestry=EUR,Date=April2024
  --storage-dtype {float64,float32,int16,int8}
                        The data type for the entries of the LD matrix.
  --compute-spectral-properties
                        Compute and store the spectral properties of the LD matrix (e.g. eigenvalues, eigenvectors).
  --compressor {lz4,zlib,zstd,gzip}
                        The compressor name or compression algorithm to use for the LD matrix.
  --compression-level COMPRESSION_LEVEL
                        The compression level to use for the entries of the LD matrix (1-9).
  --ld-window LD_WINDOW
                        Maximum number of neighboring SNPs to consider when computing LD.
  --ld-window-kb LD_WINDOW_KB
                        Maximum distance (in kilobases) between pairs of variants when computing LD.
  --ld-window-cm LD_WINDOW_CM
                        Maximum distance (in centi Morgan) between pairs of variants when computing LD.
  --ld-blocks LD_BLOCKS
                        Path to the file with the LD block boundaries, in LDetect format (e.g. chr start stop, tab-separated)
  --genmap-Ne GENMAP_NE
                        The effective population size for the population from which the genetic map was derived.
  --genmap-sample-size GENMAP_SS
                        The sample size for the dataset used to infer the genetic map.
  --shrinkage-cutoff SHRINK_CUTOFF
                        The cutoff value below which we assume that the correlation between variants is zero.

 
```