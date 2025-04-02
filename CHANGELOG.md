# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2025-04-01

### Changed

- Updated behavior of `.load` of `LDMatrix`. Now, by default it loads
an `LDLinearOperator` object.
  - Now the cached loaded data is in the form of a `LDLinearOperator` object.
- Moved a lot of the functionality of converting LD data to `CSR` format 
to the `LDLinearOperator`.
- Removed printing where possible in the package and changed it to use 
the `logging` module.
- Resolved some issues in how the `pandas-plink` genotype matrix is 
handled in case of splitting by chromosomes/variants.
- Removed `fill_na` from the `standardize` method in `stats.transforms.genotype`.
- Fixed how the package interfaces with `tempfile` to properly cleanup 
temporary files/directories.
- Made the tests for `LDMatrix` a bit more comprehensive.

### Added

- Added preliminary tests to the CLI scripts (`magenpy_ld` and `magenpy_simulate`)
- Support for block iterator for the `LDMatrix`.
- `rank_one_update` for the `LDLinearOperator` class.
- Unified method to map variants to genomic blocks `map_variants_to_genomic_blocks`.
- Added `summary` method to `LDMatrix` to provide a summary of the LD matrix.
- Added `__repr__` and `__repr_html__` methods to `LDMatrix`.
- Added functionality to allow slicing of `LDLinearOperator` and outputting 
subsets of the data as a numpy array directly.
- Added implementation of the `PUMAS` procedure for sampling summary data 
conditional on the LD matrix. Relevant functions: 
  - `sumstats_train_test_split`
  - `multivariate_normal_conditional_sampling`
- Added a faster intersection implementation `intersect_multiple_arrays`.
- Added preliminary `bedReaderGenotypeMatrix` to support using the `bed-reader`
package as a backend (still needs more development and testing).
- Added convenience method `setup_logger` to set up logging in the package.


## [0.1.4] - 2024-10-01

### Changed

- Updated the data type for the index pointer in the `LDMatrix` object to be `int64`. `int32` does
not work well for very large datasets with millions of variants and it causes overflow errors.
- Updated the way we determine the `pandas` chunksize when converting from `plink` tables to `zarr`.
- Simplified the way we compute the quantization scale in `model_utils`.
- Fixed major bug in how LD window thresholds that are passed to `plink1.9` are computed.
- Fixed in-place `fillna` in `from_plink_table` in `LDMatrix` to conform to latest `pandas` API.
- Update `run_shell_script` to check for and capture errors.
- Refactored code to slightly reduce import/load times.
- Cleaned up `load_data` method of `LDMatrix` and subsumed functionality in `load_rows`.
- Fixed bugs in `match_snp_tables`.
- Fixed bugs and re-wrote how the `block` LD estimator is computed using both the `plink` and `xarray` backends.
- Updated `from_plink_table` method in `LDMatrix` to handle cases where boundaries are different from what 
`plink` computes.
- Fixed bug in `symmetrize_ut_csr_matrix` utility functions.
- Changed default storage data type for LD matrices to `int16`.

### Added

- Added extra validation checks in `LDMatrix` to ensure that the index pointer is formatted correctly.
- `LDLinearOperator` class to allow for efficient linear algebra operations on the LD matrix without
representing the full symmetric matrix in memory.
- Added utility methods to `LDMatrix` class to allow for computing eigenvalues, performing SVD, etc.
- Added `Spectral properties` to the attributes of LD matrices.
- Added support to slice/retrieve entries of LD matrix by using SNP rsIDs.
- Added support to reading LD matrices from AWS s3 storage.
- Added utility method to detect if a file contains header information.
- Added utility method to generate overlapping windows over a sequence.
- Added `compute_extremal_eigenvalues` to allow the user to compute extremal (minimum and maximum) eigenvalues 
of LD matrices.
- Added the utility function `combine_ld_matrices` to allow for combining LD matrices from different sources.

## [0.1.3] - 2024-05-21

### Changed

- Updated the logic for `detect_outliers` in phenotype transforms to actually reflect the function
name (before it was returning true for inliers...).
- Updated `quantize` and `dequantize` to minimize data copying as much as possible.
- Updated `LDMatrix.load_rows()` method to minimize data copying.
- Fixed bug in `LDMatrix.n_neighbors` implementation.
- Updated `dask` version in `requirements.txt` to avoid installing `dask-expr`.


### Added

- Added `get_peak_memory_usage` to `system_utils` to inspect peak memory usage of a process.
- Placeholder method to perform QC on `SumstatsTable` objects (needs to be implemented still).
- New attached dataset for long-range LD regions.
- New method in SumstatsTable to impute rsID (if missing).
- Preliminary support for matching with CHR+POS in SumstatsTable (still needs more work).
- LDMatrix updates:
  - New method to filter long-range LD regions.
  - New method to prune LD matrix.
- New algorithm for symmetrizing upper triangular and block diagonal LD matrices.
  - Much faster and more memory efficient than using `scipy`.
  - New `LDMatrix` class has efficient data loading in `.load_data` method.
  - We still retain `load_rows` because it is useful for loading a subset of rows.

## [0.1.2] - 2024-04-24

### Changed

- Fixed `manhattan` plot implementation to support various new features.
- Added a warning when accessing `csr_matrix` property of `LDMatrix` when it hasn't been loaded
previously.

### Added

- `reset_mask` method for magenpy `LDMatrix`.
- `Dockerfile`s for both `cli` and `jupyter` modes.
- A helper script to convert LD matrices from old format to new format.

## [0.1.1] - 2024-04-12

### Changed

- Fixed bugs in how covariates are processed in `SampleTable`.
- Fixed bugs / issues in implementation of GWAS with `xarray` backend.
- Streamlined implementation of `manhattan` plotting function.

## [0.1.0] - 2024-04-04

A large scale restructuring of the code base to improve efficiency and usability.

### Changed

- Bug fixes across the entire code base.
- Simulator classes have been renamed from `GWASimulator` to `PhenotypeSimulator`.
- Moved plotting script to its own separate module.
- Updated some method names / commandline flags to be consistent throughout.

### Added

- Basic integration testing with `pytest` and GitHub workflows.
- Documentation for the entire package using `mkdocs`.
- Integration testing / automating building with GitHub workflows.
- New implementation of the LD matrix that uses CSR matrix data structures.
  - Quantization / float precision specification when storing LD matrices.
  - Allow user to specify Compressor / Compressor options for Zarr storage.
- New implementation of `magenpy_simulate` script.
  - Allow users to set random seed.
  - Now accept `--prop-causal` instead of specifying full mixing proportions.
- Tried to incorporate `genome_build` into various data structures. This will be useful in the 
future to ensure consistent genome builds across different data types.
- Allow user to pass various metadata to `magenpy_ld` to save information about dataset 
characteristics.
- New sumstats parsers:
  - Saige sumstats format.
  - plink1.9 sumstats format.
  - GWAS Catalog sumstats format.
- Chained transform function for transforming phenotypes.

## [0.0.12] - 2023-02-12

### Changed

- Removed the `--fast-math` compiler flag due to concerns about 
numerical precision (e.g. [Beware of fast-math](https://simonbyrne.github.io/notes/fastmath/)).
- Updated implementation of `SumstatsParser` class to allow user to specify `read_csv_kwargs` at the point of instantiation.
- Updated plink executors to propagate the error messages to the user.
- Updated `merge_snp_tables` to allow for merges on columns other than `SNP`.
- Refactored, cleaned, and updated the implementation of the `AnnotationMatrix` class.
- Fixed bug in `GWADataLoader.split_by_samples()`: Need to perform `deepcopy`, otherwise splitting would not work properly.
- Updated `read_annotations` method in `GWADataLoader` to work with the latest `AnnotationMatrix` interfaces.
- Fixed bug in the `manhattan` plotting function.

### Added

- Added parsers for functional annotations and annotation files. Mainly support LDSC annotation format for now.
- Added a utility method to `GWADataLoader` called `align_with` to streamline aligning `GWADataLoader` objects across SNP and sample dimensions.
- Added utility methods for flattening the LD matrix in `LDMatrix`.
- Added a method to perform matrix-vector multiplication in `LDMatrix`.
- Added a method to perform block-wise iteration in the `LDMatrix` class.

## [0.0.11] - 2022-09-06

### Changed

- Fixed bug in implementation of `identify_mismatched_snps`.
- Fixed bugs in handling of missing information in LD matrix.
- Fixed bug in handling of covariates in `SampleTable`.
- Updated `README` file to remove line indicators `>>>` from sample code.

### Added

- Added the reference allele `A2` to the output of the `true_beta_table` 
in `GWASimulator`.

## [0.0.10] - 2022-08-22

### Changed

- Fixed a bug in the phenotype likelihood inference in `SampleTable`.
- Changed the implementation of the `merge_snp_tables` utility function to 
check for BOTH reference and alternative alleles.
- Modified implementation of `score` method of `GWADataLoader` to correct 
potential issues with the BETAS being for a subset of the chromosomes.

### Added

- A utility method to `GenotypeMatrix` called `estimate_memory_allocation`. This should 
allow the user to gauge the memory resources required to interact with the 
genotype files.

## [0.0.9] - 2022-08-09

### Changed

- Fixing bug in computing minor allele frequency with `plink`.

## [0.0.8] - 2022-08-09

### Added

- Added the reference allele `A2` to the list of attributes of `LDMatrix`.
- Added `effect_sign` as a property of `SumstatsTable`.

### Changed

- Fixed implementation of `merge_snp_tables` to detect allele differences that
are not flips between `A1`/`A2`.
- Improved implementation of `.score` method of `xarrayGenotypeMatrix`.

## [0.0.7] - 2022-07-12

### Added

- Added `tqdm` progress bars when processing multiple files/chromosomes 
  in `GWADataLoader`.
- Added `min_maf` and `min_mac` flags in `magenpy_ld` and `magenpy_simulate`.

### Changed

- Lowered default threshold for LD shrinkage to 1e-3.
- Bug fix in `SampleTable`.

## [0.0.6] - 2022-07-11

### Added

- Utility function to compute the genomic control or lambda factor.
- A method to set the causal SNPs directly in `GWASimulator`.

### Changed

- Fixed bugs in `manhattan` plotting function.
- Added alternative ways to derive Chi-Squared statistic from 
  other summary stats (e.g. p-value).
- Give user more fine-grained control on what to reset in the `.simulate()`
method of `GWASimulator`.
- Modified the LD score computation method to allow for aggregating LD scores 
by functional category or annotations.
- Streamlining module import structure to speed up loading.

## [0.0.5] - 2022-06-28

### Changed

- Fixed bugs in rechunking logic when computing LD matrices using `xarray`/`dask`

## [0.0.4] - 2022-06-28

### Added

- A new attached dataset of GWAS summary statistics for standing height from the fastGWA database.

### Changed

- Updated the data harmonization method in `GWADataLoader` to ensure that all data sources 
have the same set of chromosomes.
- Bug fixes in `SumStatsTable`, `GWASimulator`, and `GWADataLoader`.

## [0.0.3] - 2022-06-26

### Added

- New methods to split `GWADataLoader` objects by chromosome and by samples. 
The latter should come in handy for splitting the samples for training, validation and testing.

### Changed

- Updated implementation of the shrinkage estimator of LD to align it more closely 
with the original formulation in Wen and Stephens (2010) and implementations in RSS software.
- Fixed various bugs and errors in the code.
- Added proper handling for the slice objects in `plot_ld_matrix`

## [0.0.2] - 2022-06-15

### Added

- Added classes encapsulating data structures and methods for:
    - Genotype matrices: `GenotypeMatrix`
    - Sample tables: `SampleTable`
    - Summary statistics table: `SumstatsTable`
- Added a new `stats` submodule that implements utilities and functions 
to compute various statistics, including `ld` (SNP correlation matrix), 
  `h2` (heritability), `score`, `transforms`, `variant` statistics, and `gwa` 
  (genome-wide association testing).
  
- Added a modular class for summary statistics parsers `SumstatsParser`.
- Added modular interfaces for `executors`, representing external software, 
such as `plink`.
- Added support for window size specifications using number of SNPs and distance
in kilobases.
  
- Added `CHANGELOG.md` to track the latest changes and updates to `magenpy`.

### Changed

- Refactored the `GWADataLoader` class to utilize the new data structures.
- Updated plotting functions/utilities.
- Updated documentation in README file.
- Updated implementation of `MulticohortGWASimulator` (still incomplete).


## [0.0.1] - 2022-05-17

### Added

- Refactored the code for `pypi` package release.
- Added license, `.toml` file, and `MANIFEST.in`.

### Changed

- Updated `setup.py` to prepare for the package release.
- Updated `README.md` to add basic documentation.
