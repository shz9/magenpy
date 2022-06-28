# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
