# `magenpy`: *M*odeling and *A*nalysis of (Statistical) *Gen*etics data in *py*thon

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/magenpy.svg)](https://pypi.python.org/pypi/magenpy/)
[![PyPI version fury.io](https://badge.fury.io/py/magenpy.svg)](https://pypi.python.org/pypi/magenpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[![Linux CI](https://github.com/shz9/magenpy/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/ci-linux.yml)
[![MacOS CI](https://github.com/shz9/magenpy/actions/workflows/ci-osx.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/ci-osx.yml)
[![Windows CI](https://github.com/shz9/magenpy/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/ci-windows.yml)
[![Docs Build](https://github.com/shz9/magenpy/actions/workflows/ci-docs.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/ci-docs.yml)
[![Binary wheels](https://github.com/shz9/magenpy/actions/workflows/wheels.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/wheels.yml)


[![Downloads](https://static.pepy.tech/badge/magenpy)](https://pepy.tech/project/magenpy)
[![Downloads](https://static.pepy.tech/badge/magenpy/month)](https://pepy.tech/project/magenpy)

`magenpy` is a Python package for modeling and analyzing statistical genetics data. 
The package provides tools for:

* Reading and processing genotype data in `plink` BED format.
* Efficient LD matrix construction and storage in [Zarr](https://zarr.readthedocs.io/en/stable/index.html) array format.
* Data structures for harmonizing various GWAS data sources.
  * Includes parsers for commonly used GWAS summary statistics formats.
* Simulating polygenic traits (continuous and binary) using complex genetic architectures.
    * Multi-cohort simulation scenarios (beta)
    * Simulations incorporating functional annotations in the genetic architecture (beta)
* Interfaces for performing association testing on simulated and real phenotypes.
* Preliminary support for processing and integrating genomic annotations with other data sources.

### Helpful links

- [Documentation](https://shz9.github.io/magenpy/)
- [Citation / BibTeX records](./CITATION.md)
- [Report issues/bugs](https://github.com/shz9/magenpy/issues)
