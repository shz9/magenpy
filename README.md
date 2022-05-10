# `magenpy`: *M*odeling and *A*nalysis of (Statistical) *Gen*etics data in *py*thon

This repository includes code and scripts for loading, manipulating, and simulating with genotype data. The software works mostly with PLINK's `.bed` file format, with the hope that we will extend this to other data formats in the near future.

The functionalities that this package supports are:

- Efficient LD matrix construction and storage in [Zarr](https://zarr.readthedocs.io/en/stable/) array format.
- Data structures for harmonizing GWAS summary statistics
- Simulating complex traits (continuous and binary) using complex genetic architectures.
  - Multi-ethnic simulation scenarios (beta)
  - Simulations incorporating functional annotations in the genetic architecture (beta)
- Preliminary support for processing and integrating genomic annotations with other data sources.

## Installation

We are working on adding this source code into `pypi` in the near future.
In the meantime, you can manually install the source code as follows:

```
git clone https://github.com/shz9/magenpy.git
cd magenpy
pip install -r requirements.txt
pip install -r optional-requirements.txt
```

## Getting started

TODO

## Citations

Shadi Zabad, Simon Gravel, Yue Li. **Fast and Accurate Bayesian Polygenic Risk Modeling with Variational Inference** (2022)


