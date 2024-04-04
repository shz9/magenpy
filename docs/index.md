# Modeling and Analysis of Statistical Genetics data in Python (`magenpy`)

This site contains documentation, tutorials, and examples for using the `magenpy` package for the purposes of 
handling, harmonizing, and computing over genotype data to prepare them for downstream genetics analyses. 
The `magenpy` package provides tools for:

* Reading and processing genotype data in `plink` BED format.
* Efficient LD matrix construction and storage in [Zarr](https://zarr.readthedocs.io/en/stable/index.html) array format.
* Data structures for harmonizing various GWAS data sources.
  * Includes parsers for commonly used GWAS summary statistics formats.
* Simulating polygenic traits (continuous and binary) using complex genetic architectures.
    * Multi-cohort simulation scenarios (beta)
    * Simulations incorporating functional annotations in the genetic architecture (beta)
* Interfaces for performing association testing on simulated and real phenotypes.
* Preliminary support for processing and integrating genomic annotations with other data sources.

If you use `magenpy` in your research, please cite the following paper:

> Zabad, S., Gravel, S., & Li, Y. (2023). **Fast and accurate Bayesian polygenic risk modeling with variational inference.** 
The American Journal of Human Genetics, 110(5), 741–761. https://doi.org/10.1016/j.ajhg.2023.03.009


## Helpful links

* [API Reference](api/overview.md)
* [Installation](installation.md)
* [Getting Started](getting_started.md)
* [Features and Configurations](features.md)
* [Command Line Scripts](commandline/overview.md)
* [Project homepage on `GitHub`](https://github.com/shz9/magenpy)
* [Sister package `viprs`](https://github.com/shz9/viprs)


## Contact

If you have any questions or issues, please feel free to open an [issue](https://github.com/shz9/magenpy/issues) 
on the `GitHub` repository or contact us directly at:

* [Shadi Zabad](mailto:shadi.zabad@mail.mcgill.ca)
* [Yue Li](mailto:yueli@cs.mcgill.ca)
* [Simon Gravel](mailto:simon.gravel@mcgill.ca)

