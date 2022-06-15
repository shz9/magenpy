# `magenpy`: *M*odeling and *A*nalysis of (Statistical) *Gen*etics data in *py*thon

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/magenpy.svg)](https://pypi.python.org/pypi/magenpy/)
[![PyPI version fury.io](https://badge.fury.io/py/magenpy.svg)](https://pypi.python.org/pypi/magenpy/)

This repository includes modules and scripts for loading, manipulating, and simulating with genotype data. 
The software works mainly with `plink`'s `.bed` file format, with the hope that we will extend this to 
other genotype data formats in the future.

The features and functionalities that this package supports are:

- Efficient LD matrix construction and storage in [Zarr](https://zarr.readthedocs.io/en/stable/) array format.
- Data structures for harmonizing various GWAS data sources.
- Simulating complex traits (continuous and binary) using elaborate genetic architectures.
  - Multi-cohort simulation scenarios (beta)
  - Simulations incorporating functional annotations in the genetic architecture (beta)
- Interfaces for performing association testing on simulated and real phenotypes.
- Preliminary support for processing and integrating genomic annotations with other data sources.

**NOTE**: The codebase is still in active development and some of interfaces or data structures will be 
replaced or modified in future releases.

## Table of Contents

- [Installation](#Installation)
- [Getting started](#getting-started)
- [Features and Configurations](#features-and-configurations)
   - [(1) Complex trait simulation](#1-complex-trait-simulation)
   - [(2) Genome-wide Association Testing](#2-genome-wide-association-testing)
   - [(3) Calculating LD matrices](#3-calculating-ld-matrices)
      - [LD estimators and their properties](#ld-estimators-and-their-properties)
   - [(4) Data harmonization](#4-data-harmonization)
   - [(5) Using `plink` as a backend](#5-using-plink-as-backend)
   - [(6) Commandline scripts](#6-commandline-scripts)
- [Citations](#citations) 


## Installation

`magenpy` is now available on the python package index `pypi` and 
can be minimally installed using the package installer `pip`:

```shell
pip install magenpy==0.0.2
```

To access the full functionalities of `magenpy`, however, it is recommended that 
you install the full list of dependencies:

```shell
pip install magenpy[full]==0.0.2
```

If you wish to install the package from source, 
you can directly clone it from the GitHub repository and install it locally 
as follows:

```
git clone https://github.com/shz9/magenpy.git
cd magenpy
make install
```

## Getting started

`magenpy` comes with a sample dataset from the 1000G project that 
you can use to experiment and familiarize yourself with its features. 
Once the package is installed, you can run a couple of quick tests 
to verify that the main features are working properly.

For example, to simulate a quantitative trait, you can invoke 
the following commands in a `python` interpreter:

```python
>>> import magenpy as mgp
>>> g_sim = mgp.GWASimulator(mgp.tgp_eur_data_path(),
                             h2=0.1)
>>> g_sim.simulate()
>>> g_sim.to_phenotype_table()
         FID      IID  phenotype
0    HG00096  HG00096   0.795651
1    HG00097  HG00097   0.550914
2    HG00099  HG00099  -0.928486
3    HG00100  HG00100   0.893626
4    HG00101  HG00101  -0.670106
..       ...      ...        ...
373  NA20815  NA20815   0.246071
374  NA20818  NA20818   1.821426
375  NA20819  NA20819  -0.457994
376  NA20826  NA20826   0.954208
377  NA20828  NA20828   0.088412

[378 rows x 3 columns]
```

This simulates a quantitative trait with heritability set to 0.1, 
using genotype data for a subset of 378 individuals of European ancestry 
from the 1000G project and approximately 15,000 SNPs on chromosome 22. 
By default, the simulator assumes that only 10% of the SNPs are 
causal (this is drawn at random from a Bernoulli distribution with `p=0.1`).
To obtain a list of the causal SNPs in this simulation, you can invoke the 
`.get_causal_status()` method, which returns a boolean vector indicating 
whether each SNP is causal or not:

```python
>>> g_sim.get_causal_status()
{22: array([ True, False, False, ..., False, False, False])}
```

In this case, for example, the first SNP is causal for the simulated phenotype. A note 
about the design of data structures in `magenpy`. Our main data structure is a class known 
as `GWADataLoader`, which is an all-purpose object that brings together different data sources and 
harmonizes them together. In `GWADataLoader`, SNP-related data sources are stored in dictionaries, where 
the key is the chromosome number and the value is the data structure associated with that chromosome. 
Thus, in the output above, the data is for chromosome 22 and the feature is a boolean 
vector indicating whether a given SNP is causal or not. 

You can also get the full information 
about the genetic architecture by invoking the method `.to_true_beta_table()`,
which returns a `pandas` dataframe with the effect size, expected heritability contribution, 
and causal status of each variant in the simulation:

```python
>>> g_sim.to_true_beta_table()
       CHR         SNP A1  MixtureComponent  Heritability      BETA  Causal
0       22    rs131538  A                 1      0.000063 -0.008013    True
1       22   rs9605903  C                 0      0.000000  0.000000   False
2       22   rs5746647  G                 0      0.000000  0.000000   False
3       22  rs16980739  T                 0      0.000000  0.000000   False
4       22   rs9605923  A                 0      0.000000  0.000000   False
...    ...         ... ..               ...           ...       ...     ...
15933   22   rs8137951  A                 0      0.000000  0.000000   False
15934   22   rs2301584  A                 0      0.000000  0.000000   False
15935   22   rs3810648  G                 0      0.000000  0.000000   False
15936   22   rs2285395  A                 0      0.000000  0.000000   False
15937   22  rs28729663  A                 0      0.000000  0.000000   False

[15938 rows x 7 columns]
```


We can also simulate a more complex genetic architecture by, e.g. simulating effect sizes from 
4 Gaussian mixture components, instead of the standard spike-and-slab density used by default:

```python
>>> g_sim = mgp.GWASimulator(mgp.tgp_eur_data_path(),
                              pi=[.9, .03, .03, .04],
                              d=[0., .01, .1, 1.],
                              h2=0.1)
>>> g_sim.simulate()
>>> g_sim.to_phenotype_table()
         FID      IID  phenotype
0    HG00096  HG00096   0.435024
1    HG00097  HG00097   1.030874
2    HG00099  HG00099   0.042322
3    HG00100  HG00100   1.392733
4    HG00101  HG00101   0.722763
..       ...      ...        ...
373  NA20815  NA20815  -0.402506
374  NA20818  NA20818  -0.321429
375  NA20819  NA20819  -0.845630
376  NA20826  NA20826  -0.690078
377  NA20828  NA20828   0.256937

[378 rows x 3 columns]
```

The parameter `pi` specifies the mixing proportions for the Gaussian mixture 
distribution and the `d` is a multiplier on the variance (see references below). In this case, 90% of the variants 
are not causal, and the remaining 10% are divided between 3 mixture components that contribute 
differentially to the heritability. The last component, which constitutes 4% of all SNPs, contributes 100 
times and 10 times to the heritability than components 2 an 3, respectively.

## Features and Configurations

### (1) Complex trait simulation

`magenpy` may be used for complex trait simulation employing a variety of different 
genetic architectures and phenotype likelihoods. For example, to simulate a quantitative 
trait with heritability set to 0.25 and where a random subset of 15% of the variants are causal, 
you may invoke the following command:

```python
>>> import magenpy as mgp
>>> g_sim = mgp.GWASimulator(mgp.tgp_eur_data_path(),
                             pi=[.85, .15],
                             h2=0.25)
>>> g_sim.simulate()
```

Then, you can export the simulated phenotype to a `pandas` dataframe as follows:

```python
>>> g_sim.to_phenotype_table()
         FID      IID  phenotype
0    HG00096  HG00096  -2.185944
1    HG00097  HG00097  -1.664984
2    HG00099  HG00099  -0.208703
3    HG00100  HG00100   0.257040
4    HG00101  HG00101  -0.068826
..       ...      ...        ...
373  NA20815  NA20815  -1.770358
374  NA20818  NA20818   1.823890
375  NA20819  NA20819   0.835763
376  NA20826  NA20826  -0.029256
377  NA20828  NA20828  -0.088353

[378 rows x 3 columns]
```

To simulate a binary, case-control trait, the interface is very similar. First, 
you need to specify that the likelihood for the phenotype is binomial (`phenotype_likelihood='binomial'`), and then 
specify the prevalence of the positive cases in the population. For example, 
to simulate a case-control trait with heritability of 0.3 and prevalence of 8%, we can invoke the following 
command:

```python
>>> import magenpy as mgp
>>> g_sim = mgp.GWASimulator(mgp.tgp_eur_data_path(),
                             phenotype_likelihood='binomial',
                             prevalence=.08,
                             h2=0.3)
>>> g_sim.simulate()
>>> g_sim.to_phenotype_table()
         FID      IID  phenotype
0    HG00096  HG00096          0
1    HG00097  HG00097          0
2    HG00099  HG00099          0
3    HG00100  HG00100          0
4    HG00101  HG00101          0
..       ...      ...        ...
373  NA20815  NA20815          0
374  NA20818  NA20818          0
375  NA20819  NA20819          1
376  NA20826  NA20826          0
377  NA20828  NA20828          0

[378 rows x 3 columns]
```

### (2) Genome-wide Association Testing

`magenpy` is not a GWAS tool. However, we do support preliminary association 
testing functionalities either via closed-form formulas for quantitative traits, or 
by providing a `python` interface to third-party association testing tools, such as `plink`.  

If you are conducting simple tests based on simulated data, an easy way to perform 
association testing is to tell the simulator that you'd like to perform GWAS on the 
simulated trait, with the `perform_gwas=True` flag:

```python
>>> import magenpy as mgp
>>> g_sim = mgp.GWASimulator(mgp.tgp_eur_data_path(),
                             pi=[.85, .15],
                             h2=0.25)
>>> g_sim.simulate(perform_gwas=True)
```

Alternatively, you can conduct association testing on real or 
simulated phenotypes using the `.perform_gwas()` method and exporting the
summary statistics to a `pandas` dataframe with `.to_summary_statistics_table()`:

```python
>>> g_sim.perform_gwas()
>>> g_sim.to_summary_statistics_table()
       CHR         SNP       POS A1 A2  ...    N      BETA         Z        SE      PVAL
0       22    rs131538  16871137  A  G  ...  378 -0.046662 -0.900937  0.051793  0.367622
1       22   rs9605903  17054720  C  T  ...  378  0.063977  1.235253  0.051793  0.216736
2       22   rs5746647  17057138  G  T  ...  378  0.057151  1.103454  0.051793  0.269830
3       22  rs16980739  17058616  T  C  ...  378 -0.091312 -1.763029  0.051793  0.077896
4       22   rs9605923  17065079  A  T  ...  378  0.069368  1.339338  0.051793  0.180461
...    ...         ...       ... .. ..  ...  ...       ...       ...       ...       ...
15933   22   rs8137951  51165664  A  G  ...  378  0.078817  1.521782  0.051793  0.128064
15934   22   rs2301584  51171497  A  G  ...  378  0.076377  1.474658  0.051793  0.140304
15935   22   rs3810648  51175626  G  A  ...  378 -0.001448 -0.027952  0.051793  0.977701
15936   22   rs2285395  51178090  A  G  ...  378 -0.019057 -0.367949  0.051793  0.712911
15937   22  rs28729663  51219006  A  G  ...  378  0.029667  0.572805  0.051793  0.566777

[15938 rows x 11 columns]
```

If you wish to use `plink2` for association testing (highly recommended), ensure that 
you tell `GWASimulator` (or any `GWADataLoader`-derived object) to use plink by explicitly 
specifying the `backend` software that you wish to use:

```python
>>> import magenpy as mgp
>>> g_sim = mgp.GWASimulator(mgp.tgp_eur_data_path(),
                             backend='plink',
                             pi=[.85, .15],
                             h2=0.25)
>>> g_sim.simulate(perform_gwas=True)
```

When using `plink`, we sometimes create temporary intermediate files to pass to the software. To clean up 
the temporary directories and files, you can invoke the `.cleanup()` command:

```python
>>> g_sim.cleanup()
```

### (3) Calculating LD matrices

One of the main features of the `magenpy` package is an efficient interface for computing 
and storing Linkage Disequilibrium (LD) matrices. LD matrices record the pairwise SNP-by-SNP 
Pearson correlation coefficient. In general, LD matrices are computed for each chromosome separately 
or may also be computed within LD blocks from, e.g. LDetect. For large autosomal chromosomes, 
LD matrices can be huge and may require extra care from the user.

In `magenpy`, LD matrices can be computed using either `xarray` or `plink`, depending on the 
backend that the user specifies (see Section 5 below). In general, at this moment, we do not recommend using 
`xarray` as a backend for large genotype matrices, as it is less efficient than `plink`. When using the default 
`xarray` as a backend, we compute the full `X'X` (X-transpose-X) matrix first, store it on-disk in chunked 
`Zarr` arrays and then perform all sparsification procedures afterwards. When using `plink` as a 
backend, on the other hand, we only compute LD between variants that are generally in close proximity 
along the chromosome, so it is generally more efficient. In the end, both will be transformed such that 
the LD matrix is stored in sparse `Zarr` arrays.

**A note on dependencies**: If you wish to use `xarray` as a backend to compute LD matrices,
you may need to install some of the optional dependencies for `magenpy`, including e.g. `rechunker`. In this case, 
it is recommended that you install all the dependencies listed in `requirements-optional.txt`. If you wish 
to use `plink` as a backend, you may need to configure the paths for `plink` as explained in Section 5 below.

In either case, to compute an LD matrix using `magenpy`, you can invoke the `.compute_ld()` method 
of all `GWADataLoader`-derived objects, as follows:

```python
>>> # Using xarray:
>>> import magenpy as mgp
>>> gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path())
>>> gdl.compute_ld(estimator='windowed',
                   output_dir='output/ldl/',
                   window_size=100)
```

This creates a windowed LD matrix where we only measure the correlation between the focal SNP and the nearest
100 from either side. As stated above, the LD matrix will be stored on-disk and that is why we must 
specify the output directory when we call `.compute_ld()`. To use `plink` to compute the LD matrix, 
we can invoke a similar command:

```python
>>> # Using plink:
>>> import magenpy as mgp
>>> gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                            backend='plink')
>>> gdl.compute_ld(estimator='windowed',
                   output_dir='output/ldl/',
                   cm_window_size=3.)
```

In this case, we are computing a windowed LD matrix where we only measure the correlation between 
SNPs that are at most 3 centi Morgan (cM) apart along the chromosome. For this small 1000G dataset, computing 
the LD matrix takes about a minute. The LD matrices in Zarr format will be written to the path 
specified in `output_dir`, so ensure that this argument is set to the desired directory. 

To facilitate working with LD matrices stored in `Zarr` format, we created a data structure in cython called `LDMatrix`, 
which acts as an intermediary and provides various features. For example, to compute LD scores 
using this LD matrix, you can invoke the command `.compute_ld_scores()` on it:

```python
>>> gdl.ld[22]
<LDMatrix.LDMatrix at 0x7fcec882e350>
>>> gdl.ld[22].compute_ld_scores()
array([1.60969673, 1.84471792, 1.59205322, ..., 3.3126724 , 3.42234106,
       2.97252452])
```

You can also get a table that lists the properties of the SNPs included in the LD matrix:

```python
>>> gdl.ld[22].to_snp_table()
       CHR         SNP       POS A1       MAF
0       22   rs9605903  17054720  C  0.260736
1       22   rs5746647  17057138  G  0.060327
2       22  rs16980739  17058616  T  0.131902
3       22   rs9605927  17067005  C  0.033742
4       22   rs5746664  17074622  A  0.066462
...    ...         ...       ... ..       ...
14880   22   rs8137951  51165664  A  0.284254
14881   22   rs2301584  51171497  A  0.183027
14882   22   rs3810648  51175626  G  0.065440
14883   22   rs2285395  51178090  A  0.061350
14884   22  rs28729663  51219006  A  0.159509

[14885 rows x 5 columns]
```

Finally, note that the `LDMatrix` object supports an iterator interface, so in principle 
you can iterate over rows of the LD matrix without loading the entire thing into memory. 
The following example shows the first 10 entries of the first row of the matrix:

```python
>>> np.array(next(gdl.ld[22]))[:10]
array([ 1.00000262, -0.14938791, -0.27089083,  0.33311111,  0.35015815,
       -0.08077946, -0.08077946,  0.0797345 , -0.16252513, -0.23680465])
```

Finally, as of `magenpy==0.0.2`, now you can export the Zarr array into a `scipy` sparse `csr` 
matrix as follows:

```python
>>> gdl.ld[22].to_csr_matrix()
<15938x15938 sparse matrix of type '<class 'numpy.float64'>'
	with 24525854 stored elements in Compressed Sparse Row format>
```

#### LD estimators and their properties

`magenpy` supports computing LD matrices using 4 different estimators that are commonly used 
in statistical genetics applications. 
For a more thorough description of the estimators and their properties, consult our manuscript 
and the citations therein. The LD estimators are:

1) `windowed` (recommended): The windowed estimator computes the pairwise correlation coefficient between SNPs that are 
    within a pre-defined distance along the chromosome from each other. In many statistical genetics applications, the 
   recommended distance is between 1 and 3 centi Morgan (cM). As of `magenpy==0.0.2`, now you can customize 
   the distance based on three criteria: **(1)** A window size based on the number neighboring variants, **(2)** 
   distance threshold in kilobases (kb), and **(3)** distance threshold in centi Morgan (cM). When defining the 
   boundaries for each SNP, `magenpy` takes the intersection of the boundaries defined by each window.
   
```python
>>> import magenpy as mgp
>>> gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                            backend='plink')
>>> gdl.compute_ld('windowed', output_dir='output/ld/',
                   window_size=100, kb_window_size=1000, cm_window_size=2.)
>>> gdl.cleanup()
```

2) `block`: The block estimator estimates the pairwise correlation coefficient between 
variants that are in the same LD block, as defined by, e.g. LDetect. Given an LD block file, 
   we can compute a block-based LD matrix as follows:
   
```python
>>> import magenpy as mgp
>>> ld_block_url = "https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed"
>>> gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                            backend='plink')
>>> gdl.compute_ld('block', output_dir='output/ld/',
                   ld_blocks_file=ld_block_url)
>>> gdl.cleanup()
```

If you have the LD blocks file on your system, you can also pass the path to the file instead.

3) `shrinkage`: For the shrinkage estimator, we shrink the entries of the LD matrix by a 
   quantity related to the distance between SNPs along the chromosome + some additional information 
   related to the sample from which the genetic map was estimated. In particular, 
   we need to specify the effective population size and the sample size used to 
   estimate the genetic map. Also, to make the matrix sparse, we often specify a threshold value 
   below which we consider the correlation to be zero. Here's an example for the 1000G sample:
   

```python
>>> import magenpy as mgp
>>> gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                            backend='plink')
>>> gdl.compute_ld('shrinkage', output_dir='output/ld/',
                   genetic_map_ne=11400, # effective population size (Ne)
                   genetic_map_sample_size=183, # Sample size
                   threshold=1e-5) # The cutoff value
>>> gdl.cleanup()
```

4) `sample`: This estimator computes the pairwise correlation coefficient between all SNPs on 
   the same chromosome and thus results in a dense matrix. Thus, it is rarely used in practice and 
   we include it here for testing/debugging purposes mostly. To compute the sample LD matrix, you only need 
   to specify the correct estimator:
   
```python
>>> import magenpy as mgp
>>> gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                            backend='plink')
>>> gdl.compute_ld('sample', output_dir='output/ld/')
>>> gdl.cleanup()
```

### (4) Data harmonization

There are many different statistical genetics data sources and formats out there. One of the goals of 
`magenpy` is to create a friendly interface for matching and merging these data sources for 
downstream analyses. For example, for summary statistics-based methods, we often need 
to merge the LD matrix derived from a reference panel with the GWAS summary statistics estimated 
in a different cohort. While this is a simple task, it can be tricky sometimes, e.g. in 
cases where the effect allele is flipped between the two cohort.

The functionalities that we provide for this are minimal at this stage and mainly geared towards 
harmonizing `Zarr`-formatted LD matrices with GWAS summary statistics. The following example 
shows how to do this in a simple case:

```python
>>> import magenpy as mgp
>>> # First, generate some summary statistics from a simulation:
>>> g_sim = mgp.GWASimulator(mgp.tgp_eur_data_path())
>>> g_sim.simulate()
>>> g_sim.to_summary_statistics_table().to_csv(
        "chr_22.sumstats", sep="\t", index=False
    )
>>> # Then load those summary statistics and match them with previously
>>> # computed windowed LD matrix for chromosome 22:
>>> gdl = mgp.GWADataLoader(ld_store_files='output/windowed_ld/chr_22/',
                             sumstats_files='chr_22.sumstats',
                             sumstats_format='magenpy')
```

Here, the `GWADataLoader` object takes care of the harmonization step by 
automatically invoking the `.harmonize_data()` method. When you read or update 
any of the data sources, we recommend that you invoke the `.harmonize_data()` method again 
to make sure that all the data sources are aligned properly. In the near future, 
we are planning to add many other functionalities in this space. Stay tuned.

### (5) Using `plink` as backend

Many of the functionalities that `magenpy` supports require access to and performing linear algebra 
operations on the genotype matrix. By default, `magenpy` uses `xarray` and `dask` 
to carry out these operations, as these are the tools supported by our main dependency: `pandas-plink`.

However, `dask` can be quite slow and inefficient when deployed on large-scale genotype matrices. To get 
around this difficulty, for many operations, such as linear scoring or computing minor allele frequency, 
we support (and recommend) using `plink` as a backend.

To use `plink` as a backend for `magenpy`, first you may need to configure the paths 
on your system. By default, `magenpy` assumes that, in the shell, the name `plink2` invokes the `plink2` 
executable and `plink` invokes `plink1.9` software. To change this behavior, you can update the 
configuration file as follows. First, let's see the default configurations that ship with `magenpy`:

```python
>>> import magenpy as mgp
>>> mgp.print_options()
-> Section: DEFAULT
---> plink1.9_path: plink
---> plink2_path: plink2
```

The above shows the default configurations for the `plink1.9` and `plink2` paths. To change 
the path for `plink2`, for example, you can use the `set_option()` function:

```python
>>> mgp.set_option("plink2_path", "~/software/plink2/plink2")
>>> mgp.print_options()
-> Section: USER
---> plink2_path: ~/software/plink2/plink2
---> plink1.9_path: plink
-> Section: DEFAULT
---> plink1.9_path: plink
---> plink2_path: plink2
```

As you can see, this added a new section to the configuration file, named `USER`, that has the 
new path for the `plink2` software. Now, every time `magenpy` needs to invoke `plink2`, it calls 
the executable stored at `~/software/plink2/`. Note that you only need to do this once on any particular 
machine or system, as this preference is now recorded in the configuration file and will be taken into 
account for all future operations.

Note that for most of the operations, we assume that the user has `plink2` installed. We only 
use `plink1.9` for some operations that are currently not supported by `plink2`, especially for 
e.g. LD computation. This behavior may change in the near future.

Once the paths are configured, to use `plink` as a backend for the various computations and 
tools, make sure that you specify the `backend='plink'` flag in `GWADataLoader` and all of its 
derived data structures (including all the `GWASimulator` classes):

```python
>>> import magenpy as mgp
>>> gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                            backend='plink')
```

### (6) Commandline scripts

If you are not comfortable programming in `python` and would like to access some of the functionalities 
of `magenpy` with minimal interaction with `python` code, we packaged a number of commandline 
scripts that can be useful for some downstream applications.

The binaries that are currently supported are:

1) `magenpy_ld`: For computing LD matrices and storing them in `Zarr` format.
2) `magenpy_simulate`: For simulating complex traits with various genetic architectures.

Once you install `magenpy` via `pip`, these two scripts will be added to the system `PATH` 
and you can invoke them directly from the commandline, as follows:

```shell
$ magenpy_ld -h

**********************************************                            
 _ __ ___   __ _  __ _  ___ _ __  _ __  _   _ 
| '_ ` _ \ / _` |/ _` |/ _ \ '_ \| '_ \| | | |
| | | | | | (_| | (_| |  __/ | | | |_) | |_| |
|_| |_| |_|\__,_|\__, |\___|_| |_| .__/ \__, |
                 |___/           |_|    |___/
Modeling and Analysis of Genetics data in python
Version: 0.0.2 | Release date: May 2022
Author: Shadi Zabad, McGill University
**********************************************
< Compute LD matrix and output in Zarr format >

usage: magenpy_ld [-h] [--estimator {windowed,sample,shrinkage,block}] --bfile
                  BED_FILE [--keep KEEP_FILE] [--extract EXTRACT_FILE]
                  [--backend {plink,xarray}] [--temp-dir TEMP_DIR]
                  --output-dir OUTPUT_DIR [--ld-window LD_WINDOW]
                  [--ld-window-kb LD_WINDOW_KB] [--ld-window-cm LD_WINDOW_CM]
                  [--ld-blocks LD_BLOCKS] [--genmap-Ne GENMAP_NE]
                  [--genmap-sample-size GENMAP_SS]
                  [--shrinkage-cutoff SHRINK_CUTOFF]

Commandline arguments for LD matrix computation

optional arguments:
  -h, --help            show this help message and exit
  --estimator {windowed,sample,shrinkage,block}
                        The LD estimator (windowed, shrinkage, block, sample)
  --bfile BED_FILE      The path to a plink BED file
  --keep KEEP_FILE      A plink-style keep file to select a subset of
                        individuals to compute the LD matrices.
  --extract EXTRACT_FILE
                        A plink-style extract file to select a subset of SNPs
                        to compute the LD matrix for.
  --backend {plink,xarray}
                        The backend software used to compute the Linkage-
                        Disequilibrium between variants.
  --temp-dir TEMP_DIR   The temporary directory where we store intermediate
                        files.
  --output-dir OUTPUT_DIR
                        The output directory where the Zarr formatted LD
                        matrices will be stored.
  --ld-window LD_WINDOW
                        Maximum number of neighboring SNPs to consider when
                        computing LD.
  --ld-window-kb LD_WINDOW_KB
                        Maximum distance (in kilobases) between pairs of
                        variants when computing LD.
  --ld-window-cm LD_WINDOW_CM
                        Maximum distance (in centi Morgan) between pairs of
                        variants when computing LD.
  --ld-blocks LD_BLOCKS
                        Path to the file with the LD block boundaries, in
                        LDetect format (e.g. chr start stop, tab-separated)
  --genmap-Ne GENMAP_NE
                        The effective population size for the population from
                        which the genetic map was derived.
  --genmap-sample-size GENMAP_SS
                        The sample size for the dataset used to infer the
                        genetic map.
  --shrinkage-cutoff SHRINK_CUTOFF
                        The cutoff value below which we assume that the
                        correlation between variants is zero.
```

And: 

```shell
$ magenpy_simulate -h

**********************************************                            
 _ __ ___   __ _  __ _  ___ _ __  _ __  _   _ 
| '_ ` _ \ / _` |/ _` |/ _ \ '_ \| '_ \| | | |
| | | | | | (_| | (_| |  __/ | | | |_) | |_| |
|_| |_| |_|\__,_|\__, |\___|_| |_| .__/ \__, |
                 |___/           |_|    |___/
Modeling and Analysis of Genetics data in python
Version: 0.0.2 | Release date: May 2022
Author: Shadi Zabad, McGill University
**********************************************
< Simulate complex quantitative or case-control traits >

usage: magenpy_simulate [-h] --bed-files BED_FILES [--keep KEEP_FILE]
                        [--extract EXTRACT_FILE] [--backend {xarray,plink}]
                        [--temp-dir TEMP_DIR] --output-file OUTPUT_FILE
                        [--output-simulated-effects] --h2 H2
                        [--mix-prop MIX_PROP] [--var-mult VAR_MULT]
                        [--likelihood {gaussian,binomial}]
                        [--prevalence PREVALENCE]

Commandline arguments for the complex trait simulator

optional arguments:
  -h, --help            show this help message and exit
  --bed-files BED_FILES
                        The BED files containing the genotype data. You may
                        use a wildcard here (e.g. "data/chr_*.bed")
  --keep KEEP_FILE      A plink-style keep file to select a subset of
                        individuals for simulation.
  --extract EXTRACT_FILE
                        A plink-style extract file to select a subset of SNPs
                        for simulation.
  --backend {xarray,plink}
                        The backend software used for the computation.
  --temp-dir TEMP_DIR   The temporary directory where we store intermediate
                        files.
  --output-file OUTPUT_FILE
                        The path where the simulated phenotype will be stored
                        (no extension needed).
  --output-simulated-effects
                        Output a table with the true simulated effect size for
                        each variant.
  --h2 H2               Trait heritability. Ranges between 0. and 1.,
                        inclusive.
  --mix-prop MIX_PROP, -p MIX_PROP
                        Mixing proportions for the mixture density (comma
                        separated). For example, for the spike-and-slab
                        mixture density, with the proportion of causal
                        variants set to 0.1, you can specify: "--mix-prop
                        0.9,0.1 --var-mult 0,1".
  --var-mult VAR_MULT, -d VAR_MULT
                        Multipliers on the variance for each mixture
                        component.
  --likelihood {gaussian,binomial}
                        The likelihood for the simulated trait. Gaussian (e.g.
                        quantitative) or binomial (e.g. case-control).
  --prevalence PREVALENCE
                        The prevalence of cases (or proportion of positives)
                        for binary traits. Ranges between 0. and 1.
```

You can find examples of how to run the commandline scripts in the `examples` directory on GitHub. 
To request other functionalities to be packaged with `magenpy`, please contact the developers or 
open an Issue on [GitHub](https://github.com/shz9/magenpy).

## Citations

Shadi Zabad, Simon Gravel, Yue Li. **Fast and Accurate Bayesian Polygenic Risk Modeling with Variational Inference**. (2022)

```bibtex
@article {
    Zabad2022.05.10.491396,
    author = {Zabad, Shadi and Gravel, Simon and Li, Yue},
    title = {Fast and Accurate Bayesian Polygenic Risk Modeling with Variational Inference},
    elocation-id = {2022.05.10.491396},
    year = {2022},
    doi = {10.1101/2022.05.10.491396},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2022/05/11/2022.05.10.491396},
    journal = {bioRxiv}
}
```
