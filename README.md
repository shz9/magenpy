# `magenpy`: *M*odeling and *A*nalysis of (Statistical) *Gen*etics data in *py*thon

This repository includes modules and scripts for loading, manipulating, and simulating with genotype data. 
The software works mostly with `plink`'s `.bed` file format, with the hope that we will extend this to 
other data formats in the future.

The features and functionalities that this package supports are:

- Efficient LD matrix construction and storage in [Zarr](https://zarr.readthedocs.io/en/stable/) array format.
- Data structures for harmonizing various GWAS data sources.
- Simulating complex traits (continuous and binary) using complex genetic architectures.
  - Multi-ethnic simulation scenarios (beta)
  - Simulations incorporating functional annotations in the genetic architecture (beta)
- Interfaces for performing association testing on simulated and real phenotypes.
- Preliminary support for processing and integrating genomic annotations with other data sources.

**NOTE**: The codebase is still in active development and some of interfaces or data structures will be 
replaced or modified in future releases.

## Installation

`magenpy` is now available on the python package index `pypi` and 
can be minimally installed using the package installer `pip`:

```shell
pip install magenpy==0.0.1
```

To access the full functionalities of `magenpy`, however, it is recommended that 
you install the full list of dependencies:

```shell
pip install magenpy[full]==0.0.1
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
>>> g_sim = mgp.GWASSimulator(mgp.tgp_eur_data_path(),
                              h2g=0.1)
>>> g_sim.simulate()
>>> g_sim.phenotypes[:10]
array([ 1.6482826 ,  0.78637006,  0.01192625,  0.94761538, -0.44302667,
       -0.64618552,  1.40570962,  0.038859  ,  1.91665207,  0.27427175])
```

This simulates a quantitative trait with heritability set to 0.1, 
using genotype data for a subset of 378 individuals of European ancestry 
from the 1000G project and approximately 15,000 SNPs on chromosome 22. 
By default, the simulator assumes that only 10% of the SNPs are 
causal (this is drawn at random from a Bernoulli distribution with p=0.1).
To obtain a list of the causal SNPs in this simulation, you can invoke the 
`.get_causal_status()` method, which returns a boolean vector indicating 
whether each SNP is causal or not:

```python
>>> g_sim.get_causal_status()
{22: array([False, False, False, ...,  False, False, True])}
```

In this case, for example, the last SNP is causal for the simulated phenotype. A note 
about the design of data structures in `magenpy`. In most cases, data for 
SNPs are stored in dictionaries, where the key is the chromosome number 
and the value is a vector of features for each SNP. Thus, in the output above, 
the data is for chromosome 22 and the feature is a boolean indicating whether 
a given SNP is causal or not. You can also get the full information 
about the genetic architecture by invoking the method `.to_true_beta_table()`,
which returns a `pandas` table with the effect size, expected heritability contribution, 
and causal status of each variant in the simulation:

```python
>>> g_sim.to_true_beta_table()
       CHR         SNP A1  MixtureComponent  Heritability      BETA  Causal
0       22    rs131538  A                 0      0.000000  0.000000   False
1       22   rs9605903  C                 0      0.000000  0.000000   False
2       22   rs5746647  G                 0      0.000000  0.000000   False
3       22  rs16980739  T                 0      0.000000  0.000000   False
4       22   rs9605923  A                 0      0.000000  0.000000   False
...    ...         ... ..               ...           ...       ...     ...
15933   22   rs8137951  A                 0      0.000000  0.000000   False
15934   22   rs2301584  A                 0      0.000000  0.000000   False
15935   22   rs3810648  G                 0      0.000000  0.000000   False
15936   22   rs2285395  A                 0      0.000000  0.000000   False
15937   22  rs28729663  A                 1      0.000125  0.001446    True

[15938 rows x 7 columns]
```


We can also simulate a more complex genetic architecture by, e.g. simulating with 4 mixture 
components:

```python
>>> g_sim = mgp.GWASSimulator(mgp.tgp_eur_data_path(),
                              pi=[.9, .03, .03, .04],
                              d=[0., .01, .1, 1.],
                              h2g=0.1)
>>> g_sim.simulate()
>>> g_sim.phenotypes[:10]
array([-1.11029618, -0.99254766, -2.37268932, -0.55944617,  0.24877759,
        0.74470583,  2.58372899, -0.51890023, -0.05431463, -0.30771234])
```

The parameter `pi` specifies the mixing proportions for the Gaussian mixture 
distribution and the `d` is a multiplier on the variance. In this case, 90% of the variants 
are not causal, and the remaining 10% are divided between 3 mixture components that contribute 
differentially to the heritability. The last component, which constitutes 4% of all SNPs, contributes 100 
times and 10 times to the heritability than components 2 an 3, respectively.

## Features and Configurations

### (1) Complex trait simulation

You can use `magenpy` for complex trait simulation using a variety of different 
genetic architectures and phenotype likelihoods. For example, to simulate a quantitative 
trait with heritability set to 0.25 and where a random subset of 15% of the variants are causal, 
you may invoke this simple command:

```python
>>> import magenpy as mgp
>>> g_sim = mgp.GWASSimulator(mgp.tgp_eur_data_path(),
                              pi=[.85, .15],
                              h2g=0.25)
>>> g_sim.simulate()
```

Then, you can export the simulated phenotype to a `pandas` table as follows:

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
specify prevalence of the positive cases in the sample. For example, 
to simulate a case-control trait with heritability of 0.3 and prevalence of 8%, we can invoke the following 
command:

```python
>>> import magenpy as mgp
>>> g_sim = mgp.GWASSimulator(mgp.tgp_eur_data_path(),
                              phenotype_likelihood='binomial',
                              prevalence=.08,
                              h2g=0.3)
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
by providing a `python` interface to third-party association testing tools, such as `plink2`.  

If you are conducting simple tests based on simulated data, an easy way to perform 
association testing is to tell the simulator that you'd like to perform GWAS on the 
simulated trait, with the `perform_gwas=True` flag:

```python
>>> import magenpy as mgp
>>> g_sim = mgp.GWASSimulator(mgp.tgp_eur_data_path(),
                              pi=[.85, .15],
                              h2g=0.25)
>>> g_sim.simulate(perform_gwas=True)
```

Alternatively, you can conduct association testing on real or 
simulated phenotypes using the `.perform_gwas()` command and exporting the
summary statistics to a `pandas` table with `.to_snp_table()`:

```python
>>> g_sim.perform_gwas()
>>> g_sim.to_snp_table()
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
you tell `GWASSimulator` (or any `GWASDataLoader`-derived object) to use plink:

```python
>>> import magenpy as mgp
>>> g_sim = mgp.GWASSimulator(mgp.tgp_eur_data_path(),
                              use_plink=True,
                              pi=[.85, .15],
                              h2g=0.25)
>>> g_sim.simulate(perform_gwas=True)
```

When using `plink`, we often create temporary intermediate files to pass to the software. To clean up 
the temporary directories and files, you can invoke the `.cleanup()` command:

```python
>>> g_sim.cleanup()
```

### (3) Calculating LD matrices

One of the main features of the `magenpy` package is an efficient interface for computing 
and storing Linkage Disequilibrium (LD) matrices. LD matrices record the pairwise SNP-by-SNP 
Pearson correlation coefficient. In general, LD matrices are computed for each chromosome separately 
but may also only be computed within LD blocks from, e.g. LDetect. For large autosomal chromosomes, 
LD matrices can be huge and may require extra care from the user.

In `magenpy`, LD matrices can be computed using either `dask` or `plink`, depending on the 
backend that the user specifies (see Section 5 below). In general, at this moment, we do not recommend using 
`dask` as a backend for large genotype matrices, as it is less efficient than `plink`. When using the default 
`dask` as a backend, we compute the full `X'X` (X-transpose-X) matrix first, store it on disk in chunked 
`Zarr` arrays and then perform all sparsification procedures afterwards. When using `plink` as a 
backend, on the other hand, we only compute LD between variants that are generally in close proximity 
along the chromosome, so it is generally more efficient. In the end, both will be transformed such that 
the LD matrix is stored in sparse `Zarr` arrays.

**A note on dependencies**: If you wish to use `dask` as a backend to compute LD matrices,
you may need to install some of the optional dependencies for `magenpy`, including e.g. `rechunker`. In this case, 
it is recommended that you install all the dependencies listed in `requirements-optional.txt`. If you wish 
to use `plink` as a backend, you may need to configure the paths for `plink` as explained in Section 5 below.

In either case, to compute an LD matrix using `magenpy`, you can either set the 
`compute_ld=True` flag in `GWASDataLoader` classes or invoke the `.compute_ld()` method 
directly, as follows:

```python
>>> # Using dask:
>>> import magenpy as mgp
>>> gdl = mgp.GWASDataLoader(mgp.tgp_eur_data_path(),
                             compute_ld=True,
                             output_dir="output/ld/")
```

or 

```python
>>> # Using plink:
>>> import magenpy as mgp
>>> gdl = mgp.GWASDataLoader(mgp.tgp_eur_data_path(),
                             use_plink=True,
                             output_dir="output/ld/")
>>> gdl.compute_ld()
```

For this small 1000G dataset, computing the LD matrix takes about a minute. The LD matrices in Zarr 
format will be written to the path specified in `output_dir`, so ensure that this argument is set to 
the desired directory. 

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

#### LD estimators and their properties

`magenpy` supports computing LD matrices using 4 different commonly-used estimators. 
For a more thorough description of the estimators and their properties, consult our manuscript 
and the citations therein. The LD estimators are:

1) `windowed` (default): The windowed estimator computes the pairwise correlation coefficient between SNPs that are 
    within a pre-defined distance along the chromosome from each other. In many statistical genetics applications, the 
   recommended distance is between 1 and 3 centi Morgan (cM). 
   To use the `windowed` estimator, you may need to specify a couple of things when 
   initializing a `GWASDataLoader` object: The window unit as well as the distance cutoff. Currently, we mainly support 
   using centi Morgan as a unit of distance, though we will add others pretty soon. Here's an example that constructs 
   windowed LD matrices with the distance cutoff set to 3cM:
   
```python
>>> import magenpy as mgp
>>> gdl = mgp.GWASDataLoader(mgp.tgp_eur_data_path(),
                             ld_estimator='windowed',
                             window_unit='cM',
                             cm_window_cutoff=3.,
                             compute_ld=True,
                             use_plink=True,
                             output_dir="output/windowed_ld/")
>>> gdl.cleanup()
```

2) `block`: The block estimator estimates the pairwise correlation coefficient between 
variants that are in the same LD block, as defined by, e.g. LDetect. Given an LD block file, 
   we can compute a block-based LD matrix as follows:
   
```python
>>> import magenpy as mgp
>>> ld_block_url = "https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed"
>>> gdl = mgp.GWASDataLoader(mgp.tgp_eur_data_path(),
                             ld_estimator='block',
                             ld_block_files=ld_block_url,
                             compute_ld=True,
                             use_plink=True,
                             output_dir="output/block_ld/")
>>> gdl.cleanup()
```

If you have the LD blocks file on your system, you can also pass the path to the file instead.

3) `shrinkage`: For the shrinkage estimator, we shrink the entries in the LD matrix by a 
   quantity related to the distance between SNPs along the chromosome + some additional information 
   related to the sample from which the genetic map was estimated. In particular, 
   we need to specify the effective population size and the sample size for the genetic map. Also, 
   to make the matrix sparse, we often specify a threshold value below which we consider 
   the correlation to be zero. Here's an example for the 1000G sample:
   

```python
>>> import magenpy as mgp
>>> gdl = mgp.GWASDataLoader(mgp.tgp_eur_data_path(),
                             ld_estimator='shrinkage',
                             genmap_Ne=11400,
                             genmap_sample_size=183,
                             shrinkage_cutoff=1e-5,
                             compute_ld=True,
                             use_plink=True,
                             output_dir="output/shrinkage_ld/")
>>> gdl.cleanup()
```

4) `sample`: This estimator computes the pairwise correlation coefficient between all SNPs on 
   the same chromosome and thus results in a dense matrix. Thus, it is rarely used in practice and 
   we include here for testing/debugging purposes mostly. To compute the sample LD matrix, you only need 
   to specify the correct estimator:
   
```python
>>> import magenpy as mgp
>>> gdl = mgp.GWASDataLoader(mgp.tgp_eur_data_path(),
                             ld_estimator='sample',
                             compute_ld=True,
                             use_plink=True,
                             output_dir="output/sample_ld/")
>>> gdl.cleanup()
```

### (4) Data harmonization

There are many different statistical genetics data sources and formats out there. One of the goals of 
`magenpy` is to create a friendly interface for matching and merging these data sources for 
downstream analyses. For example, for summary statistics-based method, we often need 
to merge the LD matrix derived from a reference panel with the GWAS summary statistics estimated 
in a different cohort. While this is a simple task, it can be tricky sometimes, e.g. in 
cases where the effect allele is flipped between the two cohort.

The functionalities that we provide for this are minimal at this stage and mainly geared towards 
harmonizing `Zarr`-formatted LD matrices with GWAS summary statistics. The following example 
shows how to do this in a simple case:

```python
>>> import magenpy as mgp
>>> # First, generate some summary statistics from a simulation:
>>> g_sim = mgp.GWASSimulator(mgp.tgp_eur_data_path())
>>> g_sim.simulate()
>>> g_sim.to_snp_table().to_csv("chr_22.sumstats", sep="\t", index=False)
>>> # Then load those summary statistics and match them with previously
>>> # computed windowed LD matrix for chromosome 22:
>>> gdl = mgp.GWASDataLoader(ld_store_files='output/windowed_ld/chr_22/',
                             sumstats_files='chr_22.sumstats',
                             sumstats_format='magenpy')
```

Here, the `GWASDataLoader` object takes care of the harmonization step by 
automatically invoking the `.harmonize_data()` method. In the near future, 
we are planning to add many other functionalities in this space. Stay tuned.

### (5) Using `plink` as backend

Many of the functionalities that `magenpy` supports require access to and performing linear algebra 
operations on top of the genotype matrix. By default, `magenpy` uses `xarray` and `dask` 
to carry out these operations, as these are the tools supported by our main dependency: `pandas-plink`.

However, `dask` can be quite slow and inefficient when deployed on large-scale genotype matrices. To get 
around this difficulty, for many operations, such as individual scoring or computing minor allele frequency, 
we support (and recommend) using `plink` as a backend.

To use `plink` as a backend for `magenpy`, first you may need to configure the paths 
on your system. By default, `magenpy` assumes that, in the shell, the name `plink2` invokes the `plink2` 
software and `plink` invokes `plink1.9` software. To change this behavior, you can update the 
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

As you see, this added a new section to the configuration file, named `USER`, that has the 
new path for the `plink2` software. Now, every time `magenpy` needs to invoke `plink2`, it calls 
the binary stored at `~/software/plink2/`. Note that you only need to do this once on any particular 
machine or system, as this preference is now recorded in the config file and will be taken into 
account for all future operations.

Note that for most of the operations, we assume that the user has `plink2` installed. We only 
use `plink1.9` for some operations that are currently not supported by `plink2`, especially for 
e.g. LD computation. This behavior may change in the near future.

Once the paths are configured, to use `plink` as a backend for the various computations and 
tools, make sure that you specify the `use_plink=True` flag in `GWASDataLoader` and all of its 
derived data structures (including all the `GWASSimulator` classes):

```python
>>> import magenpy as mgp
>>> gdl = mgp.GWASDataLoader(mgp.tgp_eur_data_path(),
                             use_plink=True)
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
$ magenpy_ld

**********************************************                            
 _ __ ___   __ _  __ _  ___ _ __  _ __  _   _ 
| '_ ` _ \ / _` |/ _` |/ _ \ '_ \| '_ \| | | |
| | | | | | (_| | (_| |  __/ | | | |_) | |_| |
|_| |_| |_|\__,_|\__, |\___|_| |_| .__/ \__, |
                 |___/           |_|    |___/
Modeling and Analysis of Genetics data in python
Version: 0.0.1 | Release date: May 2022
Author: Shadi Zabad, McGill University
**********************************************
< Compute LD matrix and output in Zarr format >

usage: magenpy_ld [-h] [--estimator {sample,windowed,shrinkage,block}] --bfile BED_FILE [--keep KEEP_FILE] [--extract EXTRACT_FILE] [--backend {dask,plink}]
                  [--temp-dir TEMP_DIR] --output-dir OUTPUT_DIR [--cm-dist CM_DIST] [--ld-blocks LD_BLOCKS] [--genmap-Ne GENMAP_NE] [--genmap-sample-size GENMAP_SS]
                  [--shrinkage-cutoff SHRINK_CUTOFF]
magenpy_ld: error: the following arguments are required: --bfile, --output-dir
```

And: 

```shell
$ magenpy_simulate

**********************************************                            
 _ __ ___   __ _  __ _  ___ _ __  _ __  _   _ 
| '_ ` _ \ / _` |/ _` |/ _ \ '_ \| '_ \| | | |
| | | | | | (_| | (_| |  __/ | | | |_) | |_| |
|_| |_| |_|\__,_|\__, |\___|_| |_| .__/ \__, |
                 |___/           |_|    |___/
Modeling and Analysis of Genetics data in python
Version: 0.0.1 | Release date: May 2022
Author: Shadi Zabad, McGill University
**********************************************
< Simulate complex quantitative or case-control traits >

usage: magenpy_simulate [-h] --bed-files BED_FILES [--keep KEEP_FILE] [--extract EXTRACT_FILE] [--backend {plink,dask}] [--temp-dir TEMP_DIR] --output-file OUTPUT_FILE
                        [--output-simulated-effects] --h2g H2G [--mix-prop MIX_PROP] [--var-mult VAR_MULT] [--likelihood {binomial,gaussian}] [--prevalence PREVALENCE]
magenpy_simulate: error: the following arguments are required: --bed-files, --output-file, --h2g
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
