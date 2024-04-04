
## (1) Complex trait simulation

`magenpy` may be used for complex trait simulation employing a variety of different 
genetic architectures and phenotype likelihoods. For example, to simulate a quantitative 
trait with heritability set to 0.25 and where a random subset of 15% of the variants are causal, 
you may invoke the following command:

```python linenums="1"
import magenpy as mgp
g_sim = mgp.PhenotypeSimulator(mgp.tgp_eur_data_path(),  # Path to 1000G genotype data
                               pi=[.85, .15],  # Proportion of non-causal and causal variants
                               h2=0.25)  # Heritability
# Export simulated phenotype to pandas dataframe:
g_sim.to_phenotype_table()
```

```
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

To simulate a binary, or case-control, trait, the interface is very similar. First, 
you need to specify that the likelihood for the phenotype is binomial (`phenotype_likelihood='binomial'`), and then 
specify the prevalence of the positive cases in the population. For example, 
to simulate a case-control trait with heritability of 0.3 and prevalence of 8%, we can invoke the following 
command:

```python linenums="1"
import magenpy as mgp
g_sim = mgp.PhenotypeSimulator(mgp.tgp_eur_data_path(),
                               phenotype_likelihood='binomial',
                               prevalence=.08,
                               h2=0.3)
g_sim.simulate()
g_sim.to_phenotype_table()
```

```
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

## (2) Genome-wide Association Testing (GWAS)

`magenpy` is **not** a GWAS tool. However, we do support preliminary association 
testing functionalities either via closed-form formulas for quantitative traits, or 
by providing a `python` interface to third-party association testing tools, such as `plink`.  

If you are conducting simple tests based on simulated data, an easy way to perform 
association testing is to tell the simulator that you'd like to perform GWAS on the 
simulated trait, with the `perform_gwas=True` flag:

```python linenums="1"
import magenpy as mgp
g_sim = mgp.PhenotypeSimulator(mgp.tgp_eur_data_path(),
                               pi=[.85, .15],
                               h2=0.25)
g_sim.simulate(perform_gwas=True)
```

Alternatively, you can conduct association testing on real or 
simulated phenotypes using the `.perform_gwas()` method and exporting the
summary statistics to a `pandas` dataframe with `.to_summary_statistics_table()`:

```python linenums="1"
g_sim.perform_gwas()
g_sim.to_summary_statistics_table()
```

```
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
you tell `PhenotypeSimulator` (or any `GWADataLoader`-derived object) to use plink by explicitly 
specifying the `backend` software that you wish to use:

```python linenums="1"
import magenpy as mgp
g_sim = mgp.PhenotypeSimulator(mgp.tgp_eur_data_path(),
                               backend='plink', # Set the backend
                               pi=[.85, .15],
                               h2=0.25)
g_sim.simulate(perform_gwas=True)
g_sim.cleanup() # Clean up temporary files
```

When using `plink`, we sometimes create temporary intermediate files to pass to the software. To clean up 
the temporary directories and files, you can invoke the `.cleanup()` command.

## (3) Calculating LD matrices

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

In either case, to compute an LD matrix using `magenpy`, you can invoke the `.compute_ld()` method 
of all `GWADataLoader`-derived objects, as follows:

```python linenums="1"
# Using xarray:
import magenpy as mgp
gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path())
gdl.compute_ld(estimator='windowed',
               output_dir='output/ldl/',
               window_size=100)
gdl.cleanup()
```

This creates a windowed LD matrix where we only measure the correlation between the focal SNP and the nearest
100 variants from either side. As stated above, the LD matrix will be stored on-disk and that is why we must 
specify the output directory when we call `.compute_ld()`. To use `plink` to compute the LD matrix, 
we can invoke a similar command:

```python linenums="1"
# Using plink:
import magenpy as mgp
gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                        backend='plink')
gdl.compute_ld(estimator='windowed',
               output_dir='output/ld/',
               cm_window_size=3.)
gdl.cleanup()
```

In this case, we are computing a windowed LD matrix where we only measure the correlation between 
SNPs that are at most 3 centi Morgan (cM) apart along the chromosome. For this small 1000G dataset, computing 
the LD matrix takes about a minute. The LD matrices in Zarr format will be written to the path 
specified in `output_dir`, so ensure that this argument is set to the desired directory. 

To facilitate working with LD matrices stored in `Zarr` format, we created a data structure in python called `LDMatrix`, 
which acts as an intermediary and provides various features. For example, to compute LD scores 
using this LD matrix, you can invoke the command `.compute_ld_scores()` on it:

```python linenums="1"
gdl.ld[22].compute_ld_scores()
```

```
array([1.60969673, 1.84471792, 1.59205322, ..., 3.3126724 , 3.42234106,
       2.97252452])
```

You can also get a table that lists the properties of the SNPs included in the LD matrix:

```python linenums="1"
gdl.ld[22].to_snp_table()
```

```
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

### LD estimators and their properties

`magenpy` supports computing LD matrices using 4 different estimators that are commonly used 
in statistical genetics applications. 
For a more thorough description of the estimators and their properties, consult our manuscript 
and the citations therein. The LD estimators are:

1) `windowed` (recommended): The windowed estimator computes the pairwise correlation coefficient between SNPs that are 
    within a pre-defined distance along the chromosome from each other. In many statistical genetics applications, the 
   recommended distance is between 1 and 3 centi Morgan (cM). As of `magenpy>=0.0.2`, now you can customize 
   the distance based on three criteria: **(1)** A window size based on the number neighboring variants, **(2)** 
   distance threshold in kilobases (kb), and **(3)** distance threshold in centi Morgan (cM). When defining the 
   boundaries for each SNP, `magenpy` takes the intersection of the boundaries defined by each window.
   
```python linenums="1"
import magenpy as mgp
gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                        backend='plink')
gdl.compute_ld('windowed', 
               output_dir='output/ld/',
               window_size=100, kb_window_size=1000, cm_window_size=2.)
gdl.cleanup()
```

2) `block`: The block estimator estimates the pairwise correlation coefficient between 
variants that are in the same LD block, as defined by, e.g. LDetect. Given an LD block file, 
   we can compute a block-based LD matrix as follows:
   
```python linenums="1"
import magenpy as mgp
ld_block_url = "https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed"
gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                        backend='plink')
gdl.compute_ld('block', 
               output_dir='output/ld/',
               ld_blocks_file=ld_block_url)
gdl.cleanup()
```

If you have the LD blocks file on your system, you can also pass the path to the file instead.

3) `shrinkage`: For the shrinkage estimator, we shrink the entries of the LD matrix by a 
   quantity related to the distance between SNPs along the chromosome + some additional information 
   related to the sample from which the genetic map was estimated. In particular, 
   we need to specify the effective population size and the sample size used to 
   estimate the genetic map. Also, to make the matrix sparse, we often specify a threshold value 
   below which we consider the correlation to be zero. Here's an example for the 1000G sample:
   

```python linenums="1"
import magenpy as mgp
gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                        backend='plink')
gdl.compute_ld('shrinkage', 
               output_dir='output/ld/',
               genetic_map_ne=11400, # effective population size (Ne)
               genetic_map_sample_size=183, # Sample size
               threshold=1e-3) # The cutoff value
gdl.cleanup()
```

4) `sample`: This estimator computes the pairwise correlation coefficient between all SNPs on 
   the same chromosome and thus results in a dense matrix. Thus, it is rarely used in practice and 
   we include it here for testing/debugging purposes mostly. To compute the sample LD matrix, you only need 
   to specify the correct estimator:
   
```python linenums="1"
import magenpy as mgp
gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                        backend='plink')
gdl.compute_ld('sample', output_dir='output/ld/')
gdl.cleanup()
```

## (4) Data harmonization

There are many different statistical genetics data sources and formats out there. One of the goals of 
`magenpy` is to create a friendly interface for matching and merging these data sources for 
downstream analyses. For example, for summary statistics-based methods, we often need 
to merge the LD matrix derived from a reference panel with the GWAS summary statistics estimated 
in a different cohort. While this is a simple task, it can be tricky sometimes, e.g. in 
cases where the effect allele is flipped between the two cohort.

The functionalities that we provide for this are minimal at this stage and mainly geared towards 
harmonizing `Zarr`-formatted LD matrices with GWAS summary statistics. The following example 
shows how to do this in a simple case:

```python linenums="1"
import magenpy as mgp
# First, generate some summary statistics from a simulation:
g_sim = mgp.PhenotypeSimulator(mgp.tgp_eur_data_path())
g_sim.simulate()
g_sim.to_summary_statistics_table().to_csv(
    "chr_22.sumstats", sep="\t", index=False
)
# Then load those summary statistics and match them with previously
# computed windowed LD matrix for chromosome 22:
gdl = mgp.GWADataLoader(ld_store_files='output/windowed_ld/chr_22/',
                        sumstats_files='chr_22.sumstats',
                        sumstats_format='magenpy')
```

Here, the `GWADataLoader` object takes care of the harmonization step by 
automatically invoking the `.harmonize_data()` method. When you read or update 
any of the data sources, we recommend that you invoke the `.harmonize_data()` method again 
to make sure that all the data sources are aligned properly. In the near future, 
we are planning to add many other functionalities in this space. Stay tuned.

## (5) Using `plink` as backend

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

```python linenums="1"
import magenpy as mgp
mgp.print_options()
```

```
-> Section: DEFAULT
---> plink1.9_path: plink
---> plink2_path: plink2
```

The above shows the default configurations for the `plink1.9` and `plink2` paths. To change 
the path for `plink2`, for example, you can use the `set_option()` function:

```python linenums="1"
mgp.set_option("plink2_path", "~/software/plink2/plink2")
mgp.print_options()
```

```
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
derived data structures (including all the `PhenotypeSimulator` classes):

```python linenums="1"
import magenpy as mgp
gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                        backend='plink')
```