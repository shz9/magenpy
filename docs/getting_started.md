`magenpy` is a `python` package that aims to streamline working with statistical genetics data 
in order to facilitate downstream analyses. The package comes with a sample dataset from the 1000G project that 
you can use to experiment and familiarize yourself with its features. 
Once the package is installed, you can run a couple of quick tests 
to verify that the main features are working properly.

For example, to simulate a quantitative trait, you can invoke 
the following commands in a `python` interpreter:

```python linenums="1"
import magenpy as mgp
g_sim = mgp.PhenotypeSimulator(mgp.tgp_eur_data_path(),  # Provide path to 1000G data
                               h2=0.1) # Heritability set to 0.1
g_sim.simulate() # Simulate the phenotype
g_sim.to_phenotype_table()

```

```
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

```python linenums="1"
g_sim.get_causal_status()
```

```
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

```python linenums="1"
g_sim.to_true_beta_table()
```

```
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
4-component sparse Gaussian mixture density, instead of the standard spike-and-slab density used by default:

```python linenums="1"
g_sim = mgp.PhenotypeSimulator(mgp.tgp_eur_data_path(),
                               pi=[.9, .03, .03, .04],  # Mixing proportions
                               d=[0., .01, .1, 1.],  # Variance multipliers
                               h2=0.1)
g_sim.simulate()
g_sim.to_phenotype_table()
```

```
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
