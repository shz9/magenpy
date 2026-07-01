In addition to the python package interface, users may also opt to use some of `magenpy`'s functionalities 
via commandline scripts. The commandline interface supports simulating complex traits, computing LD matrices,
and extracting, pruning, or expanding variant sets from pre-computed LD matrices.

When you install `magenpy` using `pip`, the commandline scripts are automatically installed on your system and 
are available for use. The available scripts are:

* [`mgp_compute_ld`](mgp_compute_ld.md): Compute LD matrices from genotype data in `plink` BED format.
    The script provides a variety of options for the user to customize the LD computation process, including the 
    choice of LD estimator, storage and compression options, etc.

* [`mgp_simulate`](mgp_simulate.md): Simulate complex traits with a variety of genetic 
    architectures. The script provides a variety of options for the user to customize the simulation process, 
    including the choice of genetic architecture, the proportion of causal variants, the effect sizes, etc.

* [`mgp_extract_ld`](mgp_extract_ld.md): Extract dense LD submatrices for SNP lists or genomic regions.

* [`mgp_prune_ld`](mgp_prune_ld.md): Prune variants by LD threshold using pre-computed LD matrices.

* [`mgp_expand_ld`](mgp_expand_ld.md): Expand focal SNPs to include LD neighbors from pre-computed LD matrices.

If you use `uv`, you can also run the scripts without permanently installing `magenpy` into the active environment:

```bash
uvx --from magenpy mgp_compute_ld -h
uvx --from magenpy mgp_simulate -h
```
