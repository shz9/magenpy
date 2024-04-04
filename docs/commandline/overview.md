In addition to the python package interface, users may also opt to use some of `magenpy`'s functionalities 
via commandline scripts. The commandline interface is limited at this point to mainly simulating complex traits 
and computing LD matrices.

When you install `magenpy` using `pip`, the commandline scripts are automatically installed on your system and 
are available for use. The available scripts are:

* [`magenpy_ld`](magenpy_ld.md): This script is used to compute LD matrices from genotype data in `plink` BED format. 
    The script provides a variety of options for the user to customize the LD computation process, including the 
    choice of LD estimator, storage and compression options, etc.

* [`magenpy_simulate`](magenpy_simulate.md): This script is used to simulate complex traits with a variety of genetic 
    architectures. The script provides a variety of options for the user to customize the simulation process, 
    including the choice of genetic architecture, the proportion of causal variants, the effect sizes, etc.