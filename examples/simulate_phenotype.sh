#!/bin/bash

# This bash script provides examples of how to use the commandline script
# for phenotype simulation that is attached with the `magenpy` package.
# Once you install the `magenpy` package using `pip`, the script `magenpy_simulate`
# will be added to the system paths, so you should be able to access it by just typing
# magenpy_simulate into the shell.

# Before we show some examples, it is a good idea to invoke the script with the `-h` flag
# to see the flags/options/arguments supported and what they mean:

magenpy_simulate -h

# Then, we need to obtain the path to the 1000G data attached with `magenpy`:

TGP_PATH=$(python -c "import magenpy as mgp; print(mgp.tgp_eur_data_path())")

# Example 1: Simulate a heritable quantitative trait (h2 = 0.2) for the 1000G individuals

magenpy_simulate --bfile "$TGP_PATH" \
                 --output-file "output/simulations/example_1" \
                 --h2 0.2

# Example 2: Simulate a heritable case-control trait (h2 = 0.3, prevalence=.2) for the 1000G individuals:

magenpy_simulate --bfile "$TGP_PATH" \
                 --output-file "output/simulations/example_2" \
                 --likelihood "binomial" \
                 --h2 0.3 \
                 --prevalence 0.2

# Example 3: Use plink as a backend for operations on the genotype matrix (recommended):

magenpy_simulate --bfile "$TGP_PATH" \
                 --backend "plink" \
                 --output-file "output/simulations/example_3" \
                 --h2 0.2

# Example 4: Use a mixture of 4 Gaussians for the effect sizes:

magenpy_simulate --bfile "$TGP_PATH" \
                 --backend "plink" \
                 --output-file "output/simulations/example_4" \
                 --h2 0.2 \
                 --mix-prop 0.9,0.03,0.03,0.04 \
                 --var-mult 0.0,0.01,0.1,1.0

# Example 5: Output the simulated phenotype + simulated effect sizes per variant:

magenpy_simulate --bfile "$TGP_PATH" \
                 --output-file "output/simulations/example_5" \
                 --h2 0.2 \
                 -p 0.9,0.1 \
                 -d 0.,1. \
                 --output-simulated-beta
