#!/bin/bash

# This bash script provides examples of how to use the commandline script
# for computing LD matrices that is attached with the `magenpy` package.
# Once you install the `magenpy` package using `pip`, the script `magenpy_ld`
# will be added to the system paths, so you should be able to access it by just typing
# magenpy_ld into the shell.

# Before we show some examples, it is a good idea to invoke the script with the `-h` flag
# to see the flags/options/arguments supported and what they mean:

magenpy_ld -h

# Then, we need to obtain the path to the 1000G data attached with `magenpy`:

TGP_PATH=$(python -c "import magenpy as mgp; print(mgp.tgp_eur_data_path())")

# Example 1: Compute the windowed LD matrix from the 1000G dataset:

magenpy_ld --bfile "$TGP_PATH" \
           --estimator "windowed" \
           --output-dir "output/ld/example_1/"

# Example 2: Use plink as a backend for LD calculation (recommended):

magenpy_ld --bfile "$TGP_PATH" \
           --backend "plink" \
           --estimator "windowed" \
           --output-dir "output/ld/example_2/"

# Example 3: Compute LD within LD blocks:

LD_URL="https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed"

magenpy_ld --bfile "$TGP_PATH" \
           --backend "plink" \
           --estimator "block" \
           --ld-blocks "$LD_URL" \
           --output-dir "output/ld/example_3/"

