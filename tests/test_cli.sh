#!/bin/bash

if [[ -t 1 ]]; then
  set -e  # Enable exit on error, only in non-interactive sessions
fi


BFILE_PATH=$(python3 -c "import magenpy as mgp; print(mgp.tgp_eur_data_path())")


# -------------------------------------------------------------------
# Test the `magenpy_ld` cli script:

# Test the commandline argument parsing:
echo "> Testing the magenpy_ld script..."
magenpy_ld --help

echo "> Estimating LD using the windowed estimator:"
magenpy_ld --estimator "windowed" \
           --bfile "$BFILE_PATH" \
           --ld-window-cm 3. \
           --output-dir "output/ld_windowed/"

# Check that there's a directory called "output/ld_windowed/chr_22/":
if [ ! -d "output/ld_windowed/chr_22/" ]; then
  echo "Error: The output directory was not created."
  exit 1
fi

# Check that the directory contains both `.zgroup` and `.zatrs` files:
if [ ! -f "output/ld_windowed/chr_22/.zgroup" ] || [ ! -f "output/ld_windowed/chr_22/.zattrs" ]; then
  echo "Error: The output directory does not contain the expected files."
  exit 1
fi

# Clean up after computation:
rm -rf output/ld_windowed
rm -rf temp/

# -------------------------------------------------------------------
# Test the `magenpy_simulate` cli script:

# Test the commandline argument parsing:
echo "> Testing the magenpy_simulate script..."
magenpy_simulate --help

echo "> Simulating genotypes using the magenpy_simulate script:"
magenpy_simulate --bfile "$BFILE_PATH" \
                 --h2 0.5 \
                 --output-file "output/pheno_1" \
                 --output-simulated-beta

# Check that the output file exists:
if [ ! -f "output/pheno_1.SimPheno" ]; then
  echo "Error: The output file was not created."
  exit 1
fi

# Check that the true betas file exists:
if [ ! -f "output/pheno_1.SimEffect" ]; then
  echo "Error: The true betas file was not created."
  exit 1
fi

# Clean up after computation:
rm -rf output/
rm -rf temp/
