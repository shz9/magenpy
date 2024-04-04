#!/bin/bash

# A script to test the package with different Python versions manually using conda
# May be useful for sanity checks before pushing changes to the repository.

# Usage:
# $ source tests/conda_manual_testing.sh

# ==============================================================================
# Define Python versions (add more here if needed)
python_versions=("3.8" "3.9" "3.10" "3.11" "3.12")

# ==============================================================================

# Loop over Python versions
for version in "${python_versions[@]}"
do
    # Create a new conda environment for the Python version
    conda create --name "magenpy$version" python="$version" -y

    # Activate the conda environment
    conda activate "magenpy$version"

    # Add some of the required dependencies:
    conda install -c conda-forge -c anaconda pip wheel compilers -y

    # Check python version:
    python --version

    # Install magenpy
    make clean
    python -m pip install -v -e .[test]

    # List the installed packages:
    python -m pip list

    # Run pytest
    python -m pytest -v

    magenpy_ld -h
    magenpy_simulate -h

    # Deactivate the conda environment
    conda deactivate

    # Remove the conda environment
    conda env remove --name "magenpy$version" -y
done
