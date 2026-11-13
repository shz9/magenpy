#!/bin/bash

if [[ -t 1 ]]; then
  set -e  # Enable exit on error, only in non-interactive sessions
fi


PYTHON_BIN="${PYTHON_BIN:-python3}"

BFILE_PATH=$("$PYTHON_BIN" -c "import magenpy as mgp; print(mgp.tgp_eur_data_path())")


# -------------------------------------------------------------------
# Test the `mgp_compute_ld` cli script:

# Test the commandline argument parsing:
echo "> Testing the mgp_compute_ld script..."
mgp_compute_ld --help

echo "> Estimating LD using the windowed estimator:"
mgp_compute_ld --estimator "windowed" \
               --bfile "$BFILE_PATH" \
               --ld-window-cm 3. \
               --output-dir "output/ld_windowed/" \
               --consolidate-metadata \
               --zip

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

"$PYTHON_BIN" - "output/ld_windowed/chr_22.zip" <<'PY'
import re
import sys
import zipfile

import magenpy as mgp
import numpy as np
import pandas as pd

with zipfile.ZipFile(sys.argv[1]) as archive:
    members = archive.infolist()

if not members:
    raise SystemExit("The LD ZIP archive is empty.")
if any(member.compress_type != zipfile.ZIP_STORED for member in members):
    raise SystemExit("The LD ZIP archive contains compressed members.")
if members[-1].filename != ".zmetadata":
    raise SystemExit("Consolidated metadata is not the final ZIP member.")

normal_names = [member.filename for member in members[:-1]]
version_key = lambda name: tuple(
    (1, int(part)) if part.isdigit() else (0, part)
    for part in re.split(r"(\d+)", name)
)
if normal_names != sorted(normal_names, key=version_key):
    raise SystemExit("Normal ZIP members are not in version-sorted order.")

original = mgp.LDMatrix.from_path("output/ld_windowed/chr_22")
archived = mgp.LDMatrix.from_path(sys.argv[1])
pd.testing.assert_frame_equal(archived.to_snp_table(), original.to_snp_table())
np.testing.assert_array_equal(archived.indptr[:], original.indptr[:])
np.testing.assert_array_equal(archived.data[:], original.data[:])
PY

LD_STORE_PATH="output/ld_windowed/chr_22"
CLI_TEST_DIR="output/cli_ld_tests"
mkdir -p "$CLI_TEST_DIR"

SNP_1=$(awk 'NR == 1 {print $2}' "${BFILE_PATH}.bim")
SNP_2=$(awk 'NR == 2 {print $2}' "${BFILE_PATH}.bim")
printf "%s\n%s\n" "$SNP_1" "$SNP_2" > "$CLI_TEST_DIR/snps.txt"

# -------------------------------------------------------------------
# Test the `mgp_extract_ld` cli script:

# Test the commandline argument parsing:
echo "> Testing the mgp_extract_ld script..."
mgp_extract_ld --help

echo "> Extracting a dense LD submatrix using the mgp_extract_ld script:"
mgp_extract_ld --ld "$LD_STORE_PATH" \
               --snp-file "$CLI_TEST_DIR/snps.txt" \
               --output-file "$CLI_TEST_DIR/extracted_ld.csv"

if [ ! -f "$CLI_TEST_DIR/extracted_ld.csv" ]; then
  echo "Error: The extracted LD matrix was not created."
  exit 1
fi

"$PYTHON_BIN" - "$CLI_TEST_DIR/extracted_ld.csv" <<'PY'
import sys
import pandas as pd

ld = pd.read_csv(sys.argv[1], index_col=0)
if ld.shape != (2, 2):
    raise SystemExit("Extracted LD matrix does not have the expected 2 x 2 shape.")
PY

# -------------------------------------------------------------------
# Test the `mgp_prune_ld` cli script:

# Test the commandline argument parsing:
echo "> Testing the mgp_prune_ld script..."
mgp_prune_ld --help

echo "> Pruning variants using the mgp_prune_ld script:"
mgp_prune_ld --ld "$LD_STORE_PATH" \
             --variants-file "$CLI_TEST_DIR/snps.txt" \
             --r2-threshold 0.1 \
             --output-file "$CLI_TEST_DIR/pruned_snps.tsv"

if [ ! -f "$CLI_TEST_DIR/pruned_snps.tsv" ]; then
  echo "Error: The pruned SNP output file was not created."
  exit 1
fi

if [ "$(tail -n +2 "$CLI_TEST_DIR/pruned_snps.tsv" | wc -l)" -lt 1 ]; then
  echo "Error: The pruned SNP output file is empty."
  exit 1
fi

# -------------------------------------------------------------------
# Test the `mgp_expand_ld` cli script:

# Test the commandline argument parsing:
echo "> Testing the mgp_expand_ld script..."
mgp_expand_ld --help

echo "> Expanding variants using the mgp_expand_ld script:"
mgp_expand_ld --ld "$LD_STORE_PATH" \
              --snp-file "$CLI_TEST_DIR/snps.txt" \
              --r2-threshold 0.1 \
              --output-file "$CLI_TEST_DIR/expanded_snps.tsv"

if [ ! -f "$CLI_TEST_DIR/expanded_snps.tsv" ]; then
  echo "Error: The expanded SNP output file was not created."
  exit 1
fi

if ! grep -q "$SNP_1" "$CLI_TEST_DIR/expanded_snps.tsv"; then
  echo "Error: The expanded SNP output file does not contain the input SNP."
  exit 1
fi

echo "> Expanding variants while retaining LD relationships:"
mgp_expand_ld --ld "$LD_STORE_PATH" \
              --snp-file "$CLI_TEST_DIR/snps.txt" \
              --r2-threshold 0.1 \
              --include-ld \
              --output-file "$CLI_TEST_DIR/expanded_snps_with_ld.tsv"

"$PYTHON_BIN" - "$CLI_TEST_DIR/expanded_snps_with_ld.tsv" <<'PY'
import sys

import pandas as pd

relationships = pd.read_csv(sys.argv[1], sep="\t")
expected_columns = ["Original_SNP", "SNP", "R"]
if relationships.columns.tolist() != expected_columns:
    raise SystemExit("The LD relationship output has unexpected columns.")
if (relationships["Original_SNP"] == relationships["SNP"]).any():
    raise SystemExit("The LD relationship output contains self-LD rows.")
if not relationships.empty:
    if ((relationships["R"] ** 2) < 0.1 - 1e-7).any():
        raise SystemExit("A reported neighbor is below the requested R2 threshold.")
PY

# Clean up after computation:
rm -rf output/ld_windowed
rm -rf temp/

# -------------------------------------------------------------------
# Test the `mgp_simulate` cli script:

# Test the commandline argument parsing:
echo "> Testing the mgp_simulate script..."
mgp_simulate --help

echo "> Simulating genotypes using the mgp_simulate script:"
mgp_simulate --bfile "$BFILE_PATH" \
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
