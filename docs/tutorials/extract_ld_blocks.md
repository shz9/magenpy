Extract LD Sub-Blocks
---

This tutorial shows how to extract dense LD submatrices from pre-computed
`magenpy` LD stores.

Assume you already computed an LD matrix:

```text
output/ld_windowed/chr_22/
```

## Extract By SNP IDs With The CLI

For a short list, pass comma-separated SNP IDs:

```bash
mgp_extract_ld \
  --ld output/ld_windowed/chr_22/ \
  --snps rs9605903,rs5746647,rs16980739 \
  --output-file output/ld_extract/snps.csv
```

For a file of SNP IDs:

```bash
mgp_extract_ld \
  --ld output/ld_windowed/chr_22/ \
  --snp-file variants.txt \
  --output-file output/ld_extract/snps.npz
```

The script logs how many requested variants matched the LD matrix before it
writes the submatrix.

## Extract By Genomic Region With The CLI

One region:

```bash
mgp_extract_ld \
  --ld output/ld_windowed/chr_22/ \
  --region chr22:20000000-20500000 \
  --output-file output/ld_extract/region.csv
```

Multiple regions:

```bash
mgp_extract_ld \
  --ld "output/ld_windowed/chr_*" \
  --region chr22:20000000-20500000,chr22:21000000-21250000 \
  --output-file output/ld_extract/regions.csv
```

You can also use a BED-like file with chromosome, start, and end columns:

```bash
mgp_extract_ld \
  --ld "output/ld_windowed/chr_*" \
  --bed-file regions.bed \
  --output-file output/ld_extract/regions.npy
```

Supported output formats are `csv`, `npy`, and `npz`. CSV output includes SNP IDs
as row and column labels. NPZ output stores both the matrix and SNP IDs.

## Extract With Python

Load the matrix and filter to the SNPs you want:

```python
import numpy as np
import pandas as pd
import magenpy as mgp

ld = mgp.LDMatrix.from_path("output/ld_windowed/chr_22/")

keep_snps = ["rs9605903", "rs5746647", "rs16980739"]
ld.filter_snps(extract_snps=np.array(keep_snps))

submatrix = ld.to_csr(return_symmetric=True, dtype="float32").toarray()

out = pd.DataFrame(submatrix, index=ld.snps, columns=ld.snps)
out.to_csv("output/ld_extract/snps_from_python.csv")

ld.reset_mask()
```

## Extract A Region With Python

Use the variant metadata to select SNPs by base-pair position:

```python
import magenpy as mgp

ld = mgp.LDMatrix.from_path("output/ld_windowed/chr_22/")

bp = ld.get_metadata("bp", apply_mask=False)
snps = ld.get_metadata("snps", apply_mask=False)

region_mask = (bp >= 20_000_000) & (bp <= 20_500_000)
region_snps = snps[region_mask]

print(f"{len(region_snps)} variants overlap the region")

ld.filter_snps(extract_snps=region_snps)
region_ld = ld.to_csr(return_symmetric=True, dtype="float32")

ld.reset_mask()
```

For very large regions, prefer keeping the result sparse or writing NPZ output
rather than materializing a dense CSV.
