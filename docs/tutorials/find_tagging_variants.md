Find Tagging Variants With LD
---

This tutorial shows how to expand a focal SNP list to include nearby variants
that tag the focal variants in LD.

The CLI accepts an `r^2` threshold and converts it internally to an absolute
correlation threshold.

## Find Tagging Variants With The CLI

Create a file with focal SNP IDs:

```text
rs9605903
rs5746647
rs16980739
```

Run:

```bash
mgp_expand_ld \
  --ld output/ld_windowed/chr_22/ \
  --snp-file focal_snps.txt \
  --r2-threshold 0.1 \
  --output-file output/tagging_variants.tsv
```

The output includes the focal SNPs and all variants that meet the LD threshold.
The script reports how many focal variants were found in the LD matrix before
expansion.

For multiple chromosomes, pass a glob:

```bash
mgp_expand_ld \
  --ld "output/ld_windowed/chr_*" \
  --snp-file focal_snps.txt \
  --r2-threshold 0.1 \
  --output-file output/tagging_variants.tsv
```

## Find Tagging Variants With Python

Load the LD matrix and convert focal SNP IDs to LD indices:

```python
import numpy as np
import pandas as pd
import magenpy as mgp

ld = mgp.LDMatrix.from_path("output/ld_windowed/chr_22/")

focal = pd.read_csv("focal_snps.txt", header=None)[0].astype(str).tolist()

ld_snps = ld.snps.astype(str)
snp_to_idx = {snp: i for i, snp in enumerate(ld_snps)}

focal_idx = np.array(
    [snp_to_idx[snp] for snp in focal if snp in snp_to_idx],
    dtype=np.int32,
)

print(f"{len(focal_idx)} of {len(focal)} focal variants are present in LD")
```

Find tagging variants at `r^2 >= 0.1`:

```python
corr_threshold = np.sqrt(0.1)

tagged = ld.find_tagging_variants(
    focal_idx,
    threshold=corr_threshold,
    return_value="snps",
)

output = list(dict.fromkeys(focal + tagged.astype(str).tolist()))

pd.DataFrame({"SNP": output}).to_csv(
    "output/tagging_variants_from_python.tsv",
    sep="\t",
    index=False,
)
```

## Inspect The Tagging Set

You can filter the LD matrix to the tagging set and inspect its metadata:

```python
ld.filter_snps(extract_snps=np.array(output))

tag_table = ld.to_snp_table(
    col_subset=["CHR", "SNP", "POS", "A1", "A2", "MAF", "LDScore"]
)

print(tag_table.head())

ld.reset_mask()
```

For large focal lists, run the operation chromosome-by-chromosome or use
`GWADataLoader(ld_store_files="output/ld_windowed/chr_*")` to manage multiple
LD stores.
