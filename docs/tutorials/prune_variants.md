Prune Variants With LD
---

LD pruning removes variants that are highly correlated with variants already
kept earlier in a priority order.

`magenpy` pruning thresholds are based on the absolute Pearson correlation
coefficient. The CLI accepts an `r^2` threshold and converts it internally to
`sqrt(r^2)`.

## Prune A Variant File With The CLI

Start with a PLINK-style file containing SNP IDs:

```text
rs9605903
rs5746647
rs16980739
```

Run pruning:

```bash
mgp_prune_ld \
  --ld output/ld_windowed/chr_22/ \
  --variants-file variants.txt \
  --r2-threshold 0.1 \
  --output-file output/pruned_variants.tsv
```

The script reports how many input variants were present in the LD matrix before
pruning. This is useful when the variant list comes from a different reference
panel or genome build.

## Use A Rank Column

When two variants are in high LD, the earlier/higher-priority variant is kept.
If your file has SNP IDs and a rank column, pass both column indices:

```bash
mgp_prune_ld \
  --ld "output/ld_windowed/chr_*" \
  --variants-file ranked_variants.tsv \
  --snp-column 0 \
  --rank-column 1 \
  --r2-threshold 0.1 \
  --output-file output/pruned_ranked_variants.tsv
```

By default, smaller rank values are higher priority. This matches p-values. Use
`--rank-descending` when larger values should be prioritized.

## Prune Summary Statistics With The CLI

Summary statistics can be read and harmonized against LD through `GWADataLoader`:

```bash
mgp_prune_ld \
  --ld "output/ld_windowed/chr_*" \
  --sumstats-file gwas.tsv \
  --sumstats-format magenpy \
  --rank-column PVAL \
  --r2-threshold 0.1 \
  --output-file output/pruned_sumstats_variants.tsv
```

If `--rank-column` is omitted, the script tries common p-value column names.

## Prune With Python

Load the LD matrix and map requested SNPs to LD indices:

```python
import numpy as np
import pandas as pd
import magenpy as mgp

ld = mgp.LDMatrix.from_path("output/ld_windowed/chr_22/")

variants = pd.read_csv("variants.txt", header=None, names=["SNP"])
requested = variants["SNP"].astype(str).tolist()

ld_snps = ld.snps.astype(str).tolist()
snp_to_idx = {snp: i for i, snp in enumerate(ld_snps)}

matched = [snp for snp in requested if snp in snp_to_idx]
variant_order = np.array([snp_to_idx[snp] for snp in matched], dtype=np.int32)

print(f"{len(matched)} of {len(requested)} variants are present in LD")
```

Prune at `r^2 <= 0.1`:

```python
corr_threshold = np.sqrt(0.1)

kept = ld.prune(
    threshold=corr_threshold,
    variant_order=variant_order,
    return_value="snps",
)

pd.DataFrame({"SNP": kept}).to_csv(
    "output/pruned_variants_from_python.tsv",
    sep="\t",
    index=False,
)
```

If you do not provide `variant_order`, `LDMatrix.prune()` uses the stored LD
matrix order.
