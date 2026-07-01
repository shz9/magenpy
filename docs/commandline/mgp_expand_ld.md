Expand Variants By LD (`mgp_expand_ld`)
---

`mgp_expand_ld` expands a focal SNP list to include variants in LD with those SNPs
using pre-computed magenpy LD matrices. The LD threshold is provided as an R-squared
threshold with `--r2-threshold`.

```bash
mgp_expand_ld --help
```

## Basic Usage

```bash
mgp_expand_ld \
  --ld "output/ld/chr_*" \
  --snp-file focal_snps.txt \
  --r2-threshold 0.1 \
  --output-file output/expanded_variants.tsv
```

The SNP file is read as a PLINK-style filter file. Use `--snp-column` to select a
zero-based SNP column index. The output includes the focal SNPs and all LD neighbors
that meet the threshold.

Use `--log-level` to control command-line logging.
