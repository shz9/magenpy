Prune Variants By LD (`mgp_prune_ld`)
---

`mgp_prune_ld` prunes variants using pre-computed magenpy LD matrices. The pruning
threshold is provided as an R-squared threshold with `--r2-threshold`.

```bash
mgp_prune_ld --help
```

## Plain Variant File

```bash
mgp_prune_ld \
  --ld "output/ld/chr_*" \
  --variants-file variants.txt \
  --r2-threshold 0.1 \
  --output-file output/pruned_variants.tsv
```

Plain variant files use PLINK-style filter-file parsing. Use `--snp-column` to select
a zero-based column index.

## Ranked Variant File

If the variant file has a rank column, provide its zero-based index:

```bash
mgp_prune_ld \
  --ld "output/ld/chr_*" \
  --variants-file ranked_variants.tsv \
  --snp-column 0 \
  --rank-column 1 \
  --r2-threshold 0.1 \
  --output-file output/pruned_variants.tsv
```

By default, smaller rank values are treated as higher priority. Use
`--rank-descending` when larger values should be retained first.

## Summary Statistics

Summary statistics are read and harmonized with LD through `GWADataLoader`:

```bash
mgp_prune_ld \
  --ld "output/ld/chr_*" \
  --sumstats-file gwas.tsv \
  --sumstats-format magenpy \
  --r2-threshold 0.1 \
  --output-file output/pruned_sumstats_variants.tsv
```

If no `--rank-column` is provided for summary statistics, the script uses a p-value
column when available.

Use `--log-level` to control command-line logging.
