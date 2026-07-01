Extract LD Blocks (`mgp_extract_ld`)
---

`mgp_extract_ld` extracts dense LD submatrices from pre-computed magenpy LD stores.
LD stores can be provided as a single path or a glob.

```bash
mgp_extract_ld --help
```

## Extract SNPs

```bash
mgp_extract_ld \
  --ld output/ld/chr_22/ \
  --snps rs123,rs456,rs789 \
  --output-file output/ld_block.csv
```

To read SNP IDs from a PLINK-style filter file:

```bash
mgp_extract_ld \
  --ld "output/ld/chr_*" \
  --snp-file variants.txt \
  --output-file output/ld_block.npz
```

## Extract Regions

Regions can be specified directly:

```bash
mgp_extract_ld \
  --ld "output/ld/chr_*" \
  --region chr22:20000000-21000000,chr22:22000000-22500000 \
  --output-file output/ld_regions.csv
```

or through a BED-like file with chromosome, start, and end columns:

```bash
mgp_extract_ld \
  --ld "output/ld/chr_*" \
  --bed-file regions.bed \
  --output-file output/ld_regions.npy
```

## Output Formats

The output format can be `csv`, `npy`, or `npz`. If `--output-format auto` is used,
the format is inferred from the output file extension. Matrices are written as
symmetric `float32` arrays.

Use `--log-level` to control command-line logging.
