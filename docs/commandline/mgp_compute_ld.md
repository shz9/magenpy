Compute LD Matrices (`mgp_compute_ld`)
---

`mgp_compute_ld` computes linkage-disequilibrium (LD) matrices from genotype data
stored in PLINK BED format and writes magenpy Zarr LD stores.

```bash
mgp_compute_ld --help
```

## Basic Usage

```bash
mgp_compute_ld \
  --bfile data/chr_22 \
  --estimator windowed \
  --ld-window-cm 3.0 \
  --output-dir output/ld_windowed/
```

The `--bfile` argument may point to a BED prefix or BED path. The command supports
the `windowed`, `block`, `shrinkage`, and `sample` LD estimators. Estimator-specific
arguments include:

* `--ld-window`, `--ld-window-kb`, `--ld-window-cm` for the windowed estimator.
* `--ld-blocks` for the block estimator.
* `--genmap-Ne`, `--genmap-sample-size`, and `--shrinkage-cutoff` for the shrinkage estimator.

## Filtering

```bash
mgp_compute_ld \
  --bfile data/chr_22 \
  --estimator windowed \
  --ld-window-kb 1000 \
  --extract variants.txt \
  --keep samples.txt \
  --min-maf 0.01 \
  --output-dir output/ld/
```

`--extract` and `--keep` use PLINK-style filter files.

## Storage And Logging

Use `--storage-dtype`, `--compressor`, and `--compression-level` to control LD storage.
Pass `--consolidate-metadata` to write a root `.zmetadata` file in each chromosome
store. Pass `--zip` to consolidate the metadata and additionally create a
zero-compression `chr_N.zip` archive alongside each `chr_N` store:

```bash
mgp_compute_ld \
  --bfile data/chr_22 \
  --ld-window-cm 3 \
  --output-dir output/ld \
  --consolidate-metadata \
  --zip
```

ZIP members are relative to the chromosome store: `.zgroup`, `.zattrs`,
`matrix/`, and `metadata/` are at the archive root, without a `chr_N/` wrapper.
This layout can be read directly with `LDMatrix.from_path("output/ld/chr_22.zip")`
or Zarr's `ZipStore`, and is also supported by magenpy's HTTP and Hugging Face
ZIP readers. Normal files are added in version-sorted order, and `.zmetadata`
is appended last. Zarr chunks keep their original compression; the ZIP layer
adds no further compression.

Package `INFO` messages are hidden by default so they do not duplicate progress-bar
updates. Use `--log-level` to change command-line logging:

```bash
mgp_compute_ld --bfile data/chr_22 --ld-window-cm 3 --output-dir output/ld --log-level INFO
```
