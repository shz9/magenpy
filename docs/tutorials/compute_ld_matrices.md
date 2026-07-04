Compute LD Matrices From BED Files
---

This tutorial shows how to compute `magenpy` LD matrices from PLINK BED files
and how to inspect the resulting stores with the Python API.

The examples assume a PLINK BED prefix:

```text
data/chr_22.bed
data/chr_22.bim
data/chr_22.fam
```

Use the prefix without the `.bed` extension when passing `--bfile`.

## Compute LD With The CLI

For most applications, a windowed LD matrix is the best starting point. This
stores correlations only for nearby variants.

```bash
mgp_compute_ld \
  --bfile data/chr_22 \
  --backend magenpy \
  --estimator windowed \
  --ld-window-cm 3.0 \
  --storage-dtype int16 \
  --output-dir output/ld_windowed/
```

The command writes one LD store per chromosome. For chromosome 22, the store is:

```text
output/ld_windowed/chr_22/
```

You can filter samples or variants before computing LD:

```bash
mgp_compute_ld \
  --bfile data/chr_22 \
  --estimator windowed \
  --ld-window-kb 1000 \
  --extract variants.txt \
  --keep samples.txt \
  --min-maf 0.01 \
  --output-dir output/ld_filtered/
```

Other supported estimators are `block`, `shrinkage`, and `sample`. The `sample`
estimator computes all within-chromosome pairwise correlations and can be very
large. Use with caution.

## Compute LD through the Python API

The same operation can be done through `GWADataLoader`:

```python
import magenpy as mgp

gdl = mgp.GWADataLoader(
    bed_files="data/chr_22",
    backend="magenpy",
)

gdl.compute_ld(
    estimator="windowed",
    output_dir="output/ld_windowed/",
    cm_window_size=3.0,
    dtype="int16",
)

ld = gdl.ld[22]
```

## Load And Inspect An LDMatrix

You can load a saved matrix directly:

```python
import magenpy as mgp

ld = mgp.LDMatrix.from_path("output/ld_windowed/chr_22/")

print(ld.chromosome)
print(ld.ld_estimator)
print(ld.sample_size)
print(ld.shape)
print(ld.stored_dtype)
```

Inspect variant metadata:

```python
snps = ld.get_metadata("snps")
bp = ld.get_metadata("bp")
maf = ld.get_metadata("maf")

print(snps[:5])
print(bp[:5])
print(maf[:5])
```

Or build a tidy SNP table:

```python
snp_table = ld.to_snp_table(
    col_subset=["CHR", "SNP", "POS", "A1", "A2", "MAF", "LDScore"]
)
print(snp_table.head())
```

## LD Scores

LD scores are the sum of squared correlations for each variant. If LD scores are
not already stored, `ld.ld_score` computes and caches them when possible.

```python
ld_scores = ld.ld_score
print(ld_scores[:5])
```

You can also compute them explicitly:

```python
ld_scores = ld.compute_ld_scores(chunk_size=5000)
```

## Pairwise LD Entries

Use integer indices:

```python
r = ld[0, 1]
print(r)
```

Or SNP IDs:

```python
snp_a = ld.snps[0]
snp_b = ld.snps[1]

r = ld[snp_a, snp_b]
print(r)
```

The returned value is Pearson correlation between `snp_a` and `snp_b`.

## Extract Dense Or Sparse Blocks

For a small block, load rows as a SciPy CSR matrix and convert to dense:

```python
block = ld.load_data(
    start_row=0,
    end_row=100,
    return_symmetric=True,
    return_as_csr=True,
    dtype="float32",
)

dense_block = block.toarray()
print(dense_block.shape)
```

To work with selected SNPs by ID, use a mask:

```python
keep_snps = ld.snps[:100]
ld.filter_snps(extract_snps=keep_snps)

submatrix = ld.to_csr(return_symmetric=True, dtype="float32")
print(submatrix.shape)

ld.reset_mask()
```

## LD Matrix Multiplication

`LDMatrix` can multiply by a vector without forcing you to manually build a
dense matrix:

```python
import numpy as np

weights = np.ones(ld.n_snps, dtype=np.float32)
out = ld.dot(weights)
```

For repeated linear algebra operations, use `LDLinearOperator`:

```python
ld_op = ld.to_linear_operator(return_symmetric=True, dtype="float32")

vector_result = ld_op @ weights

W = np.column_stack([weights, weights])
matrix_result = ld_op.matmat(W)

print(vector_result.shape)
print(matrix_result.shape)
```

Release cached LD data when you are done:

```python
ld.release()
gdl.cleanup()
```
