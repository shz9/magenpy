Simulate Complex Traits (`mgp_simulate`)
---

`mgp_simulate` simulates quantitative or case-control phenotypes from PLINK BED
genotype data using magenpy's phenotype simulation machinery.

```bash
mgp_simulate --help
```

## Basic Usage

```bash
mgp_simulate \
  --bfile data/chr_22 \
  --h2 0.5 \
  --prop-causal 0.1 \
  --output-file output/pheno_1
```

This writes `output/pheno_1.SimPheno`. Add `--output-simulated-beta` to also write
`output/pheno_1.SimEffect`.

## Genetic Architecture

For a simple spike-and-slab architecture, use `--prop-causal`:

```bash
mgp_simulate --bfile data/chr_22 --h2 0.5 --prop-causal 0.05 --output-file output/pheno
```

For custom mixture proportions and variance multipliers, use `--mix-prop` with
`--var-mult`:

```bash
mgp_simulate \
  --bfile data/chr_22 \
  --h2 0.5 \
  --mix-prop 0.9,0.1 \
  --var-mult 0,1 \
  --output-file output/pheno
```

## Case-Control Traits

```bash
mgp_simulate \
  --bfile data/chr_22 \
  --h2 0.3 \
  --phenotype-likelihood binomial \
  --prevalence 0.08 \
  --output-file output/case_control
```

Use `--log-level` to control command-line logging.
