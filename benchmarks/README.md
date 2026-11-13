# Continuous native benchmarks

[Continuous benchmark](../.github/workflows/continuous-benchmark.yml) runs on
pushes, pull requests, and manual dispatch. It measures the current revision and
the PR base (or preceding push commit) **on the same Ubuntu runner**, at one and
two native threads. Both extensions are compiled from source with OpenMP using
one shared Python 3.13 environment.
[requirements-benchmark.txt](../requirements-benchmark.txt) includes the runtime
requirements and unpinned build tools; dependencies are installed once for both
builds, and the resolved versions are retained in `dependencies.txt` in the
artifact. The runner verifies
that each worker imports the intended checkout. BLAS/NumExpr threads are fixed
at one so native threading is measured independently.

If the event's baseline commit is missing locally, the workflow tries fetching
it from `origin`. If an automatic baseline is unavailable (for example, after a
force push), it warns and uses `HEAD^`. A checkout without a parent records an
initial run without a regression comparison. An invalid explicitly requested
manual baseline fails rather than silently comparing against another revision.
Early failures before any outputs exist do not cause a second artifact-upload
failure; partial outputs are retained when available.

The suite covers allele frequencies/counts, single-score calculation, standardized
scoring, 16 simultaneous scores, GWAS with missing-call exclusion and mean
imputation, windowed LD, and block LD. Inputs are the bundled 1000 Genomes EUR
chromosome 22 BED/BIM/FAM files. Samples/SNPs are selected before MAC QC; duplicate
variant identities and monomorphic variants are removed before measurement.
No genotype data is downloaded. Windowed LD uses a 250-SNP window; block LD uses
artificial consecutive blocks of approximately 1,000 variants, not biological
recombination blocks. Both store `int16` LD with Zstd level 7, including LD scores.

Each case runs in a fresh process, warms once, and reports the median of five
repetitions. Fast in-memory cases are batched to approximately 150 ms per repeat.
Timing excludes loading/QC, imports, warm-up, external result checks, and temporary
file cleanup. LD includes estimator construction, BED reads, compression, LD
scores, internal validation, and complete Zarr output. This is a warm filesystem
benchmark; OS disk caches are not flushed. Peak RSS is the worker's exact lifetime
high-water mark (including setup and validation), not an allocation delta. PLINK
child-process memory is recorded separately for optional local comparisons.

Runtime increases exceeding **25% and 5 ms per call**, or peak worker RSS increases
exceeding **25% and 16 MiB**, fail the comparison. These floors reduce false alarms
for small workloads on shared runners. Compare only identical data/selection,
configuration, CPU, platform, Python, dependencies, and OpenMP fingerprints. New
GWAS-imputation cases can establish a baseline against older versions without
that option; missing capabilities in the current revision fail. Functional
correctness remains covered by the separate mandatory backend validation suite.

Job summaries and 90-day artifacts retain the comparison, raw timings, metadata,
and CSV history. Successful default-branch pushes also commit
[results.csv](results.csv) and [results.md](results.md), following the VIPRS
history convention. Results have separate history tables for each LD estimator,
scoring mode, GWAS mode, and variant-statistics function. Each row records the
linked commit ID, actual push date when available, benchmark run date, thread
count, runtime, memory, and host. CSV rows retain full commit IDs and timestamps.
Local runs automatically record HEAD and mark uncommitted changes; missing push
dates are left blank rather than inferred from commit author dates.
Fork/PR jobs have read-only permissions and never push or
post comments. Results-only commits are excluded from the benchmark trigger.
The repository must permit the history job to write to the default branch;
protected branch rules still apply.

Run from the repository root in an environment with the native extensions built:

```sh
python benchmarks/benchmark_native.py --threads 1 2 --output /tmp/current.json
python benchmarks/benchmark_report.py --current /tmp/current.json \
    --baseline /tmp/base.json --output-dir /tmp/benchmark-results
```

Omit `--baseline` to record an initial run. `--bfile` accepts another single-
chromosome BED prefix. For a quick smoke run, use `--snps 128 --samples 48
--block-snps 32 --window-size 16 --repeats 1 --min-seconds .01`.
The same harness can be run against a separately installed baseline using
`--source-root /path/to/baseline`; it must not import the current checkout.

Compare block implementations locally (PLINK 1.9 must be on `PATH`):

```sh
python tools/install_plink.py --destination .venv/plink-bin
PATH="$PWD/.venv/plink-bin:$PATH" python benchmarks/benchmark_native.py \
    --threads 1 --cases ld_block ld_block_fallback ld_block_plink \
    --output /tmp/block-comparison.json
```

The fallback is the current native per-block implementation, not a timing of a
historical release. PLINK uses pairwise deletion for missing calls; native LD
uses mean imputation. On Apple Silicon the pinned PLINK 1.9 executable also runs
through Rosetta, so its speed ratio is specific to that environment.
The existing [LD product microbenchmark](benchmark_ld_products.py) remains
available for compressed matrix/vector operations.
