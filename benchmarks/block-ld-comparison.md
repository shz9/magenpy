# Block LD comparison

Measured on the bundled 1000 Genomes EUR chromosome 22 panel using the local working tree.

Commit ID: [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes.
Push date: not recorded (local run).

Run date (UTC): 2026-11-13T18:50:10.205726+00:00. Host: Apple M5 Pro / macOS ARM64; native OpenMP disabled.

378 samples; 15,938 variants; 16 artificial consecutive blocks of about 1,000 variants.
Five timed repetitions after one warm-up per implementation; one requested thread;
`int16` output, Zstd level 7. Timing includes estimator construction, BED reads,
LD scores, compression, and complete Zarr output. Input setup and temporary
cleanup are excluded; filesystem caches are warm.

| Implementation | Median wall time (s) | Range (s) | Relative to native direct | Worker peak RSS (MiB) |
|---|---:|---:|---:|---:|
| Native single pass | 1.367 | 1.354–1.433 | 1.00x | 310.3 |
| Native per-block fallback | 5.276 | 5.080–5.542 | 3.86x | 163.4 |
| PLINK 1.9 per block | 15.340 | 14.616–15.806 | 11.22x | 327.0 |

The native direct and fallback outputs matched exactly across all 7,931,953
stored pairs (`int16` data and sparse indices). The direct path trades higher
peak memory for lower elapsed time.
RSS is the Python worker lifetime peak including setup/validation; PLINK child
memory is excluded from that column. This compares the current direct and
fallback implementations, not two historical releases.

PLINK 1.9 is the pinned x86-64 macOS build running through Rosetta on this ARM64
host. PLINK also uses pairwise deletion for missing calls, while native LD uses
mean imputation. These are environment-specific timings and do not establish
architecture-independent speedups or numerical equivalence between those methods.

Reproduction commands and methodology are in [README.md](README.md).
The initial [results.csv](results.csv) and [results.md](results.md) record all eight
native workloads plus these optional comparators. CI will add Linux/OpenMP runs.
