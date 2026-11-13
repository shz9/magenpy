# Magenpy continuous benchmark results

Latest revision: [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes. Push date (UTC): not recorded (local/PR/manual run). Benchmark run (UTC): 2026-11-13T18:50:10.205726+00:00.

Latest host: Apple M5 Pro (Darwin/arm64); Python 3.11.15; OpenMP: False.
Panel: 378 samples, 15,938 variants, 16 artificial blocks. Configuration: `{"block_snps": 1000, "dtype": "int16", "min_seconds": 0.15, "repeats": 5, "samples": 0, "snps": 0, "window_size": 250}`.

Each function has its own history table, newest runs first. Commit links identify the measured source; '+ local changes' marks an uncommitted checkout. Push dates come from push-event metadata; a dash means no push date was recorded. Run dates are recorded separately for local, PR, and manual benchmarks.

Times are medians after one warm-up in fresh worker processes. LD includes complete Zarr output; temporary-file cleanup is excluded. RSS is the worker's lifetime high-water mark including setup and validation, excluding PLINK children.

## LD calculation

### Windowed LD

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 776.407 | 223.4 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Block LD (native single pass)

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 1366.951 | 310.3 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Block LD (native per-block fallback)

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 5275.641 | 163.4 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Block LD (PLINK 1.9)

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 15340.291 | 327.0 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## Scoring

### Single score

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 33.127 | 115.4 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Standardized score

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 21.954 | 115.3 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### 16-score batch

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 46.743 | 115.1 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## GWAS

### Exclude missing calls per SNP

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 21.477 | 165.1 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Mean imputation

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 21.882 | 165.0 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## Variant statistics

### Allele frequencies and sample counts

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 18.681 | 115.2 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## Latest regression check

No detected regressions in comparable cases. New cases establish a baseline.

[results.csv](results.csv) retains full commit IDs, push/run timestamps, function/category labels, comparison ratios, and workload fingerprints. Raw run artifacts also retain individual timing samples and dependency versions. Compare matching workload fingerprints; ratios above 1 indicate increased time or memory.
