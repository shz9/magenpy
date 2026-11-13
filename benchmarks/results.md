# Magenpy continuous benchmark results

Latest revision: [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa). Push date (UTC): 2026-11-13T19:30:33+00:00. Benchmark run (UTC): 2026-11-13T19:32:55.605460+00:00.

Latest host: AMD EPYC 9V74 80-Core Processor (Linux/x86_64); Python 3.12.16; OpenMP: True.
Panel: 378 samples, 15,938 variants, 16 artificial blocks. Configuration: `{"block_snps": 1000, "dtype": "int16", "min_seconds": 0.15, "repeats": 5, "samples": 0, "snps": 0, "window_size": 250}`.

Each function has its own history table, newest runs first. Commit links identify the measured source; '+ local changes' marks an uncommitted checkout. Push dates come from push-event metadata; a dash means no push date was recorded. Run dates are recorded separately for local, PR, and manual benchmarks.

Times are medians after one warm-up in fresh worker processes. LD includes complete Zarr output; temporary-file cleanup is excluded. RSS is the worker's lifetime high-water mark including setup and validation, excluding PLINK children.

## LD calculation

### Windowed LD

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 1712.890 | 188.8 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 1022.735 | 187.7 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 1135.319 | 188.1 | 0.98x | 1.01x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 744.929 | 187.3 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 776.407 | 223.4 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Block LD (native single pass)

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 3364.472 | 255.0 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 1994.034 | 254.5 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 2222.969 | 254.1 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 1419.206 | 254.1 | 0.98x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
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
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 76.771 | 125.4 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 64.678 | 125.5 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 55.860 | 124.6 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 43.449 | 124.7 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 33.127 | 115.4 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Standardized score

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 64.302 | 125.4 | 0.98x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 58.733 | 125.4 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 41.483 | 124.7 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 35.570 | 124.6 | 0.98x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 21.954 | 115.3 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### 16-score batch

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 98.466 | 125.3 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 77.317 | 125.6 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 77.917 | 124.8 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 54.665 | 124.5 | 1.01x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 46.743 | 115.1 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## GWAS

### Exclude missing calls per SNP

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 61.939 | 183.5 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 59.829 | 183.6 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 38.884 | 182.9 | 0.96x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 36.498 | 182.6 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 21.477 | 165.1 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Mean imputation

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 62.384 | 183.7 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 61.249 | 183.3 | 1.02x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 38.533 | 183.1 | new | — | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 36.946 | 182.7 | new | — | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 21.882 | 165.0 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## Variant statistics

### Allele frequencies and sample counts

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 52.801 | 126.8 | 1.01x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 51.842 | 125.2 | 0.98x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 29.805 | 125.9 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 30.309 | 124.3 | 1.01x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 18.681 | 115.2 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## Latest regression check

No detected regressions in comparable cases. New cases establish a baseline.

[results.csv](results.csv) retains full commit IDs, push/run timestamps, function/category labels, comparison ratios, and workload fingerprints. Raw run artifacts also retain individual timing samples and dependency versions. Compare matching workload fingerprints; ratios above 1 indicate increased time or memory.
