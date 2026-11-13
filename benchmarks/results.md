# Magenpy continuous benchmark results

Latest revision: [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519). Push date (UTC): 2026-11-13T21:40:06+00:00. Benchmark run (UTC): 2026-11-13T21:42:37.295995+00:00.

Latest host: Intel(R) Xeon(R) 6975P-C (Linux/x86_64); Python 3.13.18; OpenMP: True.
Panel: 378 samples, 15,938 variants, 16 artificial blocks. Configuration: `{"block_snps": 1000, "dtype": "int16", "min_seconds": 0.15, "repeats": 5, "samples": 0, "snps": 0, "window_size": 250}`.

Each function has its own history table, newest runs first. Commit links identify the measured source; '+ local changes' marks an uncommitted checkout. Push dates come from push-event metadata; a dash means no push date was recorded. Run dates are recorded separately for local, PR, and manual benchmarks.

Times are medians after one warm-up in fresh worker processes. LD includes complete Zarr output; temporary-file cleanup is excluded. RSS is the worker's lifetime high-water mark including setup and validation, excluding PLINK children.

## LD calculation

### Windowed LD

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 1 | 930.322 | 171.7 | 1.00x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 2 | 595.629 | 171.3 | 0.97x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 1 | 1138.407 | 171.8 | 0.98x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 2 | 742.034 | 171.3 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 1 | 1373.448 | 172.6 | 1.03x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 2 | 799.798 | 171.7 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 1 | 1586.676 | 188.6 | 1.00x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 2 | 943.965 | 188.2 | 1.00x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 1 | 822.313 | 187.6 | 0.96x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 2 | 512.878 | 187.5 | 0.97x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 1712.890 | 188.8 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 1022.735 | 187.7 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 1135.319 | 188.1 | 0.98x | 1.01x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 744.929 | 187.3 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 776.407 | 223.4 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Block LD (native single pass)

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 1 | 1816.285 | 234.3 | 0.99x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 2 | 1165.172 | 234.1 | 1.00x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 1 | 2225.302 | 234.4 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 2 | 1458.658 | 234.2 | 1.01x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 1 | 2793.812 | 234.9 | 1.05x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 2 | 1551.882 | 235.1 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 1 | 3119.417 | 254.8 | 1.01x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 2 | 1866.947 | 254.6 | 0.98x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 1 | 1615.743 | 254.2 | 1.00x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 2 | 969.754 | 254.1 | 0.99x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
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
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 1 | 43.900 | 122.2 | 1.01x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 2 | 31.285 | 122.3 | 0.96x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 1 | 57.094 | 121.9 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 2 | 44.619 | 122.0 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 1 | 60.735 | 123.7 | 1.01x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 2 | 51.763 | 123.9 | 1.03x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 1 | 66.205 | 125.4 | 1.00x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 2 | 55.482 | 125.4 | 1.02x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 1 | 46.577 | 124.6 | 0.98x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 2 | 40.772 | 124.6 | 0.98x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 76.771 | 125.4 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 64.678 | 125.5 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 55.860 | 124.6 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 43.449 | 124.7 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 33.127 | 115.4 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Standardized score

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 1 | 29.795 | 121.9 | 0.96x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 2 | 24.796 | 122.0 | 0.97x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 1 | 42.132 | 122.2 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 2 | 36.294 | 122.0 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 1 | 51.635 | 123.7 | 1.03x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 2 | 46.640 | 123.8 | 1.02x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 1 | 57.322 | 125.4 | 1.01x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 2 | 50.250 | 125.4 | 1.01x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 1 | 36.230 | 124.7 | 0.98x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 2 | 34.791 | 124.5 | 1.01x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 64.302 | 125.4 | 0.98x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 58.733 | 125.4 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 41.483 | 124.7 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 35.570 | 124.6 | 0.98x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 21.954 | 115.3 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### 16-score batch

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 1 | 62.692 | 121.9 | 1.00x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 2 | 41.448 | 122.1 | 1.00x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 1 | 78.539 | 122.1 | 0.98x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 2 | 54.465 | 122.1 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 1 | 77.385 | 123.7 | 1.01x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 2 | 60.606 | 123.7 | 1.02x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 1 | 85.773 | 125.3 | 1.00x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 2 | 65.540 | 125.5 | 1.02x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 1 | 57.936 | 124.5 | 0.97x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 2 | 46.564 | 124.7 | 1.01x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 98.466 | 125.3 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 77.317 | 125.6 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 77.917 | 124.8 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 54.665 | 124.5 | 1.01x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 46.743 | 115.1 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## GWAS

### Exclude missing calls per SNP

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 1 | 28.876 | 182.1 | 1.02x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 2 | 25.582 | 181.8 | 0.97x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 1 | 38.482 | 181.9 | 0.98x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 2 | 36.605 | 181.5 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 1 | 49.241 | 184.4 | 1.02x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 2 | 47.046 | 184.1 | 1.02x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 1 | 52.289 | 183.9 | 0.99x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 2 | 50.142 | 183.4 | 1.00x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 1 | 35.547 | 182.9 | 1.01x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 2 | 36.329 | 182.8 | 1.02x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 61.939 | 183.5 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 59.829 | 183.6 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 38.884 | 182.9 | 0.96x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 36.498 | 182.6 | 0.99x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 21.477 | 165.1 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

### Mean imputation

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 1 | 28.398 | 181.7 | 1.00x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 2 | 26.419 | 181.6 | 0.99x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 1 | 37.853 | 182.0 | 0.95x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 2 | 36.765 | 181.8 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 1 | 49.079 | 184.4 | 1.02x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 2 | 46.450 | 184.2 | 1.00x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 1 | 52.488 | 183.4 | 1.01x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 2 | 50.218 | 183.1 | 1.01x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 1 | 35.710 | 183.2 | 1.01x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 2 | 35.696 | 182.6 | 1.03x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 62.384 | 183.7 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 61.249 | 183.3 | 1.02x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 38.533 | 183.1 | new | — | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 36.946 | 182.7 | new | — | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 21.882 | 165.0 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## Variant statistics

### Allele frequencies and sample counts

| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 1 | 21.322 | 123.4 | 1.04x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`c7bf5e6`](https://github.com/shz9/magenpy/commit/c7bf5e64b31edbd0c65ee04a2d1388cc92e74519) | 2026-11-13 | 2026-11-13 | 2 | 19.920 | 121.8 | 0.97x | 1.00x | Intel(R) Xeon(R) 6975P-C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 1 | 30.323 | 123.3 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`7f7b8e1`](https://github.com/shz9/magenpy/commit/7f7b8e1a1598370276f96a6764911c5b1a2c6274) | 2026-11-13 | 2026-11-13 | 2 | 30.098 | 121.9 | 1.01x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 1 | 41.081 | 125.2 | 1.01x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`141d43f`](https://github.com/shz9/magenpy/commit/141d43f18e73832b3831601522cd0c7412e17082) | 2026-11-13 | 2026-11-13 | 2 | 41.612 | 123.7 | 0.99x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 1 | 42.384 | 126.8 | 1.01x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`0d15c73`](https://github.com/shz9/magenpy/commit/0d15c73845192dd2e4df8b09536eb1cdfd717961) | 2026-11-13 | 2026-11-13 | 2 | 44.011 | 125.0 | 1.02x | 1.00x | AMD EPYC 7763 64-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 1 | 30.061 | 125.8 | 1.00x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`e77e0df`](https://github.com/shz9/magenpy/commit/e77e0df7cf9265426cfcdde875081e86c763d5b8) | 2026-11-13 | 2026-11-13 | 2 | 32.033 | 124.4 | 0.98x | 1.00x | AMD EPYC 9V45 96-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 1 | 52.801 | 126.8 | 1.01x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`71a49f8`](https://github.com/shz9/magenpy/commit/71a49f86ab3fdea758e57e6c5c36356f975507fa) | 2026-11-13 | 2026-11-13 | 2 | 51.842 | 125.2 | 0.98x | 1.00x | AMD EPYC 9V74 80-Core Processor (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 1 | 29.805 | 125.9 | 1.00x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`0b93d6c`](https://github.com/shz9/magenpy/commit/0b93d6c392ca6d597804118215c588b003c14084) | 2026-11-13 | 2026-11-13 | 2 | 30.309 | 124.3 | 1.01x | 1.00x | INTEL(R) XEON(R) PLATINUM 8573C (Linux/x86_64; OMP True) |
| [`93692df`](https://github.com/shz9/magenpy/commit/93692df762cff0723e467f878c7f98df9f891a39) + local changes | — | 2026-11-13 | 1 | 18.681 | 115.2 | — | — | Apple M5 Pro (Darwin/arm64; OMP False) |

## Latest regression check

No detected regressions in comparable cases. New cases establish a baseline.

[results.csv](results.csv) retains full commit IDs, push/run timestamps, function/category labels, comparison ratios, and workload fingerprints. Raw run artifacts also retain individual timing samples and dependency versions. Compare matching workload fingerprints; ratios above 1 indicate increased time or memory.
