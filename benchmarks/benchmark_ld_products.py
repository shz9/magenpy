"""Reproducible compressed LD benchmark; run from the repository root.

Example: PYTHONPATH=. python benchmarks/benchmark_ld_products.py --threads 4
Timing excludes construction and RHS packing. Columns compares one batched call
with repeated optimized vector calls. Compare the same command across revisions
for end-to-end speedups. OpenMP availability is reported rather than assumed.
"""

import argparse
from statistics import median
from time import perf_counter

import numpy as np

from magenpy.LDLinearOperator import LDLinearOperator
from magenpy.stats.ld import c_utils


def timed(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        samples.append(perf_counter() - start)
    return median(samples) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=12000)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.size, args.width, args.threads, args.repeats) < 1:
        parser.error("All parameters must be positive.")
    rng = np.random.default_rng(42)
    n = args.size
    counts = np.minimum(args.width, n - 1 - np.arange(n))
    ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    left = np.arange(1, n + 1, dtype=np.int32)
    print(
        f"n={n}, width={args.width}, nnz={ptr[-1]}, threads={args.threads}, OpenMP={getattr(c_utils, 'openmp_enabled', lambda: 'unknown')()}"
    )
    print("storage  RHS  batched_ms  columns_ms  batch_speedup")
    for storage in (np.int8, np.float32):
        data = rng.integers(-100, 100, size=ptr[-1]).astype(storage)
        if storage == np.float32:
            data *= 0.01
        op = LDLinearOperator(ptr, data, left, threads=args.threads)
        for k in (1, 16, 64):
            x = rng.normal(size=(n, k)).astype(np.float32)
            columns = [np.ascontiguousarray(x[:, j]) for j in range(k)]
            batch = lambda: op @ x
            separate = lambda: np.column_stack([op @ col for col in columns])
            np.testing.assert_allclose(batch(), separate(), rtol=2e-5, atol=2e-5)
            bt, ct = timed(batch, args.repeats), timed(separate, args.repeats)
            print(
                f"{np.dtype(storage).name:8} {k:3} {bt:11.3f} {ct:11.3f} {ct / bt:13.2f}x"
            )


if __name__ == "__main__":
    main()
