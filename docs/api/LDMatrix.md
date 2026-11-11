::: magenpy.LDMatrix.LDMatrix
## Fast vector and matrix products

Use the linear operator to multiply without expanding or dequantizing the LD
matrix in memory:

```python
import numpy as np

op = ld_matrix.to_linear_operator()
x = np.ones(op.shape[1], dtype=np.float32)
y = op @ x
X = np.ones((op.shape[1], 32), dtype=np.float32)
Y = op @ X
YT = op.T @ X
```

Pass all right-hand sides as columns of one matrix to reuse LD data across them.
C-contiguous float32/float64 arrays avoid input packing; strided and Fortran
arrays are packed once. Floating inputs retain their precision, integer inputs
are promoted to floating point, and complex products use two real products.
Scalar and row-specific diagonal shifts work for both vectors and matrices.

Input normalization and reshaping borrow existing buffers whenever possible;
read-only contiguous inputs also avoid a copy. LD storage is always borrowed.
Diagonal shifts are added directly into the result by a compiled loop, avoiding
a full `X * shift` temporary. Native float/integer shift arrays retain their
buffers and strides; float16 and non-native byte order require one conversion.
Complex products fill a single final complex output buffer from the two real
products, without intermediate complex multiply/add arrays.

The kernels convert quantized entries in registers, use independent dot-product
accumulators, and vectorize across contiguous RHS columns. They add no dependency
and need no expanded matrix, per-row conversion buffer, or per-thread output
array. Matrix products tile RHS columns to limit the active output working set.
Floating-point reassociation can change the last few bits of results.

Set `op.threads` to a positive integer to use multiple workers in an OpenMP
build (the default is one). Without OpenMP the same code runs serially. Check
`magenpy.stats.ld.c_utils.openmp_enabled()` for build support. Each worker owns
a disjoint set of output rows, including reflected lower-triangle contributions;
this fixes overlapping writes in the previous threaded triangular kernel.
Additional workers scan row pointers to locate reflected contributions, so
threading may not help small matrices or very narrow bands. Benchmark the
actual workload rather than assuming more threads will help.

For reproducible timings, from a source checkout run:

```sh
PYTHONPATH=. python benchmarks/benchmark_ld_products.py --threads 1
PYTHONPATH=. python benchmarks/benchmark_ld_products.py --threads 4
```

The benchmark checks batched results against separate vector products and
reports median timings after warmup. Compare the same workload and build flags
across revisions for overall speedups; the column comparison measures batching
within the current revision. Rebuild the extension after changing Cython or C++
sources (`python setup.py build_ext --inplace --force`).
