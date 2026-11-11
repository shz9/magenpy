#ifndef MAGENPY_LD_PRODUCTS_HPP
#define MAGENPY_LD_PRODUCTS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "../../utils/linear_algebra_utils.hpp"

// Four independent accumulators expose instruction-level parallelism even in
// builds without OpenMP. Conversion happens in registers, never in a row buffer.
// Reassociation changes rounding, but does not require global fast-math flags.
template<typename T, typename U, typename I>
inline T ld_product_dot(const T* x, const U* a, I n) {
    T s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    I j = 0;
    #ifdef _OPENMP
    #pragma omp simd reduction(+:s0,s1,s2,s3)
    #endif
    for (I q = 0; q < n / 4; ++q) {
        const I k = 4 * q;
        s0 += x[k] * static_cast<T>(a[k]);
        s1 += x[k+1] * static_cast<T>(a[k+1]);
        s2 += x[k+2] * static_cast<T>(a[k+2]);
        s3 += x[k+3] * static_cast<T>(a[k+3]);
    }
    T sum = (s0 + s1) + (s2 + s3);
    for (j = n - n % 4; j < n; ++j) sum += x[j] * static_cast<T>(a[j]);
    return sum;
}

// Each worker owns a disjoint interval of output rows. For triangular storage,
// lower contributions are clipped to that interval: no atomics, races, expanded
// matrix, or O(threads * n) scratch buffers. Scanning row pointers costs O(n *
// threads). Upper rows are revisited for each fixed-width RHS tile.
// RHS columns are tiled to bound the active output row working set. A contiguous
// inner loop enables SIMD on x86 and ARM, including quantized LD storage.
template<typename T, typename U, typename I>
void ld_matrix_product(int n, int nrhs, const int* left, const I* ptr,
                       const U* data, const T* x, T* out, T scale,
                       bool stored_full, bool lower, bool upper, bool diag,
                       int threads) {
    if (n == 0 || nrhs == 0) return;
    #ifdef _OPENMP
    #pragma omp parallel num_threads(threads) if(threads > 1)
    #endif
    {
        int worker = 0, workers = 1;
        #ifdef _OPENMP
        worker = omp_get_thread_num();
        workers = omp_get_num_threads();
        #endif
        const int begin = static_cast<int>(static_cast<int64_t>(n) * worker / workers);
        const int end = static_cast<int>(static_cast<int64_t>(n) * (worker + 1) / workers);
        // Initialize all owned rows before scattering lower contributions.
        for (int row = begin; row < end; ++row) {
            T* y = out + static_cast<size_t>(row) * nrhs;
            const T* xr = x + static_cast<size_t>(row) * nrhs;
            for (int k = 0; k < nrhs; ++k) y[k] = (!stored_full && diag) ? xr[k] : T(0);
        }
        const int source_begin = (!stored_full && lower) ? 0 : begin;
        for (int row = source_begin; row < end; ++row) {
            if (row >= begin && (stored_full || upper)) {
                T* y = out + static_cast<size_t>(row) * nrhs;
                const int first = stored_full ? left[row] : row + 1;
                const I count = ptr[row+1] - ptr[row];
                if (nrhs == 1) {
                    if (count) y[0] += scale * ld_product_dot(x + first, data + ptr[row], count);
                } else {
                    // Fixed-size accumulators stay in SIMD registers rather
                    // than repeatedly loading/storing the output for each edge.
                    int kb = 0;
                    for (; kb + 8 <= nrhs; kb += 8) {
                        T sums[8] = {};
                        for (I j = 0; j < count; ++j) {
                            const T a = static_cast<T>(data[ptr[row] + j]);
                            const T* xc = x + (static_cast<size_t>(first) + j) * nrhs + kb;
                            #ifdef _OPENMP
                            #pragma omp simd
                            #endif
                            for (int k = 0; k < 8; ++k) sums[k] += a * xc[k];
                        }
                        for (int k = 0; k < 8; ++k) y[kb+k] += scale * sums[k];
                    }
                    // A short tail also accumulates in registers.
                    for (; kb < nrhs; ++kb) {
                        T sum = 0;
                        for (I j = 0; j < count; ++j)
                            sum += static_cast<T>(data[ptr[row] + j]) *
                                   x[(static_cast<size_t>(first) + j) * nrhs + kb];
                        y[kb] += scale * sum;
                    }
                }
            }
            if (!stored_full && lower) {
                const int first = std::max(begin, row + 1);
                const int last = static_cast<int>(std::min<I>(end, row + 1 + ptr[row+1] - ptr[row]));
                const T* xr = x + static_cast<size_t>(row) * nrhs;
                if (nrhs == 1) {
                    const T a = scale * xr[0];
                    #ifdef _OPENMP
                    #pragma omp simd
                    #endif
                    for (int col = first; col < last; ++col)
                        out[col] += a * static_cast<T>(data[ptr[row] + col - row - 1]);
                } else {
                    for (int col = first; col < last; ++col) {
                        const T a = scale * static_cast<T>(data[ptr[row] + col - row - 1]);
                        T* y = out + static_cast<size_t>(col) * nrhs;
                        #ifdef _OPENMP
                        #pragma omp simd
                        #endif
                        for (int k = 0; k < nrhs; ++k) y[k] += a * xr[k];
                    }
                }
            }
        }
    }
}

// Cython cannot always reconcile NumPy's float typedefs with the corresponding
// native C++ types while deducing T from multiple arguments. These adapters fix
// T explicitly and leave the independent storage and index types deducible.
template<typename U, typename I>
inline void ld_matrix_product_float(
        int n, int nrhs, const int* left, const I* ptr, const U* data,
        const float* x, float* out, float scale, bool stored_full, bool lower,
        bool upper, bool diag, int threads) {
    using data_type = typename std::remove_cv<U>::type;
    using index_type = typename std::remove_cv<I>::type;
    ld_matrix_product<float, data_type, index_type>(
        n, nrhs, left, ptr, data, x, out, scale, stored_full, lower, upper,
        diag, threads);
}

template<typename U, typename I>
inline void ld_matrix_product_double(
        int n, int nrhs, const int* left, const I* ptr, const U* data,
        const double* x, double* out, double scale, bool stored_full,
        bool lower, bool upper, bool diag, int threads) {
    using data_type = typename std::remove_cv<U>::type;
    using index_type = typename std::remove_cv<I>::type;
    ld_matrix_product<double, data_type, index_type>(
        n, nrhs, left, ptr, data, x, out, scale, stored_full, lower, upper,
        diag, threads);
}

#endif
