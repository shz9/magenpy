#ifndef LD_UTLS_H
#define LD_UTLS_H

#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <type_traits>

// Check for and include `cblas`:
#ifdef HAVE_CBLAS
    #include <cblas.h>
#endif

// Check for and include `omp`:
#ifdef _OPENMP
    #include <omp.h>
#endif


/* ----------------------------- */
// Helper system-related functions to check for BLAS and OpenMP support

bool omp_supported() {
    /* Check if OpenMP is supported by examining compiler flags. */
    #ifdef _OPENMP
        return true;
    #else
        return false;
    #endif
}

bool blas_supported() {
    /* Check if BLAS is supported by examining compiler flags. */
    #ifdef HAVE_CBLAS
        return true;
    #else
        return false;
    #endif
}

/* ------------------------------ */
// Dot product functions

// Define a function pointer for the dot product functions `dot` and `blas_dot`:
template <typename T, typename U>
using dot_func_pt = typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type (*)(T*, U*, int);

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type
dot(T* x, U* y, int size) {
    /* Perform dot product between two vectors x and y, each of length `size`

    :param x: Pointer to the first element of the first vector
    :param y: Pointer to the first element of the second vector
    :param size: Length of the vectors

    */

    T s = 0.;

    #ifdef _OPENMP
        #ifndef _WIN32
            #pragma omp simd
        #endif
    #endif
    for (int i = 0; i < size; ++i) {
        s += x[i]*static_cast<T>(y[i]);
    }
    return s;
}

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type
blas_dot(T* x, U* y, int size) {
    /*
        Use BLAS (if available) to perform dot product
        between two vectors x and y, each of length `size`.

        :param x: Pointer to the first element of the first vector
        :param y: Pointer to the first element of the second vector
        :param size: Length of the vectors
    */

    #ifdef HAVE_CBLAS
        int incx = 1;
        int incy = 1;

        if constexpr (std::is_same<T, float>::value) {
            if constexpr (std::is_same<U, float>::value) {
                return cblas_sdot(size, x, incx, y, incy);
            } else {
                // Handles the case where y is any data type that is not a float:
                std::vector<float> y_float(size);
                std::transform(y, y + size, y_float.begin(),  [](U val) { return static_cast<float>(val);});
                return cblas_sdot(size, x, incx, y_float.data(), incy);
            }
        }
        else if constexpr (std::is_same<T, double>::value) {
            if constexpr (std::is_same<U, double>::value) {
                return cblas_ddot(size, x, incx, y, incy);
            } else {
                // Handles the case where y is any data type that is not a double:
                std::vector<double> y_double(size);
                std::transform(y, y + size, y_double.begin(),  [](U val) { return static_cast<double>(val);});
                return cblas_ddot(size, x, incx, y_double.data(), incy);
            }
        }
    #else
        return dot(x, y, size);
    #endif
}

/* * * * * */

// Define a function pointer for the axpy functions `axpy` and `blas_axpy`:
template <typename T, typename U>
using axpy_func_pt = typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type (*)(T*, U*, T, int);

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type
axpy(T* x, U* y, T alpha, int size) {
    /*
        Perform axpy operation on two vectors x and y, each of length `size`.
       axpy is a standard linear algebra operation that performs
       element-wise addition and multiplication:
       x := x + a*y.
    */

    #ifdef _OPENMP
        #ifndef _WIN32
            #pragma omp simd
        #endif
    #endif
    for (int i = 0; i < size; ++i) {
        x[i] += static_cast<T>(y[i]) * alpha;
    }
}

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type
blas_axpy(T *y, U *x, T alpha, int size) {
    /*
        Use BLAS (if available) to perform axpy operation on two vectors x and y,
        each of length `size`.
       axpy is a standard linear algebra operation that performs
       element-wise addition and multiplication:
       x := x + a*y.
    */

    #ifdef HAVE_CBLAS
        int incx = 1;
        int incy = 1;

        if constexpr (std::is_same<T, float>::value) {
            if constexpr (std::is_same<U, float>::value) {
                cblas_saxpy(size, alpha, x, incx, y, incy);
            } else {
                // Handles the case where x is any data type that is not a float:
                std::vector<float> x_float(size);
                std::transform(x, x + size, x_float.begin(),  [](U val) { return static_cast<float>(val);});
                cblas_saxpy(size, alpha, x_float.data(), incx, y, incy);
            }
        }
        else if constexpr (std::is_same<T, double>::value) {
            if constexpr (std::is_same<U, double>::value) {
                cblas_daxpy(size, alpha, x, incx, y, incy);
            } else {
                // Handles the case where x is any data type that is not a float:
                std::vector<double> x_double(size);
                std::transform(x, x + size, x_double.begin(),  [](U val) { return static_cast<double>(val);});
                cblas_daxpy(size, alpha, x_double.data(), incx, y, incy);
            }
        }
    #else
        axpy(y, x, alpha, size);
    #endif
}


template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
ld_matrix_dot(int c_size,
              int* ld_left_bound,
              I* ld_indptr,
              U* ld_data,
              T* vec,
              T* out,
              T dq_scale,
              int threads) {
    /*
        Perform matrix-vector multiplication between a Linkage-Disequilibrium (LD) matrix where the entries
        are stored in a compressed format (CSR) and a vector `vec`. This function assumes that
        the entries of the matrix are contiguous around the diagonal.
        The result is stored in the output vector `out` of the same length and data type as `vec`.

        The function is parallelized using OpenMP if the compiler supports it. It also uses BLAS
        if the library is available on the user's system. Finally, the function is templated to allow
        the user to pass quantized matrices and vectors of different data types (e.g., float, double, etc.).
        If the matrix is quantized, ensure to pass the correct scaling factor `dq_scale` to the function.
    */

    I ld_start, ld_end;

    #ifdef _OPENMP
        #pragma omp parallel for private(ld_start, ld_end) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {

        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];

        out[j] += dq_scale*blas_dot(vec + ld_left_bound[j], ld_data + ld_start, ld_end - ld_start);
    }
}


template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
ut_ld_matrix_dot(int c_size,
                 I* ld_indptr,
                 U* ld_data,
                 T* vec,
                 T* out,
                 T dq_scale,
                 bool include_lower_triangle,
                 bool include_upper_triangle,
                 bool include_diag,
                 int threads) {
    /*
        Perform matrix-vector multiplication between an upper-triangular matrix where the entries
        are stored in a compressed format (CSR) and a vector `vec`. This function assumes that
        the entries of the matrix are contiguous along the diagonal (not including the diagonal itself).
        The result is stored in the output vector `out` of the same length and data type as `vec`.

        The function is parallelized using OpenMP if the compiler supports it. It also uses BLAS
        if the library is available on the user's system. Finally, the function is templated to allow
        the user to pass quantized matrices and vectors of different data types (e.g., float, double, etc.).
        If the matrix is quantized, ensure to pass the correct scaling factor `dq_scale` to the function.

        The function gives flexibility to perform matrix-vector product assuming a full symmetric matrix,
        with and without the diagonal, and with and without the lower and upper triangles. These options
        can be specified by setting the corresponding boolean flags `include_lower_triangle`,
        `include_upper_triangle`, and `include_diag`.
    */

    I ld_start, ld_end;

    #ifdef _OPENMP
        #pragma omp parallel for private(ld_start, ld_end) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {

        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];

        if (include_upper_triangle){
            out[j] += dq_scale*blas_dot(vec + j + 1, ld_data + ld_start, ld_end - ld_start);
        }

        if (include_lower_triangle){
            blas_axpy(out + j + 1, ld_data + ld_start, dq_scale*vec[j], ld_end - ld_start);
        }

        if (include_diag) {
            out[j] += vec[j];
        }

    }
}


template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
ld_rank_one_update(int c_size,
                   int* ld_left_bound,
                   I* ld_indptr,
                   U* ld_data,
                   T* vec,
                   T* out,
                   T alpha,
                   T dq_scale,
                   int threads) {
    /*
        Perform rank-one update or perturbation on the LD matrix, which we define as:

        out = R + alpha * vec * vec^T,

        where R is the original LD matrix, vec is a vector, and alpha is a scaling factor. The input parameters
        are as follows:

        - `c_size`: The number of rows/columns in the LD matrix.
        - `ld_left_bound`: An array of size `c_size` that contains the left bound of each column in the full matrix.
        - `ld_indptr`: An array of size `c_size + 1` that contains the indices of the start and end of each column in `ld_data`.
        - `ld_data`: An array of size `ld_indptr[c_size]` that contains the non-zero entries of the LD matrix.
        - `vec`: A vector of size `c_size` that is used to perform the rank-one update.
        - `out`: A vector of size `ld_indptr[c_size]` that stores the result of the rank-one update.
        - `alpha`: A scaling factor that multiplies the outer product of `vec` with itself.
        - `dq_scale`: A scaling factor used to quantize the matrix. If the matrix is not quantized, set this to 1.
        - `threads`: The number of threads to use for parallelization.

        The function is parallelized using OpenMP if the compiler supports it. It also uses BLAS
        if the library is available on the user's system. Finally, the function is templated to allow
        the user to pass quantized matrices and vectors of different data types (e.g., float, double, etc.).
        If the matrix is quantized, ensure to pass the correct scaling factor `dq_scale` to the function.
    */

    I ld_start, ld_end;
    int col_idx;

    #ifdef _OPENMP
        #pragma omp parallel for private(ld_start, ld_end, col_idx) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {

        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];

        for (int i = 0; i < ld_end - ld_start; ++i) {
            col_idx = ld_left_bound[j] + i; // The column index of the entry in the full matrix
            out[ld_start + i] = ld_data[ld_start + i]*dq_scale + alpha*vec[j]*vec[col_idx];
        }
    }
}

#endif // LD_UTLS_H
