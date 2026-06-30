#ifndef LINEAR_ALGEBRA_UTILS_H
#define LINEAR_ALGEBRA_UTILS_H

#include <algorithm>
#include <type_traits>
#include <vector>

// Check for and include `cblas`:
#ifdef HAVE_CBLAS
    #include <cblas.h>
#endif

// Check for and include `omp`:
#ifdef _OPENMP
    #include <omp.h>
#endif

inline bool omp_supported() {
    #ifdef _OPENMP
        return true;
    #else
        return false;
    #endif
}

inline bool blas_supported() {
    #ifdef HAVE_CBLAS
        return true;
    #else
        return false;
    #endif
}

// Define a function pointer for the dot product functions `dot` and `blas_dot`:
template <typename T, typename U>
using dot_func_pt = typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type (*)(const T*, const U*, int);

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type
dot(const T* x, const U* y, int size) {
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

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type
blas_dot(const T* x, const U* y, int size) {
    #ifdef HAVE_CBLAS
        int incx = 1;
        int incy = 1;

        if constexpr (std::is_same<T, float>::value) {
            if constexpr (std::is_same<U, float>::value) {
                return cblas_sdot(size, x, incx, y, incy);
            } else {
                std::vector<float> y_float(size);
                std::transform(y, y + size, y_float.begin(),  [](U val) { return static_cast<float>(val);});
                return cblas_sdot(size, x, incx, y_float.data(), incy);
            }
        }
        else if constexpr (std::is_same<T, double>::value) {
            if constexpr (std::is_same<U, double>::value) {
                return cblas_ddot(size, x, incx, y, incy);
            } else {
                std::vector<double> y_double(size);
                std::transform(y, y + size, y_double.begin(),  [](U val) { return static_cast<double>(val);});
                return cblas_ddot(size, x, incx, y_double.data(), incy);
            }
        }
        else {
            return dot(x, y, size);
        }
    #else
        return dot(x, y, size);
    #endif
}

// Define a function pointer for the axpy functions `axpy` and `blas_axpy`:
template <typename T, typename U>
using axpy_func_pt = typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type (*)(T*, U*, T, int);

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type
axpy(T* x, U* y, T alpha, int size) {
    #ifdef _OPENMP
        #ifndef _WIN32
            #pragma omp simd
        #endif
    #endif
    for (int i = 0; i < size; ++i) {
        x[i] += static_cast<T>(y[i]) * alpha;
    }
}

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type
blas_axpy(T *y, U *x, T alpha, int size) {
    #ifdef HAVE_CBLAS
        int incx = 1;
        int incy = 1;

        if constexpr (std::is_same<T, float>::value) {
            if constexpr (std::is_same<U, float>::value) {
                cblas_saxpy(size, alpha, x, incx, y, incy);
            } else {
                std::vector<float> x_float(size);
                std::transform(x, x + size, x_float.begin(),  [](U val) { return static_cast<float>(val);});
                cblas_saxpy(size, alpha, x_float.data(), incx, y, incy);
            }
        }
        else if constexpr (std::is_same<T, double>::value) {
            if constexpr (std::is_same<U, double>::value) {
                cblas_daxpy(size, alpha, x, incx, y, incy);
            } else {
                std::vector<double> x_double(size);
                std::transform(x, x + size, x_double.begin(),  [](U val) { return static_cast<double>(val);});
                cblas_daxpy(size, alpha, x_double.data(), incx, y, incy);
            }
        }
        else {
            axpy(y, x, alpha, size);
        }
    #else
        axpy(y, x, alpha, size);
    #endif
}

#endif // LINEAR_ALGEBRA_UTILS_H
