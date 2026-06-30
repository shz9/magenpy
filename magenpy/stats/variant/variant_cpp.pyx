# distutils: language = c++
# sources: stats/variant/variant_utils.hpp

from libc.stdint cimport int8_t, int16_t, int32_t
from libcpp.string cimport string
from libcpp cimport bool as cpp_bool
import numpy as np


cdef extern from "variant_utils.hpp" nogil:
    bint blas_supported() noexcept nogil
    bint omp_supported() noexcept nogil

    void cpp_extract_genotype_matrix "extract_genotype_matrix"[T](string bed_filename,
                                                                  const int* snp_indices,
                                                                  const int* sample_indices,
                                                                  int total_samples,
                                                                  int num_samples,
                                                                  int num_snps,
                                                                  T* out,
                                                                  int threads) except + nogil

    void cpp_compute_variant_stats "compute_variant_stats"[T](string bed_filename,
                                                              const int* snp_indices,
                                                              const int* sample_indices,
                                                              int total_samples,
                                                              int num_samples,
                                                              int num_snps,
                                                              T* allele_frequencies,
                                                              int* n_per_snp,
                                                              int threads) except + nogil

    void cpp_compute_gwa_linear_stats "compute_gwa_linear_stats"[T](string bed_filename,
                                                                    const int* snp_indices,
                                                                    const int* sample_indices,
                                                                    int total_samples,
                                                                    int num_samples,
                                                                    int num_snps,
                                                                    const T* centered_phenotype,
                                                                    cpp_bool standardize_genotype,
                                                                    T* allele_frequencies,
                                                                    int* n_per_snp,
                                                                    T* x_dot_y,
                                                                    T* sum_x_sq,
                                                                    int threads) except + nogil


cpdef extract_genotype_matrix(bed_filename,
                              const int[::1] snp_indices,
                              const int[::1] sample_indices,
                              int total_samples,
                              int threads=1,
                              dtype=np.int8):
    """
    Extract selected PLINK BED dosages into a samples-by-SNPs NumPy array.

    Missing values are encoded as -1 for integer dtypes and NaN for floating
    dtypes.
    """

    cdef:
        string c_bed_filename
        const int* snp_ptr = NULL
        const int* sample_ptr = NULL
        int8_t[:, ::1] out_int8
        int16_t[:, ::1] out_int16
        int32_t[:, ::1] out_int32
        float[:, ::1] out_float32
        double[:, ::1] out_float64
        int8_t* out_int8_ptr = NULL
        int16_t* out_int16_ptr = NULL
        int32_t* out_int32_ptr = NULL
        float* out_float32_ptr = NULL
        double* out_float64_ptr = NULL
        int num_snps
        int num_samples

    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")

    if snp_indices.shape[0] > 2147483647:
        raise ValueError("Too many SNPs for the C++ genotype extraction backend.")
    if sample_indices.shape[0] > 2147483647:
        raise ValueError("Too many selected samples for the C++ genotype extraction backend.")

    num_snps = <int> snp_indices.shape[0]
    num_samples = <int> sample_indices.shape[0]

    if num_snps > 0:
        snp_ptr = &snp_indices[0]
    if num_samples > 0:
        sample_ptr = &sample_indices[0]

    c_bed_filename = bed_filename.encode()
    dtype = np.dtype(dtype)

    if dtype == np.dtype(np.int8):
        out_int8 = np.empty((num_samples, num_snps), dtype=np.int8)
        if num_samples > 0 and num_snps > 0:
            out_int8_ptr = &out_int8[0, 0]
        cpp_extract_genotype_matrix[int8_t](c_bed_filename,
                                            snp_ptr,
                                            sample_ptr,
                                            total_samples,
                                            num_samples,
                                            num_snps,
                                            out_int8_ptr,
                                            threads)
        return np.asarray(out_int8)

    elif dtype == np.dtype(np.int16):
        out_int16 = np.empty((num_samples, num_snps), dtype=np.int16)
        if num_samples > 0 and num_snps > 0:
            out_int16_ptr = &out_int16[0, 0]
        cpp_extract_genotype_matrix[int16_t](c_bed_filename,
                                             snp_ptr,
                                             sample_ptr,
                                             total_samples,
                                             num_samples,
                                             num_snps,
                                             out_int16_ptr,
                                             threads)
        return np.asarray(out_int16)

    elif dtype == np.dtype(np.int32):
        out_int32 = np.empty((num_samples, num_snps), dtype=np.int32)
        if num_samples > 0 and num_snps > 0:
            out_int32_ptr = &out_int32[0, 0]
        cpp_extract_genotype_matrix[int32_t](c_bed_filename,
                                             snp_ptr,
                                             sample_ptr,
                                             total_samples,
                                             num_samples,
                                             num_snps,
                                             out_int32_ptr,
                                             threads)
        return np.asarray(out_int32)

    elif dtype == np.dtype(np.float32):
        out_float32 = np.empty((num_samples, num_snps), dtype=np.float32)
        if num_samples > 0 and num_snps > 0:
            out_float32_ptr = &out_float32[0, 0]
        cpp_extract_genotype_matrix[float](c_bed_filename,
                                           snp_ptr,
                                           sample_ptr,
                                           total_samples,
                                           num_samples,
                                           num_snps,
                                           out_float32_ptr,
                                           threads)
        return np.asarray(out_float32)

    elif dtype == np.dtype(np.float64):
        out_float64 = np.empty((num_samples, num_snps), dtype=np.float64)
        if num_samples > 0 and num_snps > 0:
            out_float64_ptr = &out_float64[0, 0]
        cpp_extract_genotype_matrix[double](c_bed_filename,
                                            snp_ptr,
                                            sample_ptr,
                                            total_samples,
                                            num_samples,
                                            num_snps,
                                            out_float64_ptr,
                                            threads)
        return np.asarray(out_float64)

    else:
        raise ValueError("dtype must be one of int8, int16, int32, float32, or float64.")


cpdef compute_gwa_linear_stats(bed_filename,
                               const int[::1] snp_indices,
                               const int[::1] sample_indices,
                               int total_samples,
                               const double[::1] centered_phenotype,
                               bint standardize_genotype=False,
                               int threads=1):
    """
    Compute per-SNP sufficient statistics for simple linear GWAS from a PLINK
    BED file without materializing the genotype matrix.

    :return: Tuple `(allele_frequencies, n_per_snp, x_dot_y, sum_x_sq)`.
    """

    cdef:
        string c_bed_filename
        double[::1] allele_frequencies
        int[::1] n_per_snp
        double[::1] x_dot_y
        double[::1] sum_x_sq
        const int* snp_ptr = NULL
        const int* sample_ptr = NULL
        const double* centered_phenotype_ptr = NULL
        double* allele_frequency_ptr = NULL
        int* n_per_snp_ptr = NULL
        double* x_dot_y_ptr = NULL
        double* sum_x_sq_ptr = NULL
        int num_snps
        int num_samples
        cpp_bool c_standardize_genotype

    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")

    if snp_indices.shape[0] > 2147483647:
        raise ValueError("Too many SNPs for the C++ GWAS backend.")
    if sample_indices.shape[0] > 2147483647:
        raise ValueError("Too many selected samples for the C++ GWAS backend.")

    if sample_indices.shape[0] != centered_phenotype.shape[0]:
        raise ValueError("centered_phenotype must have one value per selected sample.")

    num_snps = <int> snp_indices.shape[0]
    num_samples = <int> sample_indices.shape[0]
    c_standardize_genotype = <cpp_bool> standardize_genotype

    if num_snps > 0:
        snp_ptr = &snp_indices[0]
    if num_samples > 0:
        sample_ptr = &sample_indices[0]
        centered_phenotype_ptr = &centered_phenotype[0]

    c_bed_filename = bed_filename.encode()
    allele_frequencies = np.zeros(num_snps, dtype=np.float64)
    n_per_snp = np.zeros(num_snps, dtype=np.int32)
    x_dot_y = np.zeros(num_snps, dtype=np.float64)
    sum_x_sq = np.zeros(num_snps, dtype=np.float64)

    if num_snps > 0:
        allele_frequency_ptr = &allele_frequencies[0]
        n_per_snp_ptr = &n_per_snp[0]
        x_dot_y_ptr = &x_dot_y[0]
        sum_x_sq_ptr = &sum_x_sq[0]

    cpp_compute_gwa_linear_stats[double](c_bed_filename,
                                         snp_ptr,
                                         sample_ptr,
                                         total_samples,
                                         num_samples,
                                         num_snps,
                                         centered_phenotype_ptr,
                                         c_standardize_genotype,
                                         allele_frequency_ptr,
                                         n_per_snp_ptr,
                                         x_dot_y_ptr,
                                         sum_x_sq_ptr,
                                         threads)

    return (np.asarray(allele_frequencies),
            np.asarray(n_per_snp),
            np.asarray(x_dot_y),
            np.asarray(sum_x_sq))


cpdef compute_variant_stats(bed_filename,
                            const int[::1] snp_indices,
                            const int[::1] sample_indices,
                            int total_samples,
                            int threads=1):
    """
    Compute allele frequencies and non-missing sample counts for selected
    variants and samples from a PLINK BED file.

    :param bed_filename: Path to PLINK BED file.
    :param snp_indices: Sorted selected SNP indices in original BED coordinates.
    :param sample_indices: Sorted selected sample indices in original BED coordinates.
    :param total_samples: Total number of samples in the BED file.
    :param threads: Number of OpenMP threads.
    :return: Tuple `(allele_frequencies, n_per_snp)`.
    """

    cdef:
        string c_bed_filename
        double[::1] allele_frequencies
        int[::1] n_per_snp
        const int* snp_ptr = NULL
        const int* sample_ptr = NULL
        double* allele_frequency_ptr = NULL
        int* n_per_snp_ptr = NULL
        int num_snps
        int num_samples

    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")

    if snp_indices.shape[0] > 2147483647:
        raise ValueError("Too many SNPs for the C++ variant-stat backend.")
    if sample_indices.shape[0] > 2147483647:
        raise ValueError("Too many selected samples for the C++ variant-stat backend.")

    num_snps = <int> snp_indices.shape[0]
    num_samples = <int> sample_indices.shape[0]

    if num_snps > 0:
        snp_ptr = &snp_indices[0]
    if num_samples > 0:
        sample_ptr = &sample_indices[0]

    c_bed_filename = bed_filename.encode()
    allele_frequencies = np.zeros(num_snps, dtype=np.float64)
    n_per_snp = np.zeros(num_snps, dtype=np.int32)
    if num_snps > 0:
        allele_frequency_ptr = &allele_frequencies[0]
        n_per_snp_ptr = &n_per_snp[0]

    cpp_compute_variant_stats[double](c_bed_filename,
                                      snp_ptr,
                                      sample_ptr,
                                      total_samples,
                                      num_samples,
                                      num_snps,
                                      allele_frequency_ptr,
                                      n_per_snp_ptr,
                                      threads)

    return np.asarray(allele_frequencies), np.asarray(n_per_snp)
