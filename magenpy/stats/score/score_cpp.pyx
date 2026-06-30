# distutils: language = c++
# sources: stats/score/score.hpp

from libcpp.string cimport string
from libcpp cimport bool
from cython cimport floating
import numpy as np

cdef extern from "score.hpp" nogil:
    bint blas_supported() noexcept nogil
    bint omp_supported() noexcept nogil

    void calculate_scores[T](string bed_filename,
                             const T* effect_sizes,
                             const int* snp_indices,
                             const int* sample_indices,
                             int total_samples,
                             int num_samples,
                             int num_snps,
                             int num_scores,
                             T* scores,
                             int threads,
                             const T* allele_frequencies,
                             bool standardize_genotype,
                             bool impute_missing) except + nogil


cpdef calculate_pgs(bed_filename,
                    const double[:, ::1] effect_sizes, #floating[:, ::1] effect_sizes,
                    const int[::1] snp_indices,
                    const int[::1] sample_indices,
                    int total_samples,
                    int threads,
                    allele_frequencies=None,
                    bint standardize_genotype=False,
                    bint impute_missing=False):

    """
    Calculate polygenic scores for a set of SNPs and samples using custom C++
    script, written for speed/efficiency.

    NOTE: Assumes SNP and sample indices are sorted!

    :param bed_filename: Path to PLINK BED file
    :param effect_sizes: Numpy array or matrix of effect sizes for each SNP
    :param snp_indices: Numpy array of SNP indices to use for PGS calculation
    :param sample_indices: Numpy array of sample indices to use for PGS calculation
    :param total_samples: Total number of samples stored in the BED file
    :param threads: Number of threads to use for PGS calculation
    :param allele_frequencies: Optional allele-frequency vector with one value per SNP index
    :param standardize_genotype: If True, score standardized genotypes using allele frequencies
    :param impute_missing: If True, mean-impute missing genotypes using allele frequencies
    """

    cdef:
        string c_bed_filename
        double[:, ::1] pgs
        const double[::1] allele_frequency_view
        const double* allele_frequency_ptr = NULL
        int num_samples
        int num_snps
        int num_scores
        bool c_standardize_genotype
        bool c_impute_missing

    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")

    if effect_sizes.shape[0] != snp_indices.shape[0]:
        raise ValueError("effect_sizes must have one row per SNP index.")

    if effect_sizes.shape[1] <= 0:
        raise ValueError("effect_sizes must have at least one score column.")

    if sample_indices.shape[0] > 2147483647:
        raise ValueError("Too many selected samples for the C++ scoring backend.")

    if snp_indices.shape[0] > 2147483647:
        raise ValueError("Too many SNPs for the C++ scoring backend.")

    if effect_sizes.shape[1] > 2147483647:
        raise ValueError("Too many score columns for the C++ scoring backend.")

    num_samples = <int> sample_indices.shape[0]
    num_snps = <int> snp_indices.shape[0]
    num_scores = <int> effect_sizes.shape[1]
    c_standardize_genotype = <bool> standardize_genotype
    c_impute_missing = <bool> impute_missing

    if allele_frequencies is not None:
        allele_frequencies = np.ascontiguousarray(allele_frequencies, dtype=np.float64)
        allele_frequency_view = allele_frequencies
        if allele_frequency_view.shape[0] != num_snps:
            raise ValueError("allele_frequencies must have one value per SNP index.")
        if allele_frequency_view.shape[0] > 0:
            allele_frequency_ptr = &allele_frequency_view[0]
    elif (standardize_genotype or impute_missing) and num_snps > 0:
        raise ValueError("allele_frequencies are required when standardize_genotype or impute_missing is True.")

    c_bed_filename = bed_filename.encode()
    pgs = np.zeros((num_samples, num_scores))

    if num_samples == 0 or num_snps == 0:
        return np.array(pgs)

    #if floating is float:
    #    pgs = np.zeros((sample_indices.shape[0], effect_sizes.shape[1]), dtype=np.float32)
    #else:
    #    pgs = np.zeros((sample_indices.shape[0], effect_sizes.shape[1]), dtype=np.float64)

    calculate_scores[double](c_bed_filename,
                             &effect_sizes[0, 0],
                             &snp_indices[0],
                             &sample_indices[0],
                             total_samples,
                             num_samples,
                             num_snps,
                             num_scores,
                             &pgs[0, 0],
                             threads,
                             allele_frequency_ptr,
                             c_standardize_genotype,
                             c_impute_missing)

    return np.array(pgs)
