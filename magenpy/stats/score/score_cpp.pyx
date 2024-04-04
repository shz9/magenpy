# distutils: language = c++
# sources: stats/score/score.hpp

from libcpp.string cimport string
from cython cimport floating
import numpy as np

#string bed_filename,

cdef extern from "score.hpp" nogil:
    bint blas_supported() noexcept nogil
    bint omp_supported() noexcept nogil

    void calculate_scores[T](string bed_filename,
                             T* effect_sizes,
                             int* snp_indices,
                             int* sample_indices,
                             int num_samples,
                             int num_snps,
                             int num_scores,
                             T* scores,
                             int threads) noexcept nogil


cpdef calculate_pgs(bed_filename,
                    double[:, ::1] effect_sizes, #floating[:, ::1] effect_sizes,
                    int[:] snp_indices,
                    int[:] sample_indices, #int[:] sample_indices,
                    int threads):

    """
    Calculate polygenic scores for a set of SNPs and samples using custom C++
    script, written for speed/efficiency.

    NOTE: Assumes SNP and sample indices are sorted!

    :param bed_filename: Path to PLINK BED file
    :param effect_sizes: Numpy array or matrix of effect sizes for each SNP
    :param snp_indices: Numpy array of SNP indices to use for PGS calculation
    :param sample_indices: Numpy array of sample indices to use for PGS calculation
    :param threads: Number of threads to use for PGS calculation
    """

    cdef:
        string c_bed_filename = bed_filename.encode()
        double[:, ::1] pgs = np.zeros((sample_indices.shape[0], effect_sizes.shape[1]))

    #if floating is float:
    #    pgs = np.zeros((sample_indices.shape[0], effect_sizes.shape[1]), dtype=np.float32)
    #else:
    #    pgs = np.zeros((sample_indices.shape[0], effect_sizes.shape[1]), dtype=np.float64)

    calculate_scores(c_bed_filename,
                     &effect_sizes[0, 0],
                     &snp_indices[0],
                     &sample_indices[0],
                     sample_indices.shape[0],
                     snp_indices.shape[0],
                     effect_sizes.shape[1],
                     &pgs[0, 0],
                     threads)

    return np.array(pgs)
