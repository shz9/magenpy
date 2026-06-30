#ifndef LD_UTLS_H
#define LD_UTLS_H

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <type_traits>

#include "../../utils/linear_algebra_utils.hpp"
#include "../../utils/plink_bed_utils.hpp"

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
decode_standardized_ld_bed_row(const unsigned char* bed_row,
                               const std::vector<SelectedSampleByte>& sample_groups,
                               int num_samples,
                               T allele_frequency,
                               bool impute_missing,
                               const PlinkDosageLookup& lookup,
                               T* values,
                               uint8_t* observed) {

    if (!std::isfinite(static_cast<double>(allele_frequency)) ||
        allele_frequency < T(0) || allele_frequency > T(1)) {
        throw std::out_of_range("Allele frequencies must be in [0, 1].");
    }

    const T imputed_dosage = static_cast<T>(2) * allele_frequency;
    T sum = T(0);
    int count = 0;

    std::fill(values, values + num_samples, T(0));
    if (observed != nullptr) {
        std::fill(observed, observed + num_samples, static_cast<uint8_t>(0));
    }

    for (const SelectedSampleByte& group : sample_groups) {
        const unsigned char byte = bed_row[group.byte_index];

        for (int slot = 0; slot < 4; ++slot) {
            const int output_index = group.output_index[slot];
            if (output_index < 0) {
                continue;
            }

            const int8_t dosage = lookup.dosage[byte][slot];

            if (dosage >= 0) {
                values[output_index] = static_cast<T>(dosage);
                sum += values[output_index];
                ++count;
                if (observed != nullptr) {
                    observed[output_index] = 1;
                }
            }
            else if (impute_missing) {
                values[output_index] = imputed_dosage;
                sum += values[output_index];
                ++count;
                if (observed != nullptr) {
                    observed[output_index] = 1;
                }
            }
        }
    }

    if (count == 0) {
        return;
    }

    const T mean = sum / static_cast<T>(count);
    T sum_squares = T(0);

    for (int sample_pos = 0; sample_pos < num_samples; ++sample_pos) {
        if (impute_missing || observed[sample_pos]) {
            const T centered = values[sample_pos] - mean;
            values[sample_pos] = centered;
            sum_squares += centered * centered;
        }
    }

    if (sum_squares <= T(0)) {
        std::fill(values, values + num_samples, T(0));
        return;
    }

    const T inverse_sd = static_cast<T>(
        std::sqrt(static_cast<double>(count) / static_cast<double>(sum_squares))
    );

    for (int sample_pos = 0; sample_pos < num_samples; ++sample_pos) {
        values[sample_pos] *= inverse_sd;
    }
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


template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
compute_ld_from_bed(std::string bed_filename,
                    const int* ref_snp_indices,
                    const int* alt_snp_indices,
                    int num_pairs,
                    const int* sample_indices,
                    int total_samples,
                    int num_samples,
                    const T* allele_frequencies,
                    bool impute_missing,
                    T* ld_data,
                    int threads) {
    /*
        Compute LD directly from a SNP-major PLINK BED file for paired variant
        indices.

        Each output entry corresponds to one pair:
            ld_data[pair_idx] = corr(ref_snp_indices[pair_idx], alt_snp_indices[pair_idx])

        Allele frequencies are expected to be indexed by BED variant index, so
        the frequency for SNP `snp_index` is `allele_frequencies[snp_index]`.
        If missing genotypes are imputed, the imputed dosage is `2p`.
        Genotypes are then centered and scaled by their empirical mean and
        standard deviation over the selected samples.

        If `impute_missing` is true, missing dosages are mean-imputed and
        therefore contribute zero after standardization. Otherwise, missing
        genotypes are excluded from each pair's denominator.
    */

    if (total_samples <= 0) {
        throw std::invalid_argument("Total BED sample count must be positive.");
    }
    if (num_pairs < 0 || num_samples < 0) {
        throw std::invalid_argument("Invalid LD calculation dimensions.");
    }
    if ((num_pairs > 0 && ref_snp_indices == nullptr) ||
        (num_pairs > 0 && alt_snp_indices == nullptr) ||
        (num_samples > 0 && sample_indices == nullptr) ||
        allele_frequencies == nullptr ||
        (num_pairs > 0 && ld_data == nullptr)) {
        throw std::invalid_argument("Null pointer passed to compute_ld_from_bed.");
    }

    if (num_pairs == 0) {
        return;
    }

    validate_plink_bed_file(bed_filename);

    const std::vector<SelectedSampleByte> sample_groups =
        group_selected_samples_by_byte(sample_indices, total_samples, num_samples);

    if (num_samples == 0) {
        std::fill(ld_data, ld_data + num_pairs, T(0));
        return;
    }

    const std::streamoff variant_stride = plink_bed_variant_stride(total_samples);
    const size_t stride = static_cast<size_t>(variant_stride);
    const int num_threads = threads > 0 ? threads : 1;
    const PlinkDosageLookup lookup;

    for (int pair_idx = 0; pair_idx < num_pairs; ++pair_idx) {
        const int ref_snp_index = ref_snp_indices[pair_idx];
        const int alt_snp_index = alt_snp_indices[pair_idx];

        if (ref_snp_index < 0 || alt_snp_index < 0) {
            throw std::out_of_range("Reference SNP index must be non-negative.");
        }

        const T ref_allele_frequency = allele_frequencies[ref_snp_index];
        const T alt_allele_frequency = allele_frequencies[alt_snp_index];
        if (!std::isfinite(static_cast<double>(ref_allele_frequency)) ||
            !std::isfinite(static_cast<double>(alt_allele_frequency)) ||
            ref_allele_frequency < T(0) || ref_allele_frequency > T(1) ||
            alt_allele_frequency < T(0) || alt_allele_frequency > T(1)) {
            throw std::out_of_range("Allele frequencies must be in [0, 1].");
        }
    }

    #ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
        {
    #endif
            std::ifstream bed_file(bed_filename, std::ios::binary);
            if (!bed_file.is_open()) {
                throw std::runtime_error("Error opening BED file.");
            }

            std::vector<unsigned char> ref_row(stride);
            std::vector<unsigned char> alt_row(stride);
            std::vector<T> ref_values(static_cast<size_t>(num_samples));
            std::vector<T> alt_values(static_cast<size_t>(num_samples));
            std::vector<uint8_t> ref_observed;
            std::vector<uint8_t> alt_observed;
            if (!impute_missing) {
                ref_observed.resize(static_cast<size_t>(num_samples));
                alt_observed.resize(static_cast<size_t>(num_samples));
            }

            #ifdef _OPENMP
                #pragma omp for schedule(static)
            #endif
            for (int pair_idx = 0; pair_idx < num_pairs; ++pair_idx) {
                const int ref_snp_index = ref_snp_indices[pair_idx];
                const int alt_snp_index = alt_snp_indices[pair_idx];

                read_plink_bed_row(bed_file, ref_snp_index, variant_stride, ref_row.data());
                read_plink_bed_row(bed_file, alt_snp_index, variant_stride, alt_row.data());

                uint8_t* ref_observed_ptr = impute_missing ? nullptr : ref_observed.data();
                uint8_t* alt_observed_ptr = impute_missing ? nullptr : alt_observed.data();

                decode_standardized_ld_bed_row(ref_row.data(),
                                               sample_groups,
                                               num_samples,
                                               allele_frequencies[ref_snp_index],
                                               impute_missing,
                                               lookup,
                                               ref_values.data(),
                                               ref_observed_ptr);

                decode_standardized_ld_bed_row(alt_row.data(),
                                               sample_groups,
                                               num_samples,
                                               allele_frequencies[alt_snp_index],
                                               impute_missing,
                                               lookup,
                                               alt_values.data(),
                                               alt_observed_ptr);

                T cross_product = blas_dot(ref_values.data(),
                                           alt_values.data(),
                                           num_samples);

                int denominator = num_samples;
                if (!impute_missing) {
                    denominator = 0;
                    for (int sample_pos = 0; sample_pos < num_samples; ++sample_pos) {
                        if (ref_observed[sample_pos] && alt_observed[sample_pos]) {
                            ++denominator;
                        }
                    }
                }

                ld_data[pair_idx] = denominator > 0 ?
                    cross_product / static_cast<T>(denominator) : T(0);
            }
    #ifdef _OPENMP
        }
    #endif
}


template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
compute_ut_ld_from_bed(std::string bed_filename,
                       const int* snp_indices,
                       const int* ld_boundaries_end,
                       const int64_t* ld_indptr,
                       int num_snps,
                       const int* sample_indices,
                       int total_samples,
                       int num_samples,
                       const T* allele_frequencies,
                       bool impute_missing,
                       T* ld_data,
                       int threads) {
    /*
        Compute the flat data array for an upper-triangular LD matrix directly
        from a SNP-major PLINK BED file.

        For each selected row `j`, this computes LD with selected columns
        `[j + 1, ld_boundaries_end[j])`. Output entries are stored in CSR order:

            ld_data[ld_indptr[j] + k] = corr(snp_indices[j], snp_indices[j + 1 + k])

        Allele frequencies are indexed by original BED variant index.
    */

    if (total_samples <= 0) {
        throw std::invalid_argument("Total BED sample count must be positive.");
    }
    if (num_snps < 0 || num_samples < 0) {
        throw std::invalid_argument("Invalid LD calculation dimensions.");
    }
    if ((num_snps > 0 && snp_indices == nullptr) ||
        (num_snps > 0 && ld_boundaries_end == nullptr) ||
        (num_snps > 0 && ld_indptr == nullptr) ||
        (num_samples > 0 && sample_indices == nullptr) ||
        allele_frequencies == nullptr ||
        (num_snps > 0 && ld_indptr[num_snps] > 0 && ld_data == nullptr)) {
        throw std::invalid_argument("Null pointer passed to compute_ut_ld_from_bed.");
    }

    if (num_snps == 0) {
        return;
    }

    if (ld_indptr[0] != 0) {
        throw std::invalid_argument("LD indptr must start at zero.");
    }

    for (int j = 0; j < num_snps; ++j) {
        const int snp_index = snp_indices[j];
        const int row_start = j + 1;
        const int row_end = ld_boundaries_end[j];
        const int row_len = std::max(row_end - row_start, 0);
        const int64_t stored_row_len = ld_indptr[j + 1] - ld_indptr[j];

        if (snp_index < 0) {
            throw std::out_of_range("SNP index must be non-negative.");
        }
        if (row_end < 0 || row_end > num_snps) {
            throw std::out_of_range("LD boundary end is outside the selected SNP range.");
        }
        if (ld_indptr[j + 1] < ld_indptr[j] ||
            stored_row_len != static_cast<int64_t>(row_len)) {
            throw std::invalid_argument("LD indptr is inconsistent with LD boundaries.");
        }

        const T allele_frequency = allele_frequencies[snp_index];
        if (!std::isfinite(static_cast<double>(allele_frequency)) ||
            allele_frequency < T(0) || allele_frequency > T(1)) {
            throw std::out_of_range("Allele frequencies must be in [0, 1].");
        }

    }

    validate_plink_bed_file(bed_filename);

    const std::vector<SelectedSampleByte> sample_groups =
        group_selected_samples_by_byte(sample_indices, total_samples, num_samples);

    if (num_samples == 0) {
        std::fill(ld_data, ld_data + ld_indptr[num_snps], T(0));
        return;
    }

    const std::streamoff variant_stride = plink_bed_variant_stride(total_samples);
    const size_t stride = static_cast<size_t>(variant_stride);
    const int num_threads = threads > 0 ? threads : 1;
    const PlinkDosageLookup lookup;

    const size_t values_per_variant = static_cast<size_t>(num_samples);
    const size_t num_selected_variants = static_cast<size_t>(num_snps);
    if (values_per_variant > 0 &&
        num_selected_variants > std::numeric_limits<size_t>::max() / values_per_variant) {
        throw std::length_error("Selected genotype matrix is too large to allocate.");
    }

    const size_t genotype_values_size = num_selected_variants * values_per_variant;
    std::vector<T> genotype_values(genotype_values_size);
    std::vector<uint8_t> observed_values;
    if (!impute_missing) {
        observed_values.resize(genotype_values_size);
    }

    #ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
        {
    #endif
            std::ifstream bed_file(bed_filename, std::ios::binary);
            if (!bed_file.is_open()) {
                throw std::runtime_error("Error opening BED file.");
            }

            std::vector<unsigned char> bed_row(stride);

            #ifdef _OPENMP
                #pragma omp for schedule(static)
            #endif
            for (int j = 0; j < num_snps; ++j) {
                const int snp_index = snp_indices[j];
                T* genotype_row = genotype_values.data() +
                    static_cast<size_t>(j) * values_per_variant;
                uint8_t* observed_row = impute_missing ? nullptr :
                    observed_values.data() + static_cast<size_t>(j) * values_per_variant;

                read_plink_bed_row(bed_file, snp_index, variant_stride, bed_row.data());
                decode_standardized_ld_bed_row(bed_row.data(),
                                               sample_groups,
                                               num_samples,
                                               allele_frequencies[snp_index],
                                               impute_missing,
                                               lookup,
                                               genotype_row,
                                               observed_row);
            }
    #ifdef _OPENMP
        }
    #endif

    #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(num_threads)
    #endif
    for (int j = 0; j < num_snps; ++j) {
        const int row_start = j + 1;
        const int row_end = ld_boundaries_end[j];

        if (row_end <= row_start) {
            continue;
        }

        const T* ref_values = genotype_values.data() +
            static_cast<size_t>(j) * values_per_variant;
        const uint8_t* ref_observed = impute_missing ? nullptr :
            observed_values.data() + static_cast<size_t>(j) * values_per_variant;

        for (int alt_pos = row_start; alt_pos < row_end; ++alt_pos) {
            const int64_t data_idx = ld_indptr[j] + (alt_pos - row_start);
            const T* alt_values = genotype_values.data() +
                static_cast<size_t>(alt_pos) * values_per_variant;

            T cross_product = T(0);
            int denominator = num_samples;

            if (impute_missing) {
                cross_product = dot(ref_values, alt_values, num_samples);
            }
            else {
                denominator = 0;
                const uint8_t* alt_observed = observed_values.data() +
                    static_cast<size_t>(alt_pos) * values_per_variant;

                #ifdef _OPENMP
                    #ifndef _WIN32
                        #pragma omp simd reduction(+:cross_product, denominator)
                    #endif
                #endif
                for (int sample_pos = 0; sample_pos < num_samples; ++sample_pos) {
                    cross_product += ref_values[sample_pos] * alt_values[sample_pos];
                    if (ref_observed[sample_pos] && alt_observed[sample_pos]) {
                        ++denominator;
                    }
                }
            }

            ld_data[data_idx] = denominator > 0 ?
                cross_product / static_cast<T>(denominator) : T(0);
        }
    }
}

#endif // LD_UTLS_H
