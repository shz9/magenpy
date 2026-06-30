#ifndef VARIANT_UTILS_H
#define VARIANT_UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../utils/linear_algebra_utils.hpp"
#include "../../utils/plink_bed_utils.hpp"

template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
missing_genotype_value() {
    return static_cast<T>(-1);
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
missing_genotype_value() {
    return std::numeric_limits<T>::quiet_NaN();
}

template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, void>::type
extract_genotype_matrix(std::string bed_filename,
                        const int* snp_indices,
                        const int* sample_indices,
                        int total_samples,
                        int num_samples,
                        int num_snps,
                        T* out,
                        int threads) {
    /*
        Extract selected PLINK BED dosages into a row-major samples-by-SNPs
        matrix. Dosage decoding follows the shared convention:
            00 -> 2, 01 -> missing, 10 -> 1, 11 -> 0.

        Missing values are encoded as -1 for integer outputs and NaN for
        floating-point outputs.
    */

    if (total_samples <= 0) {
        throw std::invalid_argument("Total BED sample count must be positive.");
    }
    if (num_samples < 0 || num_snps < 0) {
        throw std::invalid_argument("Invalid genotype extraction dimensions.");
    }
    if ((num_snps > 0 && snp_indices == nullptr) ||
        (num_samples > 0 && sample_indices == nullptr) ||
        (num_samples > 0 && num_snps > 0 && out == nullptr)) {
        throw std::invalid_argument("Null pointer passed to extract_genotype_matrix.");
    }

    validate_plink_bed_file(bed_filename);

    for (int snp_pos = 0; snp_pos < num_snps; ++snp_pos) {
        if (snp_indices[snp_pos] < 0) {
            throw std::out_of_range("SNP index must be non-negative.");
        }
    }

    const std::vector<SelectedSampleByte> sample_groups =
        group_selected_samples_by_byte(sample_indices, total_samples, num_samples);

    if (num_samples == 0 || num_snps == 0) {
        return;
    }

    const PlinkDosageLookup lookup;
    const std::streamoff variant_stride = plink_bed_variant_stride(total_samples);
    const size_t stride = static_cast<size_t>(variant_stride);
    const int num_threads = threads > 0 ? threads : 1;
    constexpr size_t target_block_bytes = 32u * 1024u * 1024u;
    const int snps_per_block = std::max<int>(
        1,
        static_cast<int>(std::min<size_t>(
            static_cast<size_t>(std::numeric_limits<int>::max()),
            std::max<size_t>(1, target_block_bytes / std::max<size_t>(1, stride))
        ))
    );

    std::vector<unsigned char> bed_block;
    bed_block.resize(static_cast<size_t>(snps_per_block) * stride);

    std::ifstream bed_file(bed_filename, std::ios::binary);
    if (!bed_file.is_open()) {
        throw std::runtime_error("Error opening BED file.");
    }

    const T missing_value = missing_genotype_value<T>();

    for (int block_start = 0; block_start < num_snps; block_start += snps_per_block) {
        const int block_snps = std::min(snps_per_block, num_snps - block_start);

        for (int block_pos = 0; block_pos < block_snps; ++block_pos) {
            read_plink_bed_row(bed_file,
                               snp_indices[block_start + block_pos],
                               variant_stride,
                               bed_block.data() + static_cast<size_t>(block_pos) * stride);
        }

        #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(num_threads)
        #endif
        for (int block_pos = 0; block_pos < block_snps; ++block_pos) {
            const int snp_pos = block_start + block_pos;
            const unsigned char* bed_row =
                bed_block.data() + static_cast<size_t>(block_pos) * stride;

            for (const SelectedSampleByte& group : sample_groups) {
                const unsigned char byte = bed_row[group.byte_index];

                for (int slot = 0; slot < 4; ++slot) {
                    const int output_sample = group.output_index[slot];
                    if (output_sample < 0) {
                        continue;
                    }

                    const int8_t dosage = lookup.dosage[byte][slot];
                    out[static_cast<size_t>(output_sample) * static_cast<size_t>(num_snps) +
                        static_cast<size_t>(snp_pos)] =
                        dosage >= 0 ? static_cast<T>(dosage) : missing_value;
                }
            }
        }
    }
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
compute_gwa_linear_stats(std::string bed_filename,
                         const int* snp_indices,
                         const int* sample_indices,
                         int total_samples,
                         int num_samples,
                         int num_snps,
                         const T* centered_phenotype,
                         bool standardize_genotype,
                         T* allele_frequencies,
                         int* n_per_snp,
                         T* x_dot_y,
                         T* sum_x_sq,
                         int threads) {
    /*
        Compute per-variant sufficient statistics for simple linear GWAS from
        a SNP-major PLINK BED file without materializing the genotype matrix.

        For each selected SNP, this computes:
            allele_frequencies[snp_pos]
            n_per_snp[snp_pos]
            x_dot_y[snp_pos] = x_j' y_centered
            sum_x_sq[snp_pos] = x_j' x_j

        Missing genotypes are assigned x=0 after centering/scaling. This
        matches the existing xarray GWAS path, where missing genotypes are
        filled with 0 after centering or standardization.
    */

    if (total_samples <= 0) {
        throw std::invalid_argument("Total BED sample count must be positive.");
    }
    if (num_samples < 0 || num_snps < 0) {
        throw std::invalid_argument("Invalid GWAS calculation dimensions.");
    }
    if ((num_snps > 0 && snp_indices == nullptr) ||
        (num_samples > 0 && sample_indices == nullptr) ||
        (num_samples > 0 && centered_phenotype == nullptr) ||
        (num_snps > 0 && allele_frequencies == nullptr) ||
        (num_snps > 0 && n_per_snp == nullptr) ||
        (num_snps > 0 && x_dot_y == nullptr) ||
        (num_snps > 0 && sum_x_sq == nullptr)) {
        throw std::invalid_argument("Null pointer passed to compute_gwa_linear_stats.");
    }

    validate_plink_bed_file(bed_filename);

    for (int snp_pos = 0; snp_pos < num_snps; ++snp_pos) {
        if (snp_indices[snp_pos] < 0) {
            throw std::out_of_range("SNP index must be non-negative.");
        }
    }

    const std::vector<SelectedSampleByte> sample_groups =
        group_selected_samples_by_byte(sample_indices, total_samples, num_samples);

    if (num_snps == 0) {
        return;
    }

    if (num_samples == 0) {
        for (int snp_pos = 0; snp_pos < num_snps; ++snp_pos) {
            allele_frequencies[snp_pos] = std::numeric_limits<T>::quiet_NaN();
            n_per_snp[snp_pos] = 0;
            x_dot_y[snp_pos] = std::numeric_limits<T>::quiet_NaN();
            sum_x_sq[snp_pos] = T(0);
        }
        return;
    }

    const PlinkDosageLookup lookup;
    const std::streamoff variant_stride = plink_bed_variant_stride(total_samples);
    const size_t stride = static_cast<size_t>(variant_stride);
    const int num_threads = threads > 0 ? threads : 1;
    constexpr size_t target_block_bytes = 32u * 1024u * 1024u;
    const int snps_per_block = std::max<int>(
        1,
        static_cast<int>(std::min<size_t>(
            static_cast<size_t>(std::numeric_limits<int>::max()),
            std::max<size_t>(1, target_block_bytes / std::max<size_t>(1, stride))
        ))
    );

    std::vector<unsigned char> bed_block;
    bed_block.resize(static_cast<size_t>(snps_per_block) * stride);

    std::ifstream bed_file(bed_filename, std::ios::binary);
    if (!bed_file.is_open()) {
        throw std::runtime_error("Error opening BED file.");
    }

    for (int block_start = 0; block_start < num_snps; block_start += snps_per_block) {
        const int block_snps = std::min(snps_per_block, num_snps - block_start);

        for (int block_pos = 0; block_pos < block_snps; ++block_pos) {
            read_plink_bed_row(bed_file,
                               snp_indices[block_start + block_pos],
                               variant_stride,
                               bed_block.data() + static_cast<size_t>(block_pos) * stride);
        }

        #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(num_threads)
        #endif
        for (int block_pos = 0; block_pos < block_snps; ++block_pos) {
            T dosage_sum = T(0);
            T dosage_sum_sq = T(0);
            T dosage_dot_y = T(0);
            T observed_y_sum = T(0);
            int non_missing = 0;
            const unsigned char* bed_row =
                bed_block.data() + static_cast<size_t>(block_pos) * stride;

            for (const SelectedSampleByte& group : sample_groups) {
                const unsigned char byte = bed_row[group.byte_index];

                for (int slot = 0; slot < 4; ++slot) {
                    const int sample_pos = group.output_index[slot];
                    if (sample_pos < 0) {
                        continue;
                    }

                    const int8_t dosage = lookup.dosage[byte][slot];
                    if (dosage >= 0) {
                        const T dosage_value = static_cast<T>(dosage);
                        const T phenotype_value = centered_phenotype[sample_pos];
                        dosage_sum += dosage_value;
                        dosage_sum_sq += dosage_value * dosage_value;
                        dosage_dot_y += dosage_value * phenotype_value;
                        observed_y_sum += phenotype_value;
                        ++non_missing;
                    }
                }
            }

            const int snp_pos = block_start + block_pos;
            n_per_snp[snp_pos] = non_missing;

            if (non_missing == 0) {
                allele_frequencies[snp_pos] = std::numeric_limits<T>::quiet_NaN();
                x_dot_y[snp_pos] = std::numeric_limits<T>::quiet_NaN();
                sum_x_sq[snp_pos] = T(0);
                continue;
            }

            const T mean_dosage = dosage_sum / static_cast<T>(non_missing);
            allele_frequencies[snp_pos] =
                dosage_sum / (static_cast<T>(2) * static_cast<T>(non_missing));

            const T centered_sum_sq =
                dosage_sum_sq - dosage_sum * dosage_sum / static_cast<T>(non_missing);
            const T centered_dot_y = dosage_dot_y - mean_dosage * observed_y_sum;

            if (centered_sum_sq <= T(0)) {
                x_dot_y[snp_pos] = T(0);
                sum_x_sq[snp_pos] = T(0);
            }
            else if (standardize_genotype) {
                const T scale = static_cast<T>(
                    std::sqrt(static_cast<double>(non_missing) /
                              static_cast<double>(centered_sum_sq))
                );
                x_dot_y[snp_pos] = centered_dot_y * scale;
                sum_x_sq[snp_pos] = static_cast<T>(non_missing);
            }
            else {
                x_dot_y[snp_pos] = centered_dot_y;
                sum_x_sq[snp_pos] = centered_sum_sq;
            }
        }
    }
}


template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
compute_variant_stats(std::string bed_filename,
                      const int* snp_indices,
                      const int* sample_indices,
                      int total_samples,
                      int num_samples,
                      int num_snps,
                      T* allele_frequencies,
                      int* n_per_snp,
                      int threads) {
    /*
        Compute A1-like allele frequencies and non-missing sample counts from a
        SNP-major PLINK BED file.

        The output arrays are indexed by selected SNP position:
            allele_frequencies[snp_pos]
            n_per_snp[snp_pos]

        Dosage decoding follows the shared PLINK BED convention:
            00 -> 2, 01 -> missing, 10 -> 1, 11 -> 0.
        Allele frequency is dosage_sum / (2 * non_missing_count). If a variant
        has no observed genotypes in the selected samples, its frequency is NaN
        and its sample count is 0.
    */

    if (total_samples <= 0) {
        throw std::invalid_argument("Total BED sample count must be positive.");
    }
    if (num_samples < 0 || num_snps < 0) {
        throw std::invalid_argument("Invalid variant-stat calculation dimensions.");
    }
    if ((num_snps > 0 && snp_indices == nullptr) ||
        (num_samples > 0 && sample_indices == nullptr) ||
        allele_frequencies == nullptr ||
        n_per_snp == nullptr) {
        throw std::invalid_argument("Null pointer passed to compute_variant_stats.");
    }

    validate_plink_bed_file(bed_filename);

    for (int snp_pos = 0; snp_pos < num_snps; ++snp_pos) {
        if (snp_indices[snp_pos] < 0) {
            throw std::out_of_range("SNP index must be non-negative.");
        }
    }

    const std::vector<SelectedSampleByte> sample_groups =
        group_selected_samples_by_byte(sample_indices, total_samples, num_samples);

    if (num_snps == 0) {
        return;
    }

    if (num_samples == 0) {
        for (int snp_pos = 0; snp_pos < num_snps; ++snp_pos) {
            allele_frequencies[snp_pos] = std::numeric_limits<T>::quiet_NaN();
            n_per_snp[snp_pos] = 0;
        }
        return;
    }

    const PlinkDosageLookup lookup;
    const std::streamoff variant_stride = plink_bed_variant_stride(total_samples);
    const size_t stride = static_cast<size_t>(variant_stride);
    const int num_threads = threads > 0 ? threads : 1;
    constexpr size_t target_block_bytes = 32u * 1024u * 1024u;
    const int snps_per_block = std::max<int>(
        1,
        static_cast<int>(std::min<size_t>(
            static_cast<size_t>(std::numeric_limits<int>::max()),
            std::max<size_t>(1, target_block_bytes / std::max<size_t>(1, stride))
        ))
    );

    std::vector<unsigned char> bed_block;
    bed_block.resize(static_cast<size_t>(snps_per_block) * stride);

    std::ifstream bed_file(bed_filename, std::ios::binary);
    if (!bed_file.is_open()) {
        throw std::runtime_error("Error opening BED file.");
    }

    for (int block_start = 0; block_start < num_snps; block_start += snps_per_block) {
        const int block_snps = std::min(snps_per_block, num_snps - block_start);

        for (int block_pos = 0; block_pos < block_snps; ++block_pos) {
            read_plink_bed_row(bed_file,
                               snp_indices[block_start + block_pos],
                               variant_stride,
                               bed_block.data() + static_cast<size_t>(block_pos) * stride);
        }

        #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(num_threads)
        #endif
        for (int block_pos = 0; block_pos < block_snps; ++block_pos) {
            int dosage_sum = 0;
            int non_missing = 0;
            const unsigned char* bed_row =
                bed_block.data() + static_cast<size_t>(block_pos) * stride;

            for (const SelectedSampleByte& group : sample_groups) {
                const unsigned char byte = bed_row[group.byte_index];

                for (int slot = 0; slot < 4; ++slot) {
                    if (group.output_index[slot] < 0) {
                        continue;
                    }

                    const int8_t dosage = lookup.dosage[byte][slot];
                    if (dosage >= 0) {
                        dosage_sum += static_cast<int>(dosage);
                        ++non_missing;
                    }
                }
            }

            const int snp_pos = block_start + block_pos;
            n_per_snp[snp_pos] = non_missing;
            allele_frequencies[snp_pos] = non_missing > 0 ?
                static_cast<T>(dosage_sum) / (static_cast<T>(2) * static_cast<T>(non_missing)) :
                std::numeric_limits<T>::quiet_NaN();
        }
    }
}

#endif // VARIANT_UTILS_H
