#ifndef SCORE_H
#define SCORE_H

#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../../utils/linear_algebra_utils.hpp"
#include "../../utils/plink_bed_utils.hpp"

template<typename T>
void calculate_scores(std::string bed_filename,
                      const T* effect_sizes,
                      const int* snp_indices,
                      const int* sample_indices,
                      int total_samples,
                      int num_samples,
                      int num_snps,
                      int num_scores,
                      T* scores,
                      int threads,
                      const T* allele_frequencies = nullptr,
                      bool standardize_genotype = false,
                      bool impute_missing = false) {

    if (total_samples <= 0) {
        throw std::invalid_argument("Total BED sample count must be positive.");
    }
    if (num_samples < 0 || num_snps < 0 || num_scores <= 0) {
        throw std::invalid_argument("Invalid score calculation dimensions.");
    }
    if (effect_sizes == nullptr || snp_indices == nullptr ||
        sample_indices == nullptr || scores == nullptr) {
        throw std::invalid_argument("Null pointer passed to calculate_scores.");
    }
    if ((standardize_genotype || impute_missing) && allele_frequencies == nullptr) {
        throw std::invalid_argument("Allele frequencies are required for standardization or missing-value imputation.");
    }

    validate_plink_bed_file(bed_filename);

    for (int snp_pos = 0; snp_pos < num_snps; ++snp_pos) {
        if (snp_indices[snp_pos] < 0) {
            throw std::out_of_range("SNP index must be non-negative.");
        }
        if (allele_frequencies != nullptr &&
            (allele_frequencies[snp_pos] < T(0) || allele_frequencies[snp_pos] > T(1))) {
            throw std::out_of_range("Allele frequencies must be in [0, 1].");
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

    const int num_sample_groups = static_cast<int>(sample_groups.size());
    const bool use_frequency_adjustment = standardize_genotype || impute_missing;

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
        for (int group_pos = 0; group_pos < num_sample_groups; ++group_pos) {
            const SelectedSampleByte& group = sample_groups[group_pos];

            for (int block_pos = 0; block_pos < block_snps; ++block_pos) {
                const unsigned char byte =
                    bed_block[static_cast<size_t>(block_pos) * stride + group.byte_index];

                const uint8_t selected_nonzero_mask =
                    lookup.nonzero_mask[byte] & group.selected_mask;

                if (!use_frequency_adjustment && selected_nonzero_mask == 0) {
                    continue;
                }

                const T* snp_effects =
                    effect_sizes + static_cast<size_t>(block_start + block_pos) * num_scores;
                const T allele_frequency = use_frequency_adjustment ?
                    allele_frequencies[block_start + block_pos] : T(0);
                const T mean_dosage = static_cast<T>(2) * allele_frequency;
                const T variance = mean_dosage * (static_cast<T>(1) - allele_frequency);
                const T inverse_sd = standardize_genotype && variance > T(0) ?
                    static_cast<T>(1) / static_cast<T>(std::sqrt(static_cast<double>(variance))) :
                    T(0);

                const int output0 = group.output_index[0];
                const int output1 = group.output_index[1];
                const int output2 = group.output_index[2];
                const int output3 = group.output_index[3];

                const int8_t dosage0 = lookup.dosage[byte][0];
                const int8_t dosage1 = lookup.dosage[byte][1];
                const int8_t dosage2 = lookup.dosage[byte][2];
                const int8_t dosage3 = lookup.dosage[byte][3];

                T value0 = T(0);
                T value1 = T(0);
                T value2 = T(0);
                T value3 = T(0);

                if (standardize_genotype) {
                    value0 = dosage0 >= 0 ? (static_cast<T>(dosage0) - mean_dosage) * inverse_sd : T(0);
                    value1 = dosage1 >= 0 ? (static_cast<T>(dosage1) - mean_dosage) * inverse_sd : T(0);
                    value2 = dosage2 >= 0 ? (static_cast<T>(dosage2) - mean_dosage) * inverse_sd : T(0);
                    value3 = dosage3 >= 0 ? (static_cast<T>(dosage3) - mean_dosage) * inverse_sd : T(0);
                }
                else if (impute_missing) {
                    value0 = dosage0 >= 0 ? static_cast<T>(dosage0) : mean_dosage;
                    value1 = dosage1 >= 0 ? static_cast<T>(dosage1) : mean_dosage;
                    value2 = dosage2 >= 0 ? static_cast<T>(dosage2) : mean_dosage;
                    value3 = dosage3 >= 0 ? static_cast<T>(dosage3) : mean_dosage;
                }
                else {
                    value0 = dosage0 > 0 ? static_cast<T>(dosage0) : T(0);
                    value1 = dosage1 > 0 ? static_cast<T>(dosage1) : T(0);
                    value2 = dosage2 > 0 ? static_cast<T>(dosage2) : T(0);
                    value3 = dosage3 > 0 ? static_cast<T>(dosage3) : T(0);
                }

                T* scores0 = output0 >= 0 && value0 != T(0) ?
                             scores + static_cast<size_t>(output0) * num_scores : nullptr;
                T* scores1 = output1 >= 0 && value1 != T(0) ?
                             scores + static_cast<size_t>(output1) * num_scores : nullptr;
                T* scores2 = output2 >= 0 && value2 != T(0) ?
                             scores + static_cast<size_t>(output2) * num_scores : nullptr;
                T* scores3 = output3 >= 0 && value3 != T(0) ?
                             scores + static_cast<size_t>(output3) * num_scores : nullptr;

                for (int score_idx = 0; score_idx < num_scores; ++score_idx) {
                    const T effect = snp_effects[score_idx];

                    if (scores0 != nullptr) {
                        scores0[score_idx] += value0 * effect;
                    }
                    if (scores1 != nullptr) {
                        scores1[score_idx] += value1 * effect;
                    }
                    if (scores2 != nullptr) {
                        scores2[score_idx] += value2 * effect;
                    }
                    if (scores3 != nullptr) {
                        scores3[score_idx] += value3 * effect;
                    }
                }
            }
        }
    }

}

#endif // SCORE_H
