#ifndef PLINK_BED_UTILS_H
#define PLINK_BED_UTILS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct SelectedSampleByte {
    int byte_index;
    uint8_t selected_mask;
    std::array<int, 4> output_index;
};

struct PlinkDosageLookup {
    std::array<std::array<int8_t, 4>, 256> dosage;
    std::array<uint8_t, 256> nonzero_mask;

    PlinkDosageLookup() : dosage{}, nonzero_mask{} {
        for (int byte = 0; byte < 256; ++byte) {
            uint8_t mask = 0;

            for (int slot = 0; slot < 4; ++slot) {
                const int code = (byte >> (2 * slot)) & 0x3;
                int8_t value;

                // PLINK BED SNP-major two-bit encoding:
                // 00 -> homozygous first allele, 01 -> missing,
                // 10 -> heterozygous, 11 -> homozygous second allele.
                // For A1-like dosages this maps to 2, NA, 1, 0.
                if (code == 0) {
                    value = 2;
                }
                else if (code == 2) {
                    value = 1;
                }
                else if (code == 3) {
                    value = 0;
                }
                else {
                    value = -1;
                }

                dosage[byte][slot] = value;
                if (value > 0) {
                    mask |= static_cast<uint8_t>(1u << slot);
                }
            }

            nonzero_mask[byte] = mask;
        }
    }
};

inline std::streamoff plink_bed_variant_stride(int total_samples) {
    return (static_cast<std::streamoff>(total_samples) + 3) / 4;
}

inline void validate_plink_bed_file(const std::string& bed_filename) {
    std::ifstream initial_file(bed_filename, std::ios::binary);
    if (!initial_file.is_open()) {
        throw std::runtime_error("Error opening BED file.");
    }

    char magic_number[3];
    initial_file.read(magic_number, 3);
    if (!initial_file) {
        throw std::runtime_error("Could not read BED file header.");
    }
    if (magic_number[0] != '\x6C' || magic_number[1] != '\x1B' || magic_number[2] != '\x01') {
        throw std::runtime_error("Invalid PLINK BED file.");
    }
}

inline std::vector<SelectedSampleByte>
group_selected_samples_by_byte(const int* sample_indices,
                               int total_samples,
                               int num_samples) {

    std::vector<SelectedSampleByte> groups;
    groups.reserve(static_cast<size_t>(num_samples));

    int previous_sample_index = -1;

    for (int sample_pos = 0; sample_pos < num_samples; ++sample_pos) {
        const int sample_index = sample_indices[sample_pos];

        if (sample_index < 0 || sample_index >= total_samples) {
            throw std::out_of_range("Sample index is outside the BED sample range.");
        }

        if (sample_index <= previous_sample_index) {
            throw std::invalid_argument("Sample indices must be sorted and unique.");
        }
        previous_sample_index = sample_index;

        const int byte_index = sample_index / 4;
        const int slot = sample_index % 4;

        if (groups.empty() || groups.back().byte_index != byte_index) {
            SelectedSampleByte group;
            group.byte_index = byte_index;
            group.selected_mask = 0;
            group.output_index.fill(-1);
            groups.push_back(group);
        }

        groups.back().selected_mask |= static_cast<uint8_t>(1u << slot);
        groups.back().output_index[slot] = sample_pos;
    }

    return groups;
}

inline void read_plink_bed_row(std::ifstream& bed_file,
                               int snp_index,
                               std::streamoff variant_stride,
                               unsigned char* row_buffer) {
    if (snp_index < 0) {
        throw std::out_of_range("SNP index must be non-negative.");
    }

    bed_file.seekg(3 + static_cast<std::streamoff>(snp_index) * variant_stride,
                   std::ios::beg);
    bed_file.read(reinterpret_cast<char*>(row_buffer),
                  static_cast<std::streamsize>(variant_stride));

    if (!bed_file) {
        throw std::runtime_error("Could not read SNP row from BED file.");
    }
}

#endif // PLINK_BED_UTILS_H
