#ifndef SCORE_H
#define SCORE_H

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>

// Check for and include `cblas`:
#ifdef HAVE_CBLAS
    #include <cblas.h>
#endif

// Check for and include `omp`:
#ifdef _OPENMP
    #include <omp.h>
#endif

/* ----------------------------- */
bool omp_supported() {
    #ifdef _OPENMP
        return true;
    #else
        return false;
    #endif
}

bool blas_supported() {
    #ifdef HAVE_CBLAS
        return true;
    #else
        return false;
    #endif
}

template<typename T>
void axpy(T* x, T* y, T a, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] += y[i] * a;
    }
}

template<typename T>
void blas_axpy(T *y, T *x, T alpha, int size) {

    #ifdef HAVE_CBLAS
        int incx = 1;
        int incy = 1;

        if constexpr (std::is_same<T, float>::value) {
            cblas_saxpy(size, alpha, x, incx, y, incy);
        }
        else {
            cblas_daxpy(size, alpha, x, incx, y, incy);
        }
    #else
        axpy(y, x, alpha, size);
    #endif
}


template<typename T>
void calculate_scores(std::string bed_filename,
                      T* effect_sizes,
                      int* snp_indices,
                      int* sample_indices,
                      int num_samples,
                      int num_snps,
                      int num_scores,
                      T* scores,
                      int threads) {

    // ----------------------------------------------------
    // Check if file is a valid PLINK BED file
    std::ifstream initial_file(bed_filename, std::ios::binary);
    if (!initial_file.is_open()) {
        throw std::runtime_error("Error opening BED file.");
    }
    char magic_number[3];
    initial_file.read(magic_number, 3);
    if (magic_number[0] != '\x6C' || magic_number[1] != '\x1B' || magic_number[2] != '\x01') {
        throw std::runtime_error("Invalid PLINK BED file.");
    }
    initial_file.close();
    // ----------------------------------------------------

    T* local_scores = scores;
    bool use_local_scores = false;

    #ifdef _OPENMP
        #pragma omp parallel num_threads(threads)
    #endif
    {
        #ifdef _OPENMP
            if (omp_get_num_threads() > 1) {
                local_scores = new T[num_samples * num_scores];
                use_local_scores = true;
            }
        #endif

        //Open a separate file stream for each thread
        std::ifstream bed_file(bed_filename, std::ios::binary);

        #ifdef _OPENMP
            #pragma omp for schedule(runtime)
        #endif
        for (size_t i = 0; i < num_snps; ++i) {

            int snp_index = snp_indices[i];

            bed_file.seekg(3 + snp_index * ((num_samples + 3) / 4), std::ios::beg);
            size_t j = 0;
            size_t sample_counter = 0;

            while (sample_counter < num_samples) {
                unsigned char buffer;
                bed_file.read(reinterpret_cast<char*>(&buffer), 1);

                for (int b = 0; b < 4 && sample_counter < num_samples; ++b, ++j) {

                    int sample_index = sample_indices[sample_counter];

                    if (j == sample_index) {
                        int genotype = (buffer >> (b * 2)) & 0x3;
                        if (genotype != 1) { // Ignore missing genotypes

                            T decoded_genotype = static_cast<T>(genotype);

                            if (genotype > 0) {
                                decoded_genotype = abs(genotype - 3);
                            }
                            else {
                                decoded_genotype += 2;
                            }

                            blas_axpy(local_scores + sample_index * num_scores,
                                      effect_sizes + snp_index * num_scores,
                                      decoded_genotype,
                                      num_scores);
                        }
                        sample_counter++;
                    }
                }
            }
        }

        // Close the file stream for each thread
        bed_file.close();

        /* If multiple threads are used, add the local scores to the global scores
           in a critical section. */
        #ifdef _OPENMP
            if (use_local_scores) {
                #pragma omp critical
                {
                    for (size_t i = 0; i < num_samples; ++i) {
                        for (size_t j = 0; j < num_scores; ++j) {
                            scores[i * num_scores + j] += local_scores[i * num_scores + j];
                        }
                    }
                }
                delete [] local_scores;
            }
        #endif
    }

}

/*

Explore producer-consumer style implementation for reading the bed file.

#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T*> queue_;
    std::queue<int> index_queue_;
    std::mutex mutex_;
    std::condition_variable cond_;

public:
    void push(T* value, int index)
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        index_queue_.push(index);
        cond_.notify_one();
    }

    T* pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{ return !queue_.empty(); });
        T* value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    int pop_index() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{ return !index_queue_.empty(); });
        int value = index_queue_.front();
        index_queue_.pop();
        return value;
    }

};

void reader(ThreadSafeQueue<T>& queue, const std::string& bed_filename, int num_samples, int num_snps) {
    std::ifstream bed_file(bed_filename, std::ios::binary);
    for (size_t i = 0; i < num_snps; ++i) {
        T* snp_entries = new T[num_samples];
        bed_file.seekg(3 + i * ((num_samples + 3) / 4), std::ios::beg);
        size_t j = 0;
        size_t sample_counter = 0;
        while (sample_counter < num_samples) {
            unsigned char buffer;
            bed_file.read(reinterpret_cast<char*>(&buffer), 1);
            for (int b = 0; b < 4 && sample_counter < num_samples; ++b, ++j) {
                int genotype = (buffer >> (b * 2)) & 0x3;
                if (genotype != 1) { // Ignore missing genotypes
                    snp_entries[sample_counter] = static_cast<T>(genotype);
                    sample_counter++;
                }
            }
        }
        queue.push(snp_entries, i);
    }
    bed_file.close();
}

void reader(ThreadSafeQueue<std::vector<int>>& queue, const std::vector<int>& snp_indices) {
    for (int snp_index : snp_indices) {
        std::vector<int> snp_entries = // read SNP entries for snp_index
        queue.push(std::move(snp_entries));
    }
}

void worker(ThreadSafeQueue<std::vector<int>>& queue, int num_samples, int num_snps, int num_scores, T* effect_sizes, T* scores) {
    while (num_snps > 0) {

        std::vector<T> snp_entries = queue.pop();
        int snp_index = queue.pop_index();

        for (size_t i = 0; i < num_samples; ++i) {
            for (size_t j = 0; j < num_scores; ++j) {
                scores[i * num_scores + j] += snp_entries[i] * effect_sizes[snp_index * num_scores + j];
            }
        }

        num_snps--;
    }
}

ThreadSafeQueue<std::vector<int>> queue;

std::thread reader_thread(reader, std::ref(queue), snp_indices);
std::thread worker_thread(worker, std::ref(queue));

reader_thread.join();
worker_thread.join();

*/

#endif // SCORE_H
