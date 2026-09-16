#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define NUM_BINS 256

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)    \
                  << " at " << __FILE__ << ":" << __LINE__       \
                  << std::endl;                                    \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

// Shared-memory privatized histogram.
// Each block accumulates into its own local histogram, then merges
// the block-local counts into the global histogram with atomics.
__global__ void histogram_shared(
    const unsigned char* X,
    unsigned int* hist,
    int N)
{
    __shared__ unsigned int local_hist[NUM_BINS];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (tid < NUM_BINS) {
        local_hist[tid] = 0;
    }

    __syncthreads();

    if (i < N) {
        unsigned int bin = static_cast<unsigned int>(X[i]);
        atomicAdd(&local_hist[bin], 1u);
    }

    __syncthreads();

    if (tid < NUM_BINS) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}

int main()
{
    int N = 1 << 24;
    size_t bytes = static_cast<size_t>(N) * sizeof(unsigned char);

    std::vector<unsigned char> h_X(N);
    std::vector<unsigned int> h_hist(NUM_BINS, 0);
    std::vector<unsigned int> reference(NUM_BINS, 0);

    for (int i = 0; i < N; ++i) {
        h_X[i] = static_cast<unsigned char>((i * 13 + 7) % NUM_BINS);
        reference[h_X[i]]++;
    }

    unsigned char* d_X = nullptr;
    unsigned int* d_hist = nullptr;

    CUDA_CHECK(cudaMalloc(&d_X, bytes));
    CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    // Warm-up.
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    histogram_shared<<<grid_size, block_size>>>(d_X, d_hist, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark. Histogram modifies its output, so clear before every launch.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int repeat = 20;
    CUDA_CHECK(cudaEventRecord(start));

    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
        histogram_shared<<<grid_size, block_size>>>(d_X, d_hist, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    std::cout << "Average histogram iteration: "
              << total_ms / repeat << " ms\n";

    // Run once from a clean state for correctness.
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    histogram_shared<<<grid_size, block_size>>>(d_X, d_hist, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_hist.data(),
        d_hist,
        NUM_BINS * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int bin = 0; bin < NUM_BINS; ++bin) {
        if (h_hist[bin] != reference[bin]) {
            std::cerr << "Mismatch at bin " << bin
                      << ": expected " << reference[bin]
                      << ", got " << h_hist[bin] << "\n";
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "Correct!\n" : "Wrong!\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_hist));

    return correct ? 0 : 1;
}
