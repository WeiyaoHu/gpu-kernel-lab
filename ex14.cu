#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)    \
                  << " at " << __FILE__ << ":" << __LINE__       \
                  << std::endl;                                    \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

__device__ float warp_reduce_sum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Returns the complete block sum to every thread in the block.
__device__ float block_reduce_sum(float val)
{
    __shared__ float warp_sums[32];
    __shared__ float block_sum;

    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }

    __syncthreads();

    if (warp_id == 0) {
        float warp_val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        warp_val = warp_reduce_sum(warp_val);

        if (lane == 0) {
            block_sum = warp_val;
        }
    }

    __syncthreads();
    return block_sum;
}

// One block processes one row.
__global__ void layernorm_rows(
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    int rows,
    int cols,
    float epsilon)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    float local_sum = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += X[row * cols + col];
    }

    float sum = block_reduce_sum(local_sum);
    float mean = sum / static_cast<float>(cols);

    float local_sq_sum = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        float x = X[row * cols + col];
        float diff = x - mean;
        local_sq_sum += diff * diff;
    }

    float sq_sum = block_reduce_sum(local_sq_sum);
    float variance = sq_sum / static_cast<float>(cols);
    float inv_std = rsqrtf(variance + epsilon);

    for (int col = tid; col < cols; col += blockDim.x) {
        float x = X[row * cols + col];
        float normalized = (x - mean) * inv_std;
        Y[row * cols + col] = normalized * gamma[col] + beta[col];
    }
}

void layernorm_cpu(
    const std::vector<float>& X,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    std::vector<float>& Y,
    int rows,
    int cols,
    float epsilon)
{
    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += X[row * cols + col];
        }

        double mean = sum / cols;
        double sq_sum = 0.0;

        for (int col = 0; col < cols; ++col) {
            double d = static_cast<double>(X[row * cols + col]) - mean;
            sq_sum += d * d;
        }

        double var = sq_sum / cols;
        double inv_std = 1.0 / std::sqrt(var + epsilon);

        for (int col = 0; col < cols; ++col) {
            double normalized =
                (static_cast<double>(X[row * cols + col]) - mean) * inv_std;
            Y[row * cols + col] = static_cast<float>(
                normalized * gamma[col] + beta[col]);
        }
    }
}

int main()
{
    int rows = 4096;
    int cols = 1024;
    float epsilon = 1e-5f;

    size_t count = static_cast<size_t>(rows) * cols;
    size_t bytes = count * sizeof(float);

    std::vector<float> h_X(count);
    std::vector<float> h_Y(count);
    std::vector<float> h_ref(count);
    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            h_X[row * cols + col] =
                0.01f * static_cast<float>(((row * 17 + col * 13) % 401) - 200);
        }
    }

    for (int col = 0; col < cols; ++col) {
        h_gamma[col] = 1.0f + 0.001f * static_cast<float>(col % 17);
        h_beta[col] = 0.001f * static_cast<float>((col % 11) - 5);
    }

    float *d_X = nullptr, *d_Y = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, bytes));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = rows;

    for (int i = 0; i < 10; ++i) {
        layernorm_rows<<<grid_size, block_size>>>(
            d_X, d_gamma, d_beta, d_Y, rows, cols, epsilon);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int repeat = 100;
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        layernorm_rows<<<grid_size, block_size>>>(
            d_X, d_gamma, d_beta, d_Y, rows, cols, epsilon);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    std::cout << "Average LayerNorm kernel time: "
              << total_ms / repeat << " ms\n";

    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, bytes, cudaMemcpyDeviceToHost));

    layernorm_cpu(h_X, h_gamma, h_beta, h_ref, rows, cols, epsilon);

    float max_error = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        max_error = std::max(max_error, std::fabs(h_Y[i] - h_ref[i]));
    }

    std::cout << "Max error: " << max_error << "\n";
    std::cout << (max_error < 2e-4f ? "Correct!\n" : "Check numerical error.\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));

    return 0;
}
