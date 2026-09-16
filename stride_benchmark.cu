#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)    \
                  << " at " << __FILE__ << ":" << __LINE__       \
                  << std::endl;                                    \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

__global__ void strided_read(
    const float* x,
    float* y,
    int N,
    int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = x[static_cast<size_t>(i) * stride];
    }
}

float benchmark_stride(
    const float* d_x,
    float* d_y,
    int N,
    int stride,
    int block_size,
    int repeat)
{
    int grid_size = (N + block_size - 1) / block_size;

    for (int i = 0; i < 10; ++i) {
        strided_read<<<grid_size, block_size>>>(d_x, d_y, N, stride);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        strided_read<<<grid_size, block_size>>>(d_x, d_y, N, stride);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

int main()
{
    int N = 1 << 20;
    int max_stride = 32;

    size_t input_count = static_cast<size_t>(N) * max_stride;
    size_t input_bytes = input_count * sizeof(float);
    size_t output_bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> h_x(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        h_x[i] = static_cast<float>(i % 1000);
    }

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, output_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), input_bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int repeat = 100;
    int strides[] = {1, 2, 4, 8, 16, 32};

    std::cout << std::setw(10) << "Stride"
              << std::setw(18) << "Time(ms)"
              << std::setw(20) << "Effective GB/s"
              << '\n';

    for (int stride : strides) {
        float avg_ms = benchmark_stride(
            d_x, d_y, N, stride, block_size, repeat);

        double useful_bytes = 2.0 * N * sizeof(float); // one read + one write
        double effective_GBs = useful_bytes / (avg_ms / 1000.0) / 1e9;

        std::cout << std::setw(10) << stride
                  << std::setw(18) << std::fixed << std::setprecision(4) << avg_ms
                  << std::setw(20) << std::setprecision(2) << effective_GBs
                  << '\n';
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
