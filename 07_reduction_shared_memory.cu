#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void reduce_sum(
    const float* X,
    float* partial,
    int N)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < N) {
        sdata[tid] = X[i];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}


int main()
{
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    // -------------------------
    // Host input
    // -------------------------

    float* h_X = new float[N];

    for (int i = 0; i < N; i++) {
        h_X[i] = 1.0f;
    }

    // -------------------------
    // Device input
    // -------------------------

    float* d_X;

    cudaMalloc(
        &d_X,
        bytes
    );

    cudaMemcpy(
        d_X,
        h_X,
        bytes,
        cudaMemcpyHostToDevice
    );

    // -------------------------
    // Launch config
    // -------------------------

    int block_size = 256;

    int grid_size =
        (N + block_size - 1)
        / block_size;

    // 每个 block 输出一个 partial sum
    float* d_partial;

    cudaMalloc(
        &d_partial,
        grid_size * sizeof(float)
    );

    float* h_partial =
        new float[grid_size];

    // -------------------------
    // Warm-up
    // -------------------------

    for (int i = 0; i < 10; i++) {

        reduce_sum<<<grid_size, block_size>>>(
            d_X,
            d_partial,
            N
        );
    }

    cudaDeviceSynchronize();

    // -------------------------
    // Benchmark
    // -------------------------

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int repeat = 100;

    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {

        reduce_sum<<<grid_size, block_size>>>(
            d_X,
            d_partial,
            N
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;

    cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    );

    float avg_ms =
        total_ms / repeat;

    std::cout
        << "Average kernel time: "
        << avg_ms
        << " ms\n";

    // -------------------------
    // Copy partial sums back
    // -------------------------

    cudaMemcpy(
        h_partial,
        d_partial,
        grid_size * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // -------------------------
    // CPU final reduction
    // -------------------------

    float gpu_result = 0.0f;

    for (int i = 0; i < grid_size; i++) {
        gpu_result += h_partial[i];
    }

    // -------------------------
    // Correctness
    // -------------------------

    float expected =
        static_cast<float>(N);

    std::cout
        << "GPU result: "
        << gpu_result
        << "\n";

    std::cout
        << "Expected: "
        << expected
        << "\n";

    bool correct =
        std::fabs(
            gpu_result - expected
        ) < 1e-3f;

    std::cout
        << (correct ? "Correct!" : "Wrong!")
        << std::endl;

    // -------------------------
    // Clean up
    // -------------------------

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_X);
    cudaFree(d_partial);

    delete[] h_X;
    delete[] h_partial;

    return 0;
}