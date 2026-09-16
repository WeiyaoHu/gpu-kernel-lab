#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void saxpy(
    float a,
    const float* X,
    float* Y,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        Y[i] = a * X[i] + Y[i];
    }
}

int main()
{
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float a = 2.0f;

    // Host memory
    float* h_X = new float[N];
    float* h_Y = new float[N];
    float* h_Y_original = new float[N];

    for (int i = 0; i < N; i++) {
        h_X[i] = static_cast<float>(i);
        h_Y[i] = static_cast<float>(i);
        h_Y_original[i] = h_Y[i];
    }

    // Device memory
    float* d_X;
    float* d_Y;

    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_Y, bytes);

    cudaMemcpy(
        d_X,
        h_X,
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_Y,
        h_Y,
        bytes,
        cudaMemcpyHostToDevice
    );

    // Launch configuration
    int block_size = 256;

    int grid_size =
        (N + block_size - 1)
        / block_size;

    // -------------------------
    // Warm-up
    // -------------------------

    for (int i = 0; i < 10; i++) {
        saxpy<<<grid_size, block_size>>>(
            a,
            d_X,
            d_Y,
            N
        );
    }

    cudaDeviceSynchronize();

    // SAXPY 会修改 Y，
    // 所以 benchmark 前恢复 Y
    cudaMemcpy(
        d_Y,
        h_Y_original,
        bytes,
        cudaMemcpyHostToDevice
    );

    // -------------------------
    // Benchmark
    // -------------------------

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int repeat = 100;

    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        saxpy<<<grid_size, block_size>>>(
            a,
            d_X,
            d_Y,
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
    // Correctness
    // -------------------------

    // benchmark 修改了 Y 多次，
    // 所以恢复，再单独跑一次
    cudaMemcpy(
        d_Y,
        h_Y_original,
        bytes,
        cudaMemcpyHostToDevice
    );

    saxpy<<<grid_size, block_size>>>(
        a,
        d_X,
        d_Y,
        N
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_Y,
        d_Y,
        bytes,
        cudaMemcpyDeviceToHost
    );

    bool correct = true;

    for (int i = 0; i < N; i++) {

        float expected =
            a * h_X[i]
            + h_Y_original[i];

        if (std::fabs(
                h_Y[i] - expected
            ) > 1e-5f)
        {
            correct = false;
            break;
        }
    }

    std::cout
        << (correct ? "Correct!" : "Wrong!")
        << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_X);
    cudaFree(d_Y);

    delete[] h_X;
    delete[] h_Y;
    delete[] h_Y_original;

    return 0;
}