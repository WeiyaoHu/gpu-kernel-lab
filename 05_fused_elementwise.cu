#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void elementwise(
    const float* X,
    float* Y,
    float a,
    float b,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {

        float z = a * X[i] + b;

        Y[i] =
            z > 0.0f
            ? z
            : 0.0f;
    }
}

int main()
{
    int N = 1 << 20;

    size_t bytes =
        N * sizeof(float);

    float a = 2.0f;
    float b = -1.0f;

    // Host
    float* h_X = new float[N];
    float* h_Y = new float[N];

    for (int i = 0; i < N; i++) {

        h_X[i] =
            static_cast<float>(i % 200 - 100);
    }

    // Device
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

    // Launch config
    int block_size = 256;

    int grid_size =
        (N + block_size - 1)
        / block_size;

    // -------------------------
    // Warm-up
    // -------------------------

    for (int i = 0; i < 10; i++) {

        elementwise<<<grid_size, block_size>>>(
            d_X,
            d_Y,
            a,
            b,
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

        elementwise<<<grid_size, block_size>>>(
            d_X,
            d_Y,
            a,
            b,
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
    // Copy result
    // -------------------------

    cudaMemcpy(
        h_Y,
        d_Y,
        bytes,
        cudaMemcpyDeviceToHost
    );

    // -------------------------
    // Correctness
    // -------------------------

    bool correct = true;

    for (int i = 0; i < N; i++) {

        float z =
            a * h_X[i] + b;

        float expected =
            z > 0.0f
            ? z
            : 0.0f;

        if (
            std::fabs(
                h_Y[i] - expected
            ) > 1e-5f
        ) {
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

    return 0;
}