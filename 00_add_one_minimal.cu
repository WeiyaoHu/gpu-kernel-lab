#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void add_one(float* x, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        x[i] += 1.0f;
    }
}

int main()
{
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    // ① CPU memory
    float* h_x = new float[N];

    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i);
    }

    // ② GPU memory
    float* d_x;
    cudaMalloc(&d_x, bytes);

    // ③ CPU → GPU
    cudaMemcpy(
        d_x,
        h_x,
        bytes,
        cudaMemcpyHostToDevice
    );

    // ④ launch configuration
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    // ⑤ kernel launch
    add_one<<<grid_size, block_size>>>(d_x, N);

    // ⑥ GPU → CPU
    cudaMemcpy(
        h_x,
        d_x,
        bytes,
        cudaMemcpyDeviceToHost
    );

    // ⑦ correctness check
    bool correct = true;

    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>(i) + 1.0f;

        if (std::fabs(h_x[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "Correct\n" : "Wrong\n");

    // ⑧ clean up
    cudaFree(d_x);
    delete[] h_x;

    return 0;
}