#include <cuda_runtime.h>
#include <iostream>

#define ROWS 4
#define COLS 6

// =====================================================
// Row-major kernel
// =====================================================
__global__ void multiply_by_2(float* A)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ROWS && col < COLS)
    {
        // Row-major:
        // 同一行连续存储
        //
        // idx = row * COLS + col
        int idx = row * COLS + col;

        A[idx] *= 2.0f;
    }
}


int main()
{
    const int N = ROWS * COLS;
    const size_t bytes = N * sizeof(float);


    // =====================================================
    // CPU 上的矩阵
    // =====================================================

    // 逻辑矩阵：
    //
    // 1   2   3   4   5   6
    // 7   8   9  10  11  12
    // 13 14  15  16  17  18
    // 19 20  21  22  23  24
    //
    // Row-major 内存布局：
    //
    // 1 2 3 4 5 6 7 8 9 10 ... 24

    float h_A[N];

    for (int i = 0; i < N; i++)
    {
        h_A[i] = i + 1;
    }


    // =====================================================
    // 分配 GPU 显存
    // =====================================================

    float* d_A;

    cudaMalloc(&d_A, bytes);


    // =====================================================
    // CPU -> GPU
    // =====================================================

    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice
    );


    // =====================================================
    // 启动 kernel
    // =====================================================

    dim3 block(6, 4);
    dim3 grid(1, 1);

    multiply_by_2<<<grid, block>>>(d_A);

    cudaDeviceSynchronize();


    // =====================================================
    // GPU -> CPU
    // =====================================================

    cudaMemcpy(
        h_A,
        d_A,
        bytes,
        cudaMemcpyDeviceToHost
    );


    // =====================================================
    // 打印结果
    // =====================================================

    std::cout << "Row-major result:\n\n";

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            int idx = row * COLS + col;

            std::cout << h_A[idx] << "\t";
        }

        std::cout << "\n";
    }


    // =====================================================
    // 释放显存
    // =====================================================

    cudaFree(d_A);

    return 0;
}