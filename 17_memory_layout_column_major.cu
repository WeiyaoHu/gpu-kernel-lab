#include <cuda_runtime.h>
#include <iostream>

#define ROWS 4
#define COLS 6

// =====================================================
// Column-major kernel
// =====================================================
__global__ void multiply_by_2(float* A)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ROWS && col < COLS)
    {
        // Column-major:
        // 同一列连续存储
        //
        // idx = col * ROWS + row
        int idx = col * ROWS + row;

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
    // Column-major 内存布局：
    //
    // 1 7 13 19
    // 2 8 14 20
    // 3 9 15 21
    // ...


    float h_A[N];


    // =====================================================
    // 按 column-major 方式初始化
    // =====================================================

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            // 逻辑上的矩阵元素
            //
            // row=0,col=0 -> 1
            // row=0,col=1 -> 2
            // ...
            // row=1,col=0 -> 7

            float value =
                row * COLS + col + 1;

            // Column-major 地址
            int idx =
                col * ROWS + row;

            h_A[idx] = value;
        }
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
    // 按逻辑矩阵形式打印
    // =====================================================

    std::cout << "Column-major result:\n\n";

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            int idx =
                col * ROWS + row;

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