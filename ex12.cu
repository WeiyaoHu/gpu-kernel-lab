#include <cuda_runtime.h>
#include <math_constants.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>


// ============================================================
// CUDA error check
// ============================================================

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::cerr                                             \
            << "CUDA error: "                                 \
            << cudaGetErrorString(err)                        \
            << " at "                                         \
            << __FILE__                                       \
            << ":"                                            \
            << __LINE__                                       \
            << std::endl;                                     \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)


// ============================================================
// Warp-level SUM reduction
//
// 输入：每个 lane 自己有一个 val
// 输出：lane 0 得到整个 warp 的和
// ============================================================

__device__ float warp_reduce_sum(float val)
{
    for (int offset = warpSize / 2;
         offset > 0;
         offset >>= 1)
    {
        val += __shfl_down_sync(
            0xffffffff,
            val,
            offset
        );
    }

    return val;
}


// ============================================================
// Warp-level MAX reduction
//
// 输入：每个 lane 自己有一个 val
// 输出：lane 0 得到整个 warp 的最大值
// ============================================================

__device__ float warp_reduce_max(float val)
{
    for (int offset = warpSize / 2;
         offset > 0;
         offset >>= 1)
    {
        float other =
            __shfl_down_sync(
                0xffffffff,
                val,
                offset
            );

        val = fmaxf(val, other);
    }

    return val;
}


// ============================================================
// Softmax Kernel
//
// 输入:
//      X : [rows, cols]
//
// 输出:
//      Y : [rows, cols]
//
// 一个 block 负责一整行
// ============================================================

__global__ void softmax_rows(
    const float* X,
    float* Y,
    int rows,
    int cols)
{
    // --------------------------------------------------------
    // 一个 block 对应一行
    // --------------------------------------------------------

    int row = blockIdx.x;

    if (row >= rows)
        return;


    int tid =
        threadIdx.x;

    int lane =
        tid % warpSize;

    int warp_id =
        tid / warpSize;

    int num_warps =
        (blockDim.x + warpSize - 1)
        / warpSize;


    // --------------------------------------------------------
    // Shared Memory
    //
    // 一个 block 最多 1024 threads
    // => 最多 32 warps
    //
    // warp_values 用于保存：
    // 每个 warp 的 max 或 sum
    // --------------------------------------------------------

    __shared__ float warp_values[32];

    __shared__ float block_max;
    __shared__ float block_sum;


    // ========================================================
    // Step 1
    //
    // 每个 thread 处理多个元素
    // 求自己的 local maximum
    // ========================================================

    float local_max =
        -CUDART_INF_F;


    for (int col = tid;
         col < cols;
         col += blockDim.x)
    {
        float value =
            X[row * cols + col];

        local_max =
            fmaxf(
                local_max,
                value
            );
    }


    // ========================================================
    // Step 2
    //
    // 每个 warp 内做 MAX reduction
    // ========================================================

    float warp_max =
        warp_reduce_max(local_max);


    // --------------------------------------------------------
    // 每个 warp 的 lane 0
    // 把 warp maximum 写入 shared memory
    // --------------------------------------------------------

    if (lane == 0) {
        warp_values[warp_id] =
            warp_max;
    }


    // 等所有 warp 写完
    __syncthreads();


    // ========================================================
    // Step 3
    //
    // Warp 0 对所有 warp maxima
    // 再进行一次 reduction
    // ========================================================

    if (warp_id == 0)
    {
        float value =
            (lane < num_warps)
            ? warp_values[lane]
            : -CUDART_INF_F;


        value =
            warp_reduce_max(value);


        // lane 0 得到整个 block 的 max
        if (lane == 0) {
            block_max = value;
        }
    }


    // 让整个 block 都看到 block_max
    __syncthreads();


    float max_value =
        block_max;


    // ========================================================
    // Step 4
    //
    // 计算：
    //
    // exp(x - max)
    //
    // 并得到每个 thread 的 local sum
    // ========================================================

    float local_sum =
        0.0f;


    for (int col = tid;
         col < cols;
         col += blockDim.x)
    {
        float x =
            X[row * cols + col];

        float e =
            expf(
                x - max_value
            );

        local_sum += e;
    }


    // ========================================================
    // Step 5
    //
    // Warp-level SUM reduction
    // ========================================================

    float warp_sum =
        warp_reduce_sum(local_sum);


    if (lane == 0) {
        warp_values[warp_id] =
            warp_sum;
    }


    __syncthreads();


    // ========================================================
    // Step 6
    //
    // Warp 0 对 warp sums 做最终 reduction
    // ========================================================

    if (warp_id == 0)
    {
        float value =
            (lane < num_warps)
            ? warp_values[lane]
            : 0.0f;


        value =
            warp_reduce_sum(value);


        if (lane == 0) {
            block_sum = value;
        }
    }


    __syncthreads();


    float sum_value =
        block_sum;


    // ========================================================
    // Step 7
    //
    // Normalize
    //
    // y[i] =
    // exp(x[i] - max)
    // -----------------
    //       sum
    //
    // 注意：
    //
    // 这里重新计算了一次 exp
    // 没有保存中间 exp 值
    // ========================================================

    for (int col = tid;
         col < cols;
         col += blockDim.x)
    {
        float x =
            X[row * cols + col];

        float e =
            expf(
                x - max_value
            );

        Y[row * cols + col] =
            e / sum_value;
    }
}


// ============================================================
// CPU reference Softmax
// ============================================================

void softmax_cpu(
    const std::vector<float>& X,
    std::vector<float>& Y,
    int rows,
    int cols)
{
    for (int row = 0;
         row < rows;
         row++)
    {
        // ----------------------------------------------------
        // MAX
        // ----------------------------------------------------

        float max_value =
            -INFINITY;

        for (int col = 0;
             col < cols;
             col++)
        {
            max_value =
                std::max(
                    max_value,
                    X[row * cols + col]
                );
        }


        // ----------------------------------------------------
        // SUM EXP
        // ----------------------------------------------------

        double sum =
            0.0;

        for (int col = 0;
             col < cols;
             col++)
        {
            sum +=
                std::exp(
                    static_cast<double>(
                        X[row * cols + col]
                        - max_value
                    )
                );
        }


        // ----------------------------------------------------
        // Normalize
        // ----------------------------------------------------

        for (int col = 0;
             col < cols;
             col++)
        {
            Y[row * cols + col] =
                static_cast<float>(
                    std::exp(
                        static_cast<double>(
                            X[row * cols + col]
                            - max_value
                        )
                    )
                    / sum
                );
        }
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    // ========================================================
    // 1. Problem size
    // ========================================================

    int rows =
        4096;

    int cols =
        1024;

    size_t num_elements =
        static_cast<size_t>(rows)
        * cols;

    size_t bytes =
        num_elements
        * sizeof(float);


    std::cout
        << "Rows = "
        << rows
        << "\n";

    std::cout
        << "Cols = "
        << cols
        << "\n";

    std::cout
        << "Elements = "
        << num_elements
        << "\n";

    std::cout
        << "Data size = "
        << bytes / (1024.0 * 1024.0)
        << " MB\n\n";


    // ========================================================
    // 2. Host memory
    // ========================================================

    std::vector<float> h_X(
        num_elements
    );

    std::vector<float> h_Y(
        num_elements
    );

    std::vector<float> h_reference(
        num_elements
    );


    // ========================================================
    // 3. Initialize input
    //
    // 故意让数值在 1000 左右
    //
    // 如果直接 exp(1000)
    // 很容易 overflow。
    //
    // 但 stable softmax:
    //
    // exp(x - max)
    //
    // 就不会。
    // ========================================================

    for (int row = 0;
         row < rows;
         row++)
    {
        for (int col = 0;
             col < cols;
             col++)
        {
            int pattern =
                (col * 13 + row * 7)
                % 200;

            h_X[row * cols + col]
                =
                1000.0f
                +
                0.01f
                * static_cast<float>(
                    pattern - 100
                );
        }
    }


    // ========================================================
    // 4. Allocate Device memory
    // ========================================================

    float* d_X;
    float* d_Y;


    CUDA_CHECK(
        cudaMalloc(
            &d_X,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_Y,
            bytes
        )
    );


    // ========================================================
    // 5. H2D
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_X,
            h_X.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // 6. Launch configuration
    //
    // 一个 block 负责一行
    // ========================================================

    int block_size =
        256;

    int grid_size =
        rows;


    std::cout
        << "Grid size  = "
        << grid_size
        << "\n";

    std::cout
        << "Block size = "
        << block_size
        << "\n\n";


    // ========================================================
    // 7. Warm-up
    // ========================================================

    int warmup =
        10;


    for (int i = 0;
         i < warmup;
         i++)
    {
        softmax_rows
        <<<grid_size, block_size>>>(
            d_X,
            d_Y,
            rows,
            cols
        );
    }


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // 8. Benchmark
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop)
    );


    int repeat =
        100;


    CUDA_CHECK(
        cudaEventRecord(start)
    );


    for (int i = 0;
         i < repeat;
         i++)
    {
        softmax_rows
        <<<grid_size, block_size>>>(
            d_X,
            d_Y,
            rows,
            cols
        );
    }


    CUDA_CHECK(
        cudaEventRecord(stop)
    );


    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );


    float total_ms =
        0.0f;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );


    float avg_ms =
        total_ms
        / repeat;


    std::cout
        << "========== Benchmark ==========\n";

    std::cout
        << "Total time: "
        << total_ms
        << " ms\n";

    std::cout
        << "Average kernel time: "
        << avg_ms
        << " ms\n\n";


    // ========================================================
    // 9. Copy GPU result back
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_Y.data(),
            d_Y,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // 10. CPU reference
    // ========================================================

    softmax_cpu(
        h_X,
        h_reference,
        rows,
        cols
    );


    // ========================================================
    // 11. Correctness Check
    // ========================================================

    float max_error =
        0.0f;


    for (size_t i = 0;
         i < num_elements;
         i++)
    {
        float error =
            std::fabs(
                h_Y[i]
                -
                h_reference[i]
            );

        max_error =
            std::max(
                max_error,
                error
            );
    }


    std::cout
        << "========== Correctness ==========\n";

    std::cout
        << std::scientific
        << "Max error = "
        << max_error
        << "\n";


    bool correct =
        max_error < 1e-5f;


    std::cout
        << (
            correct
            ? "Correct!"
            : "Wrong!"
        )
        << "\n\n";


    // ========================================================
    // 12. Check row sums
    //
    // Softmax 每一行应该：
    //
    // sum ≈ 1
    // ========================================================

    std::cout
        << "========== Row Sum Check ==========\n";


    for (int row = 0;
         row < 3;
         row++)
    {
        double sum =
            0.0;

        for (int col = 0;
             col < cols;
             col++)
        {
            sum +=
                h_Y[row * cols + col];
        }


        std::cout
            << "Row "
            << row
            << " sum = "
            << std::fixed
            << std::setprecision(8)
            << sum
            << "\n";
    }


    // ========================================================
    // 13. Clean up
    // ========================================================

    CUDA_CHECK(
        cudaEventDestroy(start)
    );

    CUDA_CHECK(
        cudaEventDestroy(stop)
    );

    CUDA_CHECK(
        cudaFree(d_X)
    );

    CUDA_CHECK(
        cudaFree(d_Y)
    );


    return 0;
}
