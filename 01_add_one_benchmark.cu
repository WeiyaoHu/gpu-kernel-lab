#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// ============================================================
// CUDA Kernel
// 每个 thread 负责一个元素：x[i] += 1
// ============================================================
__global__ void add_one(float* x, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        x[i] += 1.0f;
    }
}


// ============================================================
// Main
// ============================================================
int main()
{
    // --------------------------------------------------------
    // 1. Problem size
    // --------------------------------------------------------
    int N = 1 << 20;  // 2^20 = 1,048,576 elements
    size_t bytes = N * sizeof(float);

    std::cout << "N = " << N << std::endl;
    std::cout << "Data size = "
              << bytes / (1024.0 * 1024.0)
              << " MB" << std::endl;


    // --------------------------------------------------------
    // 2. Allocate CPU (Host) memory
    // --------------------------------------------------------
    float* h_x = new float[N];

    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i);
    }


    // --------------------------------------------------------
    // 3. Allocate GPU (Device) memory
    // --------------------------------------------------------
    float* d_x;

    cudaMalloc(&d_x, bytes);


    // --------------------------------------------------------
    // 4. Copy input:
    //    CPU → GPU
    //    Host → Device
    // --------------------------------------------------------
    cudaMemcpy(
        d_x,
        h_x,
        bytes,
        cudaMemcpyHostToDevice
    );


    // --------------------------------------------------------
    // 5. Kernel launch configuration
    // --------------------------------------------------------
    int block_size = 256;

    int grid_size =
        (N + block_size - 1) / block_size;

    std::cout << "Block size = "
              << block_size << std::endl;

    std::cout << "Grid size = "
              << grid_size << std::endl;


    // ========================================================
    // 6. Warm-up
    // ========================================================

    int warmup = 10;

    for (int i = 0; i < warmup; i++) {

        add_one<<<grid_size, block_size>>>(
            d_x,
            N
        );
    }

    // 等待 warm-up 全部执行完成
    cudaDeviceSynchronize();


    // --------------------------------------------------------
    // Warm-up 已经把数据修改了 10 次。
    //
    // 为了正式 benchmark 从同样的初始数据开始，
    // 重新把原始 h_x 拷贝到 GPU。
    // --------------------------------------------------------
    cudaMemcpy(
        d_x,
        h_x,
        bytes,
        cudaMemcpyHostToDevice
    );


    // ========================================================
    // 7. Create CUDA Events
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // ========================================================
    // 8. Benchmark
    // ========================================================

    int repeat = 100;

    // 在 GPU timeline 上记录开始时间
    cudaEventRecord(start);


    // 连续运行 kernel 100 次
    for (int i = 0; i < repeat; i++) {

        add_one<<<grid_size, block_size>>>(
            d_x,
            N
        );
    }


    // 在 GPU timeline 上记录结束时间
    cudaEventRecord(stop);


    // CPU 等到 stop event 真正完成
    cudaEventSynchronize(stop);


    // --------------------------------------------------------
    // Calculate elapsed time
    // --------------------------------------------------------

    float total_ms = 0.0f;

    cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    );

    float avg_ms = total_ms / repeat;


    std::cout << "\n========== Benchmark ==========\n";

    std::cout << "Repeat: "
              << repeat << std::endl;

    std::cout << "Total kernel time: "
              << total_ms
              << " ms" << std::endl;

    std::cout << "Average kernel time: "
              << avg_ms
              << " ms" << std::endl;


    // ========================================================
    // 9. Correctness Check
    //
    // 注意：
    //
    // benchmark 已经把 x 加了 100 次。
    //
    // 为了检查“一次 add_one 是否正确”，
    // 我们重新恢复输入，再单独运行一次。
    // ========================================================

    cudaMemcpy(
        d_x,
        h_x,
        bytes,
        cudaMemcpyHostToDevice
    );


    // 单独运行一次 kernel
    add_one<<<grid_size, block_size>>>(
        d_x,
        N
    );


    // 等 GPU 完成
    cudaDeviceSynchronize();


    // --------------------------------------------------------
    // 10. GPU → CPU
    // --------------------------------------------------------

    cudaMemcpy(
        h_x,
        d_x,
        bytes,
        cudaMemcpyDeviceToHost
    );


    // --------------------------------------------------------
    // 11. Verify result
    // --------------------------------------------------------

    bool correct = true;

    for (int i = 0; i < N; i++) {

        float expected =
            static_cast<float>(i) + 1.0f;

        if (std::fabs(
                h_x[i] - expected
            ) > 1e-5f)
        {
            std::cout
                << "Mismatch at i = "
                << i
                << std::endl;

            std::cout
                << "Expected: "
                << expected
                << std::endl;

            std::cout
                << "Got: "
                << h_x[i]
                << std::endl;

            correct = false;

            break;
        }
    }


    std::cout << "\n========== Correctness ==========\n";

    if (correct) {
        std::cout << "Correct!" << std::endl;
    }
    else {
        std::cout << "Wrong!" << std::endl;
    }


    // ========================================================
    // 12. Clean up
    // ========================================================

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_x);

    delete[] h_x;


    return 0;
}