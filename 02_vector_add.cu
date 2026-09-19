#include <cuda_runtime.h>
#include <iostream>
#include <cmath>


// ============================================================
// CUDA Kernel
// C[i] = A[i] + B[i]
// ============================================================
__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


int main()
{
    // ========================================================
    // 1. Problem size
    // ========================================================

    int N = 1 << 20;

    size_t bytes =
        N * sizeof(float);


    // ========================================================
    // 2. Allocate Host memory
    // ========================================================

    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];


    // ========================================================
    // 3. Initialize input
    // ========================================================

    for (int i = 0; i < N; i++) {

        h_A[i] = static_cast<float>(i);

        h_B[i] = static_cast<float>(2 * i);
    }


    // ========================================================
    // 4. Allocate Device memory
    // ========================================================

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);


    // ========================================================
    // 5. Host → Device
    // ========================================================

    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        bytes,
        cudaMemcpyHostToDevice
    );


    // ========================================================
    // 6. Launch configuration
    // ========================================================

    int block_size = 256;

    int grid_size =
        (N + block_size - 1)
        / block_size;


    std::cout
        << "N = "
        << N
        << std::endl;

    std::cout
        << "Block size = "
        << block_size
        << std::endl;

    std::cout
        << "Grid size = "
        << grid_size
        << std::endl;


    // ========================================================
    // 7. Warm-up
    // ========================================================

    int warmup = 10;

    for (int i = 0; i < warmup; i++) {

        vector_add<<<grid_size, block_size>>>(
            d_A,
            d_B,
            d_C,
            N
        );
    }

    cudaDeviceSynchronize();


    // ========================================================
    // 8. Create CUDA Events
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // ========================================================
    // 9. Benchmark
    // ========================================================

    int repeat = 100;

    cudaEventRecord(start);


    for (int i = 0; i < repeat; i++) {

        vector_add<<<grid_size, block_size>>>(
            d_A,
            d_B,
            d_C,
            N
        );
    }


    cudaEventRecord(stop);

    cudaEventSynchronize(stop);


    // ========================================================
    // 10. Timing
    // ========================================================

    float total_ms = 0.0f;

    cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    );

    float avg_ms =
        total_ms / repeat;


    std::cout
        << "\n========== Benchmark ==========\n";

    std::cout
        << "Total time: "
        << total_ms
        << " ms\n";

    std::cout
        << "Average kernel time: "
        << avg_ms
        << " ms\n";


    // ========================================================
    // 11. Device → Host
    // ========================================================

    cudaMemcpy(
        h_C,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );


    // ========================================================
    // 12. Correctness Check
    // ========================================================

    bool correct = true;

    for (int i = 0; i < N; i++) {

        float expected =
            h_A[i] + h_B[i];

        if (
            std::fabs(
                h_C[i] - expected
            ) > 1e-5f
        ) {

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
                << h_C[i]
                << std::endl;

            correct = false;

            break;
        }
    }


    std::cout
        << "\n========== Correctness ==========\n";

    std::cout
        << (correct ? "Correct!" : "Wrong!")
        << std::endl;


    // ========================================================
    // 13. Clean up
    // ========================================================

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;


    return 0;
}