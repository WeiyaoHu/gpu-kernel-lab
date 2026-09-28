#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << "\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error code: " << static_cast<int>(s) << "\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

int main()
{
    const int M = 512;
    const int N = 512;
    const int K = 512;

    std::vector<float> h_A(static_cast<size_t>(M) * K, 1.0f);
    std::vector<float> h_B(static_cast<size_t>(K) * N, 1.0f);
    std::vector<float> h_C(static_cast<size_t>(M) * N);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major semantics. For row-major A(MxK), B(KxN), C(MxN),
    // compute C^T = B^T * A^T by swapping A/B and M/N in the column-major call.
    auto run_gemm = [&]() {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N));
    };

    for (int i = 0; i < 10; ++i) run_gemm();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int repeat = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) run_gemm();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double avg_ms = total_ms / repeat;
    const double gflops = (2.0 * M * N * K) / (avg_ms * 1.0e6);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // All inputs are one, so every C element should equal K.
    float max_error = 0.0f;
    for (float v : h_C)
        max_error = std::max(max_error, std::fabs(v - static_cast<float>(K)));

    std::cout << "cuBLAS SGEMM average: " << avg_ms << " ms\n";
    std::cout << "Throughput: " << gflops << " GFLOP/s\n";
    std::cout << "Max error: " << max_error << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
