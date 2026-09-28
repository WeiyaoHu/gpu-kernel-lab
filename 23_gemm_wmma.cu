#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Teaching WMMA kernel:
//   * one warp/block computes one 16x16 C tile
//   * A/B are FP16, accumulator/output are FP32
//   * M, N, K must be multiples of 16
//
// This makes the contrast explicit:
// scalar CUDA-core-style FMA loops -> warp-level matrix MMA.
__global__ void wmma_gemm(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K)
{
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16)
    {
        const half* A_tile = A + (tile_row * 16) * K + k0;
        const half* B_tile = B + k0 * N + tile_col * 16;

        wmma::load_matrix_sync(a_frag, A_tile, K);
        wmma::load_matrix_sync(b_frag, B_tile, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* C_tile = C + (tile_row * 16) * N + tile_col * 16;
    wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}

// Example launch:
// dim3 block(32);                      // one warp
// dim3 grid(N / 16, M / 16);
// wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
