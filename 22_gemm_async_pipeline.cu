#include <cuda/pipeline>

#define TILE 16

// Pedagogical async-copy + ping-pong GEMM.
// Assumptions for clarity:
//   * launch with dim3 block(TILE, TILE)
//   * M, N and K are multiples of TILE
//   * compile for an architecture that can accelerate cuda::memcpy_async
//     (Ampere/sm_80 or newer for hardware async global->shared copies)
//
// Double buffering answers "where does the next tile go?".
// cuda::memcpy_async + the pipeline answer "can the copy be issued early?".
__global__ void gemm_async_double_buffer(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    __shared__ float As[2][TILE][TILE];
    __shared__ float Bs[2][TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const int num_tiles = K / TILE;

    // Prologue: issue Tile 0 -> shared buffer 0.
    pipe.producer_acquire();
    cuda::memcpy_async(
        &As[0][ty][tx],
        &A[row * K + tx],
        sizeof(float),
        pipe);
    cuda::memcpy_async(
        &Bs[0][ty][tx],
        &B[ty * N + col],
        sizeof(float),
        pipe);
    pipe.producer_commit();

    for (int t = 0; t < num_tiles; ++t)
    {
        const int current = t & 1;
        const int next = current ^ 1;

        // Issue the next tile first. Do NOT wait immediately: the point is to
        // let this transfer make progress while the current tile is computed.
        if (t + 1 < num_tiles)
        {
            const int next_k = (t + 1) * TILE;

            pipe.producer_acquire();
            cuda::memcpy_async(
                &As[next][ty][tx],
                &A[row * K + next_k + tx],
                sizeof(float),
                pipe);
            cuda::memcpy_async(
                &Bs[next][ty][tx],
                &B[(next_k + ty) * N + col],
                sizeof(float),
                pipe);
            pipe.producer_commit();
        }

        // Wait for the oldest committed stage (the current tile).
        pipe.consumer_wait();
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[current][ty][k] * Bs[current][k][tx];

        // Everyone must finish consuming the current shared buffer before a
        // later iteration is allowed to overwrite it.
        __syncthreads();
        pipe.consumer_release();
    }

    C[row * N + col] = sum;
}
