#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define THREAD_COLS 4

// Launch with dim3 block(TILE_N / THREAD_COLS, TILE_M) = (8, 32).
// Each thread computes 1 x THREAD_COLS output elements.
__global__ void gemm_register_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int linear_tid = ty * blockDim.x + tx;

    const int row = blockIdx.y * TILE_M + ty;
    const int local_col_base = tx * THREAD_COLS;
    const int col_base = blockIdx.x * TILE_N + local_col_base;

    float acc[THREAD_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t)
    {
        // Cooperative load: all 256 threads together fill the complete A tile.
        for (int idx = linear_tid; idx < TILE_M * TILE_K; idx += threads_per_block)
        {
            const int r = idx / TILE_K;
            const int c = idx % TILE_K;
            const int g_row = blockIdx.y * TILE_M + r;
            const int g_col = t * TILE_K + c;

            As[r][c] = (g_row < M && g_col < K)
                ? A[g_row * K + g_col]
                : 0.0f;
        }

        // Cooperative load for B.
        for (int idx = linear_tid; idx < TILE_K * TILE_N; idx += threads_per_block)
        {
            const int r = idx / TILE_N;
            const int c = idx % TILE_N;
            const int g_row = t * TILE_K + r;
            const int g_col = blockIdx.x * TILE_N + c;

            Bs[r][c] = (g_row < K && g_col < N)
                ? B[g_row * N + g_col]
                : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k)
        {
            // One shared-memory load of A is reused across four accumulators.
            const float a = As[ty][k];

            #pragma unroll
            for (int j = 0; j < THREAD_COLS; ++j)
            {
                const int local_col = local_col_base + j;
                acc[j] += a * Bs[k][local_col];
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < THREAD_COLS; ++j)
    {
        const int col = col_base + j;
        if (row < M && col < N)
            C[row * N + col] = acc[j];
    }
}
