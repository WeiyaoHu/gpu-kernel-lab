#define TILE 16

__global__ void gemm_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // -----------------------------------
    // Shared Memory Tiles
    // -----------------------------------

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];


    // -----------------------------------
    // Thread coordinates
    // -----------------------------------

    int tx = threadIdx.x;
    int ty = threadIdx.y;


    // -----------------------------------
    // Global C coordinates
    // -----------------------------------

    int row =
        blockIdx.y * TILE + ty;

    int col =
        blockIdx.x * TILE + tx;


    // 每个 thread 最终负责一个 C[row][col]
    float sum = 0.0f;


    // -----------------------------------
    // 沿 K 方向遍历 tiles
    // -----------------------------------

    int num_tiles =
        (K + TILE - 1) / TILE;


    for (int t = 0; t < num_tiles; t++)
    {
        // ===============================
        // 1. Load A Tile
        // ===============================

        int A_col =
            t * TILE + tx;

        if (row < M && A_col < K)
        {
            As[ty][tx] =
                A[row * K + A_col];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }


        // ===============================
        // 2. Load B Tile
        // ===============================

        int B_row =
            t * TILE + ty;

        if (B_row < K && col < N)
        {
            Bs[ty][tx] =
                B[B_row * N + col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }


        // ===============================
        // 3. 等整个 Tile 加载完成
        // ===============================

        __syncthreads();


        // ===============================
        // 4. 使用 Shared Memory 计算
        // ===============================

        for (int k = 0; k < TILE; k++)
        {
            sum +=
                As[ty][k]
                *
                Bs[k][tx];
        }


        // ===============================
        // 5. 必须等大家用完当前 tile
        //
        // 否则有 thread 可能已经开始
        // 覆盖 As/Bs 加载下一块
        // ===============================

        __syncthreads();
    }


    // -----------------------------------
    // Store result
    // -----------------------------------

    if (row < M && col < N)
    {
        C[row * N + col] =
            sum;
    }
}