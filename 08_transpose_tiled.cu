#define TILE_DIM 32

__global__ void transpose_tiled(
    const float* input,
    float* output,
    int width,
    int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x =
        blockIdx.x * TILE_DIM
        + threadIdx.x;

    int y =
        blockIdx.y * TILE_DIM
        + threadIdx.y;

    // -------------------------
    // Coalesced read
    // -------------------------

    if (x < width && y < height) {

        tile[threadIdx.y][threadIdx.x]
            =
        input[y * width + x];
    }

    __syncthreads();

    // -------------------------
    // Swap block coordinates
    // -------------------------

    int out_x =
        blockIdx.y * TILE_DIM
        + threadIdx.x;

    int out_y =
        blockIdx.x * TILE_DIM
        + threadIdx.y;

    // -------------------------
    // Coalesced write
    // -------------------------

    if (out_x < height && out_y < width) {

        output[out_y * height + out_x]
            =
        tile[threadIdx.x][threadIdx.y];
    }
}