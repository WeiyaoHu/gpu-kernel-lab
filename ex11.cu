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


__global__ void reduce_v3(
    const float* X,
    float* partial,
    int N)
{
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;

    int i =
        blockIdx.x * blockDim.x * 2
        + tid;

    // ----------------------
    // 每个 thread 先处理两个元素
    // ----------------------

    float val = 0.0f;

    if (i < N)
        val += X[i];

    if (i + blockDim.x < N)
        val += X[i + blockDim.x];


    // ----------------------
    // Warp 内 reduction
    // ----------------------

    val = warp_reduce_sum(val);


    // lane id
    int lane =
        tid % warpSize;

    // warp id
    int warp_id =
        tid / warpSize;


    // ----------------------
    // 每个 warp 的 lane 0
    // 写出一个 warp sum
    // ----------------------

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }

    __syncthreads();


    // ----------------------
    // 第一个 warp
    // 对所有 warp sums 再 reduction
    // ----------------------

    int num_warps =
        (blockDim.x + warpSize - 1)
        / warpSize;

    float block_sum = 0.0f;

    if (warp_id == 0) {

        block_sum =
            (lane < num_warps)
            ? warp_sums[lane]
            : 0.0f;

        block_sum =
            warp_reduce_sum(block_sum);
    }


    // ----------------------
    // block 最终结果
    // ----------------------

    if (tid == 0) {
        partial[blockIdx.x] = block_sum;
    }
}