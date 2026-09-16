__global__ void reduce_v2(
    const float* X,
    float* partial,
    int N)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;

    int i =
        blockIdx.x * (blockDim.x * 2)
        + tid;

    float sum = 0.0f;

    if (i < N)
        sum += X[i];

    if (i + blockDim.x < N)
        sum += X[i + blockDim.x];

    sdata[tid] = sum;

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}