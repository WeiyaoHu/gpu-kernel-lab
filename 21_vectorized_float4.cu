#include <cuda_runtime.h>

// A minimal demonstration of vectorized global-memory access.
// One CUDA thread copies four consecutive FP32 values with one float4 load/store.
// BLOCK_SIZE here is still a CUDA thread-block size; vectorization changes the
// amount of data handled by each thread, not the physical DRAM bandwidth.
__global__ void vectorized_copy_float4(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = tid * 4;

    if (base + 3 < n)
    {
        // cudaMalloc returns sufficiently aligned base pointers and base is a
        // multiple of four floats, so these accesses are 16-byte aligned.
        const float4 v = reinterpret_cast<const float4*>(x)[tid];
        reinterpret_cast<float4*>(y)[tid] = v;
    }
    else
    {
        // Tail handling when n is not divisible by 4.
        for (int i = base; i < n; ++i)
            y[i] = x[i];
    }
}

// Example launch:
// int threads = 256;
// int vec_items = (N + 3) / 4;
// int blocks = (vec_items + threads - 1) / threads;
// vectorized_copy_float4<<<blocks, threads>>>(d_x, d_y, N);
