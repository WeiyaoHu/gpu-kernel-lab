# GPU Kernel Lab

English | [中文](./README_CN.md)

A hands-on CUDA and Triton learning repository focused on GPU execution, memory movement, kernel optimization, GEMM, Tensor Cores, and the transition toward Transformer/LLM kernels.

The repository is intentionally educational: examples progress from readable baselines to more hardware-aware implementations, with an emphasis on understanding *why* an optimization works rather than only reproducing optimized code.

## Current Progress

| Stage | Topic | Status |
|---|---|---|
| Stage 1 | GPU & CUDA Foundations | ✅ Completed |
| Stage 2 | CUDA Kernel Performance Optimization | ✅ Completed |
| Stage 3 | GEMM / Tiling / Tensor Cores / cuBLAS | ✅ Completed |
| Stage 4 | Triton | ✅ Completed |
| Stage 5 | Transformer / LLM Kernels | 🚧 In progress |
| Stage 6 | Serving / AI Infrastructure | Planned |
| Stage 7 | CUTLASS / CuTe | Planned |
| Stage 8 | AI for Kernel Optimization | Planned |

For detailed Stage 3–4 conceptual notes in Chinese, see [`STAGE3_4_NOTES_CN.md`](./STAGE3_4_NOTES_CN.md).

---

## Core Mental Model

```text
CUDA software model:
Kernel -> Grid -> Block -> Thread

NVIDIA execution view:
GPU -> SM -> Warp -> Lane/Thread -> Execution Units

Performance data path:
Global Memory -> L2/L1 -> Shared Memory -> Registers -> Compute
```

Key ideas learned so far:

- A CUDA thread is not permanently mapped 1:1 to a CUDA core.
- A warp contains 32 lanes/threads on current NVIDIA GPUs.
- Registers are thread-private; shared memory is block-local.
- Coalescing concerns global-memory transactions across a warp.
- Bank conflicts concern shared-memory banks.
- Occupancy helps latency hiding, but higher occupancy is not automatically faster.
- Benchmark first, profile second, optimize based on evidence.
- Data reuse and data movement are central to high-performance GEMM and AI kernels.

---

# Stage 1–2: CUDA Foundations and Kernel Optimization

The first two stages cover:

```text
Basic kernel launch / host-device flow
CUDA Events and benchmarking
Vector Add / SAXPY / ReLU
Kernel fusion
Memory coalescing and stride experiments
Shared-memory tiling
Matrix transpose
Bank conflicts and padding
Occupancy and register pressure
Warp divergence
Reduction optimization
Warp shuffle
Softmax
Histogram / atomics
LayerNorm
Nsight Systems / Nsight Compute workflow
```

The core optimization loop is:

```text
Correctness
-> Benchmark
-> Profile
-> Form a hypothesis
-> Modify kernel
-> Benchmark again
```

---

# Stage 3: GEMM and Tensor Cores

GEMM combines almost every earlier concept in one workload:

```text
C[M,N] = A[M,K] @ B[K,N]
```

The progression in this repo is:

```text
Naive GEMM
-> Shared-Memory Tiling
-> Register Tiling / Thread Coarsening
-> Vectorized Access
-> Loop Unrolling / ILP
-> Double Buffering / Async Copy
-> Tensor Core / WMMA
-> cuBLAS baseline
```

### Shared-Memory Tiling

A block computes a C tile and cooperatively loads A/B tiles:

```text
Global A/B
   -> coalesced cooperative load
Shared-memory tiles
   -> reuse
Registers / FMA
   ->
C tile
```

### Register Tiling

One thread computes multiple output elements so a value loaded from shared memory can update multiple accumulators. This increases reuse and ILP but also increases register pressure.

### Async Pipeline

Double buffering is a storage strategy; asynchronous copy is the mechanism that allows the next tile to be issued before it is needed. The goal is:

```text
compute tile t
      ||
load tile t+1
```

### Tensor Cores

WMMA changes the inner compute abstraction from scalar FMA to warp-level matrix MMA:

```text
load_matrix_sync
-> fragments
-> mma_sync
-> FP32 accumulator fragment
-> store_matrix_sync
```

### cuBLAS

cuBLAS is NVIDIA's pre-built high-performance linear algebra library. It is useful both in production and as an industrial baseline for custom GEMM kernels.

---

# Stage 4: Triton

Triton uses a higher-level, tile/program-centric programming model.

The most important distinction is:

```text
Triton program = one independent computation-task instance
BLOCK_SIZE / BLOCK_M/N/K = logical data-tile size
num_warps = execution resources used by a program
num_stages = software-pipeline depth
```

A program is not simply "a block of data" and `BLOCK_SIZE=256` does not mean 256 CUDA threads.

Examples included here cover:

```text
Vector Add
2D Matrix Add / Broadcasting
Row Reduction
Softmax
LayerNorm
GEMM with tl.dot
GEMM Autotune
RMSNorm
```

For GEMM, a Triton program can read tiles from both A and B, perform multiple K-tile iterations, accumulate a C tile, and finally store it. `tl.dot` expresses tile-level matrix multiplication; the compiler decides the final lowering based on dtype, architecture, and configuration.

Autotune does **not** write the kernel. It benchmarks configurations for a kernel we wrote, such as:

```text
BLOCK_M / BLOCK_N / BLOCK_K
num_warps
num_stages
```

This differs from cuBLAS, where the optimized GEMM implementation itself is already provided by NVIDIA.

---

# Repository Map

## CUDA Foundations / Optimization

| File | Topic |
|---|---|
| `00_add_one_minimal.cu` | Minimal CUDA kernel and host/device flow |
| `01_add_one_benchmark.cu` | CUDA Events and benchmarking |
| `02_vector_add.cu` | Coalesced vector addition |
| `03_stride_benchmark.cu` | Stride vs effective bandwidth |
| `04_saxpy.cu` | Arithmetic intensity / FMA intuition |
| `05_relu.cu` | Elementwise activation |
| `06_fused_elementwise.cu` | Kernel fusion |
| `07_reduction_shared_memory.cu` | Shared-memory reduction |
| `08_transpose_naive.cu` | Naive transpose |
| `09_transpose_tiled.cu` | Shared-memory transpose |
| `10_transpose_padded.cu` | Bank-conflict-free transpose |
| `11_reduction_two_elements.cu` | Two elements per thread |
| `12_reduction_warp_shuffle.cu` | Warp-shuffle reduction |
| `13_softmax_rows.cu` | Row-wise stable Softmax |
| `14_histogram_shared.cu` | Atomics and privatized histogram |
| `15_layernorm_rows.cu` | Row-wise LayerNorm |

## GEMM / Tensor Core / cuBLAS

| File | Topic |
|---|---|
| `16_memory_layout_row_major.cu` | Row-major indexing |
| `17_memory_layout_column_major.cu` | Column-major indexing |
| `18_gemm_naive.cu` | Naive one-thread-per-output GEMM |
| `19_gemm_tiled.cu` | Shared-memory tiled GEMM |
| `20_gemm_register_tiled.cu` | Register tiling / thread coarsening |
| `21_vectorized_float4.cu` | `float4` vectorized global access |
| `22_gemm_async_pipeline.cu` | Ping-pong buffers + async pipeline |
| `23_gemm_wmma.cu` | WMMA / Tensor Core teaching kernel |
| `24_cublas_sgemm.cu` | cuBLAS SGEMM benchmark baseline |

## Triton

| File | Topic |
|---|---|
| `25_triton_vector_add.py` | Program / offsets / mask |
| `26_triton_matrix_add.py` | 2D indexing and broadcasting |
| `27_triton_row_sum.py` | Reduction with `tl.sum` |
| `28_triton_softmax.py` | Fused row-wise Softmax |
| `29_triton_layernorm.py` | Fused row-wise LayerNorm |
| `30_triton_gemm.py` | Tiled GEMM with `tl.dot` |
| `31_triton_gemm_autotune.py` | GEMM configuration autotuning |
| `32_triton_rmsnorm.py` | First Transformer/LLM-oriented kernel |

---

# Build / Run

CUDA examples require the CUDA Toolkit and `nvcc`. Example:

```bash
nvcc -O3 24_cublas_sgemm.cu -lcublas -o cublas_gemm
./cublas_gemm
```

For WMMA / async-copy experiments, compile for a suitable GPU architecture, for example:

```bash
nvcc -O3 -arch=sm_80 file.cu -o app
```

Triton examples require PyTorch + Triton on a supported GPU environment:

```bash
python 25_triton_vector_add.py
```

Useful profiling commands:

```bash
nvcc -O3 -Xptxas -v file.cu -o app
nsys profile ./app
ncu ./app
ncu --set full ./app
```

---

# Performance Questions I Ask Now

```text
How is work mapped to blocks/programs and warps?
What is the output tile?
How does data move Global -> Shared -> Register?
Is global memory coalesced and aligned?
Is there useful data reuse?
Are there shared-memory bank conflicts?
What are the register/shared-memory costs?
Is occupancy sufficient without sacrificing ILP?
Can load and compute overlap through a pipeline?
Is the kernel memory-bound or compute-bound?
Can Tensor Cores be used effectively?
Should this be a custom kernel or a cuBLAS call?
If using Triton, which parameters should be autotuned?
```

The guiding workflow remains:

```text
Measure -> Diagnose -> Optimize -> Measure Again
```
