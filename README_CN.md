# GPU Kernel Lab

[English](./README.md) | 中文

这是一个用于记录 CUDA / Triton GPU Kernel 学习与优化过程的实践仓库。重点不是只把代码“跑起来”，而是理解 GPU 的执行模型、数据移动、内存层级、GEMM 优化、Tensor Core，以及这些知识如何进一步连接到 Transformer / LLM Kernel。

详细的 Stage 3–4 概念整理见 [`STAGE3_4_NOTES_CN.md`](./STAGE3_4_NOTES_CN.md)。

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| Stage 1 | GPU 与 CUDA 基础 | ✅ 完成 |
| Stage 2 | CUDA Kernel 性能优化 | ✅ 完成 |
| Stage 3 | GEMM / Tiling / Tensor Core / cuBLAS | ✅ 完成 |
| Stage 4 | Triton | ✅ 完成 |
| Stage 5 | Transformer / LLM Kernel | 🚧 进行中 |
| Stage 6 | Serving / AI Infrastructure | 计划中 |
| Stage 7 | CUTLASS / CuTe | 计划中 |
| Stage 8 | AI for Kernel Optimization | 计划中 |

---

## 核心执行模型

```text
CUDA 软件模型：
Kernel -> Grid -> Block -> Thread

NVIDIA 执行视角：
GPU -> SM -> Warp -> Lane/Thread -> Execution Units

数据路径：
Global Memory -> L2/L1 -> Shared Memory -> Registers -> Compute
```

目前形成的核心认识包括：

- CUDA thread 与 CUDA Core 不是固定 1:1 对应。
- 一个 warp 通常包含 32 个 threads；lane 是 thread 在所属 warp 内的位置编号。
- register 是 thread 私有；shared memory 是 block 内共享。
- global memory 重点看 coalescing / transactions；shared memory 重点看 bank conflict。
- occupancy 用于 latency hiding，但越高不代表一定越快。
- 性能优化最终必须回到 benchmark / profiler，而不是只凭直觉。
- 高性能 GEMM 和 AI kernel 的关键之一是减少数据移动、提高数据复用。

---

# Stage 1–2：CUDA 基础与性能优化

已经覆盖：

```text
Kernel launch / Host-Device flow
CUDA Events benchmark
Vector Add / SAXPY / ReLU
Kernel Fusion
Memory Coalescing / Stride Benchmark
Shared Memory
Matrix Transpose
Bank Conflict / Padding
Occupancy / Register Pressure
Warp Divergence
Reduction / Warp Shuffle
Softmax
Histogram / Atomic
LayerNorm
Nsight Systems / Nsight Compute
```

核心工作流：

```text
Correctness
-> Benchmark
-> Profile
-> 提出瓶颈假设
-> 修改 kernel
-> 再次 Benchmark
```

---

# Stage 3：GEMM / Tensor Core

矩阵乘：

```text
C[M,N] = A[M,K] @ B[K,N]
```

学习路线：

```text
Naive GEMM
-> Shared-Memory Tiling
-> Register Tiling / Thread Coarsening
-> Vectorized Access
-> Loop Unrolling / ILP
-> Double Buffering / Async Copy
-> Tensor Core / WMMA
-> cuBLAS Baseline
```

### Shared-Memory Tiling

```text
Global A/B
-> cooperative coalesced load
-> Shared-memory tiles
-> 数据复用
-> Register / FMA
-> C tile
```

### Register Tiling

从：

```text
1 thread -> 1 output
```

进一步变成：

```text
1 thread -> 多个 output accumulators
```

以更多 register 为代价，减少 shared-memory traffic 并提升 ILP。

### Async Pipeline

Double buffering 只解决“下一块数据放哪里”；真正的 load/compute overlap 还需要 async copy + software pipeline：

```text
compute tile t
      ||
load tile t+1
```

### Tensor Core / WMMA

普通 CUDA Core 更接近 scalar FMA；Tensor Core 面向 matrix MMA：

```text
D = A @ B + C
```

WMMA 通过一个 warp 共同持有 matrix fragments 并执行 `mma_sync`。

### cuBLAS

cuBLAS 是 NVIDIA 已经写好的高性能线性代数库。标准 GEMM 工程上通常优先使用 cuBLAS/cuBLASLt；自己写 CUDA/Triton GEMM 则适合学习、研究、特殊 fusion/layout 或自定义算法。

---

# Stage 4：Triton

Triton 的核心抽象不是单个 thread，而是 tile/program 级任务。

最准确的区分：

```text
program
= 一个独立的计算任务实例

BLOCK_SIZE / BLOCK_M/N/K
= 这个任务处理的逻辑数据 tile 大小

num_warps
= 执行这个 program 的 warp 资源

num_stages
= software pipeline 的深度
```

所以 `BLOCK_SIZE=256` 更接近“这个 program 逻辑处理 256 个数据位置”，而不是“启动 256 个 CUDA threads”。

已经实现：

```text
Vector Add
2D Matrix Add
Row Sum Reduction
Softmax
LayerNorm
GEMM / tl.dot
Autotune
RMSNorm
```

### Triton GEMM

一个 program 可以同时：

```text
读取 A tile
读取 B tile
沿 K 循环
执行 tl.dot
累加 C tile
最后写回 C tile
```

所以 program 不是“属于某个输入矩阵的一块数据”，而是一个完整的 tile-level computation task。

### `tl.dot`

`tl.dot` 表达 tile-level matrix multiplication。最终是否走 Tensor Core / MMA 等硬件路径，由 dtype、GPU 架构、tile shape 和 compiler lowering 共同决定。

也可以不使用 `tl.dot`，自己用普通 multiply-add / outer-product 风格表达 GEMM，但最终机器指令仍由 Triton compiler 生成。

### Ping-Pong / `num_stages`

常规 Triton 不需要像 CUDA 一样手写 `buffer[0] / buffer[1] / current / next`。通常通过规则的 K-tile loop 和 `num_stages` 让 compiler 安排 software pipelining。

### Autotune vs cuBLAS

```text
Triton Autotune：
kernel 是自己写的
-> 自动 benchmark 多组 tile / warps / stages
-> 选择最快配置

cuBLAS：
NVIDIA 已经把 GEMM kernel 写好
-> 直接调用
```

因此更准确的比较是：

```text
cuBLAS
vs
自己写的 Triton GEMM + Autotune
```

---

# 文件索引

## CUDA 基础与优化

`00_`–`15_`：从基础 kernel、benchmark、coalescing、transpose、reduction 到 Softmax / Histogram / LayerNorm。

## GEMM / Tensor Core / cuBLAS

| 文件 | 内容 |
|---|---|
| `16_memory_layout_row_major.cu` | Row-major 地址映射 |
| `17_memory_layout_column_major.cu` | Column-major 地址映射 |
| `18_gemm_naive.cu` | Naive GEMM |
| `19_gemm_tiled.cu` | Shared-memory tiled GEMM |
| `20_gemm_register_tiled.cu` | Register tiling / thread coarsening |
| `21_vectorized_float4.cu` | `float4` vectorized access |
| `22_gemm_async_pipeline.cu` | Ping-pong + async pipeline |
| `23_gemm_wmma.cu` | WMMA / Tensor Core |
| `24_cublas_sgemm.cu` | cuBLAS SGEMM baseline |

## Triton

| 文件 | 内容 |
|---|---|
| `25_triton_vector_add.py` | Program / offsets / mask |
| `26_triton_matrix_add.py` | 2D indexing / broadcasting |
| `27_triton_row_sum.py` | `tl.sum` reduction |
| `28_triton_softmax.py` | Softmax |
| `29_triton_layernorm.py` | LayerNorm |
| `30_triton_gemm.py` | `tl.dot` GEMM |
| `31_triton_gemm_autotune.py` | GEMM autotune |
| `32_triton_rmsnorm.py` | RMSNorm / LLM kernel 起点 |

---

# 编译与运行

普通 CUDA：

```bash
nvcc -O3 file.cu -o app
```

cuBLAS：

```bash
nvcc -O3 24_cublas_sgemm.cu -lcublas -o cublas_gemm
```

WMMA / async-copy 示例需要根据 GPU 选择合适架构，例如：

```bash
nvcc -O3 -arch=sm_80 file.cu -o app
```

Triton：

```bash
python 25_triton_vector_add.py
```

Profiling：

```bash
nsys profile ./app
ncu ./app
ncu --set full ./app
```

---

# 当前性能思维

```text
任务如何映射到 block / program / warp？
输出 tile 是什么？
数据如何 Global -> Shared -> Register？
访存是否 coalesced / aligned？
数据有没有被充分 reuse？
shared memory 是否有 bank conflict？
register pressure 与 occupancy 如何平衡？
能否用 pipeline 隐藏 memory latency？
是 memory-bound 还是 compute-bound？
Tensor Core 是否合适？
标准 GEMM 是否应该直接使用 cuBLAS？
自定义 Triton kernel 哪些参数应该 autotune？
```

核心仍然是：

```text
Measure -> Diagnose -> Optimize -> Measure Again
```
