# Stage 3–4 学习笔记：GEMM、Tensor Core 与 Triton

这份笔记整理了 Stage 3（GEMM）和 Stage 4（Triton）中最重要的概念。代码示例见仓库中 `18_` 之后的文件。

## 1. 矩阵在显存中仍是一维线性存储

二维矩阵最终都映射到一段线性地址。

Row-major：

```text
idx = row * num_cols + col
```

Column-major：

```text
idx = col * num_rows + row
```

所谓 row-major / column-major 说的是**数据布局**，而不是 thread 天生按行或按列执行。thread 如何映射到数据是 kernel 自己定义的。

## 2. Naive GEMM

矩阵乘法：

```text
C[M,N] = A[M,K] @ B[K,N]
C[i,j] = sum_k A[i,k] * B[k,j]
```

最直接的 CUDA 映射是：

```text
1 thread -> 1 C[i,j]
```

每个 thread 沿 K 循环，并把 accumulator 保存在 register 中。问题是 A/B 中很多值会被同一 block 的不同 threads 反复需要，global-memory reuse 很差。

## 3. Shared-Memory Tiling

一个 block 负责一个 C tile；整个 block cooperative load A/B tiles：

```text
Global A/B
   ↓ coalesced load
Shared-memory A/B tiles
   ↓ reuse
Registers / FMA
   ↓
C tile
```

每轮 K tile 的基本结构：

```text
load A/B tile
-> __syncthreads()
-> compute
-> __syncthreads()
-> next K tile
```

第一道 barrier 保证 tile 已经装满；第二道 barrier 保证所有 threads 都使用完当前 tile，避免下一轮覆盖 shared memory。

## 4. Register Tiling / Thread Coarsening

Shared-memory tiling 解决 Global -> Shared reuse；register tiling 继续解决 Shared -> Register reuse。

例如：

```text
1 thread -> 4 C outputs
```

一个 A register value 可以同时更新 4 个 accumulators：

```text
        a
    / / | \
  b0 b1 b2 b3
   ↓  ↓  ↓  ↓
 acc0 acc1 acc2 acc3
```

收益：shared-memory traffic 下降、ILP 提高。代价：register/thread 上升，occupancy 可能下降。

## 5. Vectorized Access

`float4` 表示一个 thread 可以用更宽的 load/store 一次搬 4 个连续 FP32。

重要区别：

```text
Coalescing     -> 一个 warp 内不同 threads 的地址关系
Vectorization  -> 一个 thread 单次 load/store 的宽度
```

`float4` 不会让物理显存带宽自动变 4 倍；主要价值是减少 load/store 指令并配合 thread coarsening。地址对齐和 tail handling 仍然需要考虑。

## 6. Loop Unrolling

典型 GEMM K-loop：

```cpp
#pragma unroll
for (int k = 0; k < BK; ++k) { ... }
```

可能带来：

- loop-control overhead 下降
- 编译器 scheduling 自由度提高
- ILP 提高

代价是 code size 和 register pressure 可能增加。因此 unroll 不是“越多越好”。

## 7. Double Buffering 与 Async Copy

Double buffering 首先是**存储结构**：

```text
Buffer 0 -> current tile
Buffer 1 -> next tile
```

它只回答“下一块放哪里”，并不会自动实现 load/compute overlap。

真正的异步流水需要：

```text
Double Buffer
+
Async Copy
+
Software Pipeline
```

理想时间线：

```text
compute tile t
       ||
load tile t+1
```

在现代 CUDA 中可通过 `cuda::memcpy_async` / pipeline API，以及底层 `cp.async` 思想实现 Global -> Shared 的异步搬运。

## 8. Tensor Core / WMMA

普通 CUDA Core 风格是标量 FMA：

```text
a * b + c
```

Tensor Core 面向矩阵级 MMA：

```text
D = A @ B + C
```

WMMA 的核心流程：

```text
load_matrix_sync
-> fragment
-> mma_sync
-> accumulator fragment
-> store_matrix_sync
```

fragment 是 warp-level 数据结构：一个 16x16 tile 并不是被单个 thread 完整持有，而是分布在整个 warp 的寄存器中。

## 9. cuBLAS 的定位

cuBLAS 是 NVIDIA 已经写好的高性能线性代数库，而不是 kernel 编程语言。

```text
CUDA / Triton -> 自己写 kernel
cuBLAS        -> 直接调用 NVIDIA 的高性能 GEMM 等实现
```

通常从 host-side `main()` 或 Python/PyTorch wrapper 调用 cuBLAS/cuBLASLt，而不是在 CUDA/Triton device kernel 内部直接调用。

工程上，如果只是标准 GEMM，优先考虑 cuBLAS/cuBLASLt；如果要特殊 fusion/layout/算法，再考虑自定义 Triton/CUDA kernel。

## 10. Triton Program 的准确理解

一个 Triton `program` 不是“一块数据”，也不是“一个 CUDA thread”。

更准确地说：

```text
Triton program = 一个独立的 tile-level 计算任务实例
```

例如 GEMM 中：

```text
一个 program
-> 读取 A 的一个/多个 tile
-> 读取 B 的一个/多个 tile
-> 沿 K 累加
-> 得到并写回一个 C tile
```

而：

```text
BLOCK_SIZE / BLOCK_M / BLOCK_N / BLOCK_K
```

描述的是 program 处理的**逻辑数据 tile 大小**，不是 CUDA thread 数。

## 11. `num_warps` 与 lane

NVIDIA 一个 warp 通常是 32 threads；每个 thread 在 warp 内有一个 lane ID：0–31。

```text
thread = 执行实体
lane   = 该 thread 在所属 warp 中的位置
```

Triton 的 `num_warps` 控制一个 program 使用多少 warp 资源，而 `BLOCK_SIZE` 控制逻辑数据 tile 大小。两者不需要相等。

## 12. Triton 的 1D / 2D 数据表达

一维：

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

二维通过 broadcasting 构造 tile：

```python
offs_m[:, None]   # [BM, 1]
offs_n[None, :]   # [1, BN]
```

组合后得到 `[BM, BN]` 的逻辑地址 tensor。

## 13. Triton Reduction / Softmax / LayerNorm

CUDA 中需要手写 warp shuffle、shared memory 和同步的 reduction，在 Triton 中通常直接表达为：

```python
tl.sum(x, axis=0)
tl.max(x, axis=0)
```

Softmax：

```text
load row
-> max reduction
-> exp(x-max)
-> sum reduction
-> normalize
```

LayerNorm：

```text
load row
-> mean
-> variance
-> rsqrt
-> normalize
-> gamma/beta
```

算法没变，只是 Triton 把更多 thread/warp-level implementation 交给 compiler。

## 14. Triton GEMM 与 `tl.dot`

一个 program 通常负责一个 C tile：

```text
A tile [BM,BK]
      x
B tile [BK,BN]
      ↓
tl.dot(a,b)
      ↓
C tile [BM,BN]
```

`tl.dot` 表达 tile-level matrix multiply。最终是否使用 Tensor Core / MMA 路径由 dtype、GPU 架构、tile shape 和 compiler lowering 等共同决定。

也可以不用 `tl.dot`，自己用普通 elementwise multiply-add/outer-product 风格写 GEMM；这更接近普通 CUDA-Core-style FMA 的表达。但 Triton 最终机器指令仍由 compiler 决定。

## 15. Triton 中的 Ping-Pong / Pipeline

常规 Triton GEMM 不需要像 CUDA 一样显式写：

```text
buffer[0]
buffer[1]
current/next
commit/wait
```

通常通过规则的 tile loop 配合：

```python
num_stages=2 / 3 / ...
```

让 compiler 做 software pipelining。`num_stages` 表达流水深度，不等于源码中一定存在同名的两个/三个数组。

## 16. Autotune 与 cuBLAS 不是一回事

Triton autotune：

```text
kernel 仍然是自己写
-> 给出多个 BLOCK_M/N/K、num_warps、num_stages 候选
-> Triton benchmark
-> 选择最快配置
```

cuBLAS：

```text
GEMM kernel 已经由 NVIDIA 写好
-> 直接调用
```

所以正确比较是：

```text
cuBLAS
vs
自己写的 Triton GEMM + autotune
```

而不是“cuBLAS vs autotune”。

## 17. 当前性能思维

对一个 kernel，我现在会依次问：

```text
输出 tile 如何划分？
数据从 Global -> Shared -> Register 怎么移动？
是否有数据复用？
访存是否 coalesced / aligned？
shared memory 是否 bank conflict？
register pressure 与 occupancy 如何权衡？
能否通过 pipeline 隐藏 memory latency？
是 memory-bound 还是 compute-bound？
是否适合 Tensor Core？
是否应该自己写，还是直接使用 cuBLAS？
如果用 Triton，哪些参数值得 autotune？
```

核心仍然是：

```text
Measure -> Diagnose -> Optimize -> Measure Again
```
