# GPU Kernel Lab

English | [中文](./README%28CN%29.md)

A hands-on CUDA learning repository focused on understanding how GPU kernels execute, how data moves through the memory hierarchy, and how to optimize kernels from a hardware-aware performance perspective.

This repo records my progression from basic CUDA execution and benchmarking to memory coalescing, shared-memory tiling, warp-level reduction, softmax, atomics, and LayerNorm. The next stage is GEMM and Tensor Core optimization.

---

## Current Progress

| Stage | Topic | Status |
|---|---|---|
| Stage 1 | GPU & CUDA Foundations | ✅ Completed |
| Stage 2 | CUDA Kernel Performance Optimization | ✅ Completed |
| Stage 3 | GEMM / Tiling / Tensor Cores | ⏳ Next |
| Stage 4 | Triton | Planned |
| Stage 5 | Transformer / LLM Kernels | Planned |
| Stage 6 | Serving / AI Infrastructure | Planned |
| Stage 7 | CUTLASS / CuTe | Planned |
| Stage 8 | AI for Kernel Optimization | Planned |

---

# What I Have Learned

## 1. GPU Execution Model

I started from the relationship between CUDA's software execution model and NVIDIA GPU hardware.

```text
CUDA software model:
Kernel -> Grid -> Block -> Thread

Hardware execution view:
GPU -> SM -> Warp -> Execution Units
```

Key ideas:

- A kernel launch creates a grid of thread blocks.
- A block is scheduled onto one SM and its threads cooperate within that SM.
- Threads are organized into warps of 32 threads.
- A warp is a core scheduling/execution unit for performance analysis.
- A thread is **not** permanently mapped 1:1 to a CUDA core.
- Warp schedulers choose ready warps and help hide memory/execution latency.
- NVIDIA follows a SIMT execution model: threads keep independent state while warps execute common instruction streams.

Example launch:

```cpp
kernel<<<100, 256>>>();
```

means:

```text
100 blocks
x 256 threads/block
= 25,600 logical CUDA threads

256 threads/block / 32 threads/warp
= 8 warps/block
```

---

## 2. SM Resources

An SM contains the hardware resources used to execute warps, including:

```text
Warp Scheduler
Registers
Shared Memory / L1
FP / INT execution units
Tensor Cores
Load / Store units
```

I learned how SM resources affect concurrency:

- Registers are private to each thread.
- Shared memory is shared by threads in the same block.
- High register usage can reduce the number of resident warps.
- Excessive register pressure can cause spilling to local memory.
- Shared-memory usage per block can also limit resident blocks per SM.

This leads directly to the idea of **occupancy**.

---

## 3. Warp Divergence

Threads in one warp should ideally follow the same control path.

Bad pattern:

```cpp
if (threadIdx.x % 2 == 0) {
    do_A();
} else {
    do_B();
}
```

A warp may contain 16 threads on each path, so the hardware must execute multiple control paths with inactive lanes.

Better work mapping tries to keep active threads clustered into whole warps whenever possible.

Example in reduction:

```cpp
if (tid < stride) {
    ...
}
```

is typically much better than selecting interleaved threads with modulo-based conditions.

---

## 4. GPU Memory Hierarchy

The memory hierarchy I use as a mental model is:

```text
Registers
    |
Shared Memory / L1
    |
L2 Cache
    |
Global Memory / VRAM
```

General intuition:

```text
Closer to execution units -> lower latency, smaller capacity
Farther from execution units -> larger capacity, higher latency
```

Important lessons:

- Global-memory variables are loaded through the memory hierarchy into registers before arithmetic is performed.
- Intermediate thread-local values should stay in registers when possible.
- Shared memory is useful when multiple threads in a block reuse the same data.
- Shared memory is explicitly managed; caches are largely hardware-managed.
- More shared memory/register usage is not automatically better because it can reduce occupancy.

---

## 5. Memory-Bound vs Compute-Bound Kernels

For a kernel such as:

```cpp
C[i] = A[i] + B[i];
```

the work per element is roughly:

```text
Read A[i]   : 4 B
Read B[i]   : 4 B
Write C[i]  : 4 B
Compute     : 1 FP32 add
```

The arithmetic intensity is very low, so vector addition is usually memory-bound.

For SAXPY:

```cpp
Y[i] = a * X[i] + Y[i];
```

there are roughly 2 FLOPs for 12 useful bytes of memory traffic, so it is also strongly memory-bound.

This introduced the idea of:

```text
Arithmetic Intensity = FLOPs / Bytes Moved
```

which will become especially important for GEMM and Roofline analysis.

---

## 6. Memory Coalescing

For global memory, the key question is not only how much data each thread loads, but how the addresses of all 32 threads in a warp are distributed.

Good:

```text
T0 -> A[0]
T1 -> A[1]
T2 -> A[2]
...
T31 -> A[31]
```

Bad:

```text
T0 -> A[0]
T1 -> A[8]
T2 -> A[16]
...
```

I learned to think in terms of memory segments/transactions:

> For the same amount of useful data, covering fewer aligned memory segments is generally more efficient.

The dedicated `stride_benchmark.cu` experiment compares access patterns with different strides and reports effective bandwidth.

---

## 7. Shared Memory and Synchronization

Shared memory is a block-local software-managed scratchpad.

Typical pattern:

```text
Global Memory
      |
Coalesced Load
      |
Shared Memory
      |
__syncthreads()
      |
Data Reuse / Rearrangement
      |
Registers / Compute
```

`__syncthreads()` is a block-wide barrier. It is required when threads need to consume shared-memory values written by other threads.

Important rule:

> A block-wide barrier must be reached consistently by all participating threads in the block.

---

## 8. Matrix Transpose and Shared-Memory Tiling

Naive matrix transpose exposes a classic memory-layout problem:

```text
Input read   -> coalesced
Output write -> strided
```

Using a shared-memory tile allows the kernel to:

```text
1. Read input coalesced
2. Rearrange data in shared memory
3. Write output coalesced
```

This is one of the first complete examples where shared memory is used not only for caching, but also for **data-layout transformation**.

---

## 9. Shared-Memory Bank Conflicts

Shared memory is divided into banks. For FP32 data, the simplified model is:

```text
shared[0]  -> bank 0
shared[1]  -> bank 1
...
shared[31] -> bank 31
shared[32] -> bank 0
```

For transpose, a tile declared as:

```cpp
__shared__ float tile[32][32];
```

can cause a 32-way bank conflict when reading columns.

Padding by one element:

```cpp
__shared__ float tile[32][33];
```

changes the bank mapping and removes the severe conflict.

Mental distinction:

```text
Global Memory -> Coalescing / Transactions
Shared Memory -> Banks / Bank Conflicts
```

---

## 10. Occupancy

Occupancy is the ratio of active resident warps to the hardware maximum.

```text
Occupancy = Active Warps / Maximum Warps
```

It is constrained by resources such as:

```text
Threads per block
Registers per thread
Shared memory per block
Maximum blocks per SM
Hardware warp/thread limits
```

The important lesson is:

> Higher occupancy does not automatically mean higher performance.

Reducing registers just to increase occupancy may trigger spilling or reduce instruction-level parallelism. Occupancy is a tool for latency hiding, not the final optimization objective.

---

## 11. Benchmarking Methodology

I use CUDA Events to measure kernel execution rather than directly wrapping asynchronous kernel launches with a CPU timer.

Benchmark workflow:

```text
Correctness Check
      |
Warm-up
      |
Repeated Launches
      |
CUDA Event Timing
      |
Average Kernel Latency
      |
Compare / Profile / Optimize
```

I also distinguish between:

```text
Kernel Time
vs.
End-to-End Time (H2D + Kernel + D2H)
```

This is important when evaluating small kernels where launch and transfer overheads can dominate application-level latency.

---

## 12. Profiling Workflow

Two NVIDIA profiling tools play different roles:

```text
Nsight Systems  -> system-level timeline
Nsight Compute  -> kernel-level performance analysis
```

The optimization loop I follow is:

```text
Benchmark
   -> identify bottleneck
   -> profile
   -> form hypothesis
   -> modify kernel
   -> benchmark again
   -> verify profiler metrics
```

Metrics/concepts I have studied include:

- memory throughput
- DRAM/L1/L2 behavior
- theoretical vs achieved occupancy
- registers per thread
- shared-memory usage
- warp stalls
- branch/warp efficiency
- bank conflicts
- instruction mix

Profiler metrics are diagnostic signals; the final objective is still correctness and actual runtime/throughput.

---

# Kernel Exercises

## Elementwise Kernels

The early kernels follow the mapping:

```text
One thread -> one output element
```

They cover:

- `add_one`
- vector addition
- SAXPY
- ReLU
- fused elementwise operations

The fused example:

```cpp
float z = a * X[i] + b;
Y[i] = max(z, 0.0f);
```

introduces **kernel fusion**: keeping intermediate values in registers instead of writing temporary tensors back to global memory.

---

## Reduction

Reduction changes the parallel pattern from:

```text
One thread -> one output
```

to:

```text
Many threads -> one partial/block result
```

### Baseline

The first implementation uses shared memory and a tree reduction:

```text
256 -> 128 -> 64 -> 32 -> ... -> 1
```

### Optimization 1: Two Elements per Thread

Each thread first loads two coalesced elements and adds them in registers:

```text
Global loads
    -> register partial sum
    -> shared-memory reduction
```

This increases useful work per thread and reduces the number of blocks/partial values.

### Optimization 2: Warp Shuffle

`__shfl_down_sync()` allows lanes in the same warp to exchange register values without routing every intermediate through shared memory.

Optimized hierarchy:

```text
2 elements/thread
      |
Register accumulation
      |
Warp shuffle reduction
      |
One sum per warp
      |
Small shared-memory exchange
      |
Warp 0 final reduction
      |
Block sum
```

This reduces shared-memory traffic and block-wide synchronization.

---

## Softmax

The current implementation performs row-wise numerically stable Softmax.

For an input shaped `[4096, 1024]`:

```text
4096 blocks
1 block -> 1 row
256 threads/block
1024 elements/row
~4 elements/thread
```

For each row:

```text
1. Each thread computes a local max
2. Warp/block MAX reduction
3. Compute exp(x - max)
4. Each thread computes a local sum
5. Warp/block SUM reduction
6. Normalize every element
```

Formula:

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

The subtraction of the row maximum improves numerical stability.

The teaching implementation intentionally recomputes `exp(x - max)` during the output pass. This exposes an important optimization trade-off:

```text
Recompute intermediate values
vs.
Store them in registers/shared/global memory
```

---

## Histogram

Histogram introduces **race conditions** and **atomic operations**.

Unsafe update:

```cpp
hist[bin]++;
```

because multiple threads may read the same old value and overwrite each other's updates.

Correct update:

```cpp
atomicAdd(&hist[bin], 1);
```

A more scalable pattern uses block-local shared-memory privatization:

```text
Threads in Block 0 -> Shared Histogram 0
Threads in Block 1 -> Shared Histogram 1
...
          |
          -> merge with fewer global atomics
```

This reduces global atomic contention.

---

## LayerNorm

LayerNorm is another row-wise cooperative kernel.

For each row:

```text
1. Reduce sum(x)
2. Compute mean
3. Reduce sum((x - mean)^2)
4. Compute variance and inverse standard deviation
5. Normalize
6. Apply gamma and beta
```

Formula:

```text
y_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i
```

The kernel reuses the same block/warp reduction ideas learned from Reduction and Softmax.

Potential future optimizations include:

- loading each input only once and keeping per-thread values in registers
- balancing register pressure against fewer global loads
- using Welford reduction for numerically stable mean/variance computation

---

# Repository Map

| File | Topic | Main Idea |
|---|---|---|
| `ex0.cu` | Minimal CUDA kernel | Basic host/device flow and `add_one` |
| `ex1.cu` | CUDA benchmark | Warm-up, CUDA Events, repeated timing, correctness |
| `ex2.cu` | Vector Add | Coalesced elementwise memory-bound kernel |
| `ex3.cu` | SAXPY | FLOPs, arithmetic intensity, FMA intuition |
| `ex4.cu` | ReLU | Elementwise activation and data-dependent control flow |
| `ex5.cu` | Fused Elementwise | Kernel fusion and reduced intermediate memory traffic |
| `ex6.cu` | Reduction v1 | Shared-memory tree reduction + synchronization |
| `ex7.cu` | Naive Transpose | Coalesced read but strided write |
| `ex8.cu` | Tiled Transpose | Shared-memory tiling for coalesced read/write |
| `ex9.cu` | Conflict-Free Transpose | `32 x 33` padding to avoid bank conflicts |
| `ex10.cu` | Reduction v2 | Two elements per thread + register pre-reduction |
| `ex11.cu` | Reduction v3 | Warp shuffle + reduced shared-memory traffic |
| `ex12.cu` | Row-wise Softmax | Stable max/sum reductions and normalization |
| `ex13.cu` | Histogram | Atomics, race conditions, shared-memory privatization |
| `ex14.cu` | Row-wise LayerNorm | Mean/variance reduction + affine normalization |
| `stride_benchmark.cu` | Coalescing Experiment | Compare stride 1/2/4/8/16/32 effective bandwidth |

---

# Build

CUDA Toolkit with `nvcc` is required. On Windows, a working MSVC C++ toolchain is also required, such as the **Desktop development with C++** workload from Visual Studio 2022.

For a standalone executable example:

```bash
nvcc ex12.cu -O3 -o softmax
./softmax
```

## Windows and VS Code

Some source files contain non-ASCII comments. When compiling manually on Windows, tell MSVC to read the source as UTF-8:

```powershell
nvcc -Xcompiler=/utf-8 ex12.cu -O3 -o softmax.exe
.\softmax.exe
```

If the project path contains non-ASCII characters and `nvcc` reports an internal path error, use the VS Code task below or move the project to an ASCII-only path before compiling manually.

The repository also includes VS Code build tasks. Open a standalone `.cu` file and press `Ctrl+Shift+B` to build and run the current file, or use **Terminal → Run Task** to choose between build-only and build-and-run. The task prefers a project-local `.cuda-env`, falls back to `nvcc` from the system `PATH`, and works around compatibility issues with non-ASCII workspace paths.

The standalone examples are `ex0.cu` through `ex6.cu`, `ex12.cu` through `ex14.cu`, and `stride_benchmark.cu`. Files `ex7.cu` through `ex11.cu` are kernel-focused comparison snippets without `main()` and cannot be run independently.

For optimization/profiling builds, useful commands include:

```bash
nvcc -O3 -Xptxas -v file.cu -o app
nsys profile ./app
ncu ./app
ncu --set full ./app
```

---

# Performance Mindset

When reading or writing a CUDA kernel, I now try to answer the following questions:

```text
How are threads mapped to data?
How are threads grouped into warps?
Is control flow warp-friendly?
Are global loads/stores coalesced?
How many memory transactions are required?
Can data be reused in shared memory or registers?
Are there shared-memory bank conflicts?
What are the register and shared-memory costs?
Is occupancy sufficient to hide latency?
Is the kernel memory-bound or compute-bound?
Can kernels be fused to reduce global-memory traffic?
Are atomics or synchronization creating contention/stalls?
What does profiling say?
Did the optimization actually improve runtime?
```

The core workflow is:

```text
Measure -> Diagnose -> Optimize -> Measure Again
```

---

# Next: Stage 3 — GEMM

The next stage is matrix multiplication optimization. It will combine most of the concepts above in one kernel:

```text
Coalesced Global Loads
        +
Shared-Memory Tiling
        +
Register Tiling
        +
Data Reuse
        +
Occupancy / Register Pressure
        +
Bank-Conflict Avoidance
        +
Arithmetic Intensity
        +
Tensor Cores
```

Planned progression:

```text
Naive GEMM
-> tiled GEMM
-> register tiling
-> vectorized memory access
-> warp-level mapping
-> Tensor Core / WMMA concepts
-> profiling and optimization
```

---

## Notes

This repository is intentionally educational. The kernels progress from simple, readable baselines toward more hardware-aware implementations. The emphasis is not only on writing correct CUDA, but on understanding **why** a particular implementation maps well or poorly onto GPU hardware.
