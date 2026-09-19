# GPU Kernel 学习实验室

[English](./README.md) | 中文

这是一个面向 **CUDA、GPU Kernel 与 AI Systems 性能优化** 的学习与实验仓库。

这个仓库记录我从 GPU 基础执行模型出发，逐步学习 CUDA 编程、Global Memory、Shared Memory、Warp-level primitives、Profiling 与典型 GPU Kernel Optimization 的过程。当前已经完成前两个阶段：**GPU & CUDA 基础**、**CUDA Kernel 性能优化**。下一阶段将进入 **GEMM、Tiling 与 Tensor Core**。

本仓库的重点不只是把程序“写对”，而是逐步建立一套面向 GPU 硬件的 Performance Analysis / Optimization 方法：

```text
代码是否正确？
 ↓
Thread 如何映射到数据？
 ↓
Warp 如何执行？
 ↓
Global Memory 访问是否 Coalesced？
 ↓
Shared Memory 是否存在 Bank Conflict？
 ↓
Register、Shared Memory 与 Occupancy 是否合理？
 ↓
Kernel 是 Memory-bound 还是 Compute-bound？
 ↓
能否减少 Synchronization、Memory Traffic 和中间结果写回？
 ↓
使用 Profiling 工具验证判断
 ↓
修改代码并重新测量
```

---

## 当前学习进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第一阶段 | GPU & CUDA 基础 | ✅ 已完成 |
| 第二阶段 | CUDA Kernel 性能优化 | ✅ 已完成 |
| 第三阶段 | GEMM、Tiling 与 Tensor Core | ⏳ 下一阶段 |
| 第四阶段 | Triton | 待学习 |
| 第五阶段 | Transformer / LLM Kernel | 待学习 |
| 第六阶段 | Serving / AI Infra | 待学习 |
| 第七阶段 | CUTLASS / CuTe | 待学习 |
| 第八阶段 | AI for Kernel | 待学习 |

---

# 一、GPU 执行模型

首先建立 CUDA 软件执行模型与 NVIDIA GPU 硬件之间的对应关系。

```text
CUDA 软件模型：
Kernel → Grid → Block → Thread

GPU 硬件执行视角：
GPU → SM → Warp → Execution Units
```

核心理解：

- 一次 Kernel launch 会创建一个 Grid，其中包含多个 Block。
- 一个 Block 会被调度到某个 SM 上执行。
- 同一个 Block 中的 Thread 可以共享 Shared Memory，并进行 Block-level Synchronization。
- NVIDIA GPU 中一个 Warp 通常包含 32 个 Thread。
- Warp 是分析 NVIDIA GPU 执行与性能时最重要的粒度之一。
- Thread 与 CUDA Core 并不是一一永久对应关系。
- Warp Scheduler 会从 ready Warp 中选择可执行工作，从而帮助隐藏 Memory Access Latency。
- NVIDIA GPU 使用 SIMT 执行模型：每个 Thread 有自己的状态，但同一 Warp 中的 Thread 通常共同推进指令流。

例如：

```cpp
kernel<<<100, 256>>>();
```

表示：

```text
100 个 Block
× 每个 Block 256 个 Thread
= 25,600 个逻辑 CUDA Thread

256 / 32
= 每个 Block 8 个 Warp
```

---

# 二、SM 内部资源

一个 SM 中包含执行 Warp 所需的核心资源，例如：

```text
Warp Scheduler
Register File
Shared Memory / L1 Cache
FP / INT 执行单元
Tensor Core
Load / Store Units
```

重要结论：

- Register 是 Thread 私有的。
- Shared Memory 由同一个 Block 中的 Thread 共享。
- 单个 Thread 的 Register 占用过高会减少一个 SM 能同时驻留的 Warp 数量。
- Register Pressure 过大可能导致 Register Spilling 到 Local Memory。
- 单个 Block 的 Shared Memory 占用过高同样会减少 SM 上同时驻留的 Block 数量。

这些资源约束最终会影响 **Occupancy**。

---

# 三、Warp Divergence

同一个 Warp 中的 Thread 最好尽量执行相同控制路径。

例如：

```cpp
if (threadIdx.x % 2 == 0) {
 do_A();
} else {
 do_B();
}
```

一个 Warp 中可能有一半 Thread 走 `A`，另一半 Thread 走 `B`。硬件需要分别处理不同路径，此时部分 Lane 会处于非活动状态，执行资源利用率下降。

更好的工作分配方式是尽量让连续 Thread 具有一致行为。

例如 Reduction 中：

```cpp
if (tid < stride) {
 ...
}
```

通常比交错选择 Thread 的方式更适合 Warp 执行。

重要认识：

> 分支本身不是问题，真正需要关注的是同一个 Warp 内部是否出现不同控制路径。

---

# 四、GPU Memory Hierarchy

目前采用的核心 Memory Hierarchy 模型：

```text
Register
 ↓
Shared Memory / L1
 ↓
L2 Cache
 ↓
Global Memory / VRAM
```

总体规律：

```text
越靠近计算单元：延迟低、容量小
越远离计算单元：容量大、延迟高
```

重要理解：

- Global Memory 中的数据会经过 GPU Memory Hierarchy，被 load 到 Register 后参与计算。
- Thread 私有的中间变量应尽可能保留在 Register 中。
- 当同一 Block 中的多个 Thread 需要重复使用同一批数据时，Shared Memory 非常有价值。
- Shared Memory 通常由程序员显式管理，而 Cache 主要由硬件管理。
- 使用更多 Register 或 Shared Memory 并不一定更快，因为它们可能降低 Occupancy。

例如：

```cpp
C[i] = A[i] + B[i];
```

可以从数据流角度理解为：

```text
Global Memory 中的 A[i]、B[i]
 ↓
 load
 ↓
 Register
 ↓
 FP32 加法
 ↓
 Register
 ↓
 store
 ↓
Global Memory 中的 C[i]
```

---

# 五、Memory-bound 与 Compute-bound

对于：

```cpp
C[i] = A[i] + B[i];
```

每个元素大致需要：

```text
读取 A[i]：4 字节
读取 B[i]：4 字节
写入 C[i]：4 字节
计算：1 次 FP32 加法
```

Arithmetic Intensity 很低，因此向量加法通常属于 **Memory-bound Kernel**。

对于 SAXPY：

```cpp
Y[i] = a * X[i] + Y[i];
```

每个元素大约：

```text
2 次浮点运算
12 字节有效数据访问
```

Arithmetic Intensity 约为：

```text
Arithmetic Intensity = FLOPs / 搬运字节数
```

这一概念会在后续 GEMM 和 Roofline 分析中变得更加重要。

---

# 六、Global Memory Coalescing

分析 Global Memory 访问时，重点不是单个 Thread 读多少，而是 **一个 Warp 中的 32 个 Thread 访问哪些地址**。

理想情况：

```text
T0 → A[0]
T1 → A[1]
T2 → A[2]
...
T31 → A[31]
```

较差情况：

```text
T0 → A[0]
T1 → A[8]
T2 → A[16]
...
```

Profiling 时需要关注访问覆盖了多少个对齐的 Memory Segment 与 Memory Transaction。

核心直觉：

> 在有效数据量相同的情况下，一个 Warp 访问覆盖的 Memory Segment 越少、越紧凑、越对齐，通常效率越高。

仓库中的 `stride_benchmark.cu` 用于测试：

```text
stride = 1 / 2 / 4 / 8 / 16 / 32
```

并比较不同 Strided Access 下的平均 Kernel time 与 Effective Bandwidth。

---

# 七、Shared Memory 与 Synchronization

Shared Memory 可以理解为一个 Block 内由程序员显式管理的高速工作区。

典型流程：

```text
Global Memory
 ↓
Coalesced Load
 ↓
Shared Memory
 ↓
__syncthreads()
 ↓
数据复用 / 数据重排
 ↓
Register / 计算
```

`__syncthreads()` 是 Block-level barrier。

它常用于保证：

1. 同一个 Block 中的 Thread 已经完成 Shared Memory 写入；
2. 后续 Thread 可以安全读取其他 Thread 写入的数据。

重要规则：

> Block-level barrier 必须由需要参与同步的 Thread 一致地到达，否则可能产生错误行为。

---

# 八、Matrix Transpose 与 Shared Memory Tiling

Naive Transpose 是一个经典的 Memory Access 问题。

对于行主序矩阵：

```text
Input Load → Coalesced
Output Store → Strided
```

使用 Shared Memory 分块后，可以：

```text
Coalesced Global Load
 ↓
Shared Memory tile
 ↓
tile 内部转置
 ↓
Coalesced Global Store
```

因此可以同时改善输入与输出的 Global Memory 访问模式。

对应文件：

- `07_transpose_naive.cu`：Naive Transpose
- `08_transpose_tiled.cu`：Shared Memory Tiled Transpose
- `09_transpose_padded.cu`：加入 Padding、避免严重 Bank Conflict 的转置

---

# 九、Shared Memory Bank Conflict

Shared Memory 内部被划分为多个 Bank。

在常见的 32-Bank 模型下，可以近似理解为：

```text
shared[0] → Bank 0
shared[1] → Bank 1
...
shared[31] → Bank 31
shared[32] → Bank 0
```

如果同一 Warp 中的多个 Thread 访问同一个 Bank 的不同地址，则可能发生 Bank Conflict，访问需要分批完成。

矩阵转置中：

```cpp
__shared__ float tile[32][32];
```

按列读取时可能导致多个 Thread 映射到同一个 Bank。

加入一列 Padding：

```cpp
__shared__ float tile[32][33];
```

会改变地址映射，使相邻 Thread 更容易落入不同 Bank，从而减少冲突。

因此：

```text
Global Memory 优化重点：Coalescing
Shared Memory 优化重点：避免 Bank Conflict
```

---

# 十、Occupancy

Occupancy 可以理解为：

```text
一个 SM 实际驻留的 Active Warps 数量
÷
该 SM 支持的最大 Warp 数量
```

影响 Occupancy 的主要因素：

- 每个 Block 的 Thread 数
- Registers / Thread
- Shared Memory / Block
- 硬件允许的最大 Thread 数、Warp 数与 Block 数

典型关系：

```text
Registers / Thread 增加
 ↓
每个 Block 的 Register 占用增加
 ↓
SM 可同时驻留 Block 减少
 ↓
Active Warps 减少
 ↓
Occupancy 可能下降
```

Shared Memory 占用同理。

但是：

> Occupancy 不是越高越好。

为了强行提高 Occupancy 而过度减少 Register，可能造成 Register Spilling，从而增加 Local Memory 访问，最终性能反而下降。

因此 Occupancy 是一种帮助隐藏延迟的手段，而不是最终目标。

---

# 十一、Profiling 方法

性能优化不能只靠猜测，需要建立完整闭环：

```text
Correctness Check
 ↓
Benchmark
 ↓
确认瓶颈
 ↓
提出优化假设
 ↓
修改 Kernel
 ↓
重新测试
 ↓
再次分析
```

主要工具：

## Nsight Systems

用于观察整个程序级时间线，例如：

- CPU 与 GPU 时间关系
- Kernel launch 与执行时间线
- H2D / D2H 拷贝时间
- GPU 空闲区间
- 多 Stream 并发情况

常用命令：

```bash
nsys profile ./program
```

## Nsight Compute

用于分析单个 CUDA Kernel 内部的性能问题，例如：

- DRAM bandwidth
- L1 / L2 行为
- theoretical / achieved Occupancy
- 每个 Thread Register 数量
- Shared Memory 使用量
- Warp stall 原因
- branch efficiency
- Bank Conflict
- 指令类型与执行单元利用率

常用命令：

```bash
ncu ./program
```

需要更多指标时：

```bash
ncu --set full ./program
```

最终最重要的指标仍然是真实运行时间与 throughput，而不是单纯追求某个 Performance Counter“更漂亮”。

---

# 十二、CUDA Benchmark 方法

仓库中的基础实验统一采用以下方法：

```text
准备输入
 ↓
cudaMalloc
 ↓
H2D 拷贝
 ↓
Warm-up
 ↓
CUDA Event 计时
 ↓
重复运行多次
 ↓
计算平均 Kernel time
 ↓
D2H 拷贝
 ↓
Correctness Check
```

使用 CUDA Event 的原因是 CUDA Kernel launch 通常对 CPU 异步，直接用 CPU 计时器包住 Kernel launch 无法准确反映 GPU Kernel 实际运行时间。

对于原地修改输入的 Kernel，例如：

```cpp
x[i] += 1;
```

或：

```cpp
Y[i] = a * X[i] + Y[i];
```

在 Warm-up 和正式测试之间、正式测试和 Correctness Check 之间，需要注意恢复原始输入。

---

# 十三、基础 CUDA Kernel 实验

## 1. add_one

```cpp
x[i] += 1.0f;
```

学习内容：

- CUDA 基本程序结构
- Host / Device 内存
- `cudaMalloc`
- `cudaMemcpy`
- Kernel launch
- 边界判断
- CUDA Event Benchmark

对应：

- `00_add_one_minimal.cu`
- `01_add_one_benchmark.cu`

---

## 2. Vector Add

```cpp
C[i] = A[i] + B[i];
```

学习内容：

- 一个 Thread 负责一个元素
- 连续 Thread 访问连续地址
- Global Memory Coalescing
- Memory-bound Kernel
- Effective Bandwidth

对应：

- `02_vector_add.cu`

---

## 3. SAXPY

```cpp
Y[i] = a * X[i] + Y[i];
```

学习内容：

- FLOPs
- Arithmetic Intensity
- FMA
- 原地写回
- Memory-bound 分析

对应：

- `03_saxpy.cu`

---

## 4. ReLU

```cpp
Y[i] = max(0, X[i]);
```

学习内容：

- 数据相关条件判断
- Warp Divergence
- predication / select 思想
- Elementwise Kernel 的 Memory-bound 特征

对应：

- `04_relu.cu`

---

## 5. 融合 Elementwise

例如：

```cpp
Y[i] = ReLU(a * X[i] + b);
```

学习内容：

- Kernel Fusion
- 减少中间结果对 Global Memory 的写回
- 提高单位 Memory Traffic 对应的计算量
- 中间结果尽量停留在 Register 中

对应：

- `05_fused_elementwise.cu`

---

# 十四、Reduction 优化

Reduction 目标：

```text
X[0] + X[1] + ... + X[N-1]
```

最初版本采用 Shared Memory Tree Reduction：

```text
256
 ↓
128
 ↓
64
 ↓
32
 ↓
16
 ↓
...
 ↓
1
```

优化过程包括：

## 1. 连续 Active Thread

相比交错 Thread 工作：

```cpp
if (tid % (2 * stride) == 0)
```

使用：

```cpp
if (tid < stride)
```

可以减少 Warp 内部的 Warp Divergence。

## 2. 每个 Thread 先处理两个元素

```text
Thread 0 → X[0] + X[256]
Thread 1 → X[1] + X[257]
...
```

先在 Register 中完成一轮 local reduction，然后再进入 Shared Memory / Warp Reduction。

## 3. Warp Shuffle

使用：

```cpp
__shfl_down_sync(...)
```

允许同一个 Warp 中的 Thread 通过硬件提供的 Warp-level 数据交换机制传递 Register 值。

典型 Warp Reduction：

```cpp
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
```

这样可以减少：

- Shared Memory 读写
- Block-level Synchronization 次数
- 无效 Thread 参与

对应：

- `06_reduction_shared_memory.cu`：基础 Reduction
- `10_reduction_two_elements.cu`：每个 Thread 处理两个元素
- `11_reduction_warp_shuffle.cu`：Warp Shuffle Reduction

---

# 十五、Softmax

按行 Softmax：

\[
y_i = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
\]

其中：

\[
m=\max_j x_j
\]

通过减去最大值提高数值稳定性。

当前实现采用：

```text
一个 Block 负责一行
 ↓
每个 Thread 处理多个元素
 ↓
每个 Thread 求 Local Max
 ↓
Warp / Block MAX Reduction
 ↓
得到整行最大值
 ↓
每个 Thread 计算 exp(x - max)
 ↓
每个 Thread 得到 Local Sum
 ↓
Warp / Block SUM Reduction
 ↓
得到整行分母
 ↓
每个 Thread 写回自己的多个 Softmax 输出
```

例如：

```text
4096 行 × 1024 列
256 threads / block
```

则：

```text
4096 个 Block
每个 Block 负责 1 行
每个 Block 输出 1024 个值
每个 Thread 平均处理 4 个元素
```

重要认识：

- 4096 个 Block 执行相同的 Kernel 代码，但处理不同的数据行。
- 同一行的 1024 个输出共享同一个 `max` 和 `sum`，但每个输出值通常不同。
- `i` 表示当前输出元素索引，`j` 表示求归一化统计量时遍历的输入索引。
- 输入最初位于 Global Memory 中，Thread 加载后在 Register 中进行局部计算。
- 当前教学版本为了节省中间存储，会重新计算一次 `exp`，体现了“重新计算”和“保存中间结果”之间的权衡。

对应：

- `12_softmax_rows.cu`

---

# 十六、Histogram、Race Condition 与 Atomic Operation

Histogram 的核心问题是：

```text
多个 Thread
 ↓
可能同时修改同一个 bin
```

直接执行：

```cpp
hist[bin]++;
```

可能产生 Race Condition，因为它本质上包含：

```text
读取
修改
写回
```

多个 Thread 同时执行时可能丢失更新。

使用：

```cpp
atomicAdd(&hist[bin], 1);
```

可以保证正确性，但大量 Thread 竞争同一地址时会产生 Atomic Contention，影响性能。

进一步优化：

```text
每个 Block
 ↓
维护一份 Shared Memory 中的 Local Histogram
 ↓
先在 Block 内局部聚合
 ↓
最后用少量 Global Atomic merge 到 Global Histogram
```

这种方法属于 **Privatization / 分层聚合** 思路。

对应：

- `13_histogram_shared.cu`

---

# 十七、LayerNorm

LayerNorm 对一行数据计算：

\[
\mu=\frac1N\sum_i x_i
\]

\[
\sigma^2=\frac1N\sum_i(x_i-\mu)^2
\]

\[
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
\]

当前实现同样采用：

```text
一个 Block 负责一行
 ↓
每个 Thread 处理多个元素
 ↓
Local Sum
 ↓
Block Reduction 得到 mean
 ↓
Local Squared-Difference Sum
 ↓
Block Reduction 得到 variance
 ↓
归一化
 ↓
gamma / beta
 ↓
写回输出
```

该问题与 Softmax 的结构非常相似：

```text
Softmax：
MAX Reduction + SUM Reduction + Elementwise

LayerNorm：
SUM Reduction + Variance Reduction + Elementwise
```

进一步优化方向：

- 将 Thread 负责的输入保留在 Register 中，减少多次读取 Global Memory
- 权衡 Register Pressure 与 Occupancy
- 使用更稳定的方差计算方法
- 后续学习 Welford 并行统计方法

对应：

- `14_layernorm_rows.cu`

---

# 十八、当前形成的 Performance Analysis / Optimization 思维

现在拿到一个 CUDA Kernel，我会优先分析：

```text
1. Thread 如何映射到数据？
2. 一个 Thread 负责多少个元素？
3. 同一 Warp 是否走一致控制路径？
4. Global Memory 访问是否连续、紧凑、对齐？
5. 有没有可以重复利用的数据？
6. 是否值得使用 Shared Memory？
7. Shared Memory 是否存在 Bank Conflict？
8. Register 占用是否过高？
9. Shared Memory 占用是否限制 Block 驻留？
10. Occupancy 是否足以隐藏延迟？
11. Kernel 是 Memory-bound 还是 Compute-bound？
12. 是否存在 Race Condition 或 Atomic Operation 热点？
13. 是否可以做 Kernel Fusion？
14. 是否可以减少同步？
15. 是否可以使用 Warp-level 原语？
16. Profiling 工具给出的证据是什么？
17. 修改之后真实运行时间是否下降？
```

最终原则：

> 性能优化不是把某一个 metric 推到最大，而是在 Memory Access、Compute、Register、Shared Memory、Parallelism、Synchronization 和 Numerical Stability 之间找到更好的整体平衡。

---

# 十九、仓库文件说明

| 文件 | 内容 |
|---|---|
| `00_add_one_minimal.cu` | 最基础的 `add_one` Kernel |
| `01_add_one_benchmark.cu` | `add_one` + CUDA Event Benchmark |
| `02_vector_add.cu` | 向量加法 |
| `03_saxpy.cu` | SAXPY |
| `04_relu.cu` | ReLU |
| `05_fused_elementwise.cu` | 融合 Elementwise Kernel |
| `06_reduction_shared_memory.cu` | 基础 Shared Memory Reduction |
| `07_transpose_naive.cu` | Naive Transpose |
| `08_transpose_tiled.cu` | Shared Memory 分块矩阵转置 |
| `09_transpose_padded.cu` | 使用 Padding 减少 Bank Conflict 的矩阵转置 |
| `10_reduction_two_elements.cu` | 每个 Thread 处理两个元素的 Reduction |
| `11_reduction_warp_shuffle.cu` | Warp Shuffle Reduction |
| `12_softmax_rows.cu` | 按行 Softmax |
| `13_histogram_shared.cu` | Histogram + Atomic + Shared Memory Privatization |
| `14_layernorm_rows.cu` | 按行 LayerNorm |
| `stride_benchmark.cu` | 不同 stride 下的 Global Memory 访问性能实验 |

> 说明：`07_transpose_naive.cu`～`11_reduction_warp_shuffle.cu` 主要用于展示和对比 Kernel 实现，本身没有 `main()`，不能直接编译为可执行程序；`00_add_one_minimal.cu`～`06_reduction_shared_memory.cu`、`12_softmax_rows.cu`～`14_layernorm_rows.cu` 和 `stride_benchmark.cu` 可以独立编译运行。

---

# 二十、编译与运行

编译需要 CUDA Toolkit（包含 `nvcc`）。在 Windows 上还需要可用的 MSVC C++ 工具链，例如 Visual Studio 2022 的“使用 C++ 的桌面开发”组件。

单个 CUDA 文件可以使用：

```bash
nvcc 02_vector_add.cu -O3 -o vector_add
./vector_add
```

## Windows 与 VS Code

部分源码包含中文注释。在 Windows 上手动编译时，建议让 MSVC 明确使用 UTF-8：

```powershell
nvcc -Xcompiler=/utf-8 02_vector_add.cu -O3 -o vector_add.exe
.\vector_add.exe
```

如果项目路径包含中文等非 ASCII 字符，并且 `nvcc` 报告内部路径错误，请使用下面的 VS Code 任务，或将项目放到纯英文路径后再手动编译。

仓库已经提供 VS Code 构建任务。打开一个可独立运行的 `.cu` 文件后，按 `Ctrl+Shift+B` 即可编译并运行当前文件；也可以从 **Terminal → Run Task** 选择仅编译或编译并运行。任务会优先使用项目内的 `.cuda-env`，否则使用系统 `PATH` 中的 `nvcc`，并处理非 ASCII 工作区路径的兼容问题。

这些任务适用于包含 `main()` 的示例，即 `00_add_one_minimal.cu`～`06_reduction_shared_memory.cu`、`12_softmax_rows.cu`～`14_layernorm_rows.cu` 和 `stride_benchmark.cu`。`07_transpose_naive.cu`～`11_reduction_warp_shuffle.cu` 是用于对比实现的 Kernel 片段，不能单独运行。

查看编译器生成的 Register 与 Shared Memory 使用信息：

```bash
nvcc -Xptxas -v 12_softmax_rows.cu -O3 -o softmax
```

使用 Nsight Systems：

```bash
nsys profile ./softmax
```

使用 Nsight Compute：

```bash
ncu ./softmax
```

---

# 二十一、下一阶段：GEMM

下一阶段将进入矩阵乘法优化。

计划重点：

```text
Naive GEMM
 ↓
分析 Global Memory 访问与数据复用
 ↓
Shared Memory 分块
 ↓
Register 分块
 ↓
Thread 级 / Warp 级工作划分
 ↓
提升 Arithmetic Intensity
 ↓
Vectorized Memory Access
 ↓
Occupancy 与 Register Pressure 权衡
 ↓
Tensor Core
 ↓
进一步学习 CUTLASS / CuTe
```

GEMM 会把前两个阶段学到的内容真正组合起来：

- Coalesced Memory Access
- Shared Memory
- Bank Conflict
- 数据复用
- Register Pressure
- Occupancy
- Warp-level 执行
- Arithmetic Intensity
- Profiling
- Tensor Core

---

## 当前阶段总结

目前已经从“知道 CUDA 可以在 GPU 上并行计算”逐步建立到以下能力：

```text
看懂基础 CUDA Kernel
 ↓
理解它在 SM / Warp 层面的执行方式
 ↓
分析数据从 Global Memory / VRAM 到 Register 的路径
 ↓
分析 Warp-level 控制流和 memory access 行为
 ↓
判断 Memory-bound 与 Compute-bound
 ↓
使用 Shared Memory、Reduction 和 Warp Shuffle
 ↓
理解 Atomic、Race Condition 与 Privatization
 ↓
理解 Softmax / LayerNorm 等典型 AI Kernel 结构
 ↓
使用 Benchmark 与 Profiling 工具验证优化效果
```

下一步将进入更系统的高性能矩阵计算与 AI Kernel 优化。
