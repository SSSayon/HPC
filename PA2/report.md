# PA2 全源最短路 实验报告

## 1. 实现方法

![pa2_block ](https://cdn.jsdelivr.net/gh/A-sock-puppet/imgbed2@main/img/2024/05/20240514193623_pa2_block%20.png)

### 1.1 phase1

没什么特殊的，按照文档来，只有一个 `BLOCK_SIZE * BLOCK_SIZE` 的块。先读入 shared memory 再进行处理。

`BLOCK_SIZE` 算半个超参数，设置为 32 或相近 2 的倍数是比较符合 GPU 的硬件结构的。但见 phase3 中的考量大概是只能取 32 为佳了。

### 1.2 phase2

考虑到此时若仍只是一系列 `BLOCK_SIZE * BLOCK_SIZE` 的块，处理整个十字区域需要的块数太大了。因此按照文档的建议，进行两级分块，引入大块（分横向和纵向两种），将之前的块称为小块，每个大块中包含一定数量 (程序中用 `SPAN` 表示) 的小块。每个大块由一个线程块处理，先将大块对应的十字区域的数据以及中心块的数据读入 shared memory；每个线程处理各个小块中对应位置的数据（`SPAN` 个）。

```cpp
__shared__ int center[BLOCK_SIZE * BLOCK_SIZE];
__shared__ int shared[SPAN][BLOCK_SIZE * BLOCK_SIZE];
```

`SPAN` 是个超参数，大了并行性不好，小了每个线程没啥工作量。测试后采用 `SPAN = 4`。但事实上这里再怎么凹性能也没啥用，打印下数据就能发现 phase1 和 phase2 的所用时间连 phase3 的零头都不到，毕竟后者有巨多的块。

<div style="page-break-after: always;"></div>

### 1.3 phase3

采用与 phase2 相近的分块方式，只是每个大块所含小块数量与 phase2 不同。注意这里的块数会非常大，因此大块中小块的数量不再是一个超参数，而应做到最大限度使用 shared memory：

```cpp
/* 
`FIXED_SPAN` is NOT a hyper-parameter
to make full use of the shared memory, the equation below need to be met: 
2 * FIXED_SPAN * BLOCK_SIZE * BLOCK_SIZE * sizeof(int) B = 48 KB
*/
#define FIXED_SPAN 6 * 1024 / (BLOCK_SIZE * BLOCK_SIZE)
```

这里同时对 `BLOCK_SIZE` 也提出了限制，取 32 了：

```cpp
static_assert((6 * 1024) % (BLOCK_SIZE * BLOCK_SIZE) == 0, 
"BLOCK_SIZE chosen poorly that shared memory cannot be used to the full extent");
```

因此 `FIXED_SPAN = 6` 也是定的。先将每个大块对应的十字区域的数据读入 shared memory，然后每个线程处理各个小块中对应位置的数据（`FIXED_SPAN * FIXED_SPAN` 个）。

### 1.4 优化过程

首先就是分块方法及参数的选择，这在上面已经说了。

然后就没非常明确的想法了。开始优化代码结构，减少局部变量，减少分支判断，等等等等。性能有所提升，但没有多少。

但在 phase3 中进行的减少分支的优化很有效果。对于大部分的大块，线程在更新对应小块的数据时是不需要边界判断的。我把这部分单独分出来，相当于原模原样一段代码复制一份，删掉其中所有边界判断。这至少带来了 20% 多的性能提升。

## 2. 运行时间及加速比

| 图规模 n      | 1000      | 2500       | 5000        | 7500         | 10000        |
| ------------- | --------- | ---------- | ----------- | ------------ | ------------ |
| 运行时间 (ms) | 1.639607  | 14.012142  | 88.087830   | 283.395618   | 650.584813   |
| 朴素实现 (ms) | 13.766302 | 377.810864 | 2993.669812 | 10077.100457 | 22908.972556 |
| 加速比        | 8.40      | 26.96      | 33.99       | 35.56        | 35.21        |

