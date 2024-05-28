# PA3 稀疏矩阵-矩阵乘 实验报告

## 1. 实现方法

$$
A_{M\times M} \, B_{M\times K} = C_{M\times K}
$$

### 1.1 解决 warp divergence

对应 `spmm_kernel_K_32` 和 `spmm_kernel_K_256` 函数 (代码中被注释掉了，因为之后的优化中自然包含这一方法)。

以下取每个线程块恰含一个 warp (32 个线程)。

改变并行维度，在 $K=32=WARP\_SIZE$ 时，**让每个 warp 处理 $C$ 的一行**，即每个线程计算该行的一个元素。$A$ 中对应行的非零元素（以下“元素”一般均指非零元）要被每个 warp 访问 32 次，因此将其**读入 shared memory**。注意这里**每行的计算是分批的**，即每次读取 $WARP\_SIZE$ 个元素到 shared memory，计算完后再读取下一批。

在 $K=256$ 时，$C$ 的每行分配 $\frac{256/32}{K\_256\_STEP}$ 个 warp，每个 warp 处理连续的 $32 * K\_256\_STEP$ 个元素。简单尝试后取 $K\_256\_STEP=2$。其余操作与 $K=32$ 完全类似。

### 1.2 拆分数据密集行

对应 `spmm_kernel_K_32/256_with_truncated_line` 和 `preprocess_truncated_line` 函数。

绘制矩阵非零元分布图，容易发现 $A$ 每行元素个数分布及其不均匀。因此在预处理时**将每行元素划分**为至多 `TRUNCATED_STEP` 个元素的若干块，分配 warp 时不再按行分配而是**按这些块分配**。

注意，这里给 $C$ 写入数据时**要使用原子操作** `atomicAdd`，因为不同 warp 可能在写入同一个 $C$ 中元素。

这里有一个超参数 `TRUNCATED_STEP`，用脚本批量处理，对每个数据集都进行了选取。

### 1.3 合并数据稀疏行

对应 `spmm_kernel_K_32/256_with_combined_line` 和 `preprocess_trunc_combined_line` 函数。

进一步观察矩阵，容易发现在一些数据集中大多数行只有 1~2 个元素，而在上面的方法中它们也分别被分配了一整个 warp。因此在预处理时**将一些稀疏行合并**，选取每行不超过 `max_element_per_line` 个元素的行、不超过 `combined_size` 行合并在一起。先排序再合并，无进一步明显提升。

这里有两个超参数 `max_element_per_line`, `combined_size`，用脚本批处理也工程量巨大，于是**最终采用了统一的参数**。对每个数据集性能都有较小的提升。对单个数据集的测试表明，针对其矩阵元素分布选择参数，在一些数据集上能得到较大的性能提升。

<div style="page-break-after: always;"></div>

## 2. 以 am 为例分析优化影响

| reference | warp div. handled | cusparse | 性能线   | line truncated | line combined |
| --------- | ----------------- | -------- | -------- | -------------- | ------------- |
| TLE       | 25271.80          | 13394.30 | 13000.00 | 12648.40       | 12138.40      |

绘制矩阵非零元分布如下：

<img src="https://cdn.jsdelivr.net/gh/A-sock-puppet/imgbed2@main/img/2024/05/20240528230332_cusparse_1.png" alt="cusparse_1" style="zoom:67%;" />

矩阵极少数行包含了绝大多数的元素，特别是前几行，包含了极多的元素。因此可以看到，当对行进行了分割，速度得到了极大的提升（2倍）。

<div style="display: flex; justify-content: space-around;">
  <img src="https://cdn.jsdelivr.net/gh/A-sock-puppet/imgbed2@main/img/2024/05/20240528230436_cusparse_2.png" alt="Image 1" style="width: 49%;">
  <img src="https://cdn.jsdelivr.net/gh/A-sock-puppet/imgbed2@main/img/2024/05/20240528230446_cusparse_3.png" alt="Image 2" style="width: 49%;">
</div>

接下来观察矩阵局部，80000 行附近每行基本含 3 个非零元， 800000 行附近每行基本含 10-15 个非零元。选取合并参数 `max_element_per_line = 15`, `combined_size = 3`，性能再次得到了一定提升。

<div style="page-break-after: always;"></div>

## 3. 运行时间与加速比

- `len = 32`

| dataset      | cusparse (us) | opt (us) | 加速比 |
| ------------ | ------------- | -------- | ------ |
| arxiv        | 753.32        | 327.20   | 2.30   |
| collab       | 1303.84       | 628.55   | 2.07   |
| citation     | 16444.30      | 9412.78  | 1.75   |
| ddi          | 641.09        | 225.62   | 2.84   |
| protein      | 24662.30      | 8161.69  | 3.02   |
| ppa          | 18383.20      | 10056.50 | 1.83   |
| reddit.dgl   | 48661.00      | 20845.70 | 2.33   |
| products     | 55855.30      | 32139.40 | 1.74   |
| youtube      | 3643.90       | 2085.86  | 1.75   |
| amazon_cogdl | 125359.00     | 50229.90 | 2.50   |
| yelp         | 6575.16       | 3361.53  | 1.96   |
| wikikg2      | 7140.36       | 3838.86  | 1.86   |
| am           | 3740.35       | 1865.75  | 2.00   |

- `len = 256`

| dataset      | cusparse (us) | opt (us)  | 加速比 |
| ------------ | ------------- | --------- | ------ |
| arxiv        | 2995.11       | 2510.17   | 1.19   |
| collab       | 5202.80       | 4703.87   | 1.11   |
| citation     | 78848.10      | 73511.20  | 1.07   |
| ddi          | 1553.06       | 1511.50   | 1.03   |
| protein      | 80818.60      | 68547.50  | 1.18   |
| ppa          | 84968.70      | 82396.90  | 1.03   |
| reddit.dgl   | 202359.00     | 182760.00 | 1.11   |
| products     | 258368.00     | 254979.00 | 1.01   |

<div style="page-break-after: always;"></div>

<p style="text-align: right;"><b>续表</b></p>


| dataset      | cusparse (us) | opt (us)  | 加速比 |
| ------------ | ------------- | --------- | ------ |
| youtube      | 14416.50      | 14349.20  | 1.00   |
| amazon_cogdl | 517236.00     | 425727.00 | 1.21   |
| yelp         | 29972.70      | 27243.40  | 1.10   |
| wikikg2      | 16656.40      | 21434.40  | 0.78   |
| am           | 13394.30      | 12172.80  | 1.10   |
在 **25**/26 个测试中超过 cusparse 的性能。