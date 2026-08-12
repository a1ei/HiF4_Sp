# Qwen3.5-4B OPD / Entropy LFQ 实验汇总

## 实验概览

本目录包含三种 OmniQuant+LFQ 方案：

- **OPD+LFQ**：前面的 block 使用普通 MSE；最后一个 block 使用学生强迫生成的 OPD token 做完整词表 soft-label LFQ。
- **EntropyGrad+LFQ**：前面的 block 使用输出 `y` 的熵梯度重要性加权 MSE；最后一个 block 使用不加重要性权重的普通 LFQ。
- **EntropyGradNorm+LFQ**：与 EntropyGrad 相同，但对重要性做归一化；最后一个 block 同样使用普通 LFQ。

三种方法均基于 `Qwen/Qwen3.5-4B`，权重采用 HiF4 W4，校准/训练阶段激活为 A16。评测时分别测试 W4A16，以及在同一量化权重上启用 HiF4 激活 fake quant 的 W4A4。

## 统一配置

### 校准与训练

| 配置项 | 数值 |
|---|---:|
| 校准数据 | `simplescaling/s1K-1.1` |
| 随机种子 | 42 |
| 样本数 | 512 |
| 序列长度 | 4096 |
| 数据切片 | head，offset 0 |
| 权重量化 | HiF4 W4 |
| 激活校准 | A16 |
| OmniQuant | LWC + LET |
| Epoch | 10 |
| LWC 学习率 | `1e-2` |
| LET 学习率 | `5e-3` |
| 最后一层 LFQ 学习率 | `2e-3` |
| LFQ logits 分块长度 | 128 |

前 `n-1` 个 block 使用 MSE，最后一个 block 不使用 hidden-state MSE，只通过冻结的 final norm 和 LM Head 计算 LFQ soft-label cross-entropy。最后一层 LFQ 不使用 entropy importance 权重。

### 评测

| 配置项 | 数值 |
|---|---:|
| AIME | `aime25_avg5`，每题采样 5 次 |
| LiveCodeBench | `lcb:codegeneration_v6` |
| MMLU-Pro | 固定 seed 42，最多 1000 题 |
| Sampling | temperature 0.7，top-p 0.8，top-k 20 |
| 最大模型长度 / 最大新 token | 32768 / 32768 |
| Tensor Parallel | 4 |
| GPU memory utilization | 0.9 |
| Batch size | 128 |

## OPD 数据

- 生成模型：HiF4 RTN W4A16 的 Qwen3.5-4B。
- 使用 s1K 的 512 个问题进行学生强迫生成，并整理成固定的 `512 × 4096` token 张量。
- 有效 token 数：2,097,152。
- 文件：`opd_tokens_512x4096.pt`。
- 文件大小：18,876,357 bytes，约 18.0 MiB。
- 只保存 token，不保存 logits。

## 最后一层 LFQ 训练

| 方法 | Epoch | LFQ CE | FP/Q Top-1 一致率 | 有效 token | 梯度范数 |
|---|---:|---:|---:|---:|---:|
| OPD+LFQ | 0 | 0.303975 | 97.5866% | 2,097,152 | `2.996e-05` |
| OPD+LFQ | 9 | 0.301269 | 97.6561% | 2,097,152 | `1.067e-04` |
| EntropyGrad+LFQ | 0 | 0.497229 | 96.3438% | 2,097,152 | `2.504e-05` |
| EntropyGrad+LFQ | 9 | 0.496481 | 96.4557% | 2,097,152 | `5.024e-05` |
| EntropyGradNorm+LFQ | 0 | 0.497178 | 96.3618% | 2,097,152 | `2.522e-05` |
| EntropyGradNorm+LFQ | 9 | 0.496346 | 96.4703% | 2,097,152 | `5.548e-05` |

三组实验最后一层的可学习量化参数都获得了有限且非零的梯度。OPD 与另外两种方法所用的最后层输入分布不同，所以不能仅根据 LFQ CE 的绝对值判断方法优劣。

## Benchmark 结果

数值均为 `得分 ± 标准误`，粗体为同一激活精度下的最佳结果。

| 方法 | 精度 | AIME25 Avg@5 | LiveCodeBench Pass@1 | MMLU-Pro 1000 |
|---|---|---:|---:|---:|
| OPD+LFQ | W4A16 | 47.33% ± 6.75% | **26.86% ± 3.36%** | **77.90% ± 1.31%** |
| EntropyGrad+LFQ | W4A16 | 50.00% ± 6.76% | 21.71% ± 3.13% | **77.90% ± 1.31%** |
| EntropyGradNorm+LFQ | W4A16 | **54.00% ± 6.92%** | 24.57% ± 3.26% | 76.80% ± 1.34% |
| OPD+LFQ | W4A4 | 34.67% ± 6.06% | **19.43% ± 3.00%** | 75.30% ± 1.36% |
| EntropyGrad+LFQ | W4A4 | **43.33% ± 6.58%** | 18.29% ± 2.93% | **76.20% ± 1.35%** |
| EntropyGradNorm+LFQ | W4A4 | 42.00% ± 7.00% | 18.86% ± 2.97% | 75.80% ± 1.36% |

## A4 激活量化造成的下降

下表为 `W4A16 - W4A4`，单位是百分点；数值越小，说明对 A4 激活量化越稳健。

| 方法 | AIME25 | LiveCodeBench | MMLU-Pro |
|---|---:|---:|---:|
| OPD+LFQ | 12.66 | 7.43 | 2.60 |
| EntropyGrad+LFQ | **6.67** | **3.42** | 1.70 |
| EntropyGradNorm+LFQ | 12.00 | 5.71 | **1.00** |

## 结论

- **OPD+LFQ 对代码任务最有效**：W4A16 和 W4A4 的 LiveCodeBench 都是三种方案中最高，分别为 26.86% 和 19.43%。
- **EntropyGradNorm+LFQ 的 W4A16 AIME 最好**：达到 54.00%，但加入 A4 激活量化后下降较大。
- **EntropyGrad+LFQ 的 W4A4 综合表现最好**：AIME 和 MMLU-Pro 均为 W4A4 三组中的最高值，而且 AIME、LiveCodeBench 的 A4 降幅最小。
- 三种方法在启用 A4 激活量化后，三个任务的成绩都下降；当前结果没有显示 W4A4 优于 W4A16 的情况。
- OPD 的最后层 FP/Q 一致率最高，但没有在所有 benchmark 上都占优，说明最后层 token 分布拟合更好不等价于所有下游能力都更好。
- 本目录没有普通 LFQ、纯 OmniQuant 或 BF16 基线，因此这些结果只能比较三种方案之间的相对表现，不能据此计算它们相对基础方法的净增益。

## 完整性与异常记录

- OPD+LFQ 和 EntropyGradNorm+LFQ 均正常完成量化、保存和评测。
- EntropyGrad+LFQ 首次运行已完成训练并保存 `omni_parameters.pth`，但最终保存模型时遇到 `Disk quota exceeded`。
- `recover_entropy_grad_lfq.log` 随后通过已有 OmniQuant 参数以 `epochs=0` 重建并成功保存模型；其 W4A4/W4A16 评测均已完整完成。
- 六份评测日志均包含 AIME25、LiveCodeBench 和 MMLU-Pro 的最终结果。

## 主要原始文件

- 量化日志：`quant_opd_lfq.log`、`quant_entropy_grad_lfq.log`、`quant_entropy_grad_norm_lfq.log`
- 恢复日志：`recover_entropy_grad_lfq.log`
- OPD 生成日志：`generate_opd_vllm.log`
- 评测日志：`eval_*_w4a4.log`、`eval_*_w4a16.log`
- OmniQuant 参数：`omni_*/omni_parameters.pth`
