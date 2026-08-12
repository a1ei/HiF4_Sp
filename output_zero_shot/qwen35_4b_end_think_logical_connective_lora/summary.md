# Qwen3.5-4B `</think>` + 逻辑连接词 LoRA 实验汇总

## 1. 实验范围

本目录包含三种 HiF4 权重量化模型上的 LM Head LoRA 实验：

1. GPTQ + `</think>`/逻辑连接词 LoRA
2. GPTQ-EntropyGrad + `</think>`/逻辑连接词 LoRA
3. GPTQ-EntropyGradNorm + `</think>`/逻辑连接词 LoRA

每种模型均评测两种推理精度：

- W4A16：不启用激活 fake quant
- W4A4：启用 HiF4 激活 fake quant

六组评测均已生成完整结果。

## 2. 统一配置

### LoRA 训练

| 配置项 | 值 |
|---|---|
| Teacher | `Qwen/Qwen3.5-4B`（BF16） |
| 数据集 | `simplescaling/s1K-1.1` |
| 样本数 | 128 |
| 最大长度 | 16384 |
| Seed | 42 |
| Epoch | 10 |
| 学习率 | `1e-4` |
| 正位置上限 | 8 |
| 负位置数 | 8 |
| 目标 token 上限 | 32 |
| 实际目标 token 数 / LoRA rank | 28 / 28 |

目标包含 `</think>`，并在其基础上加入以下 12 个逻辑连接词：

`therefore`、`thus`、`however`、`hence`、`moreover`、`consequently`、`first`、`second`、`finally`、`otherwise`、`alternatively`、`nevertheless`。

实际目标 token ID 为：

```text
[39, 71, 1118, 2018, 3765, 4213, 4611, 5170, 5326, 5339, 5752,
 8190, 8646, 10883, 13891, 14833, 15207, 15612, 22117, 30022,
 32032, 37201, 43022, 52971, 55262, 66073, 88842, 248069]
```

目标 token 数多于 13，是因为连接词在不同上下文/空格形式下可能对应不同 token，而不是简单地假定每个词只有一个 ID。

### 统一评测

| 配置项 | 值 |
|---|---|
| 任务 | AIME25 Avg@5、LiveCodeBench v6、MMLU-Pro |
| MMLU-Pro 样本数 | 1000 |
| Seed | 42 |
| Temperature / Top-p / Top-k | 0.7 / 0.8 / 20 |
| 最大上下文 / 最大生成长度 | 32768 / 32768 |
| Batch size | 128 |
| Tensor parallel | 4 |
| GPU 显存利用率 | 0.9 |

## 3. LoRA 训练结果

| 量化底座 | Epoch 1 Train | Epoch 10 Train | Epoch 1 Val | 最佳 Val（Epoch 10） |
|---|---:|---:|---:|---:|
| GPTQ | 1.558455 | **1.123123** | 1.480904 | **1.162428** |
| GPTQ-EntropyGrad | 1.574801 | **1.131876** | 1.509988 | **1.177807** |
| GPTQ-EntropyGradNorm | 1.545052 | **1.117584** | 1.490097 | **1.174732** |

三组训练和验证损失在 10 个 epoch 内都持续下降，没有出现明显的验证损失反弹。GPTQ 底座取得最低的最终验证损失，GPTQ-EntropyGradNorm 的最终训练损失最低。

## 4. Benchmark 结果

下表保留日志中的 `[0, 1]` 原始分数；括号内为标准误。

| 量化底座 | 精度 | AIME25 Avg@5 | LiveCodeBench pass@1:16 | MMLU-Pro 1000 |
|---|---|---:|---:|---:|
| GPTQ | W4A16 | 0.5867 (±0.0779) | 0.2971 (±0.0346) | 0.7900 (±0.0129) |
| GPTQ | W4A4 | 0.4733 (±0.0770) | 0.2114 (±0.0310) | **0.7680** (±0.0134) |
| GPTQ-EntropyGrad | W4A16 | 0.5400 (±0.0803) | **0.3257** (±0.0355) | 0.7830 (±0.0130) |
| GPTQ-EntropyGrad | W4A4 | 0.4800 (±0.0771) | **0.2571** (±0.0331) | 0.7560 (±0.0136) |
| GPTQ-EntropyGradNorm | W4A16 | **0.6000** (±0.0718) | 0.2686 (±0.0336) | **0.7910** (±0.0129) |
| GPTQ-EntropyGradNorm | W4A4 | **0.4933** (±0.0741) | 0.1714 (±0.0286) | 0.7590 (±0.0135) |

加粗表示同一精度下该任务的最高分。

## 5. W4A16 相对 W4A4 的变化

| 量化底座 | AIME25 | LiveCodeBench | MMLU-Pro |
|---|---:|---:|---:|
| GPTQ | +0.1134 | +0.0857 | +0.0220 |
| GPTQ-EntropyGrad | +0.0600 | +0.0686 | +0.0270 |
| GPTQ-EntropyGradNorm | +0.1067 | +0.0972 | +0.0320 |

所有模型、所有任务上，W4A16 都高于对应的 W4A4，说明本实验中激活量化带来了稳定损失，LiveCodeBench 和 AIME 对该损失尤其敏感。

## 6. 与仅 `</think>` LoRA 的对比

对照数据来自 `output_zero_shot/qwen35_4b_quant_results.md`。这里严格比较相同量化底座和相同推理精度：

- 旧实验：LoRA 只蒸馏 `</think>` 相关位置。
- 当前实验：在原有 `</think>` 基础上，继续加入 12 个逻辑连接词位置。
- 差值定义为“`</think>` + 逻辑连接词 LoRA”减去“仅 `</think>` LoRA”；正数表示加入逻辑连接词后更好。

### 6.1 完整对照结果

| 量化底座 | 精度 | LoRA 目标 | AIME25 Avg@5 | LiveCodeBench | MMLU-Pro 1000 |
|---|---|---|---:|---:|---:|
| GPTQ | W4A16 | 仅 `</think>` | **0.6200** | 0.2857 | 0.7870 |
| GPTQ | W4A16 | `</think>` + 连接词 | 0.5867 | **0.2971** | **0.7900** |
| GPTQ | W4A4 | 仅 `</think>` | **0.4933** | **0.2229** | 0.7640 |
| GPTQ | W4A4 | `</think>` + 连接词 | 0.4733 | 0.2114 | **0.7680** |
| GPTQ-EntropyGrad | W4A16 | 仅 `</think>` | **0.6000** | 0.3257 | 0.7800 |
| GPTQ-EntropyGrad | W4A16 | `</think>` + 连接词 | 0.5400 | 0.3257 | **0.7830** |
| GPTQ-EntropyGrad | W4A4 | 仅 `</think>` | 0.4800 | 0.2343 | **0.7680** |
| GPTQ-EntropyGrad | W4A4 | `</think>` + 连接词 | 0.4800 | **0.2571** | 0.7560 |
| GPTQ-EntropyGradNorm | W4A16 | 仅 `</think>` | 0.5933 | **0.3371** | 0.7870 |
| GPTQ-EntropyGradNorm | W4A16 | `</think>` + 连接词 | **0.6000** | 0.2686 | **0.7910** |
| GPTQ-EntropyGradNorm | W4A4 | 仅 `</think>` | **0.5000** | **0.2171** | **0.7780** |
| GPTQ-EntropyGradNorm | W4A4 | `</think>` + 连接词 | 0.4933 | 0.1714 | 0.7590 |

粗体表示每一对实验中的较高值；完全相同的结果不加粗。

### 6.2 加入逻辑连接词后的分数变化

| 量化底座 | 精度 | AIME25 | LiveCodeBench | MMLU-Pro |
|---|---|---:|---:|---:|
| GPTQ | W4A16 | -0.0333 | +0.0114 | +0.0030 |
| GPTQ | W4A4 | -0.0200 | -0.0115 | +0.0040 |
| GPTQ-EntropyGrad | W4A16 | -0.0600 | 0.0000 | +0.0030 |
| GPTQ-EntropyGrad | W4A4 | 0.0000 | +0.0228 | -0.0120 |
| GPTQ-EntropyGradNorm | W4A16 | +0.0067 | -0.0685 | +0.0040 |
| GPTQ-EntropyGradNorm | W4A4 | -0.0067 | -0.0457 | -0.0190 |
| **六组平均变化** |  | **-0.0189** | **-0.0153** | **-0.0028** |

18 个“底座 × 精度 × 任务”对比中，加入逻辑连接词后有 7 项提升、9 项下降、2 项不变。

### 6.3 对比结论

- 整体上，当前结果不支持“加入多个逻辑连接词比仅蒸馏 `</think>` 更好”。三项指标的六组平均变化均为负，其中 AIME 平均下降 0.0189，LiveCodeBench 平均下降 0.0153，MMLU-Pro 基本持平但仍下降 0.0028。
- GPTQ-EntropyGrad W4A4 是最明确的局部收益：LiveCodeBench 提升 0.0228，AIME 不变，但 MMLU-Pro 下降 0.0120。
- GPTQ-EntropyGradNorm W4A16 的 AIME 和 MMLU-Pro 分别提升 0.0067、0.0040，但 LiveCodeBench 明显下降 0.0685。
- GPTQ 的 MMLU-Pro 在 W4A16/W4A4 都小幅提升，但 AIME 在两种精度下都下降。
- GPTQ-EntropyGradNorm W4A4 三项指标全部下降，是加入连接词后退化最一致的一组。
- AIME 和生成式代码评测存在采样波动，且多数组间差异接近各自标准误。因此小幅差异不能视为显著提升；不过多组同方向下降说明，至少以当前连接词集合和训练配置，还没有观察到稳定收益。

## 7. 当前实验内部的结果观察

- GPTQ-EntropyGradNorm 的推理/数学表现最好：两种精度下 AIME 都最高，W4A16 的 MMLU-Pro 也最高。
- GPTQ-EntropyGrad 的代码表现最好：W4A16 和 W4A4 的 LiveCodeBench 均为最高。
- 原始 GPTQ 在 W4A4 的 MMLU-Pro 最好，并取得三组 LoRA 中最低的验证损失。
- 没有一种底座在所有任务上都占优；EntropyGrad 更偏向代码任务，EntropyGradNorm 更偏向数学推理。
- 部分差异小于或接近报告的标准误，不能仅凭本次结果断言存在统计显著差异。

## 8. 数据来源

- 训练日志：`train_gptq.log`、`train_gptq_entropy_grad.log`、`train_gptq_entropy_grad_norm.log`
- 评测日志：各模型子目录下的 `w4a16.log` 与 `w4a4.log`
- 原始结果：`results/qwen35_4b_end_think_logical_connective_lora/`
- 仅 `</think>` LoRA 对照：`output_zero_shot/qwen35_4b_quant_results.md`
