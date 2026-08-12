# Qwen3.5-4B W4A16 / W4A4 与熵方向实验汇总

## 1. 结果口径

| 简称 | 权重量化时的激活 | 推理激活 | 是否是真正A4校准 |
|---|---|---|---|
| 正确W4A16 | A16/BF16 | A16/BF16 | 不适用 |
| 旧W4A4推理 | A16/BF16 | HiF4 A4 | 否，只在推理时加A4 |
| 新W4A4 | HiF4 A4 | HiF4 A4 | **是，Hessian及逐层传播均使用A4** |

以下“正确W4A16 vs 新W4A4”只比较第一列和第三列，不用A16校准模型测得的A4结果冒充A4校准结果。

统一设置：Qwen3.5-4B、s1K 512×4096、head切片、seed 42、HiF4 W4；非greedy使用 temperature 0.7、top-p 0.8、top-k 20，AIME每题5次，MMLU-Pro 1000题。

## 2. Greedy：正确W4A16与新W4A4

Greedy 使用 temperature 0、top-p 1、top-k 0；AIME为单次 Avg@1。W4A16加载A16校准模型，W4A4加载名称含 `A4Calib` 的A4校准模型。

| 方法 | LoRA | W4A16 AIME | 新W4A4 AIME | W4A16 LCB | 新W4A4 LCB | W4A16 MMLU | 新W4A4 MMLU |
|---|---|---:|---:|---:|---:|---:|---:|
| GPTQ | 否 | 0.4333 | 0.3333 | 0.1600 | 0.0857 | 0.7700 | 0.7350 |
| GPTQ | 是 | 0.4000 | 0.2333 | 0.1657 | 0.1257 | 0.7580 | 0.7280 |
| entropy-grad-low | 否 | 0.3333 | 0.2667 | **0.1829** | **0.1429** | 0.7530 | 0.7290 |
| entropy-grad-low | 是 | 0.4000 | 0.3000 | 0.1486 | 0.1314 | **0.7730** | **0.7380** |
| entropy-grad-norm-low | 否 | 0.4667 | 0.2667 | 0.1486 | 0.1371 | 0.7700 | 0.7350 |
| entropy-grad-norm-low | 是 | **0.5000** | 0.3000 | 0.1771 | 0.0914 | 0.7670 | **0.7380** |

| 方法 | LoRA | 新W4A4−W4A16 AIME | LCB | MMLU |
|---|---|---:|---:|---:|
| GPTQ | 否 | -0.1000 | -0.0743 | -0.0350 |
| GPTQ | 是 | -0.1667 | -0.0400 | -0.0300 |
| entropy-grad-low | 否 | -0.0666 | -0.0400 | -0.0240 |
| entropy-grad-low | 是 | -0.1000 | -0.0172 | -0.0350 |
| entropy-grad-norm-low | 否 | -0.2000 | -0.0115 | -0.0350 |
| entropy-grad-norm-low | 是 | -0.2000 | -0.0857 | -0.0290 |

Greedy的18个配对指标全部是新W4A4更低。AIME只有30题且每题一次，标准误约0.08–0.09，不能过度解释小差异。

## 3. 非greedy：正确W4A16、旧W4A4推理、新W4A4

| 方法 | LoRA | 模型口径 | 校准激活 | 推理激活 | AIME Avg@5 | LiveCodeBench | MMLU-Pro 1000 |
|---|---|---|---|---|---:|---:|---:|
| GPTQ | 🔵 Base | W4A16 | A16 | A16 | 0.6000 | 0.2800 | **0.7990** |
| GPTQ | 🔵 Base | 旧W4A4 | A16 | A4 | 0.4400 | 0.2171 | 0.7660 |
| GPTQ | 🔵 Base | 新W4A4 | A4 | A4 | 0.4333 | 0.1714 | 0.7680 |
| GPTQ | 🟠 **+LoRA** | W4A16 | A16 | A16 | **0.6200** | 0.2857 | 0.7870 |
| GPTQ | 🟠 **+LoRA** | 旧W4A4 | A16 | A4 | 0.4933 | 0.2229 | 0.7640 |
| GPTQ | 🟠 **+LoRA** | 新W4A4 | A4 | A4 | **0.5000** | 0.2114 | 0.7680 |
| entropy-grad-low | 🔵 Base | W4A16 | A16 | A16 | 0.5133 | 0.3029 | 0.7890 |
| entropy-grad-low | 🔵 Base | 旧W4A4 | A16 | A4 | **0.5133** | 0.2057 | 0.7650 |
| entropy-grad-low | 🔵 Base | 新W4A4 | A4 | A4 | 0.4667 | 0.2057 | 0.7510 |
| entropy-grad-low | 🟠 **+LoRA** | W4A16 | A16 | A16 | 0.6000 | 0.3257 | 0.7800 |
| entropy-grad-low | 🟠 **+LoRA** | 旧W4A4 | A16 | A4 | 0.4800 | 0.2343 | 0.7680 |
| entropy-grad-low | 🟠 **+LoRA** | 新W4A4 | A4 | A4 | 0.4267 | 0.2000 | **0.7710** |
| entropy-grad-norm-low | 🔵 Base | W4A16 | A16 | A16 | 0.5867 | 0.2743 | 0.7840 |
| entropy-grad-norm-low | 🔵 Base | 旧W4A4 | A16 | A4 | 0.5067 | **0.2400** | **0.7780** |
| entropy-grad-norm-low | 🔵 Base | 新W4A4 | A4 | A4 | 0.4800 | **0.2171** | 0.7620 |
| entropy-grad-norm-low | 🟠 **+LoRA** | W4A16 | A16 | A16 | 0.5933 | **0.3371** | 0.7870 |
| entropy-grad-norm-low | 🟠 **+LoRA** | 旧W4A4 | A16 | A4 | 0.5000 | 0.2171 | **0.7780** |
| entropy-grad-norm-low | 🟠 **+LoRA** | 新W4A4 | A4 | A4 | 0.4400 | 0.2114 | 0.7620 |

表中：

- 🔵 Base：未加载 `</think>` LoRA。
- 🟠 **+LoRA**：加载 `</think>` LoRA；与同方法、同模型口径的蓝色行直接配对比较。
- W4A16：A16校准、A16推理。
- 旧W4A4：A16校准，只在推理时使用A4。
- 新W4A4：A4校准、A4推理，是真正的A4Calib模型。

### 同一量化口径：Base 与 LoRA 直接对比

图例：🟢 提升；🔴 下降；⚪ 不变。每个单元格均为 `Base → +LoRA（变化量）`。

| 方法 | 模型口径 | AIME Avg@5 | LiveCodeBench | MMLU-Pro 1000 |
|---|---|---:|---:|---:|
| GPTQ | W4A16 | 0.6000 → 0.6200 🟢 `+0.0200` | 0.2800 → 0.2857 🟢 `+0.0057` | 0.7990 → 0.7870 🔴 `-0.0120` |
| GPTQ | 旧W4A4 | 0.4400 → 0.4933 🟢 `+0.0533` | 0.2171 → 0.2229 🟢 `+0.0058` | 0.7660 → 0.7640 🔴 `-0.0020` |
| GPTQ | 新W4A4 | 0.4333 → 0.5000 🟢 `+0.0667` | 0.1714 → 0.2114 🟢 `+0.0400` | 0.7680 → 0.7680 ⚪ `0.0000` |
| entropy-grad-low | W4A16 | 0.5133 → 0.6000 🟢 `+0.0867` | 0.3029 → 0.3257 🟢 `+0.0228` | 0.7890 → 0.7800 🔴 `-0.0090` |
| entropy-grad-low | 旧W4A4 | 0.5133 → 0.4800 🔴 `-0.0333` | 0.2057 → 0.2343 🟢 `+0.0286` | 0.7650 → 0.7680 🟢 `+0.0030` |
| entropy-grad-low | 新W4A4 | 0.4667 → 0.4267 🔴 `-0.0400` | 0.2057 → 0.2000 🔴 `-0.0057` | 0.7510 → 0.7710 🟢 `+0.0200` |
| entropy-grad-norm-low | W4A16 | 0.5867 → 0.5933 🟢 `+0.0066` | 0.2743 → 0.3371 🟢 `+0.0628` | 0.7840 → 0.7870 🟢 `+0.0030` |
| entropy-grad-norm-low | 旧W4A4 | 0.5067 → 0.5000 🔴 `-0.0067` | 0.2400 → 0.2171 🔴 `-0.0229` | 0.7780 → 0.7780 ⚪ `0.0000` |
| entropy-grad-norm-low | 新W4A4 | 0.4800 → 0.4400 🔴 `-0.0400` | 0.2171 → 0.2114 🔴 `-0.0057` | 0.7620 → 0.7620 ⚪ `0.0000` |

从配对结果看，LoRA 对普通 GPTQ 的 AIME 和 LiveCodeBench 三种口径均有提升，其中新W4A4提升最明显；但对两种低熵加权方法并不稳定。MMLU-Pro整体变化较小。

### 新W4A4相对正确W4A16

| 方法 | LoRA | AIME变化 | LCB变化 | MMLU变化 |
|---|---|---:|---:|---:|
| GPTQ | 🔵 Base | -0.1667 | -0.1086 | -0.0310 |
| GPTQ | 🟠 **+LoRA** | -0.1200 | -0.0743 | -0.0190 |
| entropy-grad-low | 🔵 Base | -0.0466 | -0.0972 | -0.0380 |
| entropy-grad-low | 🟠 **+LoRA** | -0.1733 | -0.1257 | -0.0090 |
| entropy-grad-norm-low | 🔵 Base | -0.1067 | -0.0572 | -0.0220 |
| entropy-grad-norm-low | 🟠 **+LoRA** | -0.1533 | -0.1257 | -0.0250 |

非greedy下，新W4A4相对正确W4A16也是18项全部下降。相对旧的“A16校准→A4推理”，A4校准只有4项提高、13项下降、1项不变，没有显示稳定收益。

## 4. 高熵与低熵对照：进行中

### 模型准备状态

| 方向 | 方法 | 校准激活 | GPTQ batch | Importance batch | 模型 | `</think>` LoRA |
|---|---|---|---:|---:|---|---|
| high | entropy-grad | A16 | 1 | 8 | 完成 | 完成 |
| high | entropy-grad-norm | A16 | 1 | 8 | 完成 | 完成 |
| high | entropy-grad | A4 | 16 | 16 | 完成 | 完成 |
| high | entropy-grad-norm | A4 | 16 | 16 | 完成 | 完成 |

高熵量化日志均明确记录 `entropy_direction='high'`；A16为 `hif4a=False`，A4为 `hif4a=True`。checkpoint口径正确。

### 已有完整低熵结果

这张表整理所有已经跑完的 low baseline。A16采用此前完整的非greedy结果，A4采用真正A4Calib模型的结果。

| 方法 | 精度/校准 | LoRA | AIME Avg@5 | LCB | MMLU |
|---|---|---|---:|---:|---:|
| entropy-grad-low | W4A16/A16 | 否 | 0.5133 | 0.3029 | 0.7890 |
| entropy-grad-low | W4A16/A16 | 是 | 0.6000 | 0.3257 | 0.7800 |
| entropy-grad-norm-low | W4A16/A16 | 否 | 0.5867 | 0.2743 | 0.7840 |
| entropy-grad-norm-low | W4A16/A16 | 是 | 0.5933 | 0.3371 | 0.7870 |
| entropy-grad-low | W4A4/A4 | 否 | 0.4667 | 0.2057 | 0.7510 |
| entropy-grad-low | W4A4/A4 | 是 | 0.4267 | 0.2000 | 0.7710 |
| entropy-grad-norm-low | W4A4/A4 | 否 | 0.4800 | 0.2171 | 0.7620 |
| entropy-grad-norm-low | W4A4/A4 | 是 | 0.4400 | 0.2114 | 0.7620 |

所以低熵结果并不是没跑完；上面8组都有可用的历史完整结果。

### 本轮 high/low 匹配评测进度

为了减少随机采样运行差异，当前脚本先重新跑 low A16，再按相同脚本和参数跑 high。本表只表示这一次匹配流水线的完成状态；A4 low直接复用上表中同参数的A4Calib结果。

| 方向 | 方法 | 精度/校准 | LoRA | 本轮AIME | 本轮LCB | 本轮MMLU | 本轮状态 |
|---|---|---|---|---:|---:|---:|---|
| low | entropy-grad | W4A16/A16 | 否 | 0.5733 | 0.2800 | 0.7740 | 完成 |
| low | entropy-grad | W4A16/A16 | 是 | 0.6200 | 0.3429 | 0.7900 | 完成 |
| low | entropy-grad-norm | W4A16/A16 | 否 | — | — | — | **本轮运行中；历史结果已完成** |
| low | entropy-grad-norm | W4A16/A16 | 是 | — | — | — | 本轮排队；历史结果已完成 |
| low | entropy-grad | W4A4/A4 | 否 | 0.4667 | 0.2057 | 0.7510 | 完成、复用 |
| low | entropy-grad | W4A4/A4 | 是 | 0.4267 | 0.2000 | 0.7710 | 完成、复用 |
| low | entropy-grad-norm | W4A4/A4 | 否 | 0.4800 | 0.2171 | 0.7620 | 完成、复用 |
| low | entropy-grad-norm | W4A4/A4 | 是 | 0.4400 | 0.2114 | 0.7620 | 完成、复用 |
| high | entropy-grad | W4A16/A16 | 否 | — | — | — | 模型/LoRA完成，评测排队 |
| high | entropy-grad | W4A16/A16 | 是 | — | — | — | 模型/LoRA完成，评测排队 |
| high | entropy-grad-norm | W4A16/A16 | 否 | — | — | — | 模型/LoRA完成，评测排队 |
| high | entropy-grad-norm | W4A16/A16 | 是 | — | — | — | 模型/LoRA完成，评测排队 |
| high | entropy-grad | W4A4/A4 | 否 | — | — | — | 模型/LoRA完成，评测排队 |
| high | entropy-grad | W4A4/A4 | 是 | — | — | — | 模型/LoRA完成，评测排队 |
| high | entropy-grad-norm | W4A4/A4 | 否 | — | — | — | 模型/LoRA完成，评测排队 |
| high | entropy-grad-norm | W4A4/A4 | 是 | — | — | — | 模型/LoRA完成，评测排队 |

### 为什么现在还没有 high 分数

截至本次重新扫描：

| 检查项 | 实际状态 |
|---|---|
| `eval`目录中的日志 | 只有3个 `low_*` 日志 |
| 完整结果JSON | 只有2个，均为 `low/entropy_grad/a16` |
| 当前运行模型 | `Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-s1k-head-512x4096`，不含 `High` |
| 当前任务进度 | low entropy-grad-norm A16 Base正在生成AIME；最终JSON尚未写出 |
| `high/.../results_*.json` | 0个 |

因此高熵已经完成的是4个量化模型和4个LoRA，不是benchmark。等出现加载路径含 `EntropyGradHigh` 或 `EntropyGradNormHigh` 的评测日志，以及对应 `high/.../results_*.json` 后，才能填入high分数。

新的 low entropy-grad W4A16 分数与历史运行不完全一致，说明vLLM并行采样即使seed相同也不是严格逐token复现。最终比较high/low，应优先使用本轮匹配流水线结果并结合标准误。

## 5. 当前结论

| 问题 | 当前数据结论 |
|---|---|
| A4校准是否优于A16校准后直接测A4？ | 没有。非greedy 18项中4升、13降、1不变。 |
| 真正W4A4是否接近正确W4A16？ | 仍有明显差距；greedy和非greedy各18项全部下降。 |
| `</think>` LoRA是否稳定有效？ | 不稳定。GPTQ新W4A4采样的AIME/LCB提升，但两个entropy新W4A4的AIME下降。 |
| high entropy是否优于low entropy？ | 尚不能判断；high模型已准备好，但benchmark尚未出分。 |

## 6. 原始数据目录

| 内容 | 目录 |
|---|---|
| Greedy正确W4A16/新W4A4 | `results/qwen35_4b_gptq_a4_calib_greedy` |
| 新A4校准非greedy W4A4 | `results/qwen35_4b_gptq_a4_calib_sampling_w4a4` |
| 旧A16校准的W4A16/W4A4 | `results/qwen35_4b_quant_comparison`、`results/qwen35_4b_entropy_grad_awq_lora` |
| 高低熵对照进行中 | `results/qwen35_4b_entropy_direction_comparison` |
| 高低熵量化/训练/评测日志 | `output_zero_shot/qwen35_4b_entropy_direction_comparison` |
