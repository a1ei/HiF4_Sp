# Qwen3.5-4B 高熵/低熵重要性实验汇总

## 实验口径

- 模型：Qwen3.5-4B，GPTQ HiF4 W4权重。
- 校准：s1k-1.1，512×4096，head切片，seed 42；所有权重均使用A16/BF16激活计算Hessian。
- W4A16：A16校准权重，A16推理。
- W4A4：同一个A16校准权重，推理时启用HiF4 A4激活；目录名为 `a4_from_a16`。
- 评测：AIME25 Avg@5、LiveCodeBench v6、MMLU-Pro 1000题。

## 完整结果

| 熵方向 | 重要性 | 精度 | LoRA | AIME25 Avg@5 | LiveCodeBench | MMLU-Pro |
|---|---|---|---|---:|---:|---:|
| 低熵 | entropy-grad | W4A16 | Base | 0.5733 | 0.2800 | 0.7740 |
| 低熵 | entropy-grad | W4A16 | +LoRA | 0.6200 | **0.3429** | 0.7900 |
| 低熵 | entropy-grad | W4A4 | Base | 0.5467 | 0.2171 | 0.7720 |
| 低熵 | entropy-grad | W4A4 | +LoRA | 0.4667 | 0.2114 | 0.7690 |
| 低熵 | entropy-grad-norm | W4A16 | Base | 0.6000 | 0.3143 | 0.7930 |
| 低熵 | entropy-grad-norm | W4A16 | +LoRA | 0.5600 | 0.3143 | **0.7950** |
| 低熵 | entropy-grad-norm | W4A4 | Base | 0.5333 | 0.1886 | 0.7780 |
| 低熵 | entropy-grad-norm | W4A4 | +LoRA | 0.5067 | 0.2000 | 0.7510 |
| 高熵 | entropy-grad | W4A16 | Base | 0.6267 | 0.2571 | 0.7820 |
| 高熵 | entropy-grad | W4A16 | +LoRA | **0.6533** | 0.2686 | 0.7790 |
| 高熵 | entropy-grad | W4A4 | Base | 0.5067 | 0.1943 | 0.7560 |
| 高熵 | entropy-grad | W4A4 | +LoRA | 0.5467 | 0.1771 | 0.7630 |
| 高熵 | entropy-grad-norm | W4A16 | Base | 0.5600 | 0.2686 | 0.7860 |
| 高熵 | entropy-grad-norm | W4A16 | +LoRA | 0.6133 | 0.3200 | 0.7930 |
| 高熵 | entropy-grad-norm | W4A4 | Base | 0.4933 | 0.2000 | 0.7570 |
| 高熵 | entropy-grad-norm | W4A4 | +LoRA | 0.5133 | 0.2000 | 0.7620 |

粗体是对应指标在16组中的最高值。

## 高熵相对低熵的直接对比

固定相同的重要性公式、精度和LoRA状态，只改变熵方向。变化量为“高熵分数−低熵分数”：🟢高熵更好，🔵低熵更好，⚪相同。

| 重要性 | 精度 | LoRA | AIME变化 | LCB变化 | MMLU变化 |
|---|---|---|---:|---:|---:|
| entropy-grad | W4A16 | Base | 🟢 +0.0534 | 🔵 -0.0229 | 🟢 +0.0080 |
| entropy-grad | W4A16 | +LoRA | 🟢 +0.0333 | 🔵 -0.0743 | 🔵 -0.0110 |
| entropy-grad | W4A4 | Base | 🔵 -0.0400 | 🔵 -0.0228 | 🔵 -0.0160 |
| entropy-grad | W4A4 | +LoRA | 🟢 +0.0800 | 🔵 -0.0343 | 🔵 -0.0060 |
| entropy-grad-norm | W4A16 | Base | 🔵 -0.0400 | 🔵 -0.0457 | 🔵 -0.0070 |
| entropy-grad-norm | W4A16 | +LoRA | 🟢 +0.0533 | 🟢 +0.0057 | 🔵 -0.0020 |
| entropy-grad-norm | W4A4 | Base | 🔵 -0.0400 | 🟢 +0.0114 | 🔵 -0.0210 |
| entropy-grad-norm | W4A4 | +LoRA | 🟢 +0.0066 | ⚪ 0.0000 | 🟢 +0.0110 |

24个配对指标中，低熵更好14项，高熵更好9项，1项相同。低熵整体胜出更多，尤其在LiveCodeBench和MMLU-Pro上；高熵的优势更多出现在AIME25，但不稳定。

## LoRA相对Base的变化

| 熵方向 | 重要性 | 精度 | AIME变化 | LCB变化 | MMLU变化 |
|---|---|---|---:|---:|---:|
| 低熵 | entropy-grad | W4A16 | +0.0467 | +0.0629 | +0.0160 |
| 低熵 | entropy-grad | W4A4 | -0.0800 | -0.0057 | -0.0030 |
| 低熵 | entropy-grad-norm | W4A16 | -0.0400 | 0.0000 | +0.0020 |
| 低熵 | entropy-grad-norm | W4A4 | -0.0266 | +0.0114 | -0.0270 |
| 高熵 | entropy-grad | W4A16 | +0.0266 | +0.0115 | -0.0030 |
| 高熵 | entropy-grad | W4A4 | +0.0400 | -0.0172 | +0.0070 |
| 高熵 | entropy-grad-norm | W4A16 | +0.0533 | +0.0514 | +0.0070 |
| 高熵 | entropy-grad-norm | W4A4 | +0.0200 | 0.0000 | +0.0050 |

## 结论

- W4A16总体优于W4A4。
- 低熵 entropy-grad + LoRA 的W4A16结果最均衡，三项都比其Base提高。
- 高熵 entropy-grad + LoRA取得最高AIME25；低熵 entropy-grad + LoRA取得最高LiveCodeBench；低熵 entropy-grad-norm + LoRA取得最高MMLU-Pro。
- W4A4下LoRA效果不稳定：高熵两种方法的AIME均提高，但低熵两种方法的AIME均下降。
- 当前结果不能说明高熵或低熵方向全面占优，收益取决于重要性公式、激活精度和任务。

## 数据位置

- 结果JSON：`results/qwen35_4b_entropy_direction_comparison/`
- 日志：`output_zero_shot/qwen35_4b_entropy_direction_comparison/`
