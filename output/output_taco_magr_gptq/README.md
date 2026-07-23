# TACO 校准 MagR / GPTQ 评测汇总

全部分数均换算为百分数。两个模型都基于 `Qwen/Qwen3.5-27B`，使用 TACO train 校准，并与 LiveCodeBench v6 test 按题面去重。校准文本由 `question + starter_code + first_solution` 组成，每个校准样本从一道新题开始；共使用 128 个 4096-token 样本。

权重采用 HiF4 `hifx4`、group size 64，排除 `lm_head`。评测时启用 `--fake_act_quant hif4`，使用 TP=2、temperature=0.7、top-p=0.8、top-k=20。AIME25 每种方法使用 seed 1234-1238 运行 5 次；LiveCodeBench v6 使用全部 175 题；MMLU-Pro 使用 3000 题。

| 方法 | AIME25 pass@1 | LiveCodeBench pass@1:16 | MMLU-Pro |
|---|---:|---:|---:|
| GPTQ | **78.00 +/- 6.06** | **40.57 +/- 3.72** | **85.40 +/- 0.64** |
| MagR | 76.00 +/- 6.41 | 34.86 +/- 3.61 | 84.43 +/- 0.66 |

其中 AIME25 的 `+/-` 是 5 次运行的样本标准差，LiveCodeBench 和 MMLU-Pro 的 `+/-` 是评测器报告的标准误。

## AIME25 五次结果

| 方法 | seed 1234 | seed 1235 | seed 1236 | seed 1237 | seed 1238 | 平均值 |
|---|---:|---:|---:|---:|---:|---:|
| GPTQ | 70.00 | 83.33 | 83.33 | 80.00 | 73.33 | **78.00** |
| MagR | 66.67 | 80.00 | 76.67 | 83.33 | 73.33 | 76.00 |

本次 `n=1`，所以每次 AIME 的 `pass@1` 与 `avg@n` 相同。

## 对比结论

- GPTQ 在三个任务上都高于 MagR：AIME25 高 2.00 个百分点、LiveCodeBench 高 5.71 个百分点、MMLU-Pro 高 0.97 个百分点。
- AIME25 只有 30 题且带采样随机性；LCB 和 MMLU-Pro 各只运行一次。当前差距均不足以稳健证明 GPTQ 显著优于 MagR，只能认为 GPTQ 呈现一致的领先趋势。
- 该目录没有 BF16/FP16 原模型基线，因此无法从这些日志判断两种量化方法相对原模型的绝对精度损失。

## 日志状态与注意事项

- 14 份评测日志全部包含“评估完成”，未发现 OOM 或 NaN。
- `quant_magr_taco_head_128_4096.log` 和 `quant_gptq_taco_head_128_4096.log` 都是在导入 PyTorch 时被 `KeyboardInterrupt` 中止的重跑日志，不是成功量化日志；当前脚本中的两个量化调用也已被注释。被评测的模型目录存在，并各自保存了 `quantization_args.json`。
- MMLU-Pro 设置了 `max_samples=3000`，只能与使用相同 3000 题设置的结果横向比较，不能直接当作完整 MMLU-Pro 分数。
- 所有评测都设置 `max_model_len=32768` 和 `max_new_tokens=32768`，因此 lighteval 报告上下文截断到 0 的警告。当前代码中的 `input[-0:]` 实际保留了完整 prompt，但生成长度最终由 vLLM 的上下文上限约束；后续建议把 `max_new_tokens` 调低到 30000 或更小，消除这一边界配置。
- 日志还包含 vLLM 版本约束和 FLA 输入格式警告。两种方法使用相同环境，因此横向比较条件一致，但 FLA 警告值得单独核查。

完整小数结果见 `summary.csv`。
