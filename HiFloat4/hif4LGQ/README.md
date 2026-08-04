# hif4LGQ: Logic Guard Quant

hif4LGQ 在现有权重量化之前调整普通 `nn.Linear` 权重，让分组量化
scale 更小，同时约束逻辑连接词位置的层输出变化。功能默认关闭，融合
`Delta W` 后不会增加推理计算。

## 公式与形状

对一个权重形状为 `[out_features, in_features]` 的 Linear：

```text
X_valid: [valid_tokens, in_features]
H = X_valid.T X_valid: [in_features, in_features]
N: [rank, in_features]
C: [out_features, rank]
Delta W = C N: [out_features, in_features]
W_tilde = W + Delta W
```

`torch.linalg.eigh` 对完整 FP32 Hessian 做精确特征分解。`low` 选择最小
的 `rank` 个特征值，`low_mid` 从
`floor(low_mid_start_quantile * in_features)` 开始选择。`W` 和 `N`
冻结，Adam 只优化完整的 `C`，不按输出行分块。

最终最小化：

```text
L = L_group + lambda_logic L_logic + lambda_reg ||C||F^2
```

逻辑保护损失使用：

```text
P = X_logic N.T
G_logic = P.T P / logic_token_count
L_logic = sum((C G_logic) * C)
        = ||X_logic Delta W.T||F^2 / logic_token_count
```

分组损失将 `W_tilde` 的最后一维切成 group：

```text
L_group = mean(amax(abs(group))^2)
```

同一个 Linear 的所有输出行、所有 group 同时进入一次目标函数。`max`
是默认实现；`logsumexp` 是可选的平滑最大值。HiF4 和 HiF4-1 自动使用
group size 64，NVFP4 自动使用 16，也可用 `--lgq_group_size` 覆盖。

## 数据流

校准集固定为 `s1k-1.1`，每条文本按以下字段连接：

```text
question

deepseek_thinking_trajectory

deepseek_attempt
```

每条样本独立前向，不做样本拼接。`--lgq_calib_seq_len 0` 保留每条完整
实际序列；正数时删除该长度之后的尾部 token。Hessian 使用所有有效
token，padding 不参与。首层输入及模型产生的 attention/position 参数
被捕获后，后续按 decoder layer 顺序前向。

同一输入位置的 Linear 共用激活统计，例如 q/k/v、gate/up，以及
Qwen3.5 linear attention 的 qkv/z/b/a。每个 Linear 保留自己的完整 `C`，
同输入组内的所有 `C` 放入同一个 Adam，每步将各 Linear 的目标相加后执行
一次反向传播和参数更新。一层融合全部 `Delta W` 后重新前向，得到下一层
校准输入。

## 逻辑关键词

默认词表是 `default_logic_keywords.json`，包括因果结论、转折、递进与
结构、答案提示、反思纠错词。可传入同格式 JSON 列表或分类对象：

```text
--lgq_logic_keywords_path path/to/keywords.json
```

每个词或短语会编码成 tokenizer ID 序列并进行连续子序列匹配。多 token
关键词的全部组成位置都进入 `X_logic`。日志会报告每个 Linear 实际
匹配并保护的 token 数；匹配数为零时直接报错。

## 参数

```text
--hif4lgq false
--lgq_calib_seq_len 0
--lgq_subspace_mode low
--lgq_subspace_rank 64
--lgq_low_mid_start_quantile 0.1
--lgq_steps 100
--lgq_log_interval 500
--lgq_lr 1e-3
--lgq_lambda_logic 1.0
--lgq_lambda_reg 1e-4
--lgq_group_size 0
--lgq_group_loss max
--lgq_group_smooth_tau 1e-3
--lgq_logic_keywords_path
--lgq_target_patterns "*"
--lgq_artifact_dir
--lgq_save_mode none|delta|weights|both
```

LGQ 与后续 GPTQ 可以使用不同的 attention backend。模型加载和 LGQ 使用
`--attn-implementation`，`--gptq_attn_implementation` 仅在 LGQ 完成、运行新的
GPTQ 校准前生效；默认 `same` 保持原 backend。Qwen3.5 完整序列 LGQ 可使用
`flash_attention_2`，固定短序列 GPTQ 可切换为 `eager`：

```bash
GPTQ_ATTN_IMPLEMENTATION=eager \
bash HiFloat4/quantize_qwen3_5_27b.sh \
  --attn-implementation flash_attention_2
```

`--lgq_target_patterns` 使用 shell 风格匹配本层名字或完整名字，例如
`"self_attn.*" "mlp.*"`。`exclude_layers` 仍按完整名字精确排除。
`cal_nsamples` 决定 LGQ 样本数；`cal_seqlen` 仍供后续 GPTQ、MagR 等
原量化流程使用，两者不要混淆。

## 日志与保存

每个目标 Linear 输出 Hessian token 数、逻辑 token 数、完整与选中谱
范围、group/logic/reg/total loss、`||C||F`、`||Delta W||F`，以及优化
前后的 group amax 均值和最大值。

`--lgq_log_interval N` 每隔 N 个 Adam step 额外输出该 Linear 的训练过程；
设为 0 关闭。过程日志同时报告未加权的 `logic`/`reg`、乘以 lambda 后
真正进入总损失的 `weighted_logic`/`weighted_reg`，以及
`||C||F`、`||Delta W||F` 和 `||grad(C)||F`。
`(hif4LGQ/train-group)` 先输出同输入组内所有 Linear 目标之和，随后相同
step 的 `(hif4LGQ/train)` 行分别输出每个 Linear 的损失。

指定 `--lgq_artifact_dir` 后始终写入 `config.json` 和 `metrics.jsonl`。
保存模式：

```text
delta   每个 Linear 保存 FP32 的 C N 到 delta/*.safetensors，并写 manifest
weights 保存融合后的 Hugging Face 模型到 optimized_model/
both    同时保存两者
none    不保存 Delta W 或模型权重
```

## 运行

先激活项目环境：

```bash
conda activate hif4
```

以 s1k-1.1 的 128 条完整 reasoning 样本做 LGQ，再运行 MagR：

```bash
CUDA_VISIBLE_DEVICES=5 \
HIF4LGQ=true \
MAGR=true AWQ=false GPTQ=false \
CAL_DATASET=s1k-1.1 CAL_NSAMPLES=128 \
LGQ_CALIB_SEQ_LEN=0 \
LGQ_ARTIFACT_DIR=output/hif4lgq_magr \
LGQ_SAVE_MODE=delta \
OUTPUT=Qmodel/Qwen3.5-27B-HiF4-LGQ-MagR \
bash HiFloat4/quantize_qwen3_5_27b.sh --save_only
```

CPU smoke test：

```bash
conda run -n hif4 python HiFloat4/hif4LGQ/smoke_test.py
```

## 性能注意

本实现严格构造 `[in_features, in_features]` 的完整 FP32 Hessian，并做
精确特征分解，时间复杂度为 `O(in_features^3)`，Hessian 内存为
`O(in_features^2)`。`lgq_calib_seq_len=0` 还会保留 128 条完整 reasoning
序列的逐层 hidden states，CPU 内存和运行时间都很大。实现不会自动截断、
近似分解或降低样本数；资源不足时应显式设置参数。
