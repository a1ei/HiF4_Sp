# Entropy-weighted GPTQ

这个选项在原 GPTQ Hessian 统计上增加位置级 next-token entropy 权重，不引入训练、反向传播或模型结构修改。默认关闭，关闭时走原始 GPTQ Hessian 路径。

## 参数

- `--token_importance none|entropy|entropy_grad`：默认 `none`。`entropy` 表示全局 entropy-weighted Hessian；`entropy_grad` 表示 module-group-specific activation-gradient importance。
- `--entropy_alpha FLOAT`：默认 `1.0`。控制 entropy 权重强度，必须大于或等于 0。
- `--entropy_norm minmax|zscore|mean`：默认 `minmax`。控制 entropy 归一化方式。
- `--importance_alpha FLOAT`：默认 `1.0`。控制 `entropy_grad` 局部权重强度，必须大于或等于 0。
- `--importance_batch_size INT`：默认 `1`。只控制 `entropy_grad` 归因前向/反向一次并行处理的校准样本数。增大后会减少逐层 CPU/GPU 权重搬运次数，但会增加显存占用；不会改变 GPTQ Hessian 阶段的 batch。
- `--importance_mean_normalize true|false`：默认 `true`。把每层每组有效 token 权重归一化到均值 1。

Shell 脚本对应环境变量：

- `TOKEN_IMPORTANCE`
- `ENTROPY_ALPHA`
- `ENTROPY_NORM`
- `IMPORTANCE_ALPHA`
- `IMPORTANCE_MEAN_NORMALIZE`

## 计算方式

量化前先使用未量化 FP 模型对固定 calibration inputs 做一次无梯度前向。位置 `i` 的 logits 预测 token `i+1`，因此使用该位置的预测分布熵：

```text
H_i = -sum_v p_i(v) log(p_i(v))
importance_i = max(normalized(H)) - normalized(H_i)
lambda_i = 1 + entropy_alpha * importance_i
```

因此 entropy 越低，`lambda` 越大，该位置对 GPTQ Hessian 的贡献越大。

最后一个有效 token 没有 next-token entropy，权重设为 `1`。Padding token 权重设为 `0`，不进入 Hessian。当前内置 calibration loader 不做 padding；Qwen3.5 现有 GPTQ 路径仍会拒绝带 padding 的 `attention_mask`，其他字典 batch 的 mask 会参与有效 token 判断。

归一化定义：

- `minmax`：先计算 `(H - H_min) / (H_max - H_min)`。
- `zscore`：先计算 `(H - mean) / std`，再减去最小 z-score，使结果非负并保持标准差单位。
- `mean`：先计算 `H / mean(H)`。

三种模式最后都使用 `max(normalized) - normalized` 反转，使低 entropy 对应高 importance。

GPTQ 捕获的 activation 展平后为 `[tokens, hidden]`，内部转置为 `[hidden, tokens]`。加权 Hessian 使用：

```text
X_weighted = X * sqrt(lambda)
H += X_weighted @ X_weighted.T
```

所有 entropy 和 lambda 都按整个校准集的有效 token 统计和归一化。启用后日志会输出 entropy、lambda 的 mean/std/min/max，并逐层显示是否启用 weighted Hessian。

## Module-group-specific entropy-gradient importance

使用 `TOKEN_IMPORTANCE=entropy_grad` 时，每个 calibration batch 单独完成一次 FP 前向和反向，不创建 optimizer，也不更新模型参数。模型参数在归因期间临时设置为 `requires_grad=False`，归因结束后恢复。

输出保护目标按下面的步骤计算：

```text
entropy_i = -sum_v p_i(v) log(p_i(v))
seed_i = 1 - minmax(entropy_i)
margin_i = top1_logit_i - top2_logit_i
loss_anchor = -sum_i(seed_i * margin_i) / sum_i(seed_i)
```

`seed` 会 detach，反向只经过 margin，不经过 entropy。最后一个位置和 padding 位置的 seed 都是 0。logits 按序列分块计算，不跨 batch 保存完整词表 logits。

每个 Transformer block 捕获四组 linear 输入：

- `qkv`：full-attention 中 `self_attn.q_proj` 的输入。该 tensor 同时被 q/k/v 使用，所以梯度包含三条分支；Qwen3.5 linear-attention 中对应 `linear_attn.in_proj_qkv` 的输入，同时共享给 `in_proj_z/b/a`。
- `o`：`self_attn.o_proj` 或 `linear_attn.out_proj` 的输入。
- `up_gate`：`mlp.up_proj` 的输入，该 tensor 同时被 up/gate 使用。
- `down`：`mlp.down_proj` 的输入，即激活和 gating 相乘后的 intermediate activation。

捕获 activation 和对应梯度的 shape 都是 `[batch, seqlen, group_hidden]`。局部重要性为：

```text
importance = norm(activation * gradient, dim=-1)  # [batch, seqlen]
```

每层每组在整个校准集的有效 token 上做 min-max：

```text
weight = 1 + importance_alpha * minmax(importance)
```

当 `importance_mean_normalize=true` 时，再除以有效 token 的 weight 均值。Padding 最终为0。得到的 `[num_samples, seqlen]` 权重按与 activation 完全相同的 batch-major 顺序 flatten。

Hessian 映射：

- q/k/v，以及 linear-attention 的 `in_proj_qkv/z/b/a`：使用当前层 `qkv` 权重。
- o/out projection：使用当前层 `o` 权重。
- up/gate：使用当前层 `up_gate` 权重。
- down：使用当前层 `down` 权重。

每个 linear 仍有独立 GPTQ 对象和独立量化流程，只共享对应 token weight。

## 启动量化

原始 GPTQ，不启用 token 权重：

```bash
GPTQ=true \
AWQ=false \
SMOOTHQUANT=false \
MAGR=false \
MODEL=Qwen/Qwen3.5-27B \
OUTPUT=Qmodel/Qwen3.5-27B-HiF4-GPTQ-c4 \
CAL_DATASET=c4 \
CAL_NSAMPLES=512 \
CAL_SEQLEN=512 \
TOKEN_IMPORTANCE=none \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

Entropy-weighted GPTQ：

```bash
GPTQ=true \
AWQ=false \
SMOOTHQUANT=false \
MAGR=false \
MODEL=Qwen/Qwen3.5-27B \
OUTPUT=Qmodel/Qwen3.5-27B-HiF4-GPTQ-s1k-entropy \
CAL_DATASET=s1k-1.1 \
CAL_NSAMPLES=512 \
CAL_SEQLEN=1536 \
CAL_SLICE_MODE=head \
TOKEN_IMPORTANCE=entropy \
ENTROPY_ALPHA=1.0 \
ENTROPY_NORM=minmax \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

Module-group-specific entropy-gradient GPTQ：

```bash
GPTQ=true \
AWQ=false \
SMOOTHQUANT=false \
MAGR=false \
MODEL=Qwen/Qwen3.5-27B \
OUTPUT=Qmodel/Qwen3.5-27B-HiF4-GPTQ-s1k-entropy-grad \
CAL_DATASET=s1k-1.1 \
CAL_NSAMPLES=512 \
CAL_SEQLEN=1536 \
CAL_SLICE_MODE=head \
TOKEN_IMPORTANCE=entropy_grad \
IMPORTANCE_ALPHA=1.0 \
IMPORTANCE_MEAN_NORMALIZE=true \
IMPORTANCE_BATCH_SIZE=4 \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

Entropy 模式会额外执行一次完整 FP 前向和一次第一层输入重新捕获，因此比原始 GPTQ 更慢。校准 activation 仍常驻 CPU。`entropy_grad` 按 `IMPORTANCE_BATCH_SIZE` 将多个等长样本一起送入每层；低熵 seed 和 anchor loss 都保持逐样本独立，因此 batch size 不改变 importance 定义，最后一个不足完整 batch 的分组也会正常处理。logits chunk 会随 batch size 自动减小，以控制词表 logits 的峰值显存。

## Sanity check

```bash
conda run -n hif4 python HiFloat4/hif4gptq/gptq/sanity_check_entropy_weighted.py
```

检查内容：

- `token_importance=none` 的 Hessian 累积与原公式逐元素一致。
- entropy 模式下 Hessian shape 不变。
- flatten 后的 token 权重数量与 activation token 数一致。
- padding 权重为 0，最后一个有效 token 权重为 1。
- `entropy_grad` 的四组 activation-gradient 均能生成与 token mask 同 shape 的权重。
- 归因后模型参数没有 gradient，`requires_grad` 状态恢复。
