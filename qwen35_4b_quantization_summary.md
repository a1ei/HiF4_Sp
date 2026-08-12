# Qwen3.5-4B 量化实验小结

## 1. 低熵 token 加权 GPTQ

数字、符号等低熵 token 的候选通常更集中，预测错误后的容错空间也更小。因此，我们在 GPTQ 校准时提高这类 token 的权重。

计算下一 token 的熵，并让低熵位置获得更大的种子权重：

```text
H_t = -sum_v p[t,v] * log(p[t,v])
s_t = 1 - (H_t - H_min) / (H_max - H_min + eps)
```

用 top-1 与 top-2 的 logit 间隔构造锚点损失：

```text
margin_t = top1_logit_t - top2_logit_t
L_anchor = -sum_t(s_t * margin_t) / sum_t(s_t)
```

反向传播后，每层、每个投影组分别计算 token 重要性：

```text
entropy-grad:      I_t = L2_norm(y_t * dL_anchor/dy_t)
entropy-grad-norm: I_t = L2_norm(dL_anchor/dy_t)
```

重要性经过 min-max 归一化，得到均值归一为 1 的权重：

```text
w_t = 1 + alpha * normalize(I_t)
Hessian ∝ sum_t w_t * x_t * x_t^T
L_GPTQ ≈ sum_t w_t * ||W*x_t - Q(W)*x_t||^2
```

因此，低熵且梯度敏感的输出位置会对 GPTQ 权重量化产生更大影响。

## 2. `</think>` LM Head LoRA

量化模型在结束思考的位置，对 `</think>` 的预测概率可能发生偏移。这里不蒸馏整段文本，也不蒸馏所有词表输出；每条样本只选择9个位置，并且每个位置只比较 `</think>` 的概率：

```text
1个正位置：思考内容的最后一个token，下一token确实是</think>
8个负位置：在思考过程内部按token序号等间隔选择，下一token不是</think>
```

例如思考区间有80个可选token，8个负位置大致取第1、12、23、34、46、57、68、80个。这样只是覆盖思考的前、中、后段，不判断该位置是数字、连接词还是其他内容。

教师和量化模型分别输出完整logits，经过softmax后，只取 `</think>` 对应的概率：

```text
p_teacher = softmax(teacher_logits)[</think>]
p_student = softmax(student_logits)[</think>]
```

这里：

- `p_teacher`：BF16教师认为当前位置下一token是 `</think>` 的概率。
- `p_student`：量化模型认为当前位置下一token是 `</think>` 的概率。
- `y`：数据给出的真实目标。正位置为1，负位置为0。

BCE用来衡量预测概率 `p` 与目标 `q` 的差距：

```text
BCE(p, q) = -q * log(p) - (1-q) * log(1-p)
```

每个选中位置包含两项损失：

```text
教师概率蒸馏损失 = BCE(p_student, p_teacher)
真实位置监督损失 = BCE(p_student, y)
总损失           = 教师概率蒸馏损失 + 真实位置监督损失
这两部分损失都是只看上面的几个位置，可以理解为均匀的从数据中采8个位置来计算，主要想不让推理过程提前输出</think>
```

举例：

```text
真正结束的位置：p_teacher=0.90，p_student=0.55，y=1
两项损失都会推动p_student升高。

思考中间的负位置：p_teacher=0.02，p_student=0.20，y=0
两项损失都会推动p_student降低，避免模型过早输出</think>。
```

计算整条样本损失时，正位置损失占一半，8个负位置的平均损失占一半，避免负位置因为数量更多而压过正位置。

LoRA只更新LM Head中 `</think>` token ID对应的那一行；其他词表行保持为0，不参与更新。W4A16 LoRA使用A16激活训练；W4A4 LoRA使用同一个A16校准权重模型，但训练student时开启HiF4 A4激活。

## 3. 结果

| 方法 | 精度 | AIME25 Avg@5 | LiveCodeBench | MMLU-Pro |
|---|---|---:|---:|---:|
| BF16 | BF16 | 0.5667 | **0.3543** | 0.7890 |
| GPTQ | W4A16 | 0.6000 | 0.2800 | **0.7990** |
| GPTQ + `</think>` LoRA | W4A16 | **0.6200** | 0.2857 | 0.7870 |
| GPTQ + entropy-grad | W4A16 | 0.5133 | 0.3029 | 0.7890 |
| GPTQ + entropy-grad + `</think>` LoRA | W4A16 | 0.6000 | 0.3257 | 0.7800 |
| GPTQ + entropy-grad-norm | W4A16 | 0.5867 | 0.2743 | 0.7840 |
| GPTQ + entropy-grad-norm + `</think>` LoRA | W4A16 | 0.5933 | **0.3371** | 0.7870 |
| GPTQ | W4A4 | 0.4400 | 0.2171 | 0.7660 |
| GPTQ + `</think>` LoRA | W4A4 | 0.4933 | 0.2229 | 0.7640 |
| GPTQ + entropy-grad | W4A4 | **0.5133** | 0.2057 | 0.7650 |
| GPTQ + entropy-grad + `</think>` LoRA | W4A4 | 0.4800 | 0.2343 | 0.7680 |
| GPTQ + entropy-grad-norm | W4A4 | 0.5067 | **0.2400** | **0.7780** |
| GPTQ + entropy-grad-norm + `</think>` LoRA | W4A4 | 0.5000 | 0.2171 | **0.7780** |

W4A4结果使用A16校准得到的W4权重，并在推理时开启HiF4 A4激活。表中的W4A4 LoRA是此前复用W4A16 adapter得到的结果；当前单独训练的W4A4 adapter结果尚未加入。

总体上，低熵加权在W4A4上有一定收益；`</think>` LoRA在W4A16的AIME25和LiveCodeBench上提升更稳定，但不是所有配置、所有任务都会提升。
