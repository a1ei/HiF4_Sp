# HiFloat4

HiFloat4 是 Float4 fake quant / quant-dequant 实验代码。本仓库中它主要用于 Qwen3.5 的 RTN / GPTQ 权重量化，并保存 Hugging Face 格式模型。

当前统一使用根目录的 `hif4` 环境。不要再单独维护 `qhif4`。

## 环境

先在仓库根目录安装主环境：

```bash
conda create -n hif4 python=3.11 -y
conda activate hif4

bash install.sh
```

安装脚本会同时编译：

- `3rdparty/vllm`，其中已经包含本项目需要的 Qwen3.5 / HiF4 适配。
- `HiFloat4/hif4_gpu` CUDA 扩展。
- HiFloat4 量化链路需要的 `accelerate` / `datasets` / `safetensors` / `tqdm` 等运行依赖。

`install.sh` 最后会自动导入检查。手动复查：

```bash
conda activate hif4
cd HiF4_Sp

python -c "import HiFloat4.main as m; from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM; print(Qwen3_5ForCausalLM.__name__, 'ok')"
```

如需单独重编 HiFloat4 CUDA 扩展：

```bash
conda activate hif4
cd HiF4_Sp/HiFloat4/hif4_gpu
bash build.sh
```

HiFloat4 CUDA kernel 要求输入 tensor 在 CUDA 上，所以量化机器必须有 NVIDIA GPU。

## 量化 Qwen3.5

默认量化 `Qwen/Qwen3.5-27B`，输出到仓库根目录下的 `Qmodel/`：

```bash
conda activate hif4
cd HiF4_Sp

bash HiFloat4/quantize_qwen3_5_27b.sh
```

更建议显式指定输出目录，避免写满仓库所在磁盘：

```bash
OUTPUT=/data/Qwen3.5-27B-HiF4-RTN \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

如果模型已经下载到本地：

```bash
MODEL=/path/to/Qwen3.5-27B \
OUTPUT=/data/Qwen3.5-27B-HiF4-RTN \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

脚本内部实际调用：

```bash
python HiFloat4/main.py \
  --model "${MODEL}" \
  --dtype "${DTYPE}" \
  --hif4w true \
  --hif4_weight_format "${HIF4_WEIGHT_FORMAT}" \
  --gptq "${GPTQ}" \
  --gptq_save_path "${OUTPUT}" \
  --cal_dataset "${CAL_DATASET}" \
  --cal_nsamples "${CAL_NSAMPLES}" \
  --cal_seqlen "${CAL_SEQLEN}" \
  --cal_slice_mode "${CAL_SLICE_MODE}" \
  --cal_slice_offset "${CAL_SLICE_OFFSET}" \
  --gptq_percdamp "${GPTQ_PERCDAMP}" \
  --block_size_linear "${BLOCK_SIZE_LINEAR}"
```

## 常用参数

```bash
# 改输出目录
OUTPUT=/data/Qwen3.5-27B-HiF4-RTN bash HiFloat4/quantize_qwen3_5_27b.sh

# 改模型来源
MODEL=/data/models/Qwen3.5-27B bash HiFloat4/quantize_qwen3_5_27b.sh

# 改保存 dtype
DTYPE=bfloat16 bash HiFloat4/quantize_qwen3_5_27b.sh

# 改 HiF4 权重量化格式
HIF4_WEIGHT_FORMAT=hif4-1 OUTPUT=/data/Qwen3.5-27B-HiF4-1-RTN bash HiFloat4/quantize_qwen3_5_27b.sh

# NVFP4 RTN + SmoothQuant scale-only + NVFP4 激活 fake quant
AWQ=false \
SMOOTHQUANT=true \
SMOOTHQUANT_SCALE_ONLY=true \
HIF4_WEIGHT_FORMAT=nvfp4 \
HIF4A=true \
ACT_QUANT_FORMAT=nvfp4 \
OUTPUT=/data/Qwen3.5-27B-NVFP4-RTN-SQScaleOnly-ACTNVFP4 \
bash HiFloat4/quantize_qwen3_5_27b.sh

# 使用 GPTQ 路径
GPTQ=true \
CAL_DATASET=c4 \
CAL_NSAMPLES=512 \
CAL_SEQLEN=512 \
OUTPUT=/data/Qwen3.5-27B-HiF4-GPTQ \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

`SMOOTHQUANT_SCALE_ONLY=true` 时只校准并应用 SmoothQuant scale，不再做 SmoothQuant 后的权重量化。Scale 已经写进前置 norm 和 Linear 权重，权重缩放不需要额外 scale 文件。`HIF4A=true` 的激活 fake quant 只在本次 `HiFloat4/main.py` 运行里生效，不会保存进普通 Hugging Face checkpoint。

使用 s1K-1.1 `question + deepseek_thinking_trajectory + deepseek_attempt` 完整序列的前或后 2048 个 token：

```bash
CAL_DATASET=s1k-1.1 CAL_SEQLEN=2048 CAL_SLICE_MODE=head \
OUTPUT=/data/Qwen3.5-27B-HiF4-s1k-head-2048 \
bash HiFloat4/quantize_qwen3_5_27b.sh

CAL_DATASET=s1k-1.1 CAL_SEQLEN=2048 CAL_SLICE_MODE=tail \
OUTPUT=/data/Qwen3.5-27B-HiF4-s1k-tail-2048 \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

使用 TACO train split 做校准时，loader 会先按 LiveCodeBench v6 test 的题面去重，再取 `question + starter_code + first_solution`；每条校准样本都从一个 `Question:` 开始，不够 `CAL_SEQLEN` 时继续拼接后续题目补足：

```bash
CAL_DATASET=taco CAL_SEQLEN=2048 CAL_SLICE_MODE=head \
OUTPUT=/data/Qwen3.5-27B-HiF4-taco-head-2048 \
bash HiFloat4/quantize_qwen3_5_27b.sh
```

没有第一份 solution 的样本会被跳过；无法凑够要求的样本数时才报错。

保存成功后，输出目录应包含：

- `config.json`
- `generation_config.json`
- tokenizer 文件
- 分片权重文件
- `model.safetensors.index.json`

中途因为磁盘满或进程中断留下的目录不是完整模型，不要继续当作可用 checkpoint。

## 原始 HiFloat4 用法

CUDA 示例：

```python
import torch
from quant_cy import QType, quant_dequant_float

qtype = QType("hifx4").dim(0)
x_sim = quant_dequant_float(x.cuda(), qtype, force_py=False, force_fp32=True)
w_sim = quant_dequant_float(w.cuda(), qtype, force_py=False, force_fp32=True)
y = torch.nn.functional.linear(x_sim, w_sim)
```

NPU 示例：

```python
import torch
from quant_cy_npu import QType, quant_dequant_float

qtype = QType("hifx4").dim(0)
x_sim = quant_dequant_float(x.npu(), qtype, force_py=False, force_fp32=True)
w_sim = quant_dequant_float(w.npu(), qtype, force_py=False, force_fp32=True)
y = torch.nn.functional.linear(x_sim, w_sim)
```

## Citation

```bibtex
@misc{luo2026hifloat4formatlanguagemodel,
      title={HiFloat4 Format for Language Model Inference},
      author={Yuanyong Luo and Jing Huang and Yu Cheng and Ziwei Yu and Kaihua Tang and Xinda Ma and Xin Wang and Anping Tong and Guipeng Hu and Mehran Taghian and Peng Wu and Guanglin Li and Yunke Peng and Tianchi Hu and Minqi Chen and Michael Bi Mi and Hu Liu and Xiping Zhou and Junsong Wang and Qiang Lin and Heng Liao},
      year={2026},
      eprint={2602.11287},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.11287},
}
```
