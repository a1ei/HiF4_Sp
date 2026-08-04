#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：请先执行 conda activate hif4" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="1"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-/root/data/.cache/huggingface}"
export VLLM_FLOAT32_MATMUL_PRECISION="highest"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
QUANT_DIR="${QUANT_DIR:-Qmodel/Qwen3.5-27B-OmniQuant-LFQ-W4A16}"
OMNI_DIR="${OMNI_DIR:-outputs/omniquant_lfq_qwen35_27b}"
RESULT_DIR="${RESULT_DIR:-results/qwen35_27b_omniquant_lfq}"
LOG_DIR="${LOG_DIR:-output_zero_shot/qwen35_27b_omniquant_lfq}"

CAL_NSAMPLES="${CAL_NSAMPLES:-128}"
CAL_SEQLEN="${CAL_SEQLEN:-2048}"
CAL_BATCH_SIZE="${CAL_BATCH_SIZE:-1}"
CAL_EPOCHS="${CAL_EPOCHS:-10}"
LFQ_CHUNK_SIZE="${LFQ_CHUNK_SIZE:-16}"
LWC_LR="${LWC_LR:-1e-2}"
LET_LR="${LET_LR:-5e-3}"
LFQ_LR="${LFQ_LR:-2e-3}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-3000}"
DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"

mkdir -p "${OMNI_DIR}" "${RESULT_DIR}" "${LOG_DIR}" "$(dirname "${QUANT_DIR}")"

model_is_complete() {
  [[ -f "${QUANT_DIR}/config.json" ]] &&
    { [[ -f "${QUANT_DIR}/model.safetensors" ]] ||
      [[ -f "${QUANT_DIR}/model.safetensors.index.json" ]]; }
}

if model_is_complete; then
  echo "复用已量化模型: ${QUANT_DIR}"
elif [[ -e "${QUANT_DIR}" ]]; then
  echo "错误：量化模型目录已存在但不完整: ${QUANT_DIR}" >&2
  exit 1
else
  echo "开始 OmniQuant+LFQ W4A16 量化: ${MODEL}"
  python -m HiFloat4.omniquant.main \
    --model "${MODEL}" \
    --calib_dataset s1k-1.1 \
    --nsamples "${CAL_NSAMPLES}" \
    --seqlen "${CAL_SEQLEN}" \
    --batch_size "${CAL_BATCH_SIZE}" \
    --epochs "${CAL_EPOCHS}" \
    --lwc_lr "${LWC_LR}" \
    --let_lr "${LET_LR}" \
    --lfq_lr "${LFQ_LR}" \
    --wbits 4 \
    --abits 16 \
    --lwc \
    --let \
    --lfq \
    --lfq_logits_chunk_size "${LFQ_CHUNK_SIZE}" \
    --output_dir "${OMNI_DIR}" \
    --save_dir "${QUANT_DIR}" \
    2>&1 | tee "${LOG_DIR}/quantize.log"
fi

if ! model_is_complete; then
  echo "错误：量化结束后模型文件不完整: ${QUANT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${OMNI_DIR}/omni_parameters.pth" ]]; then
  echo "错误：缺少 OmniQuant 参数: ${OMNI_DIR}/omni_parameters.pth" >&2
  exit 1
fi

if [[ ! -f "${QUANT_DIR}/processor_config.json" ]]; then
  echo "补齐 Qwen3.5 processor 元数据: ${QUANT_DIR}"
  python - "${MODEL}" "${QUANT_DIR}" <<PY
import sys
from transformers import AutoProcessor

AutoProcessor.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])
PY
fi

python - <<'PY'
import vllm
import vllm._C  # noqa: F401

print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

echo "开始评测: ${DATASETS}，GPU=1，seed=${SEED}"
python main.py \
  --model_path "${QUANT_DIR}" \
  --datasets "${DATASETS}" \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --batch_size "${BATCH_SIZE}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_samples "${MMLU_PRO_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --seed "${SEED}" \
  --use_chat_template \
  --output_dir "${RESULT_DIR}" \
  > "${LOG_DIR}/eval.log" 2>&1

echo "全部完成。"
echo "量化模型: ${QUANT_DIR}"
echo "评测结果: ${RESULT_DIR}"
echo "运行日志: ${LOG_DIR}"
