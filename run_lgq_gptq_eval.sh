#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：请先执行 conda activate hif4" >&2
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

MODEL_PATH="${MODEL_PATH:-Qmodel/Qwen3.5-27B-HiF4-LGQ-GPTQ}"
GPU="${GPU:-4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
FAKE_ACT_QUANT="${FAKE_ACT_QUANT:-hif4}"
RESULT_DIR="${RESULT_DIR:-results/lgq_gptq_eval}"
LOG_DIR="${LOG_DIR:-output_zero_shot/lgq_gptq_eval}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型目录不存在: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

run_eval() {
  local datasets="$1"
  local name="$2"
  local max_samples="${3:-}"
  local optional_args=()

  if [[ -n "${max_samples}" ]]; then
    optional_args+=(--max_samples "${max_samples}")
  fi

  echo "开始 ${name}: GPU=${GPU}, activation=${FAKE_ACT_QUANT}"
  CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
    --model_path "${MODEL_PATH}" \
    --datasets "${datasets}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --max_model_len 32768 \
    --max_new_tokens 32768 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --fake_act_quant "${FAKE_ACT_QUANT}" \
    --kv_quant_format none \
    --output_dir "${RESULT_DIR}/${name}" \
    "${optional_args[@]}" \
    > "${LOG_DIR}/${name}.log" 2>&1
  echo "完成 ${name}: ${LOG_DIR}/${name}.log"
}

run_eval "gsm8k,math_500" "gsm8k_math500"
run_eval "mmlu_pro" "mmlu_pro_1000" "1000"
run_eval "lcb:codegeneration_v6" "lcb_v6"

