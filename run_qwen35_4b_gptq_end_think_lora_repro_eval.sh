#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：请先执行 conda activate hif4" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-/root/data/.cache/huggingface}"
export VLLM_FLOAT32_MATMUL_PRECISION=highest
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

GPTQ_MODEL="${GPTQ_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-s1k-head-512x4096}"
LORA_DIR="${LORA_DIR:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-end-think-lora}"
RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_gptq_end_think_lora_repro}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_gptq_end_think_lora_repro}"

DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
REPEAT_SEEDS=(42 43 44)

model_complete() {
  local path="$1"
  [[ -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]] &&
    { [[ -f "${path}/model.safetensors" ]] ||
      [[ -f "${path}/model.safetensors.index.json" ]] ||
      [[ -f "${path}/pytorch_model.bin" ]] ||
      [[ -f "${path}/pytorch_model.bin.index.json" ]]; }
}

adapter_complete() {
  [[ -f "${LORA_DIR}/adapter_config.json" &&
     -f "${LORA_DIR}/adapter_model.safetensors" &&
     -f "${LORA_DIR}/training_metrics.json" ]]
}

result_complete() {
  local path="$1"
  find "${path}" -type f -path '*/results/results_*.json' -print -quit 2>/dev/null |
    grep -q .
}

run_eval() {
  local method="$1" repeat="$2" precision="$3" seed="$4" lora_path="${5:-}"
  local act_quant=none
  local output_dir="${RESULT_ROOT}/${method}/repeat_${repeat}/${precision}"
  local log_dir="${LOG_ROOT}/${method}/repeat_${repeat}"
  local log_file="${log_dir}/${precision}.log"
  local -a lora_args=()

  if result_complete "${output_dir}"; then
    echo "复用已完成结果：${method} 第${repeat}组 ${precision}"
    return
  fi
  [[ "${precision}" == "w4a4" ]] && act_quant=hif4
  [[ -n "${lora_path}" ]] && lora_args=(--lora_path "${lora_path}")
  mkdir -p "${output_dir}" "${log_dir}"

  echo "开始：${method} 第${repeat}组 ${precision}，seed=${seed}，GPU=0,1,2,3，TP=4"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path "${GPTQ_MODEL}" \
    --datasets "${DATASETS}" \
    --tensor_parallel_size 4 \
    --fake_act_quant "${act_quant}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --batch_size "${BATCH_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_samples "${MMLU_PRO_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --seed "${seed}" \
    --use_chat_template \
    --output_dir "${output_dir}" \
    "${lora_args[@]}" \
    > "${log_file}" 2>&1

  result_complete "${output_dir}" || {
    echo "错误：${method} 第${repeat}组 ${precision} 未生成完整结果，查看 ${log_file}" >&2
    return 1
  }
}

model_complete "${GPTQ_MODEL}" || {
  echo "错误：GPTQ模型不完整：${GPTQ_MODEL}" >&2
  exit 1
}
adapter_complete || {
  echo "错误：W4A16训练的end-think LoRA不完整：${LORA_DIR}" >&2
  exit 1
}

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
python - <<'PY'
import vllm
import vllm._C  # noqa: F401
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

# 先完成纯GPTQ的3组，再完成GPTQ+W4A16 LoRA的3组。
for method in gptq gptq_end_think_lora; do
  lora_path=""
  [[ "${method}" == "gptq_end_think_lora" ]] && lora_path="${LORA_DIR}"
  for index in "${!REPEAT_SEEDS[@]}"; do
    repeat=$((index + 1))
    seed="${REPEAT_SEEDS[index]}"
    run_eval "${method}" "${repeat}" w4a16 "${seed}" "${lora_path}"
    run_eval "${method}" "${repeat}" w4a4 "${seed}" "${lora_path}"
  done
done

echo "全部完成。结果：${RESULT_ROOT}；日志：${LOG_ROOT}"
