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
[[ -n "${CONDA_PREFIX:-}" ]] && \
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

SEED="${SEED:-42}"

GPTQ_MODEL="${GPTQ_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-A4Calib-s1k-head-512x4096}"
ENTROPY_GRAD_MODEL="${ENTROPY_GRAD_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-A4Calib-s1k-head-512x4096}"
ENTROPY_GRAD_NORM_MODEL="${ENTROPY_GRAD_NORM_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-A4Calib-s1k-head-512x4096}"

GPTQ_LORA="${GPTQ_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-A4Calib-end-think-lora}"
ENTROPY_GRAD_LORA="${ENTROPY_GRAD_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-A4Calib-end-think-lora}"
ENTROPY_GRAD_NORM_LORA="${ENTROPY_GRAD_NORM_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-A4Calib-end-think-lora}"

RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_gptq_a4_calib_sampling_w4a4}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_gptq_a4_calib_sampling_w4a4}"
DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
FORCE_EVAL="${FORCE_EVAL:-false}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

model_complete() {
  local path="$1"
  [[ -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]] &&
    { [[ -f "${path}/model.safetensors" ]] ||
      [[ -f "${path}/model.safetensors.index.json" ]] ||
      [[ -f "${path}/pytorch_model.bin" ]] ||
      [[ -f "${path}/pytorch_model.bin.index.json" ]]; }
}

adapter_complete() {
  local path="$1"
  [[ -f "${path}/adapter_config.json" &&
     -f "${path}/adapter_model.safetensors" &&
     -f "${path}/training_metrics.json" ]]
}

result_complete() {
  local output_dir="$1" result_file
  while IFS= read -r result_file; do
    grep -q '"aime25_avg5|0"' "${result_file}" &&
      grep -q '"lcb:codegeneration_v6|0"' "${result_file}" &&
      grep -q '"mmlu_pro|0"' "${result_file}" && return 0
  done < <(find "${output_dir}" -type f -name 'results_*.json' -print 2>/dev/null)
  return 1
}

run_eval() {
  local tag="$1" model_path="$2" lora_path="${3:-}"
  local output_dir="${RESULT_ROOT}/${tag}"
  local log_file="${LOG_ROOT}/${tag}.log"
  local -a lora_args=()

  if [[ "${FORCE_EVAL}" != "true" ]] && result_complete "${output_dir}"; then
    echo "复用已完成评测: ${tag}"
    return
  fi

  [[ -n "${lora_path}" ]] && lora_args=(--lora_path "${lora_path}")
  mkdir -p "${output_dir}"

  echo "GPU 0,1,2,3 开始 ${tag} W4A4，TP=4"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path "${model_path}" "${lora_args[@]}" \
    --datasets "${DATASETS}" --tensor_parallel_size 4 \
    --fake_act_quant hif4 --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --batch_size "${BATCH_SIZE}" --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --max_samples "${MMLU_PRO_SAMPLES}" \
    --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}" \
    --seed "${SEED}" --use_chat_template --output_dir "${output_dir}" \
    > "${log_file}" 2>&1

  if ! result_complete "${output_dir}"; then
    echo "错误：${tag} 评测未生成完整结果，请查看 ${log_file}" >&2
    return 1
  fi
}

for model_path in \
  "${GPTQ_MODEL}" \
  "${ENTROPY_GRAD_MODEL}" \
  "${ENTROPY_GRAD_NORM_MODEL}"; do
  model_complete "${model_path}" || {
    echo "错误：模型目录不存在或不完整: ${model_path}" >&2
    exit 1
  }
done

for lora_path in \
  "${GPTQ_LORA}" \
  "${ENTROPY_GRAD_LORA}" \
  "${ENTROPY_GRAD_NORM_LORA}"; do
  adapter_complete "${lora_path}" || {
    echo "错误：LoRA目录不存在或不完整: ${lora_path}" >&2
    exit 1
  }
done

python - <<'PY'
import vllm
import vllm._C  # noqa: F401
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

echo "========== 四卡 TP=4 串行执行六组非 greedy W4A4 评测 =========="
run_eval gptq_base "${GPTQ_MODEL}"
run_eval gptq_end_think_lora "${GPTQ_MODEL}" "${GPTQ_LORA}"
run_eval entropy_grad_base "${ENTROPY_GRAD_MODEL}"
run_eval entropy_grad_end_think_lora "${ENTROPY_GRAD_MODEL}" "${ENTROPY_GRAD_LORA}"
run_eval entropy_grad_norm_base "${ENTROPY_GRAD_NORM_MODEL}"
run_eval entropy_grad_norm_end_think_lora "${ENTROPY_GRAD_NORM_MODEL}" "${ENTROPY_GRAD_NORM_LORA}"

echo "全部完成。结果: ${RESULT_ROOT}；日志: ${LOG_ROOT}"
