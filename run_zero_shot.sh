#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前环境不是 hif4。请先执行: conda activate hif4" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

GPTQ_MODEL="${GPTQ_MODEL:-Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_1536}"
MAGR_MODEL="${MAGR_MODEL:-Qmodel/Qwen3.5-27b-Hif4-MagR-FullX}"
GPTQ_GPU="${GPTQ_GPU:-7}"
MAGR_GPU="${MAGR_GPU:-6}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"

RESULT_DIR="${RESULT_DIR:-results/gptq_magr_zero_shot_w4a4}"
LOG_DIR="${LOG_DIR:-output_zero_shot/gptq_magr_zero_shot_w4a4}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

for model_path in "${GPTQ_MODEL}" "${MAGR_MODEL}"; do
  if [[ ! -d "${model_path}" ]]; then
    echo "模型目录不存在: ${model_path}" >&2
    exit 1
  fi
done

python - <<'PY'
import sys

try:
    import vllm
    import vllm._C  # noqa: F401
except Exception as exc:
    print("vLLM CUDA 扩展不可用，无法启动评测。", file=sys.stderr)
    print(f"vLLM import path: {getattr(sys.modules.get('vllm'), '__file__', 'unknown')}", file=sys.stderr)
    print(f"错误: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

run_eval() {
  local tag="$1"
  local model_path="$2"
  local gpu="$3"
  local datasets="$4"
  local max_samples="$5"
  local name="${tag}_${datasets//,/_}"
  local output_dir="${RESULT_DIR}/${name}"
  local log_file="${LOG_DIR}/${name}.log"
  local optional_args=()

  if [[ -n "${max_samples}" ]]; then
    optional_args+=(--max_samples "${max_samples}")
    name="${name}_${max_samples}"
    output_dir="${RESULT_DIR}/${name}"
    log_file="${LOG_DIR}/${name}.log"
  fi

  echo "开始测试: ${name}"
  echo "模型: ${model_path}"
  echo "GPU: ${gpu}"
  echo "日志: ${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" python main.py \
    --model_path "${model_path}" \
    --datasets "${datasets}" \
    --tensor_parallel_size 1 \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --fake_act_quant hif4 \
    --output_dir "${output_dir}" \
    "${optional_args[@]}" \
    > "${log_file}" 2>&1

  echo "完成测试: ${name}"
}

run_model_suite() {
  local tag="$1"
  local model_path="$2"
  local gpu="$3"

  run_eval "${tag}" "${model_path}" "${gpu}" "gsm8k,math_500" ""
  run_eval "${tag}" "${model_path}" "${gpu}" "mmlu_pro" "1000"
}

run_model_suite "gptq_s1k_512_1536" "${GPTQ_MODEL}" "${GPTQ_GPU}" &
gptq_pid=$!

run_model_suite "magr_fullx" "${MAGR_MODEL}" "${MAGR_GPU}" &
magr_pid=$!

failed=0
if ! wait "${gptq_pid}"; then
  failed=1
fi
if ! wait "${magr_pid}"; then
  failed=1
fi

if [[ "${failed}" -ne 0 ]]; then
  echo "有评测任务失败，请查看日志目录: ${LOG_DIR}" >&2
  exit 1
fi

echo "全部测试完成。结果目录: ${RESULT_DIR}，日志目录: ${LOG_DIR}"
