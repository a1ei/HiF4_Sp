#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

MODEL_PATH="Qmodel/Qwen3.5-27b-NVFP4-RTN-s1k-head-512x4096"
RESULT_DIR="results/nvfp4_hif4"
LOG_DIR="output_zero_shot/output_nvfp4_hif4"

HIF4_LCB_GPUS="${HIF4_LCB_GPUS:-0,1}"
HIF4_MMLU_GPUS="${HIF4_MMLU_GPUS:-2,3}"
NVFP4_LCB_GPUS="${NVFP4_LCB_GPUS:-4,5}"
NVFP4_MMLU_GPUS="${NVFP4_MMLU_GPUS:-6,7}"
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型目录不存在: ${MODEL_PATH}" >&2
  exit 1
fi

python - <<'PY'
import sys
try:
    import vllm
    import vllm._C  # noqa: F401
except Exception as exc:
    print("vLLM CUDA 扩展不可用，无法启动评测。", file=sys.stderr)
    print(f"vLLM import path: {getattr(sys.modules.get('vllm'), '__file__', 'unknown')}", file=sys.stderr)
    print(f"错误: {exc}", file=sys.stderr)
    print("请重新源码编译安装 3rdparty/vllm，确保 import vllm._C 成功。", file=sys.stderr)
    raise SystemExit(1)
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

run_eval() {
  local act_quant="$1"
  local dataset="$2"
  local max_samples="$3"
  local name="$4"
  local gpus="$5"
  local log_file="${LOG_DIR}/${name}.log"
  local output_dir="${RESULT_DIR}/${name}"

  if [[ "${act_quant}" == "nvfp4" && ! -f "${MODEL_PATH}/nvfp4_activation_scales.safetensors" ]]; then
    echo "缺少 ${MODEL_PATH}/nvfp4_activation_scales.safetensors，跳过 ${name}。" >&2
    echo "main.py --fake_act_quant nvfp4 需要这个 activation scale 文件。" >&2
    return 1
  fi

  echo "开始测试: ${name}"
  echo "使用 GPU: ${gpus}"
  echo "日志: ${log_file}"

  if [[ -n "${max_samples}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpus}" python main.py \
      --model_path "${MODEL_PATH}" \
      --datasets "${dataset}" \
      --fake_act_quant "${act_quant}" \
      --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
      --max_model_len "${MAX_MODEL_LEN}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --temperature 0.7 \
      --top_p 0.8 \
      --top_k 20 \
      --max_samples "${max_samples}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --output_dir "${output_dir}" \
      > "${log_file}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpus}" python main.py \
      --model_path "${MODEL_PATH}" \
      --datasets "${dataset}" \
      --fake_act_quant "${act_quant}" \
      --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
      --max_model_len "${MAX_MODEL_LEN}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --temperature 0.7 \
      --top_p 0.8 \
      --top_k 20 \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --output_dir "${output_dir}" \
      > "${log_file}" 2>&1
  fi

  echo "完成测试: ${name}"
}

wait_jobs() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "有任务失败，请查看 ${LOG_DIR} 下对应日志。" >&2
    exit 1
  fi
}

# run_eval "hif4" "lcb:codegeneration_v6" "" "hif4_lcb_codegeneration_v6" "${HIF4_LCB_GPUS}" &
# hif4_lcb_pid=$!

# run_eval "hif4" "mmlu_pro" "3000" "hif4_mmlu_pro_3000" "${HIF4_MMLU_GPUS}" &
# hif4_mmlu_pid=$!

run_eval "nvfp4" "lcb:codegeneration_v6" "" "nvfp4_lcb_codegeneration_v6_nosmooth" "${NVFP4_LCB_GPUS}" &
nvfp4_lcb_pid=$!

run_eval "nvfp4" "mmlu_pro" "3000" "nvfp4_mmlu_pro_3000_nosmooth" "${NVFP4_MMLU_GPUS}" &
nvfp4_mmlu_pid=$!

wait_jobs \
  "${nvfp4_lcb_pid}" \
  "${nvfp4_mmlu_pid}"
  # "${hif4_lcb_pid}" \
  # "${hif4_mmlu_pid}" \