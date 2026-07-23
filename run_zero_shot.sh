#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

HIF4_MODEL="${HIF4_MODEL:-Qmodel/Qwen3.5-27b-NVFP4-RTN-SQScaleOnly-s1k-head-512x4096}"
NVFP4_MODEL="${NVFP4_MODEL:-Qmodel/Qwen3.5-27b-NVFP4-RTN-s1k-head-512x4096}"

DATASETS="${DATASETS:-aime25}"
REPEATS="${REPEATS:-4}"
GPUS="${GPUS:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"

RESULT_DIR="${RESULT_DIR:-results/aime_nvfp4_hif4}"
LOG_DIR="${LOG_DIR:-output_zero_shot/aime_nvfp4_hif4}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

if [[ ! -d "${HIF4_MODEL}" ]]; then
  echo "hif4 激活使用的模型目录不存在: ${HIF4_MODEL}" >&2
  exit 1
fi

if [[ ! -d "${NVFP4_MODEL}" ]]; then
  echo "nvfp4 激活使用的模型目录不存在: ${NVFP4_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${NVFP4_MODEL}/nvfp4_activation_scales.safetensors" ]]; then
  echo "缺少 nvfp4 activation scale 文件: ${NVFP4_MODEL}/nvfp4_activation_scales.safetensors" >&2
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
    raise SystemExit(1)
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

run_eval() {
  local run_idx="$1"
  local act_quant="$2"
  local model_path="$3"
  local tag="$4"
  local name="${tag}_run${run_idx}"
  local output_dir="${RESULT_DIR}/${name}"
  local log_file="${LOG_DIR}/${name}.log"

  echo "开始测试: ${name}"
  echo "模型: ${model_path}"
  echo "激活量化: ${act_quant}"
  echo "GPU: ${GPUS}, TP=${TENSOR_PARALLEL_SIZE}"
  echo "日志: ${log_file}"

  CUDA_VISIBLE_DEVICES="${GPUS}" python main.py \
    --model_path "${model_path}" \
    --datasets "${DATASETS}" \
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

  echo "完成测试: ${name}"
}

for run_idx in $(seq 1 "${REPEATS}"); do
  echo "========== AIME 第 ${run_idx}/${REPEATS} 轮 =========="
  run_eval "${run_idx}" "hif4" "${HIF4_MODEL}" "hif4_act_sqscaleonly_weight_${DATASETS}"
  run_eval "${run_idx}" "nvfp4" "${NVFP4_MODEL}" "nvfp4_act_rtn_weight_${DATASETS}"
done

echo "全部 AIME 测试完成。结果目录: ${RESULT_DIR}，日志目录: ${LOG_DIR}"
