#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前环境不是 hif4。请先执行: conda activate hif4" >&2
  exit 1
fi

MODEL="${MODEL:-Qmodel/Qwen3.5-27B-HiF4-GPTQ-s1k-head-128x4096}"
LORA_PATH="${LORA_PATH:-Qmodel/Qwen3.5-27B-HiF4-GPTQ-end-think-lora}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
BATCH_SIZE="${BATCH_SIZE:-128}"
FAKE_ACT_QUANT="${FAKE_ACT_QUANT:-hif4}"
SEED="${SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-3000}"
RESULT_ROOT="${RESULT_ROOT:-results/end_think_comparison}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/end_think_comparison}"
RUN_BASE="${RUN_BASE:-true}"
RUN_ADAPTER="${RUN_ADAPTER:-true}"

# AIME 2025 的 avg5 任务会对每道题生成 5 次。LiveCodeBench 固定为 v6。
DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"

if [[ ! -d "${MODEL}" ]]; then
  echo "错误：量化模型目录不存在: ${MODEL}" >&2
  exit 1
fi
if [[ ! -f "${LORA_PATH}/adapter_config.json" ]]; then
  echo "错误：缺少 adapter 配置: ${LORA_PATH}/adapter_config.json" >&2
  exit 1
fi
if [[ ! -f "${LORA_PATH}/adapter_model.safetensors" ]]; then
  echo "错误：缺少 adapter 权重: ${LORA_PATH}/adapter_model.safetensors" >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -ne "${TENSOR_PARALLEL_SIZE}" ]]; then
  echo "错误：GPUS 中有 ${#GPU_LIST[@]} 张卡，但 TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}" >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

python - <<'PY'
import sys

try:
    import vllm
    import vllm._C  # noqa: F401
except Exception as exc:
    print("错误：当前 hif4 环境无法导入评测所需的 vLLM。", file=sys.stderr)
    print(f"具体错误: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

run_eval() {
  local variant="$1"
  local output_dir="${RESULT_ROOT}/${variant}"
  local log_file="${LOG_ROOT}/${variant}.log"
  local -a lora_args=()

  if [[ "${variant}" == "gptq_end_think_adapter" ]]; then
    lora_args=(--lora_path "${LORA_PATH}")
  fi

  echo "开始评测: ${variant}"
  echo "任务: ${DATASETS}"
  echo "模型: ${MODEL}"
  echo "LoRA: ${lora_args[*]:-none}"
  echo "GPU: ${GPUS}, TP=${TENSOR_PARALLEL_SIZE}, seed=${SEED}"
  echo "结果: ${output_dir}"
  echo "日志: ${log_file}"

  CUDA_VISIBLE_DEVICES="${GPUS}" python main.py \
    --model_path "${MODEL}" \
    --datasets "${DATASETS}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --fake_act_quant "${FAKE_ACT_QUANT}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --batch_size "${BATCH_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_samples "${MMLU_PRO_SAMPLES}" \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --seed "${SEED}" \
    --use_chat_template \
    --output_dir "${output_dir}" \
    "${lora_args[@]}" \
    > "${log_file}" 2>&1

  echo "完成评测: ${variant}"
}

# 顺序运行，避免两组模型争用同一批 GPU。固定任务、seed 和 max_samples。
if [[ "${RUN_ADAPTER}" == "true" ]]; then
  run_eval "gptq_end_think_adapter"
fi
if [[ "${RUN_BASE}" == "true" ]]; then
  run_eval "gptq_base"
fi


echo "全部完成。"
echo "结果目录: ${RESULT_ROOT}"
echo "日志目录: ${LOG_ROOT}"
