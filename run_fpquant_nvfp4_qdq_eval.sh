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

FP32_MODEL="${FP32_MODEL:-Qmodel/Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-FP32-NoHadamard}"
BF16_MODEL="${BF16_MODEL:-Qmodel/Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-NoHadamard}"

EVAL_GPUS="${EVAL_GPUS:-5,6}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
CAL_GPU="${CAL_GPU:-5}"

CAL_DATASET="${CAL_DATASET:-s1k-1.1}"
CAL_NSAMPLES="${CAL_NSAMPLES:-512}"
CAL_SEQLEN="${CAL_SEQLEN:-4096}"
CAL_SLICE_MODE="${CAL_SLICE_MODE:-head}"
CAL_SLICE_OFFSET="${CAL_SLICE_OFFSET:-0}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
BASE_SEED="${BASE_SEED:-1234}"
RESUME="${RESUME:-true}"

RESULT_DIR="${RESULT_DIR:-results/fpquant_nvfp4_qdq}"
LOG_DIR="${LOG_DIR:-output_zero_shot/fpquant_nvfp4_qdq}"

for model_path in "${FP32_MODEL}" "${BF16_MODEL}"; do
  if [[ ! -d "${model_path}" ]]; then
    echo "模型目录不存在: ${model_path}" >&2
    exit 1
  fi
done

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

prepare_activation_scales() {
  local model_path="$1"
  local dtype="$2"
  local scale_path="${model_path}/nvfp4_activation_scales.safetensors"

  if [[ -f "${scale_path}" ]]; then
    echo "复用激活 scale: ${scale_path}"
    return
  fi

  echo "开始生成 ${dtype} 模型的 NVFP4 激活 scale: ${model_path}"
  CUDA_VISIBLE_DEVICES="${CAL_GPU}" python HiFloat4/export_nvfp4_activation_scales.py \
    --model_path "${model_path}" \
    --output_dir "${model_path}" \
    --cal_dataset "${CAL_DATASET}" \
    --cal_nsamples "${CAL_NSAMPLES}" \
    --cal_seqlen "${CAL_SEQLEN}" \
    --cal_slice_mode "${CAL_SLICE_MODE}" \
    --cal_slice_offset "${CAL_SLICE_OFFSET}" \
    --dtype "${dtype}" \
    --attn-implementation sdpa
  echo "完成激活 scale: ${scale_path}"
}

run_eval() {
  local tag="$1"
  local model_path="$2"
  local precision_mode="$3"
  local dataset="$4"
  local task_name="$5"
  local max_samples="$6"
  local seed="$7"
  local optional_args=()

  case "${precision_mode}" in
    w32a32)
      optional_args+=(--fp32_weights_fp32_activations)
      ;;
    w32a16)
      optional_args+=(--fp32_weights_bf16_activations)
      ;;
    w16a16)
      ;;
    *)
      echo "未知精度模式: ${precision_mode}" >&2
      return 1
      ;;
  esac

  if [[ -n "${max_samples}" ]]; then
    optional_args+=(--max_samples "${max_samples}")
  fi

  local output_dir="${RESULT_DIR}/${tag}/${task_name}"
  local log_file="${LOG_DIR}/${tag}_${task_name}.log"
  mkdir -p "${output_dir}"

  if [[ "${RESUME}" == "true" ]] \
    && [[ -f "${log_file}" ]] \
    && grep -Fq "评估完成。结果与 details 已保存至:" "${log_file}"; then
    echo "跳过已完成任务 ${tag}/${task_name}: ${log_file}"
    return
  fi

  echo "开始 ${tag}/${task_name}: GPU=${EVAL_GPUS}, TP=${TENSOR_PARALLEL_SIZE}, seed=${seed}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" python main.py \
    --model_path "${model_path}" \
    --datasets "${dataset}" \
    --fake_act_quant nvfp4 \
    --kv_quant_format none \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --use_chat_template \
    --seed "${seed}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --output_dir "${output_dir}" \
    "${optional_args[@]}" \
    > "${log_file}" 2>&1
  echo "完成 ${tag}/${task_name}: ${log_file}"
}

run_suite() {
  local tag="$1"
  local model_path="$2"
  local precision_mode="$3"

  echo "========== 开始 ${tag} =========="

  run_eval "${tag}" "${model_path}" "${precision_mode}" \
    "lcb:codegeneration_v6" "codegeneration_v6" "" \
    "${BASE_SEED}"

  run_eval "${tag}" "${model_path}" "${precision_mode}" \
    "mmlu_pro" "mmlu_pro_3000" "3000" \
    "${BASE_SEED}"

  local run_idx
  for run_idx in 1 2 3 4 5; do
    run_eval "${tag}" "${model_path}" "${precision_mode}" \
      "aime25" "aime_run${run_idx}" "" "$((BASE_SEED + run_idx - 1))"
  done

  echo "========== 完成 ${tag} =========="
}

prepare_activation_scales "${FP32_MODEL}" float32
prepare_activation_scales "${BF16_MODEL}" bfloat16

run_suite "w32a32" "${FP32_MODEL}" w32a32
run_suite "w32a16" "${FP32_MODEL}" w32a16
run_suite "w16a16" "${BF16_MODEL}" w16a16

echo "全部评测完成。结果: ${RESULT_DIR}，日志: ${LOG_DIR}"
