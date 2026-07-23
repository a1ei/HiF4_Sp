#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前环境不是 hif4。请先执行: conda activate hif4" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
CAL_DATASET="${CAL_DATASET:-taco}"
CAL_NSAMPLES="${CAL_NSAMPLES:-128}"
CAL_SEQLEN="${CAL_SEQLEN:-4096}"
CAL_SLICE_MODE="${CAL_SLICE_MODE:-head}"
CAL_SLICE_OFFSET="${CAL_SLICE_OFFSET:-0}"

MAGR_QUANT_GPUS="${MAGR_QUANT_GPUS:-4}"
GPTQ_QUANT_GPUS="${GPTQ_QUANT_GPUS:-6}"
MAGR_EVAL_GPUS="${MAGR_EVAL_GPUS:-4,5}"
GPTQ_EVAL_GPUS="${GPTQ_EVAL_GPUS:-6,7}"
MAGR_EVAL_TP="${MAGR_EVAL_TP:-2}"
GPTQ_EVAL_TP="${GPTQ_EVAL_TP:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
FAKE_ACT_QUANT="${FAKE_ACT_QUANT:-hif4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
AIME_REPEATS="${AIME_REPEATS:-5}"
BASE_SEED="${BASE_SEED:-1234}"

RUN_TAG="taco_head_${CAL_NSAMPLES}_${CAL_SEQLEN}"
MAGR_OUTPUT="${MAGR_OUTPUT:-Qmodel/Qwen3.5-27B-HiF4-MagR_${RUN_TAG}}"
GPTQ_OUTPUT="${GPTQ_OUTPUT:-Qmodel/Qwen3.5-27B-HiF4-GPTQ_${RUN_TAG}}"

LOG_DIR="${LOG_DIR:-output/output_taco_magr_gptq}"
RESULT_DIR="${RESULT_DIR:-results/taco_magr_gptq}"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

run_quant_magr() {
  local log_file="${LOG_DIR}/quant_magr_${RUN_TAG}.log"
  echo "开始 MagR 量化，GPU=${MAGR_QUANT_GPUS}，日志: ${log_file}"
  CUDA_VISIBLE_DEVICES="${MAGR_QUANT_GPUS}" \
  MODEL="${MODEL}" \
  OUTPUT="${MAGR_OUTPUT}" \
  GPTQ=false \
  AWQ=false \
  SMOOTHQUANT=false \
  MAGR=true \
  DTYPE=float16 \
  CAL_DATASET="${CAL_DATASET}" \
  CAL_NSAMPLES="${CAL_NSAMPLES}" \
  CAL_SEQLEN="${CAL_SEQLEN}" \
  CAL_SLICE_MODE="${CAL_SLICE_MODE}" \
  CAL_SLICE_OFFSET="${CAL_SLICE_OFFSET}" \
  HIF4_WEIGHT_FORMAT=hif4 \
  BLOCK_SIZE_LINEAR=64 \
  MAGR_CD_ITER=3 \
  MAGR_ALPHA=0.001 \
  MAGR_ALPHA_GROUPWISE=0.0001 \
  MAGR_PREPROCESS_ITER=200 \
  bash HiFloat4/quantize_qwen3_5_27b.sh \
    --save_only \
    > "${log_file}" 2>&1
  echo "完成 MagR 量化: ${MAGR_OUTPUT}"
}

run_quant_gptq() {
  local log_file="${LOG_DIR}/quant_gptq_${RUN_TAG}.log"
  echo "开始 GPTQ 量化，GPU=${GPTQ_QUANT_GPUS}，日志: ${log_file}"
  CUDA_VISIBLE_DEVICES="${GPTQ_QUANT_GPUS}" \
  MODEL="${MODEL}" \
  OUTPUT="${GPTQ_OUTPUT}" \
  GPTQ=true \
  AWQ=false \
  SMOOTHQUANT=false \
  MAGR=false \
  DTYPE=float16 \
  CAL_DATASET="${CAL_DATASET}" \
  CAL_NSAMPLES="${CAL_NSAMPLES}" \
  CAL_SEQLEN="${CAL_SEQLEN}" \
  CAL_SLICE_MODE="${CAL_SLICE_MODE}" \
  CAL_SLICE_OFFSET="${CAL_SLICE_OFFSET}" \
  HIF4_WEIGHT_FORMAT=hif4 \
  BLOCK_SIZE_LINEAR=64 \
  GPTQ_PERCDAMP=0.01 \
  TOKEN_IMPORTANCE=none \
  bash HiFloat4/quantize_qwen3_5_27b.sh \
    --save_only \
    > "${log_file}" 2>&1
  echo "完成 GPTQ 量化: ${GPTQ_OUTPUT}"
}

run_eval() {
  local tag="$1"
  local model_path="$2"
  local dataset="$3"
  local max_samples="$4"
  local seed="$5"
  local gpus="$6"
  local tensor_parallel_size="$7"
  local run_name="${tag}_${dataset//[:]/_}_seed${seed}"
  local output_dir="${RESULT_DIR}/${run_name}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local optional_args=()

  if [[ -n "${max_samples}" ]]; then
    optional_args+=(--max_samples "${max_samples}")
  fi

  echo "开始评测: ${run_name}，GPU=${gpus}，TP=${tensor_parallel_size}，日志=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpus}" python main.py \
    --model_path "${model_path}" \
    --datasets "${dataset}" \
    --tensor_parallel_size "${tensor_parallel_size}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --fake_act_quant "${FAKE_ACT_QUANT}" \
    --seed "${seed}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --output_dir "${output_dir}" \
    "${optional_args[@]}" \
    > "${log_file}" 2>&1
  echo "完成评测: ${run_name}"
}

run_eval_suite() {
  local tag="$1"
  local model_path="$2"
  local gpus="$3"
  local tensor_parallel_size="$4"

  if [[ ! -d "${model_path}" ]]; then
    echo "模型目录不存在: ${model_path}" >&2
    exit 1
  fi

  run_eval "${tag}" "${model_path}" "lcb:codegeneration_v6" "" "${BASE_SEED}" "${gpus}" "${tensor_parallel_size}"
  run_eval "${tag}" "${model_path}" "mmlu_pro" "3000" "${BASE_SEED}" "${gpus}" "${tensor_parallel_size}"

  local run_idx
  local seed
  for run_idx in $(seq 0 $((AIME_REPEATS - 1))); do
    seed=$((BASE_SEED + run_idx))
    run_eval "${tag}_aime_run$((run_idx + 1))" "${model_path}" "aime25" "" "${seed}" "${gpus}" "${tensor_parallel_size}"
  done
}

run_magr_pipeline() {
  # run_quant_magr
  run_eval_suite "magr_${RUN_TAG}" "${MAGR_OUTPUT}" "${MAGR_EVAL_GPUS}" "${MAGR_EVAL_TP}"
}

run_gptq_pipeline() {
  # run_quant_gptq
  run_eval_suite "gptq_${RUN_TAG}" "${GPTQ_OUTPUT}" "${GPTQ_EVAL_GPUS}" "${GPTQ_EVAL_TP}"
}

run_magr_pipeline &
magr_pid=$!
run_gptq_pipeline &
gptq_pid=$!

failed=0
if ! wait "${magr_pid}"; then
  failed=1
fi
if ! wait "${gptq_pid}"; then
  failed=1
fi
if [[ "${failed}" -ne 0 ]]; then
  echo "有任务失败，请查看日志目录: ${LOG_DIR}" >&2
  exit 1
fi

echo "全部完成。日志目录: ${LOG_DIR}，结果目录: ${RESULT_DIR}"
