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
export PYTHONPATH="${ROOT}/HiFloat4:${PYTHONPATH:-}"
export VLLM_FLOAT32_MATMUL_PRECISION=highest
[[ -n "${CONDA_PREFIX:-}" ]] && export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPTQ_MODEL="${GPTQ_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-s1k-head-512x4096}"
SEED="${SEED:-42}"
TRAIN_NSAMPLES="${TRAIN_NSAMPLES:-128}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-16384}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
LOGITS_CHUNK_SIZE="${LOGITS_CHUNK_SIZE:-256}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

OUTPUT_ROOT="${OUTPUT_ROOT:-Qmodel/qwen35_4b_lm_head_distillation_scope}"
RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_lm_head_distillation_scope}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_lm_head_distillation_scope}"

DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"
REPEATS="${REPEATS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
FORCE_EVAL="${FORCE_EVAL:-false}"

mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}/train" "${LOG_ROOT}/eval"

model_complete() {
  local path="$1"
  [[ -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]] &&
    { [[ -f "${path}/model.safetensors" ]] || [[ -f "${path}/model.safetensors.index.json" ]] ||
      [[ -f "${path}/pytorch_model.bin" ]] || [[ -f "${path}/pytorch_model.bin.index.json" ]]; }
}

adapter_complete() {
  local path="$1"
  [[ -f "${path}/adapter_config.json" && -f "${path}/adapter_model.safetensors" &&
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

wait_job() {
  local pid="$1" name="$2" log_file="$3"
  wait "${pid}" || {
    echo "错误：${name}失败，请查看 ${log_file}" >&2
    return 1
  }
}

ensure_gptq_model() {
  if model_complete "${GPTQ_MODEL}"; then
    echo "复用GPTQ模型: ${GPTQ_MODEL}"
    return
  fi
  [[ ! -e "${GPTQ_MODEL}" ]] || {
    echo "错误：GPTQ目录存在但不完整: ${GPTQ_MODEL}" >&2
    return 1
  }
  echo "GPU 0 开始标准A16 Hessian GPTQ量化"
  CUDA_VISIBLE_DEVICES=0 MODEL="${MODEL}" OUTPUT="${GPTQ_MODEL}" \
    GPTQ=true AWQ=false SMOOTHQUANT=false MAGR=false DTYPE=bfloat16 \
    CAL_DATASET=s1k-1.1 CAL_NSAMPLES=512 CAL_SEQLEN=4096 \
    CAL_SLICE_MODE=head CAL_SLICE_OFFSET=0 HIF4_WEIGHT_FORMAT=hif4 \
    BLOCK_SIZE_LINEAR=64 GPTQ_PERCDAMP=0.01 HIF4A=false \
    GPTQ_CALIB_BATCH_SIZE=1 TOKEN_IMPORTANCE=none \
    bash HiFloat4/quantize_qwen3_5_27b.sh \
      --attn-implementation "${ATTN_IMPLEMENTATION}" --seed "${SEED}" \
      --safe_serialization true --save_only > "${LOG_ROOT}/quant_gptq.log" 2>&1
  model_complete "${GPTQ_MODEL}" || {
    echo "错误：GPTQ量化结果不完整，请查看 ${LOG_ROOT}/quant_gptq.log" >&2
    return 1
  }
}

adapter_path() {
  local precision="$1" tag="$2"
  echo "${OUTPUT_ROOT}/${precision}/${tag}"
}

train_adapter() {
  local gpu="$1" precision="$2" tag="$3" position_scope="$4" adapter_scope="$5" rank="$6"
  local act_quant=none
  [[ "${precision}" == "w4a4" ]] && act_quant=hif4
  local output log_file
  output="$(adapter_path "${precision}" "${tag}")"
  log_file="${LOG_ROOT}/train/${precision}_${tag}.log"
  if adapter_complete "${output}"; then
    echo "复用adapter: ${precision}/${tag}"
    return
  fi
  [[ ! -e "${output}" ]] || {
    echo "错误：adapter目录存在但不完整: ${output}" >&2
    return 1
  }
  mkdir -p "$(dirname "${output}")"
  echo "GPU ${gpu} 开始训练 ${precision}/${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m HiFloat4.train_lm_head_distillation_adapter \
    --student_model "${GPTQ_MODEL}" --teacher_model "${MODEL}" \
    --output_dir "${output}" --dataset simplescaling/s1K-1.1 \
    --nsamples "${TRAIN_NSAMPLES}" --max_length "${TRAIN_MAX_LENGTH}" \
    --seed "${SEED}" --device cuda:0 --dtype bfloat16 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --epochs "${TRAIN_EPOCHS}" --learning_rate "${TRAIN_LR}" \
    --logits_chunk_size "${LOGITS_CHUNK_SIZE}" \
    --position_scope "${position_scope}" --adapter_scope "${adapter_scope}" \
    --rank "${rank}" --student_fake_act_quant "${act_quant}" \
    > "${log_file}" 2>&1
  adapter_complete "${output}" || {
    echo "错误：adapter训练结果不完整，请查看 ${log_file}" >&2
    return 1
  }
}

train_precision() {
  local precision="$1" failed=0
  echo "========== 四卡并行训练 ${precision} 四种LM Head LoRA =========="
  train_adapter 0 "${precision}" all_tokens all_tokens full_lm_head 8 & p0=$!
  train_adapter 1 "${precision}" connectives_end connectives_end full_lm_head 8 & p1=$!
  train_adapter 2 "${precision}" end_think_full end_think full_lm_head 8 & p2=$!
  train_adapter 3 "${precision}" end_think_row end_think end_think_row 1 & p3=$!
  wait_job "${p0}" "${precision}/all_tokens" "${LOG_ROOT}/train/${precision}_all_tokens.log" || failed=1
  wait_job "${p1}" "${precision}/connectives_end" "${LOG_ROOT}/train/${precision}_connectives_end.log" || failed=1
  wait_job "${p2}" "${precision}/end_think_full" "${LOG_ROOT}/train/${precision}_end_think_full.log" || failed=1
  wait_job "${p3}" "${precision}/end_think_row" "${LOG_ROOT}/train/${precision}_end_think_row.log" || failed=1
  [[ "${failed}" -eq 0 ]]
}

run_eval() {
  local precision="$1" repeat="$2" tag="$3" lora_path="$4"
  local act_quant=none generation_seed output_dir log_file
  [[ "${precision}" == "w4a4" ]] && act_quant=hif4
  generation_seed=$((SEED + repeat - 1))
  output_dir="${RESULT_ROOT}/${precision}/${tag}/repeat_${repeat}"
  log_file="${LOG_ROOT}/eval/${precision}_${tag}_repeat_${repeat}.log"
  if [[ "${FORCE_EVAL}" != "true" ]] && result_complete "${output_dir}"; then
    echo "复用评测: ${precision}/${tag}/repeat_${repeat}"
    return
  fi
  local -a lora_args=()
  [[ -n "${lora_path}" ]] && lora_args=(--lora_path "${lora_path}")
  mkdir -p "${output_dir}"
  echo "GPU 0,1,2,3 TP=4 开始评测 ${precision}/${tag}/repeat_${repeat}，生成seed=${generation_seed}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path "${GPTQ_MODEL}" "${lora_args[@]}" --datasets "${DATASETS}" \
    --tensor_parallel_size 4 --fake_act_quant "${act_quant}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" --batch_size "${BATCH_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_samples "${MMLU_PRO_SAMPLES}" --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" --top_k "${TOP_K}" --seed "${generation_seed}" \
    --use_chat_template --output_dir "${output_dir}" > "${log_file}" 2>&1
  result_complete "${output_dir}" || {
    echo "错误：评测结果不完整，请查看 ${log_file}" >&2
    return 1
  }
}

eval_precision() {
  local precision="$1" repeat
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    echo "========== ${precision} 第${repeat}/${REPEATS}次：每个模型四卡TP=4串行评测 =========="
    run_eval "${precision}" "${repeat}" baseline ""
    run_eval "${precision}" "${repeat}" all_tokens "$(adapter_path "${precision}" all_tokens)"
    run_eval "${precision}" "${repeat}" connectives_end "$(adapter_path "${precision}" connectives_end)"
    run_eval "${precision}" "${repeat}" end_think_full "$(adapter_path "${precision}" end_think_full)"
    run_eval "${precision}" "${repeat}" end_think_row "$(adapter_path "${precision}" end_think_row)"
  done
}

ensure_gptq_model

python - <<'PY'
import vllm
import vllm._C  # noqa: F401
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

train_precision w4a16
eval_precision w4a16
train_precision w4a4
eval_precision w4a4

echo "全部完成：5种方法×2种精度×${REPEATS}次评测。结果: ${RESULT_ROOT}；日志: ${LOG_ROOT}"
