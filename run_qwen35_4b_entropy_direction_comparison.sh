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
[[ -n "${CONDA_PREFIX:-}" ]] && export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
SEED="${SEED:-42}"
CAL_NSAMPLES="${CAL_NSAMPLES:-512}"
CAL_SEQLEN="${CAL_SEQLEN:-4096}"
A16_GPTQ_CALIB_BATCH_SIZE="${A16_GPTQ_CALIB_BATCH_SIZE:-1}"
A16_IMPORTANCE_BATCH_SIZE="${A16_IMPORTANCE_BATCH_SIZE:-8}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# All weight models use BF16/A16 activations for GPTQ Hessian calibration.
LOW_EG_A16="${LOW_EG_A16:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-s1k-head-512x4096}"
LOW_EGN_A16="${LOW_EGN_A16:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-s1k-head-512x4096}"
LOW_EG_A16_LORA="${LOW_EG_A16_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-end-think-lora}"
LOW_EGN_A16_LORA="${LOW_EGN_A16_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-end-think-lora}"
LOW_EG_A4_LORA="${LOW_EG_A4_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-W4A4-end-think-lora}"
LOW_EGN_A4_LORA="${LOW_EGN_A4_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-W4A4-end-think-lora}"

HIGH_EG_A16="${HIGH_EG_A16:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradHigh-s1k-head-512x4096}"
HIGH_EGN_A16="${HIGH_EGN_A16:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNormHigh-s1k-head-512x4096}"
HIGH_EG_A16_LORA="${HIGH_EG_A16_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradHigh-end-think-lora}"
HIGH_EGN_A16_LORA="${HIGH_EGN_A16_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNormHigh-end-think-lora}"
HIGH_EG_A4_LORA="${HIGH_EG_A4_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradHigh-W4A4-end-think-lora}"
HIGH_EGN_A4_LORA="${HIGH_EGN_A4_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNormHigh-W4A4-end-think-lora}"

RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_entropy_direction_comparison}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_entropy_direction_comparison}"
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
mkdir -p Qmodel "${RESULT_ROOT}" "${LOG_ROOT}/eval"

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

quantize_high() {
  local gpu="$1" tag="$2" mode="$3" output="$4"
  if model_complete "${output}"; then echo "复用高熵模型 ${tag}: ${output}"; return; fi
  [[ ! -e "${output}" ]] || { echo "错误：模型目录存在但不完整: ${output}" >&2; return 1; }

  echo "GPU ${gpu} 开始高熵 ${tag} A16校准"
  CUDA_VISIBLE_DEVICES="${gpu}" MODEL="${MODEL}" OUTPUT="${output}" \
    GPTQ=true AWQ=false SMOOTHQUANT=false MAGR=false DTYPE=bfloat16 \
    CAL_DATASET=s1k-1.1 CAL_NSAMPLES="${CAL_NSAMPLES}" CAL_SEQLEN="${CAL_SEQLEN}" \
    CAL_SLICE_MODE=head CAL_SLICE_OFFSET=0 HIF4_WEIGHT_FORMAT=hif4 \
    BLOCK_SIZE_LINEAR=64 GPTQ_PERCDAMP=0.01 \
    HIF4A=false ACT_QUANT_FORMAT=hif4 \
    GPTQ_CALIB_BATCH_SIZE="${A16_GPTQ_CALIB_BATCH_SIZE}" \
    TOKEN_IMPORTANCE="${mode}" ENTROPY_DIRECTION=high \
    IMPORTANCE_ALPHA=1.0 IMPORTANCE_MEAN_NORMALIZE=true \
    IMPORTANCE_BATCH_SIZE="${A16_IMPORTANCE_BATCH_SIZE}" \
    bash HiFloat4/quantize_qwen3_5_27b.sh \
      --attn-implementation "${ATTN_IMPLEMENTATION}" --seed "${SEED}" \
      --safe_serialization true --save_only > "${LOG_ROOT}/quant_${tag}.log" 2>&1
  model_complete "${output}"
}

train_lora() {
  local gpu="$1" tag="$2" student="$3" output="$4" student_act="$5"
  local log_file="${LOG_ROOT}/train_${tag}_lora.log"
  local -a command=(
    python HiFloat4/train_end_think_adapter.py
    --student_model "${student}" --teacher_model "${MODEL}" --output_dir "${output}"
    --dataset simplescaling/s1K-1.1 --nsamples 128 --max_length 16384
    --num_negative_positions 8 --seed "${SEED}"
    --teacher_device cuda:0 --student_device cuda:0 --dtype bfloat16
    --attn_implementation "${ATTN_IMPLEMENTATION}" --epochs 10 --learning_rate 1e-4
    --student_fake_act_quant "${student_act}"
  )
  if adapter_complete "${output}"; then echo "复用 ${tag} LoRA: ${output}"; return; fi
  if [[ -e "${output}" ]]; then
    [[ -f "${output}/feature_cache/examples.pt" &&
       -f "${output}/feature_cache/teacher_features.pt" ]] || {
      echo "错误：LoRA目录存在但不可恢复: ${output}" >&2; return 1; }
    if [[ ! -f "${output}/feature_cache/student_features.pt" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}" --phase student > "${log_file}" 2>&1
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}" --phase train >> "${log_file}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}" > "${log_file}" 2>&1
  fi
  adapter_complete "${output}"
}

run_eval() {
  local tag="$1" model_path="$2" lora_path="$3" act_quant="$4" output_dir="$5"
  local log_file="${LOG_ROOT}/eval/${tag}.log"; local -a lora_args=()
  if [[ "${FORCE_EVAL}" != "true" ]] && result_complete "${output_dir}"; then
    echo "复用评测结果: ${tag}"; return
  fi
  [[ -n "${lora_path}" ]] && lora_args=(--lora_path "${lora_path}")
  mkdir -p "${output_dir}"
  echo "GPU 0,1,2,3 TP=4 开始 ${tag}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path "${model_path}" "${lora_args[@]}" --datasets "${DATASETS}" \
    --tensor_parallel_size 4 --fake_act_quant "${act_quant}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" --batch_size "${BATCH_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_samples "${MMLU_PRO_SAMPLES}" --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" --top_k "${TOP_K}" --seed "${SEED}" \
    --use_chat_template --output_dir "${output_dir}" > "${log_file}" 2>&1
  result_complete "${output_dir}" || {
    echo "错误：${tag}结果不完整，请查看 ${log_file}" >&2; return 1; }
}

wait_job() {
  local pid="$1" name="$2" log_file="$3"
  wait "${pid}" || { echo "${name}失败，请查看 ${log_file}" >&2; return 1; }
}

for path in "${LOW_EG_A16}" "${LOW_EGN_A16}"; do
  model_complete "${path}" || { echo "错误：缺少低熵A16校准模型: ${path}" >&2; exit 1; }
done
for path in "${LOW_EG_A16_LORA}" "${LOW_EGN_A16_LORA}"; do
  adapter_complete "${path}" || { echo "错误：缺少低熵W4A16 LoRA: ${path}" >&2; exit 1; }
done

python - <<'PY'
import vllm
import vllm._C  # noqa: F401
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

echo "========== 阶段1/3：并行量化高熵A16模型 =========="
quantize_high 0 entropy_grad_high_a16 entropy_grad "${HIGH_EG_A16}" & q0=$!
quantize_high 1 entropy_grad_norm_high_a16 entropy_grad_norm "${HIGH_EGN_A16}" & q1=$!
failed=0
wait_job "${q0}" entropy-grad-high-A16 "${LOG_ROOT}/quant_entropy_grad_high_a16.log" || failed=1
wait_job "${q1}" entropy-grad-norm-high-A16 "${LOG_ROOT}/quant_entropy_grad_norm_high_a16.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

echo "========== 阶段2/3：训练同一A16校准模型的W4A16/W4A4 LoRA =========="
train_lora 0 entropy_grad_high_a16 "${HIGH_EG_A16}" "${HIGH_EG_A16_LORA}" none & l0=$!
train_lora 1 entropy_grad_norm_high_a16 "${HIGH_EGN_A16}" "${HIGH_EGN_A16_LORA}" none & l1=$!
failed=0
wait_job "${l0}" entropy-grad-high-W4A16-LoRA "${LOG_ROOT}/train_entropy_grad_high_a16_lora.log" || failed=1
wait_job "${l1}" entropy-grad-norm-high-W4A16-LoRA "${LOG_ROOT}/train_entropy_grad_norm_high_a16_lora.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

train_lora 0 entropy_grad_low_w4a4 "${LOW_EG_A16}" "${LOW_EG_A4_LORA}" hif4 & l0=$!
train_lora 1 entropy_grad_norm_low_w4a4 "${LOW_EGN_A16}" "${LOW_EGN_A4_LORA}" hif4 & l1=$!
train_lora 2 entropy_grad_high_w4a4 "${HIGH_EG_A16}" "${HIGH_EG_A4_LORA}" hif4 & l2=$!
train_lora 3 entropy_grad_norm_high_w4a4 "${HIGH_EGN_A16}" "${HIGH_EGN_A4_LORA}" hif4 & l3=$!
failed=0
wait_job "${l0}" entropy-grad-low-W4A4-LoRA "${LOG_ROOT}/train_entropy_grad_low_w4a4_lora.log" || failed=1
wait_job "${l1}" entropy-grad-norm-low-W4A4-LoRA "${LOG_ROOT}/train_entropy_grad_norm_low_w4a4_lora.log" || failed=1
wait_job "${l2}" entropy-grad-high-W4A4-LoRA "${LOG_ROOT}/train_entropy_grad_high_w4a4_lora.log" || failed=1
wait_job "${l3}" entropy-grad-norm-high-W4A4-LoRA "${LOG_ROOT}/train_entropy_grad_norm_high_w4a4_lora.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

echo "========== 阶段3/3：四卡TP=4串行执行高/低熵16组评测 =========="
run_eval low_entropy_grad_a16_base "${LOW_EG_A16}" "" none "${RESULT_ROOT}/low/entropy_grad/a16/base"
run_eval low_entropy_grad_a16_lora "${LOW_EG_A16}" "${LOW_EG_A16_LORA}" none "${RESULT_ROOT}/low/entropy_grad/a16/lora"
run_eval low_entropy_grad_norm_a16_base "${LOW_EGN_A16}" "" none "${RESULT_ROOT}/low/entropy_grad_norm/a16/base"
run_eval low_entropy_grad_norm_a16_lora "${LOW_EGN_A16}" "${LOW_EGN_A16_LORA}" none "${RESULT_ROOT}/low/entropy_grad_norm/a16/lora"
run_eval low_entropy_grad_a4_base "${LOW_EG_A16}" "" hif4 "${RESULT_ROOT}/low/entropy_grad/a4_from_a16/base"
run_eval low_entropy_grad_a4_lora "${LOW_EG_A16}" "${LOW_EG_A4_LORA}" hif4 "${RESULT_ROOT}/low/entropy_grad/a4_from_a16/lora"
run_eval low_entropy_grad_norm_a4_base "${LOW_EGN_A16}" "" hif4 "${RESULT_ROOT}/low/entropy_grad_norm/a4_from_a16/base"
run_eval low_entropy_grad_norm_a4_lora "${LOW_EGN_A16}" "${LOW_EGN_A4_LORA}" hif4 "${RESULT_ROOT}/low/entropy_grad_norm/a4_from_a16/lora"
run_eval high_entropy_grad_a16_base "${HIGH_EG_A16}" "" none "${RESULT_ROOT}/high/entropy_grad/a16/base"
run_eval high_entropy_grad_a16_lora "${HIGH_EG_A16}" "${HIGH_EG_A16_LORA}" none "${RESULT_ROOT}/high/entropy_grad/a16/lora"
run_eval high_entropy_grad_norm_a16_base "${HIGH_EGN_A16}" "" none "${RESULT_ROOT}/high/entropy_grad_norm/a16/base"
run_eval high_entropy_grad_norm_a16_lora "${HIGH_EGN_A16}" "${HIGH_EGN_A16_LORA}" none "${RESULT_ROOT}/high/entropy_grad_norm/a16/lora"
run_eval high_entropy_grad_a4_base "${HIGH_EG_A16}" "" hif4 "${RESULT_ROOT}/high/entropy_grad/a4_from_a16/base"
run_eval high_entropy_grad_a4_lora "${HIGH_EG_A16}" "${HIGH_EG_A4_LORA}" hif4 "${RESULT_ROOT}/high/entropy_grad/a4_from_a16/lora"
run_eval high_entropy_grad_norm_a4_base "${HIGH_EGN_A16}" "" hif4 "${RESULT_ROOT}/high/entropy_grad_norm/a4_from_a16/base"
run_eval high_entropy_grad_norm_a4_lora "${HIGH_EGN_A16}" "${HIGH_EGN_A4_LORA}" hif4 "${RESULT_ROOT}/high/entropy_grad_norm/a4_from_a16/lora"

echo "全部完成。所有权重均由A16校准；W4A4只在LoRA训练和评测时量化激活。结果: ${RESULT_ROOT}；日志: ${LOG_ROOT}"
