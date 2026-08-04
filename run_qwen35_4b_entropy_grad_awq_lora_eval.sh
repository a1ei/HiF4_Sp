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

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
SEED="${SEED:-42}"
CAL_NSAMPLES="${CAL_NSAMPLES:-512}"
CAL_SEQLEN="${CAL_SEQLEN:-4096}"
CAL_SLICE_MODE="${CAL_SLICE_MODE:-head}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
IMPORTANCE_ALPHA="${IMPORTANCE_ALPHA:-1.0}"
IMPORTANCE_BATCH_SIZE="${IMPORTANCE_BATCH_SIZE:-8}"

GPTQ_EG_MODEL="${GPTQ_EG_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-s1k-head-512x4096}"
GPTQ_EGN_MODEL="${GPTQ_EGN_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-s1k-head-512x4096}"
AWQ_EG_MODEL="${AWQ_EG_MODEL:-Qmodel/Qwen3.5-4B-HiF4-AWQ-EntropyGrad-s1k-head-512x4096}"
AWQ_MODEL="${AWQ_MODEL:-Qmodel/Qwen3.5-4B-HiF4-AWQ-s1k-head-512x4096}"
GPTQ_EG_LORA="${GPTQ_EG_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-end-think-lora}"
GPTQ_EGN_LORA="${GPTQ_EGN_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-end-think-lora}"
AWQ_EG_LORA="${AWQ_EG_LORA:-Qmodel/Qwen3.5-4B-HiF4-AWQ-EntropyGrad-end-think-lora}"
AWQ_LORA="${AWQ_LORA:-Qmodel/Qwen3.5-4B-HiF4-AWQ-end-think-lora}"

RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_entropy_grad_awq_lora}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_entropy_grad_awq_lora}"
DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
mkdir -p Qmodel "${RESULT_ROOT}" "${LOG_ROOT}"

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

check_model_dir() {
  local path="$1"
  if [[ -e "${path}" ]] && ! model_complete "${path}"; then
    echo "错误：模型目录存在但不完整: ${path}" >&2
    return 1
  fi
}

check_adapter_dir() {
  local path="$1"
  if [[ -e "${path}" ]] && ! adapter_complete "${path}"; then
    echo "错误：LoRA 目录存在但不完整: ${path}" >&2
    return 1
  fi
}

quantize_gptq_entropy_grad() {
  check_model_dir "${GPTQ_EG_MODEL}"
  if model_complete "${GPTQ_EG_MODEL}"; then
    echo "复用 GPTQ+entropy_grad: ${GPTQ_EG_MODEL}"
    return
  fi
  echo "GPU 0 开始 GPTQ+entropy_grad 量化"
  CUDA_VISIBLE_DEVICES=0 MODEL="${MODEL}" OUTPUT="${GPTQ_EG_MODEL}" \
    GPTQ=true AWQ=false SMOOTHQUANT=false MAGR=false DTYPE=bfloat16 \
    CAL_DATASET=s1k-1.1 CAL_NSAMPLES="${CAL_NSAMPLES}" CAL_SEQLEN="${CAL_SEQLEN}" \
    CAL_SLICE_MODE="${CAL_SLICE_MODE}" CAL_SLICE_OFFSET=0 \
    HIF4_WEIGHT_FORMAT=hif4 BLOCK_SIZE_LINEAR=64 GPTQ_PERCDAMP=0.01 \
    TOKEN_IMPORTANCE=entropy_grad IMPORTANCE_ALPHA="${IMPORTANCE_ALPHA}" \
    IMPORTANCE_MEAN_NORMALIZE=true IMPORTANCE_BATCH_SIZE="${IMPORTANCE_BATCH_SIZE}" \
    bash HiFloat4/quantize_qwen3_5_27b.sh \
      --attn-implementation "${ATTN_IMPLEMENTATION}" \
      --seed "${SEED}" --safe_serialization true --save_only \
      > "${LOG_ROOT}/quant_gptq_entropy_grad.log" 2>&1
  model_complete "${GPTQ_EG_MODEL}"
}

quantize_gptq_entropy_grad_norm() {
  check_model_dir "${GPTQ_EGN_MODEL}"
  if model_complete "${GPTQ_EGN_MODEL}"; then
    echo "复用 GPTQ+entropy_grad_norm: ${GPTQ_EGN_MODEL}"
    return
  fi
  echo "GPU 3 开始 GPTQ+entropy_grad_norm 量化"
  CUDA_VISIBLE_DEVICES=3 MODEL="${MODEL}" OUTPUT="${GPTQ_EGN_MODEL}" \
    GPTQ=true AWQ=false SMOOTHQUANT=false MAGR=false DTYPE=bfloat16 \
    CAL_DATASET=s1k-1.1 CAL_NSAMPLES="${CAL_NSAMPLES}" CAL_SEQLEN="${CAL_SEQLEN}" \
    CAL_SLICE_MODE="${CAL_SLICE_MODE}" CAL_SLICE_OFFSET=0 \
    HIF4_WEIGHT_FORMAT=hif4 BLOCK_SIZE_LINEAR=64 GPTQ_PERCDAMP=0.01 \
    TOKEN_IMPORTANCE=entropy_grad_norm IMPORTANCE_ALPHA="${IMPORTANCE_ALPHA}" \
    IMPORTANCE_MEAN_NORMALIZE=true IMPORTANCE_BATCH_SIZE="${IMPORTANCE_BATCH_SIZE}" \
    bash HiFloat4/quantize_qwen3_5_27b.sh \
      --attn-implementation "${ATTN_IMPLEMENTATION}" \
      --seed "${SEED}" --safe_serialization true --save_only \
      > "${LOG_ROOT}/quant_gptq_entropy_grad_norm.log" 2>&1
  model_complete "${GPTQ_EGN_MODEL}"
}

quantize_awq_entropy_grad() {
  check_model_dir "${AWQ_EG_MODEL}"
  if model_complete "${AWQ_EG_MODEL}"; then
    echo "复用 AWQ+entropy_grad: ${AWQ_EG_MODEL}"
    return
  fi
  echo "GPU 1 开始 AWQ+entropy_grad 量化"
  CUDA_VISIBLE_DEVICES=1 MODEL="${MODEL}" OUTPUT="${AWQ_EG_MODEL}" \
    GPTQ=false AWQ=true SMOOTHQUANT=false MAGR=false DTYPE=bfloat16 \
    CAL_DATASET=s1k-1.1 CAL_NSAMPLES="${CAL_NSAMPLES}" CAL_SEQLEN="${CAL_SEQLEN}" \
    CAL_SLICE_MODE="${CAL_SLICE_MODE}" CAL_SLICE_OFFSET=0 \
    HIF4_WEIGHT_FORMAT=hif4 AWQ_N_GRID=20 \
    TOKEN_IMPORTANCE=entropy_grad IMPORTANCE_ALPHA="${IMPORTANCE_ALPHA}" \
    IMPORTANCE_MEAN_NORMALIZE=true IMPORTANCE_BATCH_SIZE="${IMPORTANCE_BATCH_SIZE}" \
    bash HiFloat4/quantize_qwen3_5_27b.sh \
      --attn-implementation "${ATTN_IMPLEMENTATION}" \
      --seed "${SEED}" --safe_serialization true --save_only \
      > "${LOG_ROOT}/quant_awq_entropy_grad.log" 2>&1
  model_complete "${AWQ_EG_MODEL}"
}

quantize_awq() {
  check_model_dir "${AWQ_MODEL}"
  if model_complete "${AWQ_MODEL}"; then
    echo "复用 AWQ: ${AWQ_MODEL}"
    return
  fi
  echo "GPU 2 开始原始 AWQ 量化"
  CUDA_VISIBLE_DEVICES=2 MODEL="${MODEL}" OUTPUT="${AWQ_MODEL}" \
    GPTQ=false AWQ=true SMOOTHQUANT=false MAGR=false DTYPE=bfloat16 \
    CAL_DATASET=s1k-1.1 CAL_NSAMPLES="${CAL_NSAMPLES}" CAL_SEQLEN="${CAL_SEQLEN}" \
    CAL_SLICE_MODE="${CAL_SLICE_MODE}" CAL_SLICE_OFFSET=0 \
    HIF4_WEIGHT_FORMAT=hif4 AWQ_N_GRID=20 TOKEN_IMPORTANCE=none \
    bash HiFloat4/quantize_qwen3_5_27b.sh \
      --attn-implementation "${ATTN_IMPLEMENTATION}" \
      --seed "${SEED}" --safe_serialization true --save_only \
      > "${LOG_ROOT}/quant_awq.log" 2>&1
  model_complete "${AWQ_MODEL}"
}

train_lora() {
  local gpu="$1" tag="$2" student_model="$3" output_dir="$4"
  check_adapter_dir "${output_dir}"
  if adapter_complete "${output_dir}"; then
    echo "复用 ${tag} LoRA: ${output_dir}"
    return
  fi
  echo "GPU ${gpu} 开始训练 ${tag} 的 end-think lm_head LoRA"
  CUDA_VISIBLE_DEVICES="${gpu}" python HiFloat4/train_end_think_adapter.py \
    --student_model "${student_model}" --teacher_model "${MODEL}" \
    --output_dir "${output_dir}" --dataset simplescaling/s1K-1.1 \
    --nsamples 128 --max_length 16384 --num_negative_positions 8 --seed "${SEED}" \
    --teacher_device cuda:0 --student_device cuda:0 --dtype bfloat16 \
    --epochs 10 --learning_rate 1e-4 \
    > "${LOG_ROOT}/train_${tag}_end_think_lora.log" 2>&1
  adapter_complete "${output_dir}"
}

run_eval() {
  local tag="$1" model_path="$2" lora_path="$3" precision="$4"
  local act_quant=none
  local lora_args=()
  [[ "${precision}" == "w4a4" ]] && act_quant=hif4
  [[ -n "${lora_path}" ]] && lora_args=(--lora_path "${lora_path}")
  mkdir -p "${RESULT_ROOT}/${tag}/${precision}" "${LOG_ROOT}/${tag}"
  echo "4 卡 TP=4 开始评测 ${tag} ${precision}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path "${model_path}" "${lora_args[@]}" \
    --datasets "${DATASETS}" --tensor_parallel_size 4 \
    --fake_act_quant "${act_quant}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --batch_size "${BATCH_SIZE}" --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --max_samples "${MMLU_PRO_SAMPLES}" \
    --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}" \
    --seed "${SEED}" --use_chat_template \
    --output_dir "${RESULT_ROOT}/${tag}/${precision}" \
    > "${LOG_ROOT}/${tag}/${precision}.log" 2>&1
}

wait_job() {
  local pid="$1" name="$2" log="$3"
  if ! wait "${pid}"; then
    echo "${name} 失败，请查看 ${log}" >&2
    return 1
  fi
}

python - <<'PY'
import vllm
import vllm._C  # noqa: F401
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

echo "========== 阶段 1/6：并行量化 GPTQ =========="
quantize_gptq_entropy_grad & q0=$!
quantize_gptq_entropy_grad_norm & q3=$!
failed=0
wait_job "${q0}" "GPTQ+entropy_grad 量化" "${LOG_ROOT}/quant_gptq_entropy_grad.log" || failed=1
wait_job "${q3}" "GPTQ+entropy_grad_norm 量化" "${LOG_ROOT}/quant_gptq_entropy_grad_norm.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

echo "========== 阶段 2/6：并行训练 GPTQ lm_head LoRA =========="
train_lora 0 gptq_entropy_grad "${GPTQ_EG_MODEL}" "${GPTQ_EG_LORA}" & l0=$!
train_lora 3 gptq_entropy_grad_norm "${GPTQ_EGN_MODEL}" "${GPTQ_EGN_LORA}" & l3=$!
failed=0
wait_job "${l0}" "GPTQ+entropy_grad LoRA" "${LOG_ROOT}/train_gptq_entropy_grad_end_think_lora.log" || failed=1
wait_job "${l3}" "GPTQ+entropy_grad_norm LoRA" "${LOG_ROOT}/train_gptq_entropy_grad_norm_end_think_lora.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

echo "========== 阶段 3/6：4 卡顺序评测 GPTQ =========="
run_eval gptq_entropy_grad "${GPTQ_EG_MODEL}" "" w4a16
run_eval gptq_entropy_grad "${GPTQ_EG_MODEL}" "" w4a4
run_eval gptq_entropy_grad_norm "${GPTQ_EGN_MODEL}" "" w4a16
run_eval gptq_entropy_grad_norm "${GPTQ_EGN_MODEL}" "" w4a4
run_eval gptq_entropy_grad_lora "${GPTQ_EG_MODEL}" "${GPTQ_EG_LORA}" w4a16
run_eval gptq_entropy_grad_lora "${GPTQ_EG_MODEL}" "${GPTQ_EG_LORA}" w4a4
run_eval gptq_entropy_grad_norm_lora "${GPTQ_EGN_MODEL}" "${GPTQ_EGN_LORA}" w4a16
run_eval gptq_entropy_grad_norm_lora "${GPTQ_EGN_MODEL}" "${GPTQ_EGN_LORA}" w4a4

echo "========== 阶段 4/6：并行量化 AWQ =========="
quantize_awq_entropy_grad & q1=$!
quantize_awq & q2=$!
failed=0
wait_job "${q1}" "AWQ+entropy_grad 量化" "${LOG_ROOT}/quant_awq_entropy_grad.log" || failed=1
wait_job "${q2}" "AWQ 量化" "${LOG_ROOT}/quant_awq.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

echo "========== 阶段 5/6：并行训练 AWQ lm_head LoRA =========="
train_lora 1 awq_entropy_grad "${AWQ_EG_MODEL}" "${AWQ_EG_LORA}" & l1=$!
train_lora 2 awq "${AWQ_MODEL}" "${AWQ_LORA}" & l2=$!
failed=0
wait_job "${l1}" "AWQ+entropy_grad LoRA" "${LOG_ROOT}/train_awq_entropy_grad_end_think_lora.log" || failed=1
wait_job "${l2}" "AWQ LoRA" "${LOG_ROOT}/train_awq_end_think_lora.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

echo "========== 阶段 6/6：4 卡顺序评测 AWQ =========="
run_eval awq_entropy_grad "${AWQ_EG_MODEL}" "" w4a16
run_eval awq_entropy_grad "${AWQ_EG_MODEL}" "" w4a4
run_eval awq "${AWQ_MODEL}" "" w4a16
run_eval awq "${AWQ_MODEL}" "" w4a4
run_eval awq_entropy_grad_lora "${AWQ_EG_MODEL}" "${AWQ_EG_LORA}" w4a16
run_eval awq_entropy_grad_lora "${AWQ_EG_MODEL}" "${AWQ_EG_LORA}" w4a4
run_eval awq_lora "${AWQ_MODEL}" "${AWQ_LORA}" w4a16
run_eval awq_lora "${AWQ_MODEL}" "${AWQ_LORA}" w4a4

echo "全部完成。"
echo "模型与 LoRA: Qmodel/"
echo "结果: ${RESULT_ROOT}"
echo "日志: ${LOG_ROOT}"
