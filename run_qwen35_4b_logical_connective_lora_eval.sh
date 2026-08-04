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
GPTQ_MODEL="${GPTQ_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-s1k-head-512x4096}"
GPTQ_EG_MODEL="${GPTQ_EG_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-s1k-head-512x4096}"
GPTQ_EGN_MODEL="${GPTQ_EGN_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-s1k-head-512x4096}"
GPTQ_LORA="${GPTQ_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-end-think-logical-connective-lora}"
GPTQ_EG_LORA="${GPTQ_EG_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-end-think-logical-connective-lora}"
GPTQ_EGN_LORA="${GPTQ_EGN_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-end-think-logical-connective-lora}"
RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_end_think_logical_connective_lora}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_end_think_logical_connective_lora}"
DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
FORCE_EVAL="${FORCE_EVAL:-false}"
LORA_NSAMPLES="${LORA_NSAMPLES:-128}"
LORA_MAX_LENGTH="${LORA_MAX_LENGTH:-16384}"
LORA_MAX_POSITIONS="${LORA_MAX_POSITIONS:-8}"
LORA_NEGATIVE_POSITIONS="${LORA_NEGATIVE_POSITIONS:-8}"
LORA_EPOCHS="${LORA_EPOCHS:-10}"
LORA_LR="${LORA_LR:-1e-4}"
CONNECTIVES="${CONNECTIVES:-therefore,thus,however,hence,moreover,consequently,first,second,finally,otherwise,alternatively,nevertheless}"
mkdir -p Qmodel "${RESULT_ROOT}" "${LOG_ROOT}"

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
require_model() {
  model_complete "$1" || { echo "错误：缺少完整量化模型: $1" >&2; return 1; }
}

train_lora() {
  local gpu="$1" tag="$2" student_model="$3" output_dir="$4"
  if [[ -e "${output_dir}" ]] && ! adapter_complete "${output_dir}"; then
    echo "错误：adapter 目录存在但不完整: ${output_dir}" >&2; return 1
  fi
  if adapter_complete "${output_dir}"; then echo "复用 ${tag}: ${output_dir}"; return; fi
  echo "GPU ${gpu} 开始训练 ${tag} 的 </think> + 多逻辑连接词 LM Head LoRA"
  CUDA_VISIBLE_DEVICES="${gpu}" python HiFloat4/train_logical_connective_adapter.py \
    --student_model "${student_model}" --teacher_model "${MODEL}" --output_dir "${output_dir}" \
    --dataset simplescaling/s1K-1.1 --nsamples "${LORA_NSAMPLES}" \
    --max_length "${LORA_MAX_LENGTH}" --max_positive_positions "${LORA_MAX_POSITIONS}" \
    --num_negative_positions "${LORA_NEGATIVE_POSITIONS}" --connectives "${CONNECTIVES}" \
    --max_target_tokens 32 --seed "${SEED}" --teacher_device cuda:0 --student_device cuda:0 \
    --dtype bfloat16 --attn_implementation sdpa --epochs "${LORA_EPOCHS}" \
    --learning_rate "${LORA_LR}" > "${LOG_ROOT}/train_${tag}.log" 2>&1
  adapter_complete "${output_dir}"
}

result_complete() {
  find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .
}
run_eval() {
  local tag="$1" model_path="$2" lora_path="$3" precision="$4"
  local output_dir="${RESULT_ROOT}/${tag}/${precision}" act_quant=none
  [[ "${precision}" == "w4a4" ]] && act_quant=hif4
  if [[ "${FORCE_EVAL}" != "true" ]] && result_complete "${output_dir}"; then
    echo "复用评测结果: ${tag} ${precision}"; return
  fi
  mkdir -p "${output_dir}" "${LOG_ROOT}/${tag}"
  echo "GPU ${EVAL_GPUS} TP=${TENSOR_PARALLEL_SIZE} 开始评测 ${tag} ${precision}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" python main.py \
    --model_path "${model_path}" --lora_path "${lora_path}" --datasets "${DATASETS}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" --fake_act_quant "${act_quant}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" --batch_size "${BATCH_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_samples "${MMLU_PRO_SAMPLES}" --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" --top_k "${TOP_K}" --seed "${SEED}" --use_chat_template \
    --output_dir "${output_dir}" > "${LOG_ROOT}/${tag}/${precision}.log" 2>&1
}
wait_job() {
  if ! wait "$1"; then echo "$2 失败，请查看 $3" >&2; return 1; fi
}

python - <<'PY'
import vllm
import vllm._C  # noqa: F401
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY
require_model "${GPTQ_MODEL}"
require_model "${GPTQ_EG_MODEL}"
require_model "${GPTQ_EGN_MODEL}"

echo "========== 阶段 1/2：三卡并行训练多逻辑连接词 LoRA =========="
train_lora 0 gptq "${GPTQ_MODEL}" "${GPTQ_LORA}" & p0=$!
train_lora 1 gptq_entropy_grad "${GPTQ_EG_MODEL}" "${GPTQ_EG_LORA}" & p1=$!
train_lora 2 gptq_entropy_grad_norm "${GPTQ_EGN_MODEL}" "${GPTQ_EGN_LORA}" & p2=$!
failed=0
wait_job "${p0}" GPTQ "${LOG_ROOT}/train_gptq.log" || failed=1
wait_job "${p1}" GPTQ-entropy-grad "${LOG_ROOT}/train_gptq_entropy_grad.log" || failed=1
wait_job "${p2}" GPTQ-entropy-grad-norm "${LOG_ROOT}/train_gptq_entropy_grad_norm.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

echo "========== 阶段 2/2：四卡串行统一评测 =========="
run_eval gptq_logical_lora "${GPTQ_MODEL}" "${GPTQ_LORA}" w4a16
run_eval gptq_logical_lora "${GPTQ_MODEL}" "${GPTQ_LORA}" w4a4
run_eval gptq_entropy_grad_logical_lora "${GPTQ_EG_MODEL}" "${GPTQ_EG_LORA}" w4a16
run_eval gptq_entropy_grad_logical_lora "${GPTQ_EG_MODEL}" "${GPTQ_EG_LORA}" w4a4
run_eval gptq_entropy_grad_norm_logical_lora "${GPTQ_EGN_MODEL}" "${GPTQ_EGN_LORA}" w4a16
run_eval gptq_entropy_grad_norm_logical_lora "${GPTQ_EGN_MODEL}" "${GPTQ_EGN_LORA}" w4a4

echo "全部完成。结果: ${RESULT_ROOT}，日志: ${LOG_ROOT}"
