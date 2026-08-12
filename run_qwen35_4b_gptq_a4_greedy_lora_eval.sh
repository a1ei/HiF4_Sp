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
CAL_SLICE_MODE="${CAL_SLICE_MODE:-head}"
GPTQ_CALIB_BATCH_SIZE=16
IMPORTANCE_BATCH_SIZE=16
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

W4A16_GPTQ="${W4A16_GPTQ:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-s1k-head-512x4096}"
W4A16_EG="${W4A16_EG:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-s1k-head-512x4096}"
W4A16_EGN="${W4A16_EGN:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-s1k-head-512x4096}"
W4A16_GPTQ_LORA="${W4A16_GPTQ_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-end-think-lora}"
W4A16_EG_LORA="${W4A16_EG_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-end-think-lora}"
W4A16_EGN_LORA="${W4A16_EGN_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-end-think-lora}"

W4A4_GPTQ="${W4A4_GPTQ:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-A4Calib-s1k-head-512x4096}"
W4A4_EG="${W4A4_EG:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-A4Calib-s1k-head-512x4096}"
W4A4_EGN="${W4A4_EGN:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-A4Calib-s1k-head-512x4096}"
W4A4_GPTQ_LORA="${W4A4_GPTQ_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-A4Calib-end-think-lora}"
W4A4_EG_LORA="${W4A4_EG_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGrad-A4Calib-end-think-lora}"
W4A4_EGN_LORA="${W4A4_EGN_LORA:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-EntropyGradNorm-A4Calib-end-think-lora}"

RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_gptq_a4_calib_greedy}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_gptq_a4_calib_greedy}"
DATASETS="${DATASETS:-aime25,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
FORCE_EVAL="${FORCE_EVAL:-false}"
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
    echo "错误：LoRA目录存在但不完整: ${path}" >&2
    return 1
  fi
}

quantize_a4_gptq() {
  local gpu="$1" tag="$2" importance="$3" output="$4"
  check_model_dir "${output}"
  if model_complete "${output}"; then
    echo "复用 ${tag}: ${output}"
    return
  fi
  echo "GPU ${gpu} 开始量化 ${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" MODEL="${MODEL}" OUTPUT="${output}" \
    GPTQ=true AWQ=false SMOOTHQUANT=false MAGR=false DTYPE=bfloat16 \
    CAL_DATASET=s1k-1.1 CAL_NSAMPLES="${CAL_NSAMPLES}" CAL_SEQLEN="${CAL_SEQLEN}" \
    CAL_SLICE_MODE="${CAL_SLICE_MODE}" CAL_SLICE_OFFSET=0 \
    HIF4_WEIGHT_FORMAT=hif4 BLOCK_SIZE_LINEAR=64 GPTQ_PERCDAMP=0.01 \
    HIF4A=true ACT_QUANT_FORMAT=hif4 GPTQ_CALIB_BATCH_SIZE="${GPTQ_CALIB_BATCH_SIZE}" \
    TOKEN_IMPORTANCE="${importance}" IMPORTANCE_ALPHA=1.0 \
    IMPORTANCE_MEAN_NORMALIZE=true IMPORTANCE_BATCH_SIZE="${IMPORTANCE_BATCH_SIZE}" \
    bash HiFloat4/quantize_qwen3_5_27b.sh \
      --attn-implementation "${ATTN_IMPLEMENTATION}" \
      --seed "${SEED}" --safe_serialization true --save_only \
      > "${LOG_ROOT}/quant_${tag}.log" 2>&1
  model_complete "${output}"
}

train_lora() {
  local gpu="$1" tag="$2" student="$3" output="$4" student_act="$5"
  local log="${LOG_ROOT}/train_${tag}_lora.log"
  local train_cmd=(
    python HiFloat4/train_end_think_adapter.py
    --student_model "${student}" --teacher_model "${MODEL}" --output_dir "${output}"
    --dataset simplescaling/s1K-1.1 --nsamples 128 --max_length 16384
    --num_negative_positions 8 --seed "${SEED}"
    --teacher_device cuda:0 --student_device cuda:0 --dtype bfloat16
    --attn_implementation "${ATTN_IMPLEMENTATION}" --epochs 10 --learning_rate 1e-4
    --student_fake_act_quant "${student_act}"
  )
  if adapter_complete "${output}"; then
    echo "复用 ${tag} LoRA: ${output}"
    return
  fi
  if [[ -e "${output}" ]]; then
    [[ -f "${output}/feature_cache/examples.pt" &&
       -f "${output}/feature_cache/teacher_features.pt" ]] || {
      echo "错误：LoRA目录存在但没有可恢复的examples/teacher cache: ${output}" >&2
      return 1
    }
    if [[ ! -f "${output}/feature_cache/student_features.pt" ]]; then
      echo "GPU ${gpu} 从student特征阶段恢复 ${tag} LoRA"
      CUDA_VISIBLE_DEVICES="${gpu}" "${train_cmd[@]}" --phase student > "${log}" 2>&1
    fi
    echo "GPU ${gpu} 执行 ${tag} LoRA训练阶段"
    CUDA_VISIBLE_DEVICES="${gpu}" "${train_cmd[@]}" --phase train >> "${log}" 2>&1
    adapter_complete "${output}"
    return
  fi
  echo "GPU ${gpu} 开始训练 ${tag} 的 </think> LoRA"
  CUDA_VISIBLE_DEVICES="${gpu}" "${train_cmd[@]}" > "${log}" 2>&1
  adapter_complete "${output}"
}

result_complete() {
  local result_file
  while IFS= read -r result_file; do
    grep -q '"aime25|0"' "${result_file}" &&
      grep -q '"lcb:codegeneration_v6|0"' "${result_file}" &&
      grep -q '"mmlu_pro|0"' "${result_file}" && return 0
  done < <(find "$1" -type f -name 'results_*.json' -print 2>/dev/null)
  return 1
}

run_eval() {
  local tag="$1" model_path="$2" lora_path="$3" act_quant="$4"
  local output_dir="${RESULT_ROOT}/${tag}" lora_args=()
  if [[ "${FORCE_EVAL}" != "true" ]] && result_complete "${output_dir}"; then
    echo "复用评测结果: ${tag}"
    return
  fi
  [[ -n "${lora_path}" ]] && lora_args=(--lora_path "${lora_path}")
  mkdir -p "${output_dir}" "${LOG_ROOT}/eval"
  echo "GPU 0,1,2,3 TP=4 greedy评测 ${tag}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_path "${model_path}" "${lora_args[@]}" \
    --datasets "${DATASETS}" --tensor_parallel_size 4 \
    --fake_act_quant "${act_quant}" --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --batch_size "${BATCH_SIZE}" --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --max_samples "${MMLU_PRO_SAMPLES}" \
    --temperature 0 --top_p 1 --top_k 0 --seed "${SEED}" --use_chat_template \
    --output_dir "${output_dir}" > "${LOG_ROOT}/eval/${tag}.log" 2>&1
  result_complete "${output_dir}"
}

wait_job() {
  local pid="$1" name="$2" log="$3"
  if ! wait "${pid}"; then
    echo "${name}失败，请查看 ${log}" >&2
    return 1
  fi
}

python - <<'PY'
import vllm
import vllm._C  # noqa: F401
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

for path in "${W4A16_GPTQ}" "${W4A16_EG}" "${W4A16_EGN}"; do
  model_complete "${path}" || { echo "错误：缺少原有W4A16模型: ${path}" >&2; exit 1; }
done

echo "========== 阶段1/3：三卡并行进行A4感知GPTQ量化 =========="
quantize_a4_gptq 0 gptq_a4 none "${W4A4_GPTQ}" & q0=$!
quantize_a4_gptq 1 gptq_entropy_grad_a4 entropy_grad "${W4A4_EG}" & q1=$!
quantize_a4_gptq 2 gptq_entropy_grad_norm_a4 entropy_grad_norm "${W4A4_EGN}" & q2=$!
failed=0
wait_job "${q0}" GPTQ-A4 "${LOG_ROOT}/quant_gptq_a4.log" || failed=1
wait_job "${q1}" GPTQ-EntropyGrad-A4 "${LOG_ROOT}/quant_gptq_entropy_grad_a4.log" || failed=1
wait_job "${q2}" GPTQ-EntropyGradNorm-A4 "${LOG_ROOT}/quant_gptq_entropy_grad_norm_a4.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

echo "========== 阶段2/3：三卡并行训练新A4模型的</think> LoRA =========="
train_lora 0 gptq_a4 "${W4A4_GPTQ}" "${W4A4_GPTQ_LORA}" hif4 & l0=$!
train_lora 1 gptq_entropy_grad_a4 "${W4A4_EG}" "${W4A4_EG_LORA}" hif4 & l1=$!
train_lora 2 gptq_entropy_grad_norm_a4 "${W4A4_EGN}" "${W4A4_EGN_LORA}" hif4 & l2=$!
failed=0
wait_job "${l0}" GPTQ-A4-LoRA "${LOG_ROOT}/train_gptq_a4_lora.log" || failed=1
wait_job "${l1}" GPTQ-EntropyGrad-A4-LoRA "${LOG_ROOT}/train_gptq_entropy_grad_a4_lora.log" || failed=1
wait_job "${l2}" GPTQ-EntropyGradNorm-A4-LoRA "${LOG_ROOT}/train_gptq_entropy_grad_norm_a4_lora.log" || failed=1
[[ "${failed}" -eq 0 ]] || exit 1

for path in "${W4A16_GPTQ_LORA}" "${W4A16_EG_LORA}" "${W4A16_EGN_LORA}"; do
  adapter_complete "${path}" || { echo "错误：缺少原有W4A16 LoRA: ${path}" >&2; exit 1; }
done

echo "========== 阶段3/3：四卡TP=4严格串行执行12组greedy评测 =========="
run_eval w4a16_gptq_base "${W4A16_GPTQ}" "" none
run_eval w4a16_gptq_lora "${W4A16_GPTQ}" "${W4A16_GPTQ_LORA}" none
run_eval w4a16_entropy_grad_base "${W4A16_EG}" "" none
run_eval w4a16_entropy_grad_lora "${W4A16_EG}" "${W4A16_EG_LORA}" none
run_eval w4a16_entropy_grad_norm_base "${W4A16_EGN}" "" none
run_eval w4a16_entropy_grad_norm_lora "${W4A16_EGN}" "${W4A16_EGN_LORA}" none
run_eval w4a4_gptq_base "${W4A4_GPTQ}" "" hif4
run_eval w4a4_gptq_lora "${W4A4_GPTQ}" "${W4A4_GPTQ_LORA}" hif4
run_eval w4a4_entropy_grad_base "${W4A4_EG}" "" hif4
run_eval w4a4_entropy_grad_lora "${W4A4_EG}" "${W4A4_EG_LORA}" hif4
run_eval w4a4_entropy_grad_norm_base "${W4A4_EGN}" "" hif4
run_eval w4a4_entropy_grad_norm_lora "${W4A4_EGN}" "${W4A4_EGN_LORA}" hif4

echo "全部完成。结果: ${RESULT_ROOT}；日志: ${LOG_ROOT}"
