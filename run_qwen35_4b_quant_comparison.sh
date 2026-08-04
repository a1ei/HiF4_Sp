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
MAGR_MODEL="${MAGR_MODEL:-Qmodel/Qwen3.5-4B-HiF4-MAGR-s1k-head-512x4096}"
GPTQ_MODEL="${GPTQ_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-s1k-head-512x4096}"
LFQ_MODEL="${LFQ_MODEL:-Qmodel/Qwen3.5-4B-HiF4-OmniQuant-LFQ-W4A16-s1k-head-512x4096}"
LFQ_OMNI_DIR="${LFQ_OMNI_DIR:-outputs/qwen35_4b_hif4_omniquant_lfq_s1k_head_512x4096}"
LORA_DIR="${LORA_DIR:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-end-think-lora}"
RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_quant_comparison}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_quant_comparison}"
DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
mkdir -p Qmodel "${LFQ_OMNI_DIR}" "${RESULT_ROOT}" "${LOG_ROOT}"

model_complete() {
  local path="$1"
  [[ -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]] &&
    { [[ -f "${path}/model.safetensors" ]] ||
      [[ -f "${path}/model.safetensors.index.json" ]] ||
      [[ -f "${path}/pytorch_model.bin" ]] ||
      [[ -f "${path}/pytorch_model.bin.index.json" ]]; }
}
adapter_complete() {
  [[ -f "${LORA_DIR}/adapter_config.json" &&
     -f "${LORA_DIR}/adapter_model.safetensors" &&
     -f "${LORA_DIR}/training_metrics.json" ]]
}
check_model_dir() {
  local path="$1"
  if [[ -e "${path}" ]] && ! model_complete "${path}"; then
    echo "错误：模型目录存在但不完整: ${path}" >&2
    return 1
  fi
}

quantize_magr() {
  check_model_dir "${MAGR_MODEL}"
  if model_complete "${MAGR_MODEL}"; then echo "复用MAGR: ${MAGR_MODEL}"; return; fi
  echo "GPU 0 开始MAGR量化"
  CUDA_VISIBLE_DEVICES=0 MODEL="${MODEL}" OUTPUT="${MAGR_MODEL}" \
    GPTQ=false AWQ=false SMOOTHQUANT=false MAGR=true DTYPE=bfloat16 \
    CAL_DATASET=s1k-1.1 CAL_NSAMPLES="${CAL_NSAMPLES}" CAL_SEQLEN="${CAL_SEQLEN}" \
    CAL_SLICE_MODE="${CAL_SLICE_MODE}" CAL_SLICE_OFFSET=0 \
    HIF4_WEIGHT_FORMAT=hif4 BLOCK_SIZE_LINEAR=64 MAGR_CD_ITER=3 \
    MAGR_ALPHA=0.001 MAGR_ALPHA_GROUPWISE=0.0001 MAGR_PREPROCESS_ITER=200 \
    bash HiFloat4/quantize_qwen3_5_27b.sh \
      --seed "${SEED}" --safe_serialization true --save_only \
      > "${LOG_ROOT}/quant_magr.log" 2>&1
  model_complete "${MAGR_MODEL}"
}

quantize_gptq() {
  check_model_dir "${GPTQ_MODEL}"
  if model_complete "${GPTQ_MODEL}"; then echo "复用GPTQ: ${GPTQ_MODEL}"; return; fi
  echo "GPU 1 开始GPTQ量化"
  CUDA_VISIBLE_DEVICES=1 MODEL="${MODEL}" OUTPUT="${GPTQ_MODEL}" \
    GPTQ=true AWQ=false SMOOTHQUANT=false MAGR=false DTYPE=bfloat16 \
    CAL_DATASET=s1k-1.1 CAL_NSAMPLES="${CAL_NSAMPLES}" CAL_SEQLEN="${CAL_SEQLEN}" \
    CAL_SLICE_MODE="${CAL_SLICE_MODE}" CAL_SLICE_OFFSET=0 \
    HIF4_WEIGHT_FORMAT=hif4 BLOCK_SIZE_LINEAR=64 GPTQ_PERCDAMP=0.01 TOKEN_IMPORTANCE=none \
    bash HiFloat4/quantize_qwen3_5_27b.sh \
      --seed "${SEED}" --safe_serialization true --save_only \
      > "${LOG_ROOT}/quant_gptq.log" 2>&1
  model_complete "${GPTQ_MODEL}"
}

quantize_lfq() {
  check_model_dir "${LFQ_MODEL}"
  if model_complete "${LFQ_MODEL}"; then
    [[ -f "${LFQ_OMNI_DIR}/omni_parameters.pth" ]] || {
      echo "错误：LFQ模型存在但缺少omni_parameters.pth" >&2; return 1; }
    echo "复用LFQ: ${LFQ_MODEL}"; return
  fi
  echo "GPU 2 开始OmniQuant+LFQ量化"
  CUDA_VISIBLE_DEVICES=2 python -m HiFloat4.omniquant.main \
    --model "${MODEL}" --calib_dataset s1k-1.1 \
    --nsamples "${CAL_NSAMPLES}" --seqlen "${CAL_SEQLEN}" \
    --cal_slice_mode "${CAL_SLICE_MODE}" --batch_size 1 --seed "${SEED}" \
    --epochs 10 --wbits 4 --abits 16 --weight_quant_format hif4 --lwc --let --lfq \
    --lwc_lr 1e-2 --let_lr 5e-3 --lfq_lr 2e-3 --lfq_logits_chunk_size 128 \
    --output_dir "${LFQ_OMNI_DIR}" --save_dir "${LFQ_MODEL}" \
    > "${LOG_ROOT}/quant_lfq.log" 2>&1
  model_complete "${LFQ_MODEL}"
  [[ -f "${LFQ_MODEL}/processor_config.json" ]]
  [[ -f "${LFQ_OMNI_DIR}/omni_parameters.pth" ]]
}

train_lora() {
  if [[ -e "${LORA_DIR}" ]] && ! adapter_complete; then
    echo "错误：LoRA目录存在但不完整: ${LORA_DIR}" >&2; return 1
  fi
  if adapter_complete; then echo "复用LoRA: ${LORA_DIR}"; return; fi
  echo "GPU 3 开始训练end-think LoRA"
  CUDA_VISIBLE_DEVICES=3 python HiFloat4/train_end_think_adapter.py \
    --student_model "${GPTQ_MODEL}" --teacher_model "${MODEL}" \
    --output_dir "${LORA_DIR}" --dataset simplescaling/s1K-1.1 \
    --nsamples 128 --max_length 16384 --num_negative_positions 8 --seed "${SEED}" \
    --teacher_device cuda:0 --student_device cuda:0 --dtype bfloat16 \
    --epochs 10 --learning_rate 1e-4 \
    > "${LOG_ROOT}/train_end_think_lora.log" 2>&1
  adapter_complete
}

run_eval() {
  local gpu="$1" tag="$2" model_path="$3" precision="$4" lora_path="${5:-}"
  local tensor_parallel_size="${6:-1}"
  local act=none
  local -a lora_args=()
  [[ "${precision}" == w4a4 ]] && act=hif4
  [[ -n "${lora_path}" ]] && lora_args=(--lora_path "${lora_path}")
  mkdir -p "${RESULT_ROOT}/${tag}/${precision}" "${LOG_ROOT}/${tag}"
  echo "GPU ${gpu} 开始 ${tag} ${precision}，TP=${tensor_parallel_size}"
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py \
    --model_path "${model_path}" --datasets "${DATASETS}" --tensor_parallel_size "${tensor_parallel_size}" \
    --fake_act_quant "${act}" --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --batch_size "${BATCH_SIZE}" --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --max_samples "${MMLU_PRO_SAMPLES}" \
    --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}" \
    --seed "${SEED}" --use_chat_template \
    --output_dir "${RESULT_ROOT}/${tag}/${precision}" "${lora_args[@]}" \
    > "${LOG_ROOT}/${tag}/${precision}.log" 2>&1
}
run_suite() {
  local gpu="$1" tag="$2" model_path="$3" lora_path="${4:-}"
  run_eval "${gpu}" "${tag}" "${model_path}" w4a4 "${lora_path}"
  run_eval "${gpu}" "${tag}" "${model_path}" w4a16 "${lora_path}"
}

python - <<'PY'
import vllm
import vllm._C  # noqa: F401
print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

quantize_magr & magr_q=$!
quantize_gptq & gptq_q=$!
quantize_lfq & lfq_q=$!
if ! wait "${gptq_q}"; then
  echo "GPTQ量化失败: ${LOG_ROOT}/quant_gptq.log" >&2; exit 1
fi
train_lora & lora_q=$!
failed=0
if ! wait "${magr_q}"; then failed=1; fi
if ! wait "${lfq_q}"; then failed=1; fi
if ! wait "${lora_q}"; then failed=1; fi
[[ "${failed}" -eq 0 ]] || { echo "量化或LoRA失败，请查看 ${LOG_ROOT}" >&2; exit 1; }

run_suite 0 magr "${MAGR_MODEL}" & magr_e=$!
run_suite 1 gptq "${GPTQ_MODEL}" & gptq_e=$!
run_suite 2 lfq "${LFQ_MODEL}" & lfq_e=$!
run_suite 3 gptq_end_think_lora "${GPTQ_MODEL}" "${LORA_DIR}" & lora_e=$!
failed=0
if ! wait "${magr_e}"; then failed=1; fi
if ! wait "${gptq_e}"; then failed=1; fi
if ! wait "${lfq_e}"; then failed=1; fi
if ! wait "${lora_e}"; then failed=1; fi
[[ "${failed}" -eq 0 ]] || { echo "量化模型评测失败，请查看 ${LOG_ROOT}" >&2; exit 1; }

run_eval "0,1,2,3" bf16 "${MODEL}" bf16 "" 4

echo "全部完成。结果: ${RESULT_ROOT}，日志: ${LOG_ROOT}"
