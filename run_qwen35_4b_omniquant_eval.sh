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
QUANT_MODEL="${QUANT_MODEL:-Qmodel/Qwen3.5-4B-HiF4-OmniQuant-W4A16-s1k-head-512x4096}"
OMNI_DIR="${OMNI_DIR:-outputs/qwen35_4b_hif4_omniquant_s1k_head_512x4096}"
RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_omniquant}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_omniquant}"

SEED="${SEED:-42}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"

mkdir -p Qmodel "${OMNI_DIR}" "${RESULT_ROOT}" "${LOG_ROOT}"

model_complete() {
  local path="$1"
  [[ -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]] &&
    { [[ -f "${path}/model.safetensors" ]] ||
      [[ -f "${path}/model.safetensors.index.json" ]] ||
      [[ -f "${path}/pytorch_model.bin" ]] ||
      [[ -f "${path}/pytorch_model.bin.index.json" ]]; }
}

quantize_omniquant() {
  if [[ -e "${QUANT_MODEL}" ]] && ! model_complete "${QUANT_MODEL}"; then
    echo "错误：模型目录存在但不完整: ${QUANT_MODEL}" >&2
    return 1
  fi
  if model_complete "${QUANT_MODEL}"; then
    [[ -f "${OMNI_DIR}/omni_parameters.pth" ]] || {
      echo "错误：模型存在但缺少 ${OMNI_DIR}/omni_parameters.pth" >&2
      return 1
    }
    echo "复用纯 OmniQuant 模型: ${QUANT_MODEL}"
    return
  fi

  echo "GPU 0 开始纯 OmniQuant 量化；所有 block 使用 MSE"
  CUDA_VISIBLE_DEVICES=0 python -m HiFloat4.omniquant.main \
    --model "${MODEL}" --calib_dataset s1k-1.1 \
    --nsamples 512 --seqlen 4096 --cal_slice_mode head \
    --batch_size 1 --seed "${SEED}" \
    --epochs 10 --wbits 4 --abits 16 --weight_quant_format hif4 \
    --lwc --let --lwc_lr 1e-2 --let_lr 5e-3 \
    --output_dir "${OMNI_DIR}" --save_dir "${QUANT_MODEL}" \
    > "${LOG_ROOT}/quant_omniquant.log" 2>&1

  model_complete "${QUANT_MODEL}"
  [[ -f "${OMNI_DIR}/omni_parameters.pth" ]]
}

run_eval() {
  local precision="$1"
  local act_quant=none
  [[ "${precision}" == "w4a4" ]] && act_quant=hif4

  mkdir -p "${RESULT_ROOT}/${precision}"
  echo "GPU 0,1 开始纯 OmniQuant ${precision} 评测（TP=2）"
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model_path "${QUANT_MODEL}" \
    --datasets "${DATASETS}" \
    --tensor_parallel_size 2 \
    --fake_act_quant "${act_quant}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --batch_size "${BATCH_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_samples "${MMLU_PRO_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --seed "${SEED}" \
    --use_chat_template \
    --output_dir "${RESULT_ROOT}/${precision}" \
    > "${LOG_ROOT}/eval_${precision}.log" 2>&1
}

quantize_omniquant
run_eval w4a4
run_eval w4a16

echo "全部完成。模型: ${QUANT_MODEL}"
echo "结果: ${RESULT_ROOT}"
echo "日志: ${LOG_ROOT}"
