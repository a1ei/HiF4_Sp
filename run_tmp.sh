#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：请先执行 conda activate hif4" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPUS="${GPUS:-5,6}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BASE_SEED="${BASE_SEED:-1234}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output_zero_shot/qwen3_5_4b}"
RESULT_ROOT="${RESULT_ROOT:-results/qwen3_5_4b}"

mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"

run_eval() {
  local datasets="$1"
  local tag="$2"
  local seed="$3"

  echo "开始 ${tag}"
  CUDA_VISIBLE_DEVICES="${GPUS}" python main.py \
    --model_path "${MODEL}" \
    --datasets "${datasets}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --max_model_len 32768 \
    --max_new_tokens 32768 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "${seed}" \
    --fake_act_quant none \
    --output_dir "${RESULT_ROOT}/${tag}" \
    > "${OUTPUT_ROOT}/${tag}.log" 2>&1
  echo "完成 ${tag}"
}

run_eval "mmlu_pro" "mmlu_pro" "${BASE_SEED}"

for run_idx in 1 2 3 4 5; do
  seed=$((BASE_SEED + run_idx - 1))
  run_eval "aime25" "aime_run${run_idx}" "${seed}"
done

run_eval "lcb:codegeneration_v6" "livecodebench_v6" "${BASE_SEED}"

echo "Qwen3.5-4B 评测全部完成。"
