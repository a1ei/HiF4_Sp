export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# GPTQ=true \
# AWQ=false \
# SMOOTHQUANT=false \
# MAGR=false \
# MODEL=Qwen/Qwen3.5-27B \
# OUTPUT=Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_entropy_grad \
# CAL_DATASET=s1k-1.1 \
# CAL_NSAMPLES=512 \
# CAL_SEQLEN=512 \
# CAL_SLICE_MODE=head \
# TOKEN_IMPORTANCE=entropy_grad \
# IMPORTANCE_ALPHA=1.0 \
# IMPORTANCE_MEAN_NORMALIZE=true \
# IMPORTANCE_BATCH_SIZE=32 \
# bash HiFloat4/quantize_qwen3_5_27b.sh


# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
#     --model_path "Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_entropy_grad" \
#     --datasets "mmlu_pro" \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_new_tokens 32768 \
#     --temperature 0.7 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --max_samples 3000 \
#     --gpu_memory_utilization 0.9 \
#     > "output_zero_shot/gsm8k/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_entropy_grad_mmlu.log" 2>&1
MODEL_PATH="Qmodel/Qwen3.5-27B-HiF4-MagR_s1k_head_128_4096"
RUN_NAME="magr_s1k_head_128_4096"
LOG_DIR="output_zero_shot/${RUN_NAME}"
RESULT_DIR="results/${RUN_NAME}"
FAKE_ACT_QUANT="${FAKE_ACT_QUANT:-none}"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型目录不存在: ${MODEL_PATH}" >&2
  exit 1
fi

run_eval() {
  local datasets="$1"
  local name="$2"
  local max_samples="${3:-}"
  local optional_args=()

  if [[ -n "${max_samples}" ]]; then
    optional_args+=(--max_samples "${max_samples}")
  fi

  echo "开始测试 ${name}: datasets=${datasets}, GPU=6,7, activation=${FAKE_ACT_QUANT}"

  CUDA_VISIBLE_DEVICES=6,7 python main.py \
    --model_path "${MODEL_PATH}" \
    --datasets "${datasets}" \
    --tensor_parallel_size 2 \
    --max_model_len 32768 \
    --max_new_tokens 32768 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --batch_size 8 \
    --gpu_memory_utilization 0.9 \
    --fake_act_quant "${FAKE_ACT_QUANT}" \
    --kv_quant_format none \
    --output_dir "${RESULT_DIR}/${name}" \
    "${optional_args[@]}" \
    > "${LOG_DIR}/${name}.log" 2>&1

  echo "完成测试 ${name}，日志: ${LOG_DIR}/${name}.log"
}

# run_eval "gsm8k,math_500" "gsm8k_math500"
run_eval "mmlu_pro" "mmlu_pro_1000" "1000"
run_eval "lcb:codegeneration_v6" "lcb_v6"
