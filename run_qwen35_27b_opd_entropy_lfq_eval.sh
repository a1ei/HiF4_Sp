#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：请先执行 conda activate hif4" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
# export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-/root/data/.cache/huggingface}"

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
SEED="${SEED:-42}"
CAL_NSAMPLES="${CAL_NSAMPLES:-512}"
CAL_SEQLEN="${CAL_SEQLEN:-4096}"
EPOCHS="${EPOCHS:-10}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_27b_opd_entropy_lfq}"
RESULT_ROOT="${RESULT_ROOT:-results/qwen35_27b_opd_entropy_lfq}"
OPD_DATA="${OPD_DATA:-${LOG_ROOT}/opd_tokens_512x4096.pt}"
OPD_MODEL="${OPD_MODEL:-Qmodel/Qwen3.5-27B-OmniQuant-OPD-LFQ-W4A16}"
ENTROPY_GRAD_MODEL="${ENTROPY_GRAD_MODEL:-Qmodel/Qwen3.5-27B-OmniQuant-entropy-grad-LFQ-W4A16}"
ENTROPY_GRAD_NORM_MODEL="${ENTROPY_GRAD_NORM_MODEL:-Qmodel/Qwen3.5-27B-OmniQuant-entropy-grad-norm-LFQ-W4A16}"
DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"
mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}" Qmodel

model_complete() {
  local path="$1"
  [[ -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]] &&
    { [[ -f "${path}/model.safetensors" ]] || [[ -f "${path}/model.safetensors.index.json" ]]; }
}

generate_opd() {
  if [[ -f "${OPD_DATA}" ]]; then
    python -c 'import sys,torch; p=torch.load(sys.argv[1],map_location="cpu"); assert tuple(p["input_ids"].shape)==(int(sys.argv[2]),int(sys.argv[3])); assert p["attention_mask"].shape==p["input_ids"].shape' "${OPD_DATA}" "${CAL_NSAMPLES}" "${CAL_SEQLEN}"
    echo "复用 OPD tokens: ${OPD_DATA}"
    return
  fi
  local -a pids=()
  for gpu in 0 1 2 3; do
    if [[ -f "${OPD_DATA}.shard${gpu}" ]]; then
      echo "复用 OPD shard ${gpu}"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" python -m HiFloat4.omniquant.generate_opd_data \
      --model "${MODEL}" --output "${OPD_DATA}.shard${gpu}" \
      --nsamples "${CAL_NSAMPLES}" --seqlen "${CAL_SEQLEN}" --seed "${SEED}" \
      --shard_id "${gpu}" --num_shards 4 --dtype bfloat16 \
      > "${LOG_ROOT}/generate_opd_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" -eq 0 ]] || { echo "OPD 分片生成失败，请查看 ${LOG_ROOT}" >&2; return 1; }
  python -m HiFloat4.omniquant.generate_opd_data \
    --output "${OPD_DATA}" --nsamples "${CAL_NSAMPLES}" --seqlen "${CAL_SEQLEN}" \
    --merge_shards "${OPD_DATA}.shard0" "${OPD_DATA}.shard1" \
      "${OPD_DATA}.shard2" "${OPD_DATA}.shard3"
}

quantize() {
  local gpu="$1" tag="$2" output="$3" importance="$4" opd_data="${5:-}"
  local omni_dir="${LOG_ROOT}/omni_${tag}"
  if model_complete "${output}"; then echo "复用 ${tag}: ${output}"; return; fi
  [[ ! -e "${output}" ]] || { echo "错误：不完整模型目录 ${output}" >&2; return 1; }
  local -a mode_args=(--lfq)
  [[ -n "${opd_data}" ]] && mode_args+=(--opd --opd_data "${opd_data}")
  CUDA_VISIBLE_DEVICES="${gpu}" python -m HiFloat4.omniquant.main \
    --model "${MODEL}" --calib_dataset s1k-1.1 \
    --nsamples "${CAL_NSAMPLES}" --seqlen "${CAL_SEQLEN}" \
    --cal_slice_mode head --cal_slice_offset 0 --batch_size 1 --seed "${SEED}" \
    --epochs "${EPOCHS}" --wbits 4 --abits 16 --weight_quant_format hif4 \
    --lwc --let --lwc_lr 1e-2 --let_lr 5e-3 --lfq_lr 2e-3 \
    --lfq_logits_chunk_size 128 --token_importance "${importance}" \
    --importance_alpha 1.0 --importance_batch_size 1 --importance_mean_normalize \
    --output_dir "${omni_dir}" --save_dir "${output}" "${mode_args[@]}" \
    > "${LOG_ROOT}/quant_${tag}.log" 2>&1
  model_complete "${output}"
  [[ -f "${omni_dir}/omni_parameters.pth" ]]
}

run_eval() {
  local tag="$1" model_path="$2" precision="$3"
  local act_quant=none
  [[ "${precision}" == "w4a4" ]] && act_quant=hif4
  mkdir -p "${RESULT_ROOT}/${tag}/${precision}"
  CUDA_VISIBLE_DEVICES="0,1,2,3" python main.py \
    --model_path "${model_path}" --datasets "${DATASETS}" --tensor_parallel_size 4 \
    --fake_act_quant "${act_quant}" --gpu_memory_utilization 0.9 --batch_size 128 \
    --max_model_len 32768 --max_new_tokens 32768 --max_samples "${MMLU_PRO_SAMPLES}" \
    --temperature 0.7 --top_p 0.8 --top_k 20 --seed "${SEED}" --use_chat_template \
    --output_dir "${RESULT_ROOT}/${tag}/${precision}" \
    > "${LOG_ROOT}/eval_${tag}_${precision}.log" 2>&1
}

run_suite() {
  local tag="$1" model_path="$2"
  run_eval "${tag}" "${model_path}" w4a4
  run_eval "${tag}" "${model_path}" w4a16
}

generate_opd
quantize 0 opd_lfq "${OPD_MODEL}" none "${OPD_DATA}" & p0=$!
quantize 1 entropy_grad_lfq "${ENTROPY_GRAD_MODEL}" entropy_grad & p1=$!
quantize 2 entropy_grad_norm_lfq "${ENTROPY_GRAD_NORM_MODEL}" entropy_grad_norm & p2=$!
failed=0
for pid in "${p0}" "${p1}" "${p2}"; do wait "${pid}" || failed=1; done
[[ "${failed}" -eq 0 ]] || { echo "量化失败，请查看 ${LOG_ROOT}/quant_*.log" >&2; exit 1; }

run_suite opd_lfq "${OPD_MODEL}"
run_suite entropy_grad_lfq "${ENTROPY_GRAD_MODEL}"
run_suite entropy_grad_norm_lfq "${ENTROPY_GRAD_NORM_MODEL}"
echo "全部完成：${RESULT_ROOT}"
