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

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
SEED="${SEED:-42}"
CAL_NSAMPLES="${CAL_NSAMPLES:-512}"
CAL_SEQLEN="${CAL_SEQLEN:-4096}"
EPOCHS="${EPOCHS:-10}"
LOG_ROOT="${LOG_ROOT:-output_zero_shot/qwen35_4b_opd_entropy_lfq}"
RESULT_ROOT="${RESULT_ROOT:-results/qwen35_4b_opd_entropy_lfq}"
OPD_DATA="${OPD_DATA:-${LOG_ROOT}/opd_tokens_512x4096.pt}"
OPD_RTN_MODEL="${OPD_RTN_MODEL:-Qmodel/Qwen3.5-4B-HiF4-RTN-W4A16-OPD-generator}"
OPD_MODEL="${OPD_MODEL:-Qmodel/Qwen3.5-4B-OmniQuant-OPD-LFQ-W4A16}"
ENTROPY_GRAD_MODEL="${ENTROPY_GRAD_MODEL:-Qmodel/Qwen3.5-4B-OmniQuant-entropy-grad-LFQ-W4A16}"
ENTROPY_GRAD_NORM_MODEL="${ENTROPY_GRAD_NORM_MODEL:-Qmodel/Qwen3.5-4B-OmniQuant-entropy-grad-norm-LFQ-W4A16}"
DATASETS="${DATASETS:-aime25_avg5,lcb:codegeneration_v6,mmlu_pro}"
MMLU_PRO_SAMPLES="${MMLU_PRO_SAMPLES:-1000}"
mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}" Qmodel
exec 9>"${LOG_ROOT}/run.lock"
flock -n 9 || { echo "错误：已有同一脚本实例正在运行: ${LOG_ROOT}/run.lock" >&2; exit 1; }

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
  if ! model_complete "${OPD_RTN_MODEL}"; then
    [[ ! -e "${OPD_RTN_MODEL}" ]] || { echo "错误：不完整 RTN 模型目录 ${OPD_RTN_MODEL}" >&2; return 1; }
    echo "GPU 0 生成 HiF4 RTN W4A16 student checkpoint"
    CUDA_VISIBLE_DEVICES=0 MODEL="${MODEL}" OUTPUT="${OPD_RTN_MODEL}" \
      GPTQ=false AWQ=false SMOOTHQUANT=false MAGR=false DTYPE=bfloat16 \
      HIF4_WEIGHT_FORMAT=hif4 bash HiFloat4/quantize_qwen3_5_27b.sh \
      --seed "${SEED}" --safe_serialization true --save_only \
      > "${LOG_ROOT}/quant_opd_rtn.log" 2>&1
    model_complete "${OPD_RTN_MODEL}"
  else
    echo "复用 OPD RTN student: ${OPD_RTN_MODEL}"
  fi
  echo "GPU 0-3 使用 vLLM TP=4 批量生成 OPD tokens"
  CUDA_VISIBLE_DEVICES="0,1,2,3" python -m HiFloat4.omniquant.generate_opd_data \
    --backend vllm --model "${OPD_RTN_MODEL}" --output "${OPD_DATA}" \
    --nsamples "${CAL_NSAMPLES}" --seqlen "${CAL_SEQLEN}" --seed "${SEED}" \
    --tensor_parallel_size 4 --gpu_memory_utilization 0.9 --max_model_len 8192 \
    --dtype bfloat16 > "${LOG_ROOT}/generate_opd_vllm.log" 2>&1
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
