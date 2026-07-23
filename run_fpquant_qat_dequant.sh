#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前环境不是 hif4。请先执行: conda activate hif4" >&2
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VLLM_FLOAT32_MATMUL_PRECISION="highest"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SOURCE_MODEL="${SOURCE_MODEL:-ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4}"
FP32_MODEL="${FP32_MODEL:-Qmodel/Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-FP32-NoHadamard}"
BF16_MODEL="${BF16_MODEL:-Qmodel/Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-NoHadamard}"
SQ_MODEL="${SQ_MODEL:-Qmodel/Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-SQScaleOnly}"

PREPARE_MODELS="${PREPARE_MODELS:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
RUN_FP32_FP32A_ONLY="${RUN_FP32_FP32A_ONLY:-false}"

CAL_GPU="${CAL_GPU:-4}"
CAL_DATASET="${CAL_DATASET:-s1k-1.1}"
CAL_NSAMPLES="${CAL_NSAMPLES:-512}"
CAL_SEQLEN="${CAL_SEQLEN:-4096}"
CAL_SLICE_MODE="${CAL_SLICE_MODE:-head}"
CAL_SLICE_OFFSET="${CAL_SLICE_OFFSET:-0}"
SMOOTHQUANT_ALPHA="${SMOOTHQUANT_ALPHA:-0.5}"

FP32_GPUS="${FP32_GPUS:-4}"
BF16_GPUS="${BF16_GPUS:-5}"
SQ_HIF4_GPUS="${SQ_HIF4_GPUS:-6}"
HIF4_GPUS="${HIF4_GPUS:-7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
BASE_SEED="${BASE_SEED:-1234}"
AIME_REPEATS="${AIME_REPEATS:-5}"

RESULT_DIR="${RESULT_DIR:-results/fpquant_qat_dequant_chat}"
LOG_DIR="${LOG_DIR:-output_zero_shot/fpquant_qat_dequant_chat}"

INFERENCE_METADATA_FILES=(
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  added_tokens.json
  special_tokens_map.json
  chat_template.jinja
)

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" Qmodel

validate_dequant_model() {
  local model_dir="$1"
  local expected_dtype="$2"
  python - "${model_dir}" "${expected_dtype}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
expected_dtype = sys.argv[2]
config_path = model_dir / "config.json"
if not config_path.is_file():
    raise SystemExit(f"模型目录不完整，缺少 {config_path}")
if not (
    (model_dir / "model.safetensors").is_file()
    or (model_dir / "model.safetensors.index.json").is_file()
):
    raise SystemExit(f"模型目录没有 safetensors 权重: {model_dir}")
config = json.loads(config_path.read_text(encoding="utf-8"))
dequant = config.get("dequantization_config", {})
if dequant.get("linear_weight_dtype") != expected_dtype:
    raise SystemExit(
        f"{model_dir} 的 linear_weight_dtype={dequant.get('linear_weight_dtype')}，"
        f"期望 {expected_dtype}"
    )
if dequant.get("hadamard_folded") is not True:
    raise SystemExit(f"{model_dir} 没有标记 hadamard_folded=true")
generation_path = model_dir / "generation_config.json"
if not generation_path.is_file():
    raise SystemExit(f"模型目录缺少 {generation_path}")
generation = json.loads(generation_path.read_text(encoding="utf-8"))
eos_token_ids = generation.get("eos_token_id")
if eos_token_ids != [151645, 151643]:
    raise SystemExit(
        f"{model_dir} 的 eos_token_id={eos_token_ids}，期望 [151645, 151643]"
    )
if generation.get("pad_token_id") != 151643:
    raise SystemExit(
        f"{model_dir} 的 pad_token_id={generation.get('pad_token_id')}，期望 151643"
    )
tokenizer_path = model_dir / "tokenizer_config.json"
tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
if tokenizer.get("pad_token") != "<|endoftext|>":
    raise SystemExit(
        f"{model_dir} 的 tokenizer pad_token={tokenizer.get('pad_token')}，"
        "期望 <|endoftext|>"
    )
chat_template_path = model_dir / "chat_template.jinja"
canonical_chat_template = Path("FPQuant/qwen3_chat_template.jinja")
if not chat_template_path.is_file():
    raise SystemExit(f"模型目录缺少 {chat_template_path}")
if chat_template_path.read_text(encoding="utf-8") != canonical_chat_template.read_text(
    encoding="utf-8"
):
    raise SystemExit(f"{chat_template_path} 不是当前 Qwen3 评测模板")
PY
}

validate_sq_model() {
  local model_dir="$1"
  validate_dequant_model "${model_dir}" bfloat16
  python - "${model_dir}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
args_path = model_dir / "quantization_args.json"
if not args_path.is_file():
    raise SystemExit(f"SmoothQuant 模型缺少 {args_path}")
args = json.loads(args_path.read_text(encoding="utf-8"))
expected = {
    "smoothquant": True,
    "smoothquant_scale_only": True,
    "hif4w": False,
}
for key, value in expected.items():
    if args.get(key) != value:
        raise SystemExit(f"{model_dir} 的 {key}={args.get(key)}，期望 {value}")
PY
}

sync_inference_metadata() {
  local source_dir="$1"
  local target_dir="$2"
  local filename
  for filename in "${INFERENCE_METADATA_FILES[@]}"; do
    if [[ ! -f "${source_dir}/${filename}" ]]; then
      echo "基线模型缺少推理配置文件: ${source_dir}/${filename}" >&2
      exit 1
    fi
    cp "${source_dir}/${filename}" "${target_dir}/${filename}"
  done
}

validate_matching_inference_metadata() {
  local reference_dir="$1"
  local model_dir="$2"
  local filename
  for filename in "${INFERENCE_METADATA_FILES[@]}"; do
    if ! cmp -s "${reference_dir}/${filename}" "${model_dir}/${filename}"; then
      echo "推理配置不一致: ${reference_dir}/${filename} 与 ${model_dir}/${filename}" >&2
      exit 1
    fi
  done
}

prepare_dequant_model() {
  local dtype="$1"
  local output_dir="$2"
  if [[ -d "${output_dir}" ]]; then
    validate_dequant_model "${output_dir}" "${dtype}"
    echo "复用已存在模型: ${output_dir}"
    return
  fi
  python FPQuant/convert_checkpoint.py \
    --input_model "${SOURCE_MODEL}" \
    --output_dir "${output_dir}" \
    --output_dtype "${dtype}"
  validate_dequant_model "${output_dir}" "${dtype}"
}

prepare_sq_model() {
  if [[ -d "${SQ_MODEL}" ]]; then
    sync_inference_metadata "${BF16_MODEL}" "${SQ_MODEL}"
    validate_sq_model "${SQ_MODEL}"
    echo "复用已存在 SmoothQuant 模型: ${SQ_MODEL}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${CAL_GPU}" python HiFloat4/main.py \
    --model "${BF16_MODEL}" \
    --dtype bfloat16 \
    --hif4w false \
    --hif4a false \
    --gptq false \
    --smoothquant true \
    --smoothquant_scale_only true \
    --smoothquant_alpha "${SMOOTHQUANT_ALPHA}" \
    --awq false \
    --magr false \
    --flatquant false \
    --save_nvfp4_activation_scales false \
    --gptq_save_path "${SQ_MODEL}" \
    --cal_dataset "${CAL_DATASET}" \
    --cal_nsamples "${CAL_NSAMPLES}" \
    --cal_seqlen "${CAL_SEQLEN}" \
    --cal_slice_mode "${CAL_SLICE_MODE}" \
    --cal_slice_offset "${CAL_SLICE_OFFSET}" \
    --safe_serialization true \
    --save_only
  sync_inference_metadata "${BF16_MODEL}" "${SQ_MODEL}"
  validate_sq_model "${SQ_MODEL}"
}

if [[ "${PREPARE_MODELS}" == "true" ]]; then
  prepare_dequant_model float32 "${FP32_MODEL}"
  prepare_dequant_model bfloat16 "${BF16_MODEL}"
  prepare_sq_model
elif [[ "${PREPARE_MODELS}" != "false" ]]; then
  echo "PREPARE_MODELS 只能是 true 或 false" >&2
  exit 1
fi

if [[ "${RUN_EVAL}" == "false" ]]; then
  echo "模型准备完成，RUN_EVAL=false，跳过评测。"
  exit 0
fi
if [[ "${RUN_EVAL}" != "true" ]]; then
  echo "RUN_EVAL 只能是 true 或 false" >&2
  exit 1
fi

if (( AIME_REPEATS <= 0 )); then
  echo "AIME_REPEATS 必须是正整数" >&2
  exit 1
fi
if (( BASE_SEED < 0 )); then
  echo "BASE_SEED 必须是非负整数" >&2
  exit 1
fi

declare -A USED_GPUS=()
validate_gpu_group() {
  local label="$1"
  local value="$2"
  local ids=()
  local id
  IFS=',' read -r -a ids <<< "${value}"
  if [[ "${#ids[@]}" -ne "${TENSOR_PARALLEL_SIZE}" ]]; then
    echo "${label}=${value} 包含 ${#ids[@]} 张卡，但 TP=${TENSOR_PARALLEL_SIZE}" >&2
    exit 1
  fi
  for id in "${ids[@]}"; do
    if [[ ! "${id}" =~ ^[0-9]+$ ]]; then
      echo "${label} 包含非法 GPU 编号: ${id}" >&2
      exit 1
    fi
    if [[ -n "${USED_GPUS[${id}]:-}" ]]; then
      echo "GPU ${id} 同时分配给 ${USED_GPUS[${id}]} 和 ${label}" >&2
      exit 1
    fi
    USED_GPUS[${id}]="${label}"
  done
}

validate_gpu_group FP32_GPUS "${FP32_GPUS}"
validate_gpu_group BF16_GPUS "${BF16_GPUS}"
validate_gpu_group SQ_HIF4_GPUS "${SQ_HIF4_GPUS}"
validate_gpu_group HIF4_GPUS "${HIF4_GPUS}"

validate_dequant_model "${FP32_MODEL}" float32
validate_dequant_model "${BF16_MODEL}" bfloat16
validate_sq_model "${SQ_MODEL}"
validate_matching_inference_metadata "${BF16_MODEL}" "${FP32_MODEL}"
validate_matching_inference_metadata "${BF16_MODEL}" "${SQ_MODEL}"

python - <<'PY'
import vllm
import vllm._C  # noqa: F401

print(f"vLLM OK: {vllm.__version__} ({vllm.__file__})")
PY

run_eval() {
  local tag="$1"
  local model_path="$2"
  local act_quant="$3"
  local precision_mode="$4"
  local dataset="$5"
  local max_samples="$6"
  local seed="$7"
  local gpus="$8"
  local run_name="${tag}_${dataset//[:]/_}_seed${seed}"
  local output_dir="${RESULT_DIR}/${run_name}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local optional_args=()

  if [[ -n "${max_samples}" ]]; then
    optional_args+=(--max_samples "${max_samples}")
  fi
  case "${precision_mode}" in
    normal)
      ;;
    fp32w_bf16a)
      optional_args+=(--fp32_weights_bf16_activations)
      ;;
    fp32w_fp32a)
      optional_args+=(--fp32_weights_fp32_activations)
      ;;
    *)
      echo "未知精度模式: ${precision_mode}" >&2
      return 1
      ;;
  esac

  echo "开始: ${run_name}，GPU=${gpus}，模型=${model_path}，激活=${act_quant}"
  CUDA_VISIBLE_DEVICES="${gpus}" python main.py \
    --model_path "${model_path}" \
    --datasets "${dataset}" \
    --fake_act_quant "${act_quant}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --use_chat_template \
    --seed "${seed}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --output_dir "${output_dir}" \
    "${optional_args[@]}" \
    > "${log_file}" 2>&1
  echo "完成: ${run_name}"
}

run_suite() {
  local tag="$1"
  local model_path="$2"
  local act_quant="$3"
  local precision_mode="$4"
  local gpus="$5"

  run_eval "${tag}" "${model_path}" "${act_quant}" "${precision_mode}" \
    "lcb:codegeneration_v6" "" "${BASE_SEED}" "${gpus}"
  run_eval "${tag}" "${model_path}" "${act_quant}" "${precision_mode}" \
    "mmlu_pro" "3000" "${BASE_SEED}" "${gpus}"

  local run_idx
  local seed
  for run_idx in $(seq 0 $((AIME_REPEATS - 1))); do
    seed=$((BASE_SEED + run_idx))
    run_eval "${tag}_aime_run$((run_idx + 1))" "${model_path}" \
      "${act_quant}" "${precision_mode}" "aime25" "" "${seed}" "${gpus}"
  done
}

wait_jobs() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "有评测失败，请查看 ${LOG_DIR}。" >&2
    exit 1
  fi
}

if [[ "${RUN_FP32_FP32A_ONLY}" == "true" ]]; then
  run_suite "fp32w_fp32a" "${FP32_MODEL}" "none" fp32w_fp32a "${FP32_GPUS}"
  echo "W FP32 / A FP32 评测完成。结果: ${RESULT_DIR}，日志: ${LOG_DIR}"
  exit 0
fi
if [[ "${RUN_FP32_FP32A_ONLY}" != "false" ]]; then
  echo "RUN_FP32_FP32A_ONLY 只能是 true 或 false" >&2
  exit 1
fi

run_suite "fp32w_bf16a" "${FP32_MODEL}" "none" fp32w_bf16a "${FP32_GPUS}" &
fp32_pid=$!
run_suite "bf16w_bf16a" "${BF16_MODEL}" "none" normal "${BF16_GPUS}" &
bf16_pid=$!
run_suite "sq_bf16w_hif4a" "${SQ_MODEL}" "hif4" normal "${SQ_HIF4_GPUS}" &
sq_hif4_pid=$!
run_suite "bf16w_hif4a" "${BF16_MODEL}" "hif4" normal "${HIF4_GPUS}" &
hif4_pid=$!

wait_jobs "${fp32_pid}" "${bf16_pid}" "${sq_hif4_pid}" "${hif4_pid}"
echo "全部完成。结果: ${RESULT_DIR}，日志: ${LOG_DIR}"
