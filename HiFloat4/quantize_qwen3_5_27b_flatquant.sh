#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前环境不是 hif4。请先执行: conda activate hif4" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
OUTPUT="${OUTPUT:-Qmodel/Qwen3.5-27b-Hif4-FlatQuant}"
DTYPE="${DTYPE:-float16}"
HIF4_WEIGHT_FORMAT="${HIF4_WEIGHT_FORMAT:-hif4}"
CAL_DATASET="${CAL_DATASET:-c4}"
CAL_NSAMPLES="${CAL_NSAMPLES:-128}"
CAL_SEQLEN="${CAL_SEQLEN:-4096}"
FLATQUANT_EPOCHS="${FLATQUANT_EPOCHS:-15}"
FLATQUANT_CALI_BSZ="${FLATQUANT_CALI_BSZ:-4}"
FLATQUANT_LR="${FLATQUANT_LR:-1e-5}"
FLATQUANT_CALI_TRANS="${FLATQUANT_CALI_TRANS:-true}"
FLATQUANT_ADD_DIAG="${FLATQUANT_ADD_DIAG:-true}"
FLATQUANT_LWC="${FLATQUANT_LWC:-true}"
FLATQUANT_LAC="${FLATQUANT_LAC:-true}"
FLATQUANT_DIRECT_INV="${FLATQUANT_DIRECT_INV:-false}"
FLATQUANT_DIAG_INIT="${FLATQUANT_DIAG_INIT:-sq_style}"
FLATQUANT_DIAG_ALPHA="${FLATQUANT_DIAG_ALPHA:-0.3}"
FLATQUANT_MATRIX_PATH="${FLATQUANT_MATRIX_PATH:-none}"

cd "${REPO_ROOT}"

python HiFloat4/main.py \
  --model "${MODEL}" \
  --dtype "${DTYPE}" \
  --hif4_weight_format "${HIF4_WEIGHT_FORMAT}" \
  --flatquant true \
  --gptq_save_path "${OUTPUT}" \
  --gptq_cal_dataset "${CAL_DATASET}" \
  --gptq_cal_nsamples "${CAL_NSAMPLES}" \
  --gptq_cal_seqlen "${CAL_SEQLEN}" \
  --flatquant_epochs "${FLATQUANT_EPOCHS}" \
  --flatquant_cali_bsz "${FLATQUANT_CALI_BSZ}" \
  --flatquant_lr "${FLATQUANT_LR}" \
  --flatquant_cali_trans "${FLATQUANT_CALI_TRANS}" \
  --flatquant_add_diag "${FLATQUANT_ADD_DIAG}" \
  --flatquant_lwc "${FLATQUANT_LWC}" \
  --flatquant_lac "${FLATQUANT_LAC}" \
  --flatquant_direct_inv "${FLATQUANT_DIRECT_INV}" \
  --flatquant_diag_init "${FLATQUANT_DIAG_INIT}" \
  --flatquant_diag_alpha "${FLATQUANT_DIAG_ALPHA}" \
  --flatquant_matrix_path "${FLATQUANT_MATRIX_PATH}" \
  "$@"
