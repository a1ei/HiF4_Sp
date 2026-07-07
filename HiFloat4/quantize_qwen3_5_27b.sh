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
OUTPUT="${OUTPUT:-Qmodel/Qwen3.5-27b-Hif4-AWQ}"
GPTQ="${GPTQ:-false}"
SMOOTHQUANT="${SMOOTHQUANT:-false}"
AWQ="${AWQ:-true}"
MAGR="${MAGR:-false}"
DTYPE="${DTYPE:-float16}"
#虽然参数开头是GPTQ，但实际上是用于所有量化方法的校准数据集、样本数和序列长度
GPTQ_CAL_DATASET="${GPTQ_CAL_DATASET:-c4}"
GPTQ_CAL_NSAMPLES="${GPTQ_CAL_NSAMPLES:-512}"
GPTQ_CAL_SEQLEN="${GPTQ_CAL_SEQLEN:-512}"
GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"
BLOCK_SIZE_LINEAR="${BLOCK_SIZE_LINEAR:-64}" #magr中 -1是per_layer
HIF4_WEIGHT_FORMAT="${HIF4_WEIGHT_FORMAT:-hif4}"
SMOOTHQUANT_ALPHA="${SMOOTHQUANT_ALPHA:-0.5}"
AWQ_N_GRID="${AWQ_N_GRID:-20}"
MAGR_CD_ITER="${MAGR_CD_ITER:-1}"
MAGR_ALPHA="${MAGR_ALPHA:-0.001}"
MAGR_ALPHA_GROUPWISE="${MAGR_ALPHA_GROUPWISE:-0.0001}"
MAGR_PREPROCESS_ITER="${MAGR_PREPROCESS_ITER:-200}"

cd "${REPO_ROOT}"

python - <<'PY'
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

if "qwen3_5" not in CONFIG_MAPPING:
    raise RuntimeError(
        "当前 hif4 环境里的 Transformers 不支持 model_type=qwen3_5。"
        "请先按仓库 README 执行 bash install.sh。"
    )

if int(transformers.__version__.split(".", 1)[0]) >= 5:
    print(
        "当前 Transformers 是 "
        f"{transformers.__version__}，可以识别 Qwen3.5。"
    )
PY

python HiFloat4/main.py \
  --model "${MODEL}" \
  --dtype "${DTYPE}" \
  --hif4w true \
  --hif4_weight_format "${HIF4_WEIGHT_FORMAT}" \
  --gptq "${GPTQ}" \
  --smoothquant "${SMOOTHQUANT}" \
  --awq "${AWQ}" \
  --magr "${MAGR}" \
  --gptq_save_path "${OUTPUT}" \
  --gptq_cal_dataset "${GPTQ_CAL_DATASET}" \
  --gptq_cal_nsamples "${GPTQ_CAL_NSAMPLES}" \
  --gptq_cal_seqlen "${GPTQ_CAL_SEQLEN}" \
  --gptq_percdamp "${GPTQ_PERCDAMP}" \
  --block_size_linear "${BLOCK_SIZE_LINEAR}" \
  --smoothquant_alpha "${SMOOTHQUANT_ALPHA}" \
  --awq_n_grid "${AWQ_N_GRID}" \
  --magr_cd_iter "${MAGR_CD_ITER}" \
  --magr_alpha "${MAGR_ALPHA}" \
  --magr_alpha_groupwise "${MAGR_ALPHA_GROUPWISE}" \
  --magr_preprocess_iter "${MAGR_PREPROCESS_ITER}" \
  "$@"
