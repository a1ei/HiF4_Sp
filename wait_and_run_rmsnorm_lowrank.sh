#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：请先执行 conda activate hif4" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

GPTQ_MODEL="${GPTQ_MODEL:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-s1k-head-512x4096}"
HF_HOME="${HF_HOME:-/home/liuzhilei/.cache/huggingface}"
BASE_MODEL="${BASE_MODEL:-}"
OUTPUT="${OUTPUT:-Qmodel/Qwen3.5-4B-HiF4-GPTQ-ReasoningProjectedRMSNorm-r4-s1k-head-512x4096.pt}"
CURVATURE_CHECKPOINT="${CURVATURE_CHECKPOINT:-${OUTPUT}.curvature.pt}"
CURVATURE_MATRIX_CHECKPOINT="${CURVATURE_MATRIX_CHECKPOINT:-${CURVATURE_CHECKPOINT}.C}"
LOG_DIR="${LOG_DIR:-output/rmsnorm_lowrank}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen35_4b_reasoning_projected_rmsnorm.log}"

if [[ -z "${BASE_MODEL}" ]]; then
  HF_MODEL_ROOT="${HF_HOME}/hub/models--Qwen--Qwen3.5-4B"
  HF_MODEL_REF="${HF_MODEL_ROOT}/refs/main"
  if [[ ! -s "${HF_MODEL_REF}" ]]; then
    echo "错误：找不到本地 Qwen3.5-4B cache ref：${HF_MODEL_REF}" >&2
    exit 1
  fi
  HF_MODEL_REVISION="$(<"${HF_MODEL_REF}")"
  BASE_MODEL="${HF_MODEL_ROOT}/snapshots/${HF_MODEL_REVISION}"
fi

if [[ ! -s "${BASE_MODEL}/config.json" || ! -s "${BASE_MODEL}/model.safetensors.index.json" ]]; then
  echo "错误：本地 BF16 模型 snapshot 不完整：${BASE_MODEL}" >&2
  exit 1
fi

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

model_complete() {
  local model_dir="$1"
  MODEL_DIR="${model_dir}" conda run -n hif4 python -c '
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_DIR"])
required = [root / "config.json", root / "tokenizer_config.json"]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)

single_files = [root / "model.safetensors", root / "pytorch_model.bin"]
if any(path.is_file() and path.stat().st_size > 0 for path in single_files):
    raise SystemExit(0)

indexes = [
    root / "model.safetensors.index.json",
    root / "pytorch_model.bin.index.json",
]
for index_path in indexes:
    if not index_path.is_file() or index_path.stat().st_size == 0:
        continue
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = set(index.get("weight_map", {}).values())
    if shards and all((root / shard).is_file() and (root / shard).stat().st_size > 0 for shard in shards):
        raise SystemExit(0)
raise SystemExit(1)
' >/dev/null 2>&1
}

echo "等待 GPTQ 模型完成：${GPTQ_MODEL}"
echo "轮询间隔：${POLL_SECONDS} 秒"
while ! model_complete "${GPTQ_MODEL}"; do
  date '+%Y-%m-%d %H:%M:%S GPTQ 模型尚未完整，继续等待...'
  sleep "${POLL_SECONDS}"
done

if [[ -e "${OUTPUT}" ]]; then
  echo "错误：输出文件已经存在，拒绝覆盖：${OUTPUT}" >&2
  exit 1
fi

echo "GPTQ 模型已完整，开始 8 卡 RMSNorm low-rank：${GPTQ_MODEL}"
echo "BF16 本地模型：${BASE_MODEL}"
echo "日志：${LOG_FILE}"
echo "敏感子空间断点：${CURVATURE_CHECKPOINT}"
echo "原始 C 矩阵断点：${CURVATURE_MATRIX_CHECKPOINT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
torchrun --standalone --nproc_per_node=8 \
  HiFloat4/run_rmsnorm_lowrank.py \
  --model "${BASE_MODEL}" \
  --local-files-only true \
  --gptq true \
  --gptq-load-path "${GPTQ_MODEL}" \
  --hif4a true \
  --cal-dataset s1k-1.1 \
  --cal-nsamples 128 \
  --cal-seqlen 4096 \
  --cal-slice-mode head \
  --cal-slice-offset 0 \
  --lowrank-mode reasoning_projected_lowrank \
  --lowrank-rank 4 \
  --sensitive-rank 128 \
  --curvature-tokens-per-sample 32 \
  --curvature-token-selection grad_norm_topk \
  --lowrank-epochs 10 \
  --lowrank-lr 1e-5 \
  --lowrank-eta 0.05 \
  --max-relative-delta none \
  --lowrank-save-path "${OUTPUT}" \
  --curvature-checkpoint-path "${CURVATURE_CHECKPOINT}" \
  --curvature-matrix-checkpoint-path "${CURVATURE_MATRIX_CHECKPOINT}" \
  2>&1 | tee "${LOG_FILE}"

echo "RMSNorm low-rank 完成：${OUTPUT}"
