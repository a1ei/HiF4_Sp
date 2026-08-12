#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

bash run_qwen35_4b_gptq_a4_calib_sampling_eval.sh
bash run_qwen35_4b_entropy_direction_comparison.sh
