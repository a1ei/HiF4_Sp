#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT="https://hf-mirror.com"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p output_zero_shot/s1k

model_paths=(
  "Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512"
  "Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_entropy"
  # "Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_1536"
  # "Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_tail_512_512"
  # "Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_tail_512_1536"
)

#super_glue:boolq,glue:rte,winogrande,,openbookqa
for i in 1 2 3 4; do
  echo "开始第 ${i} 次 zero_shot 测试"

  # if [[ "${i}" == "0" ]]; then
  #   datasets="arc:easy,arc:challenge"
  # else
  #   datasets="aime25"
  # fi
  # datasets="aime25"
  datasets="gsm8k"

  for model_path in "${model_paths[@]}"; do
    model_name="${model_path##*/}"
    echo "开始测试 ${model_name}"

    CUDA_VISIBLE_DEVICES=2,5,6,7 python main.py \
      --model_path "${model_path}" \
      --datasets "${datasets}" \
      --tensor_parallel_size 4 \
      --max_model_len 32768 \
      --max_new_tokens 32768 \
      --temperature 0.7 \
      --top_p 0.8 \
      --top_k 20 \
      --gpu_memory_utilization 0.9 \
      > "output_zero_shot/gsm8k/${i}_${model_name}_zero_shot.log" 2>&1

    echo "${model_name} 测试完成"
  done

  echo "第 ${i} 次 zero_shot 测试完成"
done
