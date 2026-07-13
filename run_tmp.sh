export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=4

# GPTQ=true \
# AWQ=false \
# SMOOTHQUANT=false \
# MAGR=false \
# MODEL=Qwen/Qwen3.5-27B \
# OUTPUT=Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_entropy_low_important \
# CAL_DATASET=s1k-1.1 \
# CAL_NSAMPLES=512 \
# CAL_SEQLEN=512 \
# CAL_SLICE_MODE=head \
# TOKEN_IMPORTANCE=entropy \
# ENTROPY_ALPHA=1.0 \
# ENTROPY_NORM=minmax \
# bash HiFloat4/quantize_qwen3_5_27b.sh 


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --model_path "Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_entropy_low_important" \
    --datasets "mmlu_pro" \
    --tensor_parallel_size 4 \
    --max_model_len 32768 \
    --max_new_tokens 32768 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --max_samples 1000 \
    --gpu_memory_utilization 0.9 \
    > "output_zero_shot/gsm8k/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_entropy_low_important_mmlu.log" 2>&1


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --model_path "Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512" \
    --datasets "mmlu_pro" \
    --tensor_parallel_size 4 \
    --max_model_len 32768 \
    --max_new_tokens 32768 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --max_samples 1000 \
    --gpu_memory_utilization 0.9 \
    > "output_zero_shot/gsm8k/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_mmlu.log" 2>&1

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --model_path "Qmodel/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_entropy_grad" \
    --datasets "mmlu_pro" \
    --tensor_parallel_size 4 \
    --max_model_len 32768 \
    --max_new_tokens 32768 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --max_samples 1000 \
    --gpu_memory_utilization 0.9 \
    > "output_zero_shot/gsm8k/Qwen3.5-27B-HiF4-GPTQ_s1k_head_512_512_entropy_grad_mmlu.log" 2>&1
