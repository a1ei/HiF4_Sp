

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python main.py \
  --model_path "Qmodel/Qwen3.5-27b-Hif4-MagR" \
  --datasets aime25 \
  --tensor_parallel_size 4 \
  --max_model_len 32768 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --gpu_memory_utilization 0.9 \
  > "output/0_output_magr_gptq_aime25.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python main.py \
  --model_path "Qmodel/Qwen3.5-27b-Hif4-MagR" \
  --datasets aime25 \
  --tensor_parallel_size 4 \
  --max_model_len 32768 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --gpu_memory_utilization 0.9 \
  > "output/1_output_magr_gptq_aime25.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python main.py \
  --model_path "Qmodel/Qwen3.5-27b-Hif4-MagR" \
  --datasets aime25 \
  --tensor_parallel_size 4 \
  --max_model_len 32768 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --gpu_memory_utilization 0.9 \
  > "output/2_output_magr_gptq_aime25.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python main.py \
  --model_path "Qmodel/Qwen3.5-27b-Hif4-MagR" \
  --datasets aime25 \
  --tensor_parallel_size 4 \
  --max_model_len 32768 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --gpu_memory_utilization 0.9 \
  > "output/3_output_magr_gptq_aime25.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python main.py \
  --model_path "Qmodel/Qwen3.5-27b-Hif4-MagR" \
  --datasets aime25 \
  --tensor_parallel_size 4 \
  --max_model_len 32768 \
  --max_new_tokens 32768 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --gpu_memory_utilization 0.9 \
  > "output/4_output_magr_gptq_aime25.log" 2>&1 &