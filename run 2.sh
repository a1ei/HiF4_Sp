export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=2

GPTQ=false \
AWQ=false \
SMOOTHQUANT=false \
MAGR=true \
MODEL=Qwen/Qwen3.5-27B \
OUTPUT=Qmodel/Qwen3.5-27B-HiF4-MagR_s1k_head_512_4096 \
CAL_DATASET=s1k-1.1 \
CAL_NSAMPLES=512 \
CAL_SEQLEN=4096 \
CAL_SLICE_MODE=head \
bash HiFloat4/quantize_qwen3_5_27b.sh
