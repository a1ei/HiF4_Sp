conda activate hif4

OUTPUT=Qmodel/Qwen3.5-27b-NVFP4-RTN-s1k-head-512x4096

AWQ=false \
GPTQ=false \
SMOOTHQUANT=false \
MAGR=false \
HIF4_WEIGHT_FORMAT=nvfp4 \
CAL_DATASET=s1k-1.1 \
CAL_NSAMPLES=512 \
CAL_SEQLEN=4096 \
CAL_SLICE_MODE=head \
CAL_SLICE_OFFSET=0 \
OUTPUT="${OUTPUT}" \
bash HiFloat4/quantize_qwen3_5_27b.sh

CUDA_VISIBLE_DEVICES=0 python HiFloat4/export_nvfp4_activation_scales.py \
  --model_path "${OUTPUT}" \
  --cal_dataset s1k-1.1 \
  --cal_nsamples 512 \
  --seed 0 \
  --cal_seqlen 4096 \
  --cal_slice_mode head \
  --cal_slice_offset 0 \
  --dtype float16