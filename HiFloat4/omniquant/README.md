# Qwen3.5 OmniQuant + LFQ

This package is a minimal, independent reproduction of the official OmniQuant
layer-wise calibration flow for the dense Qwen3.5 language backbone. The
reference checkout under `3rdparty/OmniQuant` is not imported or modified.

Baseline OmniQuant uses hidden-state reconstruction MSE for every decoder
block. Passing `--lfq` changes only the final block to full-vocabulary
soft-label distillation through the frozen final norm and LM Head.

```bash
CUDA_VISIBLE_DEVICES=1 python -m HiFloat4.omniquant.main \
  --model Qwen/Qwen3.5-4B \
  --calib_dataset s1k-1.1 --nsamples 2 --seqlen 64 \
  --batch_size 1 --epochs 1 --wbits 4 --abits 16 \
  --lwc --let --output_dir outputs/omniquant_qwen35_4b
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m HiFloat4.omniquant.main \
  --model Qwen/Qwen3.5-4B \
  --calib_dataset s1k-1.1 --nsamples 2 --seqlen 64 \
  --batch_size 1 --epochs 1 --wbits 4 --abits 16 \
  --lwc --let --lfq --lfq_logits_chunk_size 16 \
  --output_dir outputs/omniquant_lfq_qwen35_4b \
  --save_dir outputs/omniquant_lfq_qwen35_4b/model
```

`omni_parameters.pth` contains only LWC bound factors and LET smooth scales.
The model under `--save_dir` contains ordinary Transformers modules with the
calibrated fake-quantized weights materialized, so LFQ adds no inference module.

Use `--weight_quant_format hif4` to select the repository-native HiF4 weight fake quantizer. The default remains `int4`.

## Entropy-driven OmniQuant

OmniQuant supports `--token_importance none|entropy|entropy_grad`. `entropy` weights its per-token block reconstruction MSE with the global low-entropy weights. `entropy_grad` averages the current layer qkv, o, up/gate and down attribution weights, then applies the result to that layer reconstruction MSE. `none` preserves the original MSE. Entropy weighting cannot be combined with `--lfq`, whose final-layer objective is distribution cross-entropy rather than OmniQuant reconstruction MSE.

```bash
python -m HiFloat4.omniquant.main \
  --model Qwen/Qwen3.5-4B --calib_dataset s1k-1.1 \
  --nsamples 512 --seqlen 4096 --cal_slice_mode head \
  --epochs 10 --wbits 4 --abits 16 --weight_quant_format hif4 \
  --lwc --let --token_importance entropy \
  --entropy_alpha 1.0 --entropy_norm minmax \
  --output_dir outputs/omni_entropy --save_dir Qmodel/Qwen3.5-4B-Omni-Entropy
```

For layer-local entropy-gradient calibration, replace the last importance options with:

```bash
--token_importance entropy_grad --importance_alpha 1.0 \
--importance_batch_size 4 --importance_mean_normalize
```
