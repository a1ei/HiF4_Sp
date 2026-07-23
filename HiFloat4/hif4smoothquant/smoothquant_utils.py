import logging
import os
import pathlib
import sys

import torch
import torch.nn as nn
import tqdm

HIF4_ROOT = pathlib.Path(__file__).resolve().parents[1]
HIF4GPTQ_ROOT = HIF4_ROOT / "hif4gptq"
if str(HIF4_ROOT) not in sys.path:
    sys.path.append(str(HIF4_ROOT))
if str(HIF4GPTQ_ROOT) not in sys.path:
    sys.path.append(str(HIF4GPTQ_ROOT))

from gptq.gptq_utils import (
    _get_layers,
    _is_qwen3_5_text_model,
    _run_layer,
    _validate_no_padding_attention_mask,
    find_qlayers,
)
from hif4_gpu.quant_cy import QType, quant_dequant_float


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SMOOTH_MIN_SCALE = 1e-5
NVFP4_ACTIVATION_SCALES_FILE = "nvfp4_activation_scales.safetensors"
NVFP4_FP4_E2M1_MAX = 6.0
NVFP4_FP8_E4M3FN_MAX = 448.0


def _is_excluded_layer(name: str, exclude_layers: list[str]) -> bool:
    return name in exclude_layers or "lm_head" in name


def _global_layer_name(layer_idx: int, local_name: str) -> str:
    return f"model.layers.{layer_idx}.{local_name}"


def _input_channel_absmax(inp: torch.Tensor) -> torch.Tensor:
    if inp.shape[-1] == 0:
        raise ValueError("Activation calibration received an empty input channel dimension.")
    return inp.detach().abs().float().view(-1, inp.shape[-1]).max(dim=0)[0]


def _make_causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(hidden_states):
        raise TypeError("Causal mask requires floating-point hidden states.")

    batch_size, seq_len, _ = hidden_states.shape
    min_value = torch.finfo(hidden_states.dtype).min
    mask = torch.full(
        (seq_len, seq_len),
        min_value,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


def _make_qwen3_5_causal_mask(model, hidden_states, layer_kwargs):
    attention_mask = layer_kwargs.get("attention_mask")
    if torch.is_tensor(attention_mask) and attention_mask.ndim >= 4:
        return attention_mask

    try:
        from transformers.masking_utils import create_causal_mask

        mask = create_causal_mask(
            config=model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=layer_kwargs.get("cache_position"),
            past_key_values=None,
            position_ids=layer_kwargs.get("position_ids"),
        )
        if mask is None:
            return _make_causal_mask(hidden_states)
        return mask
    except Exception as exc:
        logging.warning(
            "Falling back to local causal mask for Qwen3.5 activation calibration: %s",
            exc,
        )
        return _make_causal_mask(hidden_states)


def _layer_kwargs_for_current_layer(model, layer, hidden_states, base_layer_kwargs):
    layer_kwargs = dict(base_layer_kwargs)
    if not _is_qwen3_5_text_model(model):
        return layer_kwargs

    layer_kwargs["past_key_values"] = None
    layer_kwargs["use_cache"] = False

    layer_type = getattr(layer, "layer_type", None)
    if layer_type == "linear_attention":
        layer_kwargs["attention_mask"] = None
    elif layer_type == "full_attention":
        layer_kwargs["attention_mask"] = _make_qwen3_5_causal_mask(
            model, hidden_states, layer_kwargs
        )
    else:
        raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer_type}")
    return layer_kwargs


def _nvfp4_input_global_scale(absmax: torch.Tensor) -> torch.Tensor:
    max_abs = absmax.detach().float().amax()
    if not torch.isfinite(max_abs):
        raise ValueError("NVFP4 activation scale calibration produced a non-finite max.")
    if max_abs <= 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor(
        [NVFP4_FP8_E4M3FN_MAX * NVFP4_FP4_E2M1_MAX / max_abs.item()],
        dtype=torch.float32,
    )


def build_nvfp4_activation_scales(
    act_absmax: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        f"{name}.input_global_scale": _nvfp4_input_global_scale(absmax).contiguous()
        for name, absmax in act_absmax.items()
    }


@torch.no_grad()
def collect_nvfp4_activation_scales(
    model,
    dataloader,
    dev,
    args,
) -> dict[str, torch.Tensor]:
    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False
    try:
        act_absmax = _get_act_scales(
            model,
            dataloader,
            torch.device(dev),
            args,
            desc="(NVFP4 Act Scale Calib.) Layers",
        )
    finally:
        if use_cache is not None:
            model.config.use_cache = use_cache
    return build_nvfp4_activation_scales(act_absmax)


def attach_nvfp4_activation_scales(
    model,
    dataloader,
    dev,
    args,
) -> dict[str, torch.Tensor]:
    scales = collect_nvfp4_activation_scales(model, dataloader, dev, args)
    model._nvfp4_activation_scales = scales
    logging.info("Collected %s NVFP4 activation scales for vLLM.", len(scales))
    return scales


def save_nvfp4_activation_scales(
    scales: dict[str, torch.Tensor],
    output_dir: str,
) -> str:
    from safetensors.torch import save_file

    if not scales:
        raise ValueError("No NVFP4 activation scales to save.")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, NVFP4_ACTIVATION_SCALES_FILE)
    save_file(scales, path, metadata={"format": "pt"})
    logging.info("Saved NVFP4 activation scales to %s", path)
    return path


def should_save_nvfp4_activation_scales(args) -> bool:
    if not getattr(args, "save_nvfp4_activation_scales", True):
        return False
    return (
        getattr(args, "hif4_weight_format", "") == "nvfp4"
        or getattr(args, "act_quant_format", "") == "nvfp4"
    )


def _record_linear_input_absmax(act_scales: dict[str, torch.Tensor], name: str, inp) -> None:
    if isinstance(inp, tuple):
        inp = inp[0]
    coming_max = _input_channel_absmax(inp).cpu()
    if name in act_scales:
        if act_scales[name].numel() != coming_max.numel():
            raise ValueError(f"SmoothQuant input channel mismatch for {name}.")
        act_scales[name] = torch.max(act_scales[name], coming_max)
    else:
        act_scales[name] = coming_max


def _capture_first_layer_inputs(model, dataloader, device: torch.device, args):
    layers = _get_layers(model)

    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    max_samples = args.cal_nsamples
    inps = torch.zeros(
        (max_samples, args.cal_seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            idx = cache["i"]
            if idx < max_samples:
                inps[idx] = inp
            cache["i"] += 1
            for key, val in kwargs.items():
                cache[key] = val
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache["i"] >= max_samples:
            break

        if _is_qwen3_5_text_model(model):
            _validate_no_padding_attention_mask(batch)

        if isinstance(batch, (list, tuple)):
            input_ids = batch[0]
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
        else:
            input_ids = batch

        try:
            model(input_ids.to(device))
        except ValueError:
            pass

    layers[0] = layers[0].module

    nsamples = min(cache["i"], max_samples)
    if nsamples == 0:
        raise RuntimeError("Calibration dataloader produced zero samples.")

    inps = inps[:nsamples]

    layers[0] = layers[0].cpu()
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_kwargs = {k: v for k, v in cache.items() if k != "i"}
    return layers, inps, layer_kwargs, nsamples


@torch.no_grad()
def _get_act_scales(
    model,
    dataloader,
    device: torch.device,
    args,
    desc: str = "(SmoothQuant Calib.) Layers",
) -> dict[str, torch.Tensor]:
    layers, inps, layer_kwargs, nsamples = _capture_first_layer_inputs(
        model, dataloader, device, args
    )
    outs = torch.zeros_like(inps)
    act_scales: dict[str, torch.Tensor] = {}

    for i in tqdm.tqdm(range(len(layers)), desc=desc):
        layer = layers[i].to(device)
        full = find_qlayers(layer, layers=[nn.Linear])

        def stat_input_hook(name):
            def tmp(_, inp, __):
                _record_linear_input_absmax(act_scales, name, inp)

            return tmp

        handles = [
            sub_layer.register_forward_hook(
                stat_input_hook(_global_layer_name(i, name))
            )
            for name, sub_layer in full.items()
        ]

        for j in range(nsamples):
            layer_input = inps[j].unsqueeze(0)
            current_layer_kwargs = _layer_kwargs_for_current_layer(
                model, layer, layer_input, layer_kwargs
            )
            outs[j] = _run_layer(layer, layer_input, current_layer_kwargs)

        for handle in handles:
            handle.remove()

        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    return act_scales


def _qwen3_5_norm_uses_one_plus_weight(norm: nn.Module) -> bool:
    return norm.__class__.__name__ == "Qwen3_5RMSNorm"


@torch.no_grad()
def _smooth_ln_fcs_llama_like(
    ln: nn.Module,
    fcs: list[nn.Linear],
    act_scales: torch.Tensor,
    alpha: float = 0.5,
) -> None:
    if not isinstance(fcs, list):
        fcs = [fcs]
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SmoothQuant alpha must be in [0, 1].")
    if not hasattr(ln, "weight") or ln.weight is None:
        raise TypeError("SmoothQuant requires a norm layer with a weight parameter.")
    for fc in fcs:
        if not isinstance(fc, nn.Linear):
            raise TypeError("SmoothQuant can only smooth nn.Linear layers.")
        if ln.weight.numel() != fc.in_features or fc.in_features != act_scales.numel():
            raise ValueError("SmoothQuant channel size mismatch.")

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs],
        dim=0,
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=SMOOTH_MIN_SCALE)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1.0 - alpha))
        .clamp(min=SMOOTH_MIN_SCALE)
        .to(device)
        .to(dtype)
    )

    if torch.any(~torch.isfinite(scales)):
        raise ValueError("SmoothQuant produced non-finite scales.")

    if _qwen3_5_norm_uses_one_plus_weight(ln):
        ln.weight.data = ((1.0 + ln.weight.data.float()) / scales.float() - 1.0).to(
            dtype=ln.weight.dtype,
            device=ln.weight.device,
        )
    else:
        ln.weight.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


def _require_act_scale(act_scales: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in act_scales:
        raise KeyError(f"Missing SmoothQuant activation scale: {name}")
    return act_scales[name]


def _smooth_lm_layer(
    model,
    layer: nn.Module,
    layer_idx: int,
    act_scales: dict[str, torch.Tensor],
    alpha: float,
) -> None:
    model_type = getattr(model.config, "model_type", "")

    if model_type == "qwen3_5_text":
        layer_type = getattr(layer, "layer_type", None)
        if layer_type == "linear_attention":
            attn_ln = layer.input_layernorm
            qkv = [
                layer.linear_attn.in_proj_qkv,
                layer.linear_attn.in_proj_z,
                layer.linear_attn.in_proj_b,
                layer.linear_attn.in_proj_a,
            ]
            qkv_input_scales = _require_act_scale(
                act_scales, _global_layer_name(layer_idx, "linear_attn.in_proj_qkv")
            )
            _smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)
        elif layer_type == "full_attention":
            attn_ln = layer.input_layernorm
            qkv = [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ]
            qkv_input_scales = _require_act_scale(
                act_scales, _global_layer_name(layer_idx, "self_attn.q_proj")
            )
            _smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer_type}")

        ffn_ln = layer.post_attention_layernorm
        fcs = [layer.mlp.gate_proj, layer.mlp.up_proj]
        fcs_input_scales = _require_act_scale(
            act_scales, _global_layer_name(layer_idx, "mlp.gate_proj")
        )
        _smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        return

    if model_type in {"llama", "qwen3", "qwen3_5"}:
        attn_ln = layer.input_layernorm
        qkv = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        ]
        qkv_input_scales = _require_act_scale(
            act_scales, _global_layer_name(layer_idx, "self_attn.q_proj")
        )
        _smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

        ffn_ln = layer.post_attention_layernorm
        fcs = [layer.mlp.gate_proj, layer.mlp.up_proj]
        fcs_input_scales = _require_act_scale(
            act_scales, _global_layer_name(layer_idx, "mlp.gate_proj")
        )
        _smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        return

    raise NotImplementedError(
        f"Model type {model_type} is out of scope. Supported: llama, qwen3, qwen3_5, qwen3_5_text."
    )


@torch.no_grad()
def _smooth_lm(model, act_scales: dict[str, torch.Tensor], alpha: float, device: torch.device) -> None:
    layers = _get_layers(model)
    for i in tqdm.tqdm(range(len(layers)), desc="(SmoothQuant Smooth.) Layers"):
        layer = layers[i].to(device)
        _smooth_lm_layer(model, layer, i, act_scales, alpha)
        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _quantize_linear_weight(name: str, layer: nn.Linear, qparams: QType) -> None:
    weight = layer.weight.data
    quant_weight = quant_dequant_float(weight.contiguous(), qparams, force_fp32=True)
    if torch.any(torch.isnan(quant_weight)):
        raise ValueError(f"NaN in SmoothQuant HiF4 quantized weights: {name}")
    if torch.any(torch.isinf(quant_weight)):
        raise ValueError(f"Inf in SmoothQuant HiF4 quantized weights: {name}")
    layer.weight.data = quant_weight.to(dtype=weight.dtype, device=weight.device).contiguous()


@torch.no_grad()
def _quantize_model_weights(
    model,
    qparams: QType,
    device: torch.device,
    exclude_layers: list[str],
) -> None:
    layers = _get_layers(model)
    for i in tqdm.tqdm(range(len(layers)), desc="(SmoothQuant HiF4 Quant.) Layers"):
        layer = layers[i].to(device)
        full = find_qlayers(layer, layers=[nn.Linear])
        for name, sub_layer in full.items():
            global_name = _global_layer_name(i, name)
            if _is_excluded_layer(name, exclude_layers) or _is_excluded_layer(global_name, exclude_layers):
                logging.info("(SmoothQuant HiF4) Excluding layer: %s", global_name)
                continue
            _quantize_linear_weight(global_name, sub_layer, qparams)

        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def smoothquant_fwrd(model, dataloader, dev, args):
    scale_only = getattr(args, "smoothquant_scale_only", False)
    if scale_only:
        logging.info("----- HiFloat4 SmoothQuant Scale-Only Calibration -----")
    else:
        logging.info("----- HiFloat4 SmoothQuant Weight Quantization -----")
    device = torch.device(dev)
    if device.type != "cuda":
        raise RuntimeError("HiF4 SmoothQuant requires CUDA because quant_dequant_float uses a CUDA kernel.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    alpha = getattr(args, "smoothquant_alpha", 0.5) 
    weight_qtype = getattr(args, "hif4_weight_qtype", "hifx4")
    qparams = QType(weight_qtype).dim(-1)
    exclude_layers = getattr(args, "exclude_layers", ["lm_head"])

    try:
        act_scales = _get_act_scales(model, dataloader, device, args)
        _smooth_lm(model, act_scales, alpha, device)
        if scale_only:
            logging.info("Skipping SmoothQuant final weight quantization.")
        else:
            _quantize_model_weights(model, qparams, device, exclude_layers)
        if should_save_nvfp4_activation_scales(args):
            logging.info("Collecting post-SmoothQuant NVFP4 activation scales.")
            attach_nvfp4_activation_scales(model, dataloader, device, args)
    finally:
        model.config.use_cache = use_cache

    if scale_only:
        logging.info("----- HiFloat4 SmoothQuant Scale-Only Calibration Done -----")
    else:
        logging.info("----- HiFloat4 SmoothQuant Weight Quantization Done -----")
