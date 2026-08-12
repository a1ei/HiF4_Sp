import logging
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
    _compute_fp_token_entropy,
    _compute_layer_local_token_weights,
    _get_layers,
    _is_qwen3_5_text_model,
    _layer_kwargs_for_current_layer,
    _local_importance_group_for_linear,
    _normalize_entropy_importance,
    _run_layer,
    _validate_no_padding_attention_mask,
    find_qlayers,
)
from hif4_gpu.quant_cy import QType, quant_dequant_float


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False




def _is_excluded_layer(name: str, exclude_layers: list[str]) -> bool:
    return name in exclude_layers or "lm_head" in name


def _get_op_by_name(module: nn.Module, op_name: str) -> nn.Module:
    if not op_name:
        return module
    op = module
    for attr in op_name.split("."):
        op = getattr(op, attr)
    return op


def _get_op_name(module: nn.Module, op: nn.Module) -> str:
    for name, cur_op in module.named_modules():
        if cur_op is op:
            return name
    raise ValueError(f"Cannot find module name for {op}.")


@torch.no_grad()
def _get_act_scale(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2 or x.shape[-1] == 0:
        raise ValueError("AWQ activation tensor must have a non-empty feature dimension.")
    feature_sum = torch.zeros(x.shape[-1], dtype=torch.float32, device="cpu")
    token_count = 0
    for start in range(0, x.shape[0], _AWQ_SEARCH_BATCH_SIZE):
        chunk = x[start : start + _AWQ_SEARCH_BATCH_SIZE].detach().cpu().float()
        feature_sum += chunk.abs().reshape(-1, x.shape[-1]).sum(dim=0)
        token_count += chunk.numel() / x.shape[-1]
    if token_count == 0:
        raise ValueError("AWQ activation tensor contains zero tokens.")
    return (feature_sum / token_count).to(dtype=x.dtype)


def _module_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output

_AWQ_SEARCH_BATCH_SIZE = 16
@torch.no_grad()
def _module_output_minibatch(block: nn.Module, x: torch.Tensor, kwargs: dict) -> torch.Tensor:
    outs = []
    for start in range(0, x.shape[0], _AWQ_SEARCH_BATCH_SIZE):
        end = min(start + _AWQ_SEARCH_BATCH_SIZE, x.shape[0])
        device = next(block.parameters()).device
        chunk = x[start:end].to(device)
        out = _module_output(block(chunk, **kwargs))
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def _module_reconstruction_loss_minibatch(
    block: nn.Module,
    x: torch.Tensor,
    kwargs: dict,
    org_out: torch.Tensor,
    token_weights: torch.Tensor | None,
) -> float:
    device = next(block.parameters()).device
    error_sum = torch.zeros((), dtype=torch.float32, device=device)
    normalizer = torch.zeros((), dtype=torch.float32, device=device)

    for start in range(0, x.shape[0], _AWQ_SEARCH_BATCH_SIZE):
        end = min(start + _AWQ_SEARCH_BATCH_SIZE, x.shape[0])
        chunk = x[start:end].to(device)
        out = _module_output(block(chunk, **kwargs))
        token_error = (org_out[start:end].to(device) - out).float().pow(2)
        while token_error.ndim > 2:
            token_error = token_error.mean(dim=-1)

        if token_weights is None:
            error_sum += token_error.sum()
            normalizer += token_error.numel()
            continue

        weights = token_weights[start:end].to(device, token_error.dtype)
        if weights.shape != token_error.shape:
            raise ValueError(
                f"AWQ token weight shape {tuple(weights.shape)} does not match "
                f"reconstruction error shape {tuple(token_error.shape)}."
            )
        error_sum += (token_error * weights).sum()
        normalizer += weights.sum()

    if not torch.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("AWQ reconstruction loss must have a positive finite normalizer.")
    return (error_sum / normalizer).item()


def _module_kwargs_for_inspect(module_name: str, layer_kwargs: dict) -> dict:
    kwargs = dict(layer_kwargs)
    kwargs.pop("use_cache", None)

    if module_name == "linear_attn":
        linear_attn_kwargs = {}
        if "attention_mask" in kwargs:
            linear_attn_kwargs["attention_mask"] = kwargs["attention_mask"]
        if "past_key_values" in kwargs:
            linear_attn_kwargs["cache_params"] = kwargs["past_key_values"]
        return linear_attn_kwargs

    if module_name == "mlp" or module_name is None:
        return {}

    return kwargs


def _quantize_hif4(weight: torch.Tensor, qparams: QType) -> torch.Tensor:
    quant_weight = quant_dequant_float(weight.contiguous(), qparams, force_fp32=True)
    if torch.any(torch.isnan(quant_weight)):
        raise ValueError("NaN in AWQ HiF4 quantized weights.")
    if torch.any(torch.isinf(quant_weight)):
        raise ValueError("Inf in AWQ HiF4 quantized weights.")
    return quant_weight.to(dtype=weight.dtype, device=weight.device)


@torch.no_grad()
def _search_module_scale(
    block: nn.Module,
    linears2scale: list[nn.Linear],
    x: torch.Tensor,
    kwargs: dict,
    qparams: QType,
    n_grid: int,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if n_grid <= 0:
        raise ValueError("AWQ n_grid must be positive.")

    x = x.detach().cpu()
    org_out = _module_output_minibatch(block, x, kwargs)
    x_max = _get_act_scale(x)

    best_error = float("inf")
    best_ratio = -1
    best_scales = None
    history = []

    org_sd = {k: v.detach().cpu().clone() for k, v in block.state_dict().items()}
    for ratio_idx in range(n_grid):
        ratio = ratio_idx / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / torch.sqrt(scales.max() * scales.min())
        if torch.any(~torch.isfinite(scales)) or torch.any(scales <= 0):
            raise ValueError("AWQ produced invalid candidate scales.")

        for fc in linears2scale:
            if fc.in_features != scales.numel():
                raise ValueError("AWQ scale size does not match Linear input size.")
            fc.weight.data.mul_(scales.view(1, -1).to(fc.weight.device))
            fc.weight.data = (
                _quantize_hif4(fc.weight.data, qparams).detach()
                / scales.view(1, -1).to(fc.weight.device)
            )

        loss = _module_reconstruction_loss_minibatch(
            block, x, kwargs, org_out, token_weights
        )
        history.append(loss)
        if loss < best_error:
            best_error = loss
            best_ratio = ratio
            best_scales = scales.detach().cpu()

        block.load_state_dict(org_sd)

    if best_ratio == -1 or best_scales is None:
        raise RuntimeError(f"AWQ failed to find a scale. Loss history: {history}")
    if torch.any(torch.isnan(best_scales)):
        raise ValueError("AWQ produced NaN scales.")
    return best_scales.view(-1)


def _auto_get_scale(
    module: nn.Module,
    prev_op: nn.Module,
    layers: list[nn.Linear],
    inp: torch.Tensor,
    qparams: QType,
    n_grid: int,
    module2inspect: nn.Module | None = None,
    kwargs: dict | None = None,
    token_weights: torch.Tensor | None = None,
) -> tuple[str, tuple[str, ...], torch.Tensor]:
    if kwargs is None:
        kwargs = {}
    if module2inspect is None:
        if len(layers) != 1:
            raise ValueError("AWQ requires module2inspect when scaling multiple Linear layers.")
        module2inspect = layers[0]

    scales = _search_module_scale(
        module2inspect,
        layers,
        inp,
        kwargs,
        qparams,
        n_grid,
        token_weights,
    )
    return (
        _get_op_name(module, prev_op),
        tuple(_get_op_name(module, layer) for layer in layers),
        scales.detach().cpu(),
    )


def _require_input_feat(input_feat: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in input_feat:
        raise KeyError(f"Missing AWQ input feature: {name}")
    return input_feat[name]


def _auto_scale_block(
    model,
    layer: nn.Module,
    input_feat: dict[str, torch.Tensor],
    layer_kwargs: dict,
    qparams: QType,
    n_grid: int,
    token_weights: torch.Tensor | None = None,
    group_token_weights: dict[str, torch.Tensor] | None = None,
) -> list[tuple[str, tuple[str, ...], torch.Tensor]]:
    model_type = getattr(model.config, "model_type", "")
    scales_list = []

    def kwargs_for(module_name: str) -> dict:
        return _module_kwargs_for_inspect(module_name, layer_kwargs)

    def add(prev_name: str, layer_names: list[str], inp_name: str, inspect_name: str | None = None) -> None:
        prev_op = _get_op_by_name(layer, prev_name)
        target_layers = [_get_op_by_name(layer, name) for name in layer_names]
        module2inspect = _get_op_by_name(layer, inspect_name) if inspect_name is not None else None
        scales_list.append(
            _auto_get_scale(
                module=layer,
                prev_op=prev_op,
                layers=target_layers,
                inp=_require_input_feat(input_feat, inp_name),
                qparams=qparams,
                n_grid=n_grid,
                module2inspect=module2inspect,
                kwargs=kwargs_for(inspect_name),
                token_weights=(
                    group_token_weights[_local_importance_group_for_linear(inp_name)]
                    if group_token_weights is not None
                    else token_weights
                ),
            )
        )

    if model_type == "qwen3_5_text":
        layer_type = getattr(layer, "layer_type", None)
        if layer_type == "linear_attention":
            add(
                "input_layernorm",
                [
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.in_proj_b",
                    "linear_attn.in_proj_a",
                ],
                "linear_attn.in_proj_qkv",
                "linear_attn",
            )
        elif layer_type == "full_attention":
            add(
                "input_layernorm",
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "self_attn.q_proj",
                "self_attn",
            )
            if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
                add("self_attn.v_proj", ["self_attn.o_proj"], "self_attn.o_proj")
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer_type}")

        add(
            "post_attention_layernorm",
            ["mlp.gate_proj", "mlp.up_proj"],
            "mlp.gate_proj",
            "mlp",
        )
        add("mlp.up_proj", ["mlp.down_proj"], "mlp.down_proj")
        return scales_list

    if model_type in {"llama", "qwen3", "qwen3_5"}:
        add(
            "input_layernorm",
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "self_attn.q_proj",
            "self_attn",
        )
        if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
            add("self_attn.v_proj", ["self_attn.o_proj"], "self_attn.o_proj")
        add(
            "post_attention_layernorm",
            ["mlp.gate_proj", "mlp.up_proj"],
            "mlp.gate_proj",
            "mlp",
        )
        add("mlp.up_proj", ["mlp.down_proj"], "mlp.down_proj")
        return scales_list

    raise NotImplementedError(
        f"Model type {model_type} is out of scope. Supported: llama, qwen3, qwen3_5, qwen3_5_text."
    )


@torch.no_grad()
def _scale_ln_fcs(ln: nn.Module, fcs: list[nn.Linear], scales: torch.Tensor) -> None:
    if not isinstance(fcs, list):
        fcs = [fcs]
    if not hasattr(ln, "weight") or ln.weight is None:
        raise TypeError("AWQ requires a norm layer with a weight parameter.")

    scales = scales.to(ln.weight.device).to(ln.weight.dtype)
    if ln.__class__.__name__ == "Qwen3_5RMSNorm":
        ln.weight.data = ((1.0 + ln.weight.data.float()) / scales.float() - 1.0).to(
            dtype=ln.weight.dtype,
            device=ln.weight.device,
        )
    else:
        ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device).to(fc.weight.dtype))

    for p in ln.parameters():
        if torch.any(torch.isnan(p)):
            raise ValueError("NaN in AWQ scaled norm parameters.")
    for fc in fcs:
        for p in fc.parameters():
            if torch.any(torch.isnan(p)):
                raise ValueError("NaN in AWQ scaled Linear parameters.")


@torch.no_grad()
def _scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor) -> None:
    if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
        raise TypeError("AWQ Linear-to-Linear scaling requires two Linear modules.")

    scales = scales.to(fc1.weight.device).to(fc1.weight.dtype)
    if fc1.weight.shape[0] < scales.numel():
        raise ValueError("AWQ previous Linear output size is smaller than scale size.")
    if fc2.in_features != scales.numel():
        raise ValueError("AWQ target Linear input size does not match scale size.")

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1).to(fc2.weight.device).to(fc2.weight.dtype))

    for p in fc1.parameters():
        if torch.any(torch.isnan(p)):
            raise ValueError("NaN in AWQ scaled previous Linear parameters.")
    for p in fc2.parameters():
        if torch.any(torch.isnan(p)):
            raise ValueError("NaN in AWQ scaled target Linear parameters.")


def _is_norm_module(module: nn.Module) -> bool:
    return not isinstance(module, nn.Linear) and hasattr(module, "weight") and module.weight is not None


@torch.no_grad()
def _apply_scale(
    module: nn.Module,
    scales_list: list[tuple[str, tuple[str, ...], torch.Tensor]],
    input_feat_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    device = next(module.parameters()).device
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = _get_op_by_name(module, prev_op_name)
        layers = [_get_op_by_name(module, name) for name in layer_names]

        prev_op.to(device)
        for layer in layers:
            layer.to(device)
        scales = scales.to(device)

        if isinstance(prev_op, nn.Linear):
            if len(layers) != 1:
                raise ValueError("AWQ Linear prev_op expects exactly one target layer.")
            _scale_fc_fc(prev_op, layers[0], scales)
        elif _is_norm_module(prev_op):
            _scale_ln_fcs(prev_op, layers, scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device).to(inp.dtype))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()


def _quantize_linear_weight(name: str, layer: nn.Linear, qparams: QType) -> None:
    weight = layer.weight.data
    quant_weight = _quantize_hif4(weight, qparams)
    layer.weight.data = quant_weight.to(dtype=weight.dtype, device=weight.device).contiguous()


@torch.no_grad()
def awq_fwrd(model, dataloader, dev, args):
    logging.info("----- HiFloat4 AWQ Weight Quantization -----")
    device = torch.device(dev)
    if device.type != "cuda":
        raise RuntimeError("HiF4 AWQ requires CUDA because quant_dequant_float uses a CUDA kernel.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

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
    outs = torch.zeros_like(inps)

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
    valid_token_mask = torch.ones((nsamples, args.cal_seqlen), dtype=torch.bool)
    token_importance = getattr(args, "token_importance", "none")
    entropy_direction = getattr(args, "entropy_direction", "low")
    token_weights = None
    layer_local_weights = None
    if token_importance == "entropy":
        entropy = _compute_fp_token_entropy(model, layers, inps.cpu(), layer_kwargs, device)
        token_weights = _normalize_entropy_importance(
            entropy,
            valid_token_mask,
            alpha=args.entropy_alpha,
            norm_mode=args.entropy_norm,
            entropy_direction=entropy_direction,
        )
        del entropy
    elif token_importance == "entropy_grad":
        layer_local_weights = _compute_layer_local_token_weights(
            model, layers, inps.cpu(), layer_kwargs, valid_token_mask, device,
            alpha=args.importance_alpha,
            mean_normalize=args.importance_mean_normalize,
            batch_size=args.importance_batch_size,
            entropy_direction=entropy_direction,
        )
    elif token_importance != "none":
        raise ValueError(f"Unsupported AWQ token importance mode: {token_importance}")
    logging.info("AWQ token-importance reconstruction mode: %s", token_importance)
    weight_qtype = getattr(args, "hif4_weight_qtype", "hifx4")
    qparams = QType(weight_qtype).dim(-1)
    exclude_layers = getattr(args, "exclude_layers", ["lm_head"])
    n_grid = getattr(args, "awq_n_grid", 20)

    for i in tqdm.tqdm(range(len(layers)), desc="(AWQ Quant.) Layers"):
        layer = layers[i].to(device)
        full = find_qlayers(layer, layers=[nn.Linear])
        full = {
            name: sub_layer
            for name, sub_layer in full.items()
            if not _is_excluded_layer(name, exclude_layers)
        }

        input_feat = {}

        def cache_input_hook(name):
            def tmp(_, inp, __):
                inp = inp[0]
                input_feat.setdefault(name, []).append(inp.detach().cpu())

            return tmp

        handles = [
            sub_layer.register_forward_hook(cache_input_hook(name))
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

        input_feat = {
            name: torch.cat(features, dim=0)
            for name, features in input_feat.items()
        }

        search_layer_kwargs = _layer_kwargs_for_current_layer(
            model,
            layer,
            inps[0].unsqueeze(0),
            layer_kwargs,
        )
        scales_list = _auto_scale_block(
            model,
            layer,
            input_feat,
            search_layer_kwargs,
            qparams,
            n_grid,
            token_weights=token_weights,
            group_token_weights=(
                layer_local_weights[i] if layer_local_weights is not None else None
            ),
        )
        _apply_scale(layer, scales_list, input_feat_dict=input_feat)
        layer = layer.to(device)

        for name, sub_layer in full.items():
            _quantize_linear_weight(name, sub_layer, qparams)

        for j in range(nsamples):
            layer_input = inps[j].unsqueeze(0)
            current_layer_kwargs = _layer_kwargs_for_current_layer(
                model, layer, layer_input, layer_kwargs
            )
            outs[j] = _run_layer(layer, layer_input, current_layer_kwargs)

        layers[i] = layer.cpu()
        del layer
        del input_feat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    logging.info("----- HiFloat4 AWQ Weight Quantization Done -----")
