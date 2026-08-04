"""Learnable equivalent transformations for Qwen3.5 decoder blocks."""

import torch

from .modules import QuantLinear, iter_quant_linears


MIN_SCALE = 1e-5


def _get_groups(layer):
    if layer.layer_type == "full_attention":
        attn_fcs = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
    elif layer.layer_type == "linear_attention":
        attn_fcs = [
            layer.linear_attn.in_proj_qkv,
            layer.linear_attn.in_proj_z,
            layer.linear_attn.in_proj_b,
            layer.linear_attn.in_proj_a,
        ]
    else:
        raise ValueError(f"Unsupported Qwen3.5 layer type: {layer.layer_type}")
    return [
        ("attn_smooth_scale", layer.input_layernorm, attn_fcs),
        ("fc1_smooth_scale", layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj]),
    ]


def _initial_scale(act_max, fcs, alpha):
    weight_max = torch.cat([fc.weight.detach().abs().amax(dim=0, keepdim=True) for fc in fcs], dim=0)
    weight_max = weight_max.amax(dim=0).clamp(min=MIN_SCALE)
    return (act_max.clamp(min=MIN_SCALE).pow(alpha) / weight_max.pow(1.0 - alpha)).clamp(min=MIN_SCALE)


@torch.no_grad()
def initialize_let(layer, fp_inputs, run_layer, alpha):
    """Collect group inputs and register official-style smooth scale parameters."""
    maxima = {}
    handles = []
    for name, _, fcs in _get_groups(layer):
        maxima[name] = torch.zeros(fcs[0].in_features, dtype=torch.float32, device=fp_inputs.device)

        def hook(_module, inputs, key=name):
            value = inputs[0].detach().float().abs().amax(dim=(0, 1))
            maxima[key].copy_(torch.maximum(maxima[key], value))

        handles.append(fcs[0].register_forward_pre_hook(hook))
    try:
        for sample in fp_inputs:
            run_layer(layer, sample.unsqueeze(0))
    finally:
        for handle in handles:
            handle.remove()

    for name, norm, fcs in _get_groups(layer):
        if not all(isinstance(fc, QuantLinear) for fc in fcs):
            raise TypeError("LET groups must contain QuantLinear modules.")
        layer.register_parameter(name, torch.nn.Parameter(_initial_scale(maxima[name], fcs, alpha)))

        def norm_hook(_module, _inputs, output, key=name):
            if getattr(layer, "_let_temporary", False):
                return output / getattr(layer, key).to(output.dtype)
            return output

        norm.register_forward_hook(norm_hook)
    layer._let_temporary = False


def prepare_temporary(layer, use_let):
    for linear in iter_quant_linears(layer):
        linear.temp_weight = linear.weight
        linear.temp_bias = linear.bias
        linear.use_temporary_parameter = True
    layer._let_temporary = bool(use_let)
    if use_let:
        for name, _, fcs in _get_groups(layer):
            scale = getattr(layer, name)
            for fc in fcs:
                fc.temp_weight = fc.temp_weight * scale.to(fc.temp_weight.dtype).view(1, -1)
    for linear in iter_quant_linears(layer):
        linear.temp_weight = linear.weight_quantizer(linear.temp_weight)


def clear_temporary(layer):
    layer._let_temporary = False
    for linear in iter_quant_linears(layer):
        for name in ("temp_weight", "temp_bias"):
            if hasattr(linear, name):
                delattr(linear, name)
        linear.use_temporary_parameter = False


@torch.no_grad()
def smooth_and_quant_inplace(layer, use_let):
    if use_let:
        for name, norm, fcs in _get_groups(layer):
            scale = getattr(layer, name).detach().clamp(min=MIN_SCALE)
            if norm.__class__.__name__ == "Qwen3_5RMSNorm":
                norm.weight.copy_((1.0 + norm.weight.float()) / scale.float() - 1.0)
            else:
                norm.weight.div_(scale.to(norm.weight.dtype))
            for fc in fcs:
                fc.weight.mul_(scale.to(fc.weight.dtype).view(1, -1))
    for linear in iter_quant_linears(layer):
        linear.weight.copy_(linear.weight_quantizer(linear.weight))
    clear_temporary(layer)


def let_parameters(layer):
    return [parameter for name, parameter in layer.named_parameters() if "smooth_scale" in name]


def lwc_parameters(layer):
    return [parameter for name, parameter in layer.named_parameters() if "bound_factor" in name]


def omni_parameters(layer):
    return let_parameters(layer) + lwc_parameters(layer)


def remove_let_parameters(layer):
    for name in ("attn_smooth_scale", "fc1_smooth_scale"):
        if hasattr(layer, name):
            delattr(layer, name)


def omni_state_dict(layer):
    return {
        name: value.detach().cpu()
        for name, value in layer.named_parameters()
        if "smooth_scale" in name or "bound_factor" in name
    }
