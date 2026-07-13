import json
import logging
import os

import torch

from .function_utils import get_paras_dict_by_name


def kronecker_matmul(x, had_l, had_r):
    init_shape = x.shape
    x = x.reshape(-1, had_l.shape[0], had_r.shape[0])
    x = torch.matmul(x, had_r)
    x = torch.matmul(had_l.t(), x)
    return x.reshape(init_shape)


@torch.no_grad()
def reparameterize_qwen3_5_rmsnorm(norm, trans):
    weight = norm.weight.data.to(torch.float64)
    scale = trans.diag_scale.to(device=weight.device, dtype=torch.float64)
    norm.weight.data = ((1.0 + weight) * scale - 1.0).to(norm.weight.dtype)
    trans.use_diag = False


@torch.no_grad()
def reparameterize_model(model, device=None):
    for layer in model.model.layers:
        if device is not None:
            layer = layer.to(device)
        token_mixer = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
        token_mixer.reparameterize()
        layer.mlp.reparameterize()
        if token_mixer.ln_trans.add_diag:
            reparameterize_qwen3_5_rmsnorm(layer.input_layernorm, token_mixer.ln_trans)
        if layer.mlp.up_gate_trans.add_diag:
            reparameterize_qwen3_5_rmsnorm(layer.post_attention_layernorm, layer.mlp.up_gate_trans)
        if device is not None:
            layer.cpu()
            torch.cuda.empty_cache()
    return model


def save_flat_matrices(model, path):
    os.makedirs(path, exist_ok=True)
    flat_matrices = {}
    for idx, layer in enumerate(model.model.layers):
        token_mixer = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
        token_mixer.rep_matrix_only()
        layer.mlp.rep_matrix_only()
        flat_matrices[idx] = get_paras_dict_by_name(
            layer,
            required_names=["trans.matrix", "trans.diag_scale", "clip_factor_w", "clip_factor_a"],
        )
    matrices_path = os.path.join(path, "flat_matrices.pth")
    torch.save(flat_matrices, matrices_path)
    logging.info("Saved FlatQuant matrices to %s", matrices_path)


def load_flat_matrices(model, matrix_path):
    matrices_path = matrix_path
    if os.path.isdir(matrix_path):
        matrices_path = os.path.join(matrix_path, "flat_matrices.pth")
    if not os.path.isfile(matrices_path):
        raise FileNotFoundError(f"Missing FlatQuant matrix file: {matrices_path}")
    flat_matrices = torch.load(matrices_path, map_location="cpu", weights_only=True)
    if len(flat_matrices) != len(model.model.layers):
        raise ValueError("FlatQuant matrix layer count does not match the model.")
    for idx, layer in enumerate(model.model.layers):
        token_mixer = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
        token_mixer.rep_matrix_only()
        layer.mlp.rep_matrix_only()
        expected = set(
            get_paras_dict_by_name(
                layer,
                required_names=["trans.matrix", "trans.diag_scale", "clip_factor_w","clip_factor_a"],
            )
        )
        received = set(flat_matrices[idx])
        if received != expected:
            raise ValueError(
                f"FlatQuant matrix keys do not match layer {idx}: "
                f"missing={sorted(expected - received)}, extra={sorted(received - expected)}"
            )
        result = layer.load_state_dict(flat_matrices[idx], strict=False)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected FlatQuant matrix keys in layer {idx}: {result.unexpected_keys}")
    return model


def _standard_hf_state_dict(model):
    state_dict = {}
    for name, tensor in model.state_dict().items():
        if "_trans." in name or ".act_quantizer." in name or ".clip_factor_w_" in name:
            continue
        name = name.replace(".linear.weight", ".weight")
        name = name.replace(".linear.bias", ".bias")
        if name in state_dict:
            raise ValueError(f"Duplicate HuggingFace state key after FlatQuant mapping: {name}")
        state_dict[name] = tensor
    return state_dict


def _write_hif4_flatquant_config(args, path):
    config = {
        "format": "hif4_flatquant_fake_quant",
        "qtype": getattr(args, "hif4_weight_qtype", "hifx4"),
        "model_type": "qwen3_5_text",
        "tensor_parallel_size": 1,
        "kv_cache_quantized": False,
        "gdn_recurrent_state_quantized": False,
        "direct_inv": args.flatquant_direct_inv,
        "add_diag": args.flatquant_add_diag,
        "lwc": args.flatquant_lwc,
        "lac": args.flatquant_lac,
    }
    with open(os.path.join(path, "hif4_flatquant_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _write_quantization_args(args, path):
    with open(os.path.join(path, "quantization_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def save_hif4_flatquant_model(model, tokenizer, path, args):
    os.makedirs(path, exist_ok=True)
    matrices_path = os.path.join(path, "flat_matrices.pth")
    if not os.path.isfile(matrices_path):
        raise FileNotFoundError(f"Missing FlatQuant matrix file before save: {matrices_path}")
    model.save_pretrained(
        path,
        state_dict=_standard_hf_state_dict(model),
        safe_serialization=False,
        max_shard_size="5GB",
    )
    model.config.architectures = ["Qwen3_5Hif4FlatQuantForCausalLM"]
    model.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    _write_hif4_flatquant_config(args, path)
    _write_quantization_args(args, path)
    logging.info("Saved HiF4 FlatQuant model to %s", path)
