import json
import os
import types

import torch
from einops import rearrange
from torch import nn

from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5Model,
)
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

from .layers import EvalDecomposeTransMatrix, Hif4ActivationQuantizer


def _read_flatquant_config(vllm_config):
    model_path = vllm_config.model_config.model
    if not os.path.isdir(model_path):
        raise ValueError("HiF4 FlatQuant vLLM loading requires a local model directory.")
    config_path = os.path.join(model_path, "hif4_flatquant_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing HiF4 FlatQuant config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("format") != "hif4_flatquant_fake_quant":
        raise ValueError("Invalid HiF4 FlatQuant config format.")
    if config.get("qtype") not in {"hifx4", "hifx4_1"}:
        raise ValueError("HiF4 FlatQuant vLLM inference only supports qtype=hifx4 or hifx4_1.")
    if vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError("HiF4 FlatQuant vLLM inference currently requires tensor_parallel_size=1.")
    if vllm_config.lora_config is not None:
        raise ValueError("HiF4 FlatQuant vLLM inference does not support LoRA.")
    return model_path, config


def _add_mlp_flatquant(module, config):
    module.up_gate_trans = EvalDecomposeTransMatrix(
        module.gate_up_proj.input_size, config["add_diag"], use_diag=False
    )
    module.down_trans = EvalDecomposeTransMatrix(
        module.down_proj.input_size, config["add_diag"], use_diag=False
    )
    module.up_gate_quant = Hif4ActivationQuantizer(config["lac"], config["qtype"])
    module.down_quant = Hif4ActivationQuantizer(config["lac"], config["qtype"])
    module.forward = types.MethodType(_mlp_forward, module)


def _mlp_forward(self, x):
    x = self.up_gate_quant(self.up_gate_trans(x))
    gate_up, _ = self.gate_up_proj(x)
    x = self.act_fn(gate_up)
    x = self.down_quant(self.down_trans(x))
    x, _ = self.down_proj(x)
    return x


def _add_attention_flatquant(module, config):
    module.ln_trans = EvalDecomposeTransMatrix(
        module.qkv_proj.input_size, config["add_diag"], use_diag=False
    )
    module.o_trans = EvalDecomposeTransMatrix(
        module.o_proj.input_size, config["add_diag"], use_diag=True
    )
    module.qkv_quant = Hif4ActivationQuantizer(config["lac"], config["qtype"])
    module.o_quant = Hif4ActivationQuantizer(config["lac"], config["qtype"])
    module.forward = types.MethodType(_attention_forward, module)


def _attention_forward(self, positions, output, hidden_states):
    hidden_states = self.qkv_quant(self.ln_trans(hidden_states))
    qkv, _ = self.qkv_proj(hidden_states)
    if self.attn_output_gate:
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig_shape, -1)
        gate = gate.reshape(*orig_shape, -1)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
    k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    if self.attn_output_gate:
        attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_quant(self.o_trans(attn_output))
    output[:], _ = self.o_proj(attn_output)


def _add_gdn_flatquant(module, config):
    module.ln_trans = EvalDecomposeTransMatrix(
        module.hidden_size, config["add_diag"], use_diag=False
    )
    module.out_trans = EvalDecomposeTransMatrix(
        module.value_dim, config["add_diag"], use_diag=True
    )
    module.in_quant = Hif4ActivationQuantizer(config["lac"], config["qtype"])
    module.out_quant = Hif4ActivationQuantizer(config["lac"], config["qtype"])
    module.forward = types.MethodType(_gdn_forward, module)


def _gdn_forward(self, hidden_states, output):
    num_tokens = hidden_states.size(0)
    hidden_states = self.in_quant(self.ln_trans(hidden_states))
    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    ba, _ = self.in_proj_ba(hidden_states)
    qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
    z_size = self.value_dim // self.tp_size
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.reshape(z.size(0), -1, self.head_v_dim)
    b, a = ba.chunk(2, dim=-1)
    b = b.contiguous()
    a = a.contiguous()
    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a, core_attn_out, self.prefix)
    z_shape = z.shape
    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    core_attn_out = rearrange(core_attn_out.reshape(z_shape), "... h d -> ... (h d)")
    core_attn_out = self.out_quant(self.out_trans(core_attn_out))
    output[:num_tokens], _ = self.out_proj(core_attn_out)


def _map_flat_key(layer, key):
    if ".clip_factor_w_" in key:
        return None
    if key.startswith("self_attn.q_proj.act_quantizer."):
        return key.replace("self_attn.q_proj.act_quantizer.", "self_attn.qkv_quant.")
    if key.startswith("self_attn.o_proj.act_quantizer."):
        return key.replace("self_attn.o_proj.act_quantizer.", "self_attn.o_quant.")
    if key.startswith("linear_attn.in_proj_qkv.act_quantizer."):
        return key.replace("linear_attn.in_proj_qkv.act_quantizer.", "linear_attn.in_quant.")
    if key.startswith("linear_attn.out_proj.act_quantizer."):
        return key.replace("linear_attn.out_proj.act_quantizer.", "linear_attn.out_quant.")
    if key.startswith("mlp.gate_proj.act_quantizer."):
        return key.replace("mlp.gate_proj.act_quantizer.", "mlp.up_gate_quant.")
    if key.startswith("mlp.down_proj.act_quantizer."):
        return key.replace("mlp.down_proj.act_quantizer.", "mlp.down_quant.")
    if ".act_quantizer." in key:
        return None
    return key


class Qwen3_5Hif4FlatQuantModel(Qwen3_5Model):
    def __init__(self, *, vllm_config, prefix=""):
        self.flatquant_model_path, self.flatquant_config = _read_flatquant_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.config.model_type != "qwen3_5_text":
            raise ValueError("HiF4 FlatQuant vLLM model only supports Qwen3.5 Dense.")
        for layer in self.layers:
            if layer.layer_type == "linear_attention":
                _add_gdn_flatquant(layer.linear_attn, self.flatquant_config)
            elif layer.layer_type == "full_attention":
                _add_attention_flatquant(layer.self_attn, self.flatquant_config)
            else:
                raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer.layer_type}")
            _add_mlp_flatquant(layer.mlp, self.flatquant_config)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        matrices_path = os.path.join(self.flatquant_model_path, "flat_matrices.pth")
        if not os.path.isfile(matrices_path):
            raise FileNotFoundError(f"Missing FlatQuant matrices: {matrices_path}")
        flat_matrices = torch.load(matrices_path, map_location="cpu", weights_only=True)
        if len(flat_matrices) != len(self.layers):
            raise ValueError("FlatQuant matrix layer count does not match the vLLM model.")
        for idx in range(self.start_layer, self.end_layer):
            mapped = {}
            for key, value in flat_matrices[idx].items():
                mapped_key = _map_flat_key(self.layers[idx], key)
                if mapped_key is not None:
                    mapped[mapped_key] = value
            expected = {
                key
                for key in self.layers[idx].state_dict()
                if "_trans." in key or "_quant.clip_factor_a_" in key
            }
            received = set(mapped)
            if received != expected:
                raise ValueError(
                    f"FlatQuant vLLM matrix keys do not match layer {idx}: "
                    f"missing={sorted(expected - received)}, extra={sorted(received - expected)}"
                )
            result = self.layers[idx].load_state_dict(mapped, strict=False)
            if result.unexpected_keys:
                raise ValueError(f"Unexpected FlatQuant vLLM matrix keys in layer {idx}: {result.unexpected_keys}")
            loaded.update(f"layers.{idx}.{key}" for key in mapped)
        return loaded


class Qwen3_5Hif4FlatQuantForCausalLM(Qwen3_5ForCausalLMBase):
    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError("Qwen3.5 currently requires mamba_cache_mode=align.")
        self.quant_config = vllm_config.quant_config
        nn.Module.__init__(self)
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = Qwen3_5Hif4FlatQuantModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
