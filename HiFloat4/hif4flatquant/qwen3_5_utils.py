import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5

from .flat_linear import HiF4FlatQuantizedLinear
from .function_utils import get_decompose_dim, get_init_scale
from .trans_utils import InvDecomposeTransMatrix, SVDDecomposeTransMatrix


def _record_absmax(current, x):
    coming = x.detach().reshape(-1, x.shape[-1]).abs().max(0)[0]
    return torch.maximum(current, coming)


def _make_trans(args, size):
    left, right = get_decompose_dim(size)
    cls = InvDecomposeTransMatrix if args.flatquant_direct_inv else SVDDecomposeTransMatrix
    return cls(left, right, add_diag=args.flatquant_add_diag)


class HiF4FlatQuantQwen3_5MLP(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.config = module.config
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.act_fn = module.act_fn
        self.gate_proj = HiF4FlatQuantizedLinear(args, module.gate_proj)
        self.up_proj = HiF4FlatQuantizedLinear(args, module.up_proj)
        self.down_proj = HiF4FlatQuantizedLinear(args, module.down_proj)
        self.up_proj.act_quantizer = self.gate_proj.act_quantizer
        self.up_gate_trans = _make_trans(args, self.hidden_size)
        self.down_trans = _make_trans(args, self.intermediate_size)
        self._ori_mode = False
        self.diag_init = args.flatquant_diag_init
        if self.diag_init == "sq_style":
            self.up_smax = torch.ones(self.hidden_size) * 1e-5
            self.down_smax = torch.ones(self.intermediate_size) * 1e-5

    def _ori_forward(self, x):
        if self.diag_init == "sq_style":
            self.up_smax = _record_absmax(self.up_smax.to(x.device), x)
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        if self.diag_init == "sq_style":
            self.down_smax = _record_absmax(self.down_smax.to(x.device), x)
        return self.down_proj._ori_forward(x)

    def _trans_forward(self, x):
        x = self.up_gate_trans(x)
        gate = self.gate_proj(x, qa_trans=self.up_gate_trans)
        up = self.up_proj(x, qa_trans=self.up_gate_trans)
        x = self.act_fn(gate) * up
        x = self.down_trans(x)
        return self.down_proj(x, qa_trans=self.down_trans)

    def forward(self, x):
        return self._ori_forward(x) if self._ori_mode else self._trans_forward(x)

    def init_diag_scale(self, alpha=0.5):
        if self.diag_init == "one_style":
            return
        if self.diag_init != "sq_style":
            raise ValueError(f"Unsupported FlatQuant diag init: {self.diag_init}")
        up_wmax = torch.cat([self.gate_proj.linear.weight, self.up_proj.linear.weight], dim=0).abs().max(0)[0]
        down_wmax = self.down_proj.linear.weight.abs().max(0)[0]
        self.up_gate_trans.diag_scale.data = get_init_scale(up_wmax, self.up_smax.to(up_wmax), alpha)
        self.down_trans.diag_scale.data = get_init_scale(down_wmax, self.down_smax.to(down_wmax), alpha)
        del self.up_smax, self.down_smax
        self.diag_init = None

    def rep_matrix_only(self):
        self.up_gate_trans.to_eval_mode()
        self.down_trans.to_eval_mode()

    @torch.no_grad()
    def reparameterize(self):
        self.rep_matrix_only()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        self.up_gate_trans.use_diag = False
        if self.down_trans.add_diag:
            weight = self.up_proj.linear.weight.data.to(torch.float64)
            weight = weight.t().mul(self.down_trans.diag_scale.to(torch.float64)).t()
            self.up_proj.linear.weight.data = weight.to(self.up_proj.linear.weight.dtype)
            self.down_trans.use_diag = False


class HiF4FlatQuantQwen3_5Attention(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.head_dim = module.head_dim
        self.num_key_value_groups = module.num_key_value_groups
        self.scaling = module.scaling
        self.attention_dropout = module.attention_dropout
        self.is_causal = module.is_causal
        self.q_proj = HiF4FlatQuantizedLinear(args, module.q_proj)
        self.k_proj = HiF4FlatQuantizedLinear(args, module.k_proj)
        self.v_proj = HiF4FlatQuantizedLinear(args, module.v_proj)
        self.o_proj = HiF4FlatQuantizedLinear(args, module.o_proj)
        self.k_proj.act_quantizer = self.q_proj.act_quantizer
        self.v_proj.act_quantizer = self.q_proj.act_quantizer
        self.q_norm = module.q_norm
        self.k_norm = module.k_norm
        self.ln_trans = _make_trans(args, module.q_proj.in_features)
        self.o_trans = _make_trans(args, module.o_proj.in_features)
        self._ori_mode = False
        self.diag_init = args.flatquant_diag_init
        if self.diag_init == "sq_style":
            self.ln_smax = torch.ones(module.q_proj.in_features) * 1e-5
            self.o_smax = torch.ones(module.o_proj.in_features) * 1e-5

    def _project_qkv(self, hidden_states):
        if self._ori_mode:
            if self.diag_init == "sq_style":
                self.ln_smax = _record_absmax(self.ln_smax.to(hidden_states.device), hidden_states)
            return (
                self.q_proj._ori_forward(hidden_states),
                self.k_proj._ori_forward(hidden_states),
                self.v_proj._ori_forward(hidden_states),
            )
        hidden_states = self.ln_trans(hidden_states)
        return (
            self.q_proj(hidden_states, qa_trans=self.ln_trans),
            self.k_proj(hidden_states, qa_trans=self.ln_trans),
            self.v_proj(hidden_states, qa_trans=self.ln_trans),
        )

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query, key, value = self._project_qkv(hidden_states)
        query_states, gate = torch.chunk(query.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key.view(hidden_shape)).transpose(1, 2)
        value_states = value.view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = qwen3_5.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        attention_interface = qwen3_5.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, qwen3_5.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        if self._ori_mode:
            if self.diag_init == "sq_style":
                self.o_smax = _record_absmax(self.o_smax.to(attn_output.device), attn_output)
            return self.o_proj._ori_forward(attn_output), attn_weights
        attn_output = self.o_trans(attn_output)
        return self.o_proj(attn_output, qa_trans=self.o_trans), attn_weights

    def init_diag_scale(self, alpha=0.5):
        if self.diag_init == "one_style":
            return
        if self.diag_init != "sq_style":
            raise ValueError(f"Unsupported FlatQuant diag init: {self.diag_init}")
        qkv_wmax = torch.cat(
            [self.q_proj.linear.weight, self.k_proj.linear.weight, self.v_proj.linear.weight], dim=0
        ).abs().max(0)[0]
        o_wmax = self.o_proj.linear.weight.abs().max(0)[0]
        self.ln_trans.diag_scale.data = get_init_scale(qkv_wmax, self.ln_smax.to(qkv_wmax), alpha)
        self.o_trans.diag_scale.data = get_init_scale(o_wmax, self.o_smax.to(o_wmax), alpha)
        del self.ln_smax, self.o_smax
        self.diag_init = None

    def rep_matrix_only(self):
        self.ln_trans.to_eval_mode()
        self.o_trans.to_eval_mode()

    @torch.no_grad()
    def reparameterize(self):
        self.rep_matrix_only()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        self.v_proj.reparameterize(qa_trans=self.ln_trans)
        self.o_proj.reparameterize(qa_trans=self.o_trans)


class HiF4FlatQuantQwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        for name in (
            "hidden_size", "num_v_heads", "num_k_heads", "head_k_dim", "head_v_dim",
            "key_dim", "value_dim", "conv_kernel_size", "layer_idx", "activation",
            "act", "layer_norm_epsilon", "causal_conv1d_fn", "causal_conv1d_update",
            "chunk_gated_delta_rule", "recurrent_gated_delta_rule",
        ):
            setattr(self, name, getattr(module, name))
        self.conv_dim = module.conv_dim
        self.conv1d = module.conv1d
        self.dt_bias = module.dt_bias
        self.A_log = module.A_log
        self.norm = module.norm
        self.in_proj_qkv = HiF4FlatQuantizedLinear(args, module.in_proj_qkv)
        self.in_proj_z = HiF4FlatQuantizedLinear(args, module.in_proj_z)
        self.in_proj_b = HiF4FlatQuantizedLinear(args, module.in_proj_b)
        self.in_proj_a = HiF4FlatQuantizedLinear(args, module.in_proj_a)
        self.out_proj = HiF4FlatQuantizedLinear(args, module.out_proj)
        self.in_proj_z.act_quantizer = self.in_proj_qkv.act_quantizer
        self.in_proj_b.act_quantizer = self.in_proj_qkv.act_quantizer
        self.in_proj_a.act_quantizer = self.in_proj_qkv.act_quantizer
        self.ln_trans = _make_trans(args, self.hidden_size)
        self.out_trans = _make_trans(args, self.value_dim)
        self._ori_mode = False
        self.diag_init = args.flatquant_diag_init
        if self.diag_init == "sq_style":
            self.ln_smax = torch.ones(self.hidden_size) * 1e-5
            self.out_smax = torch.ones(self.value_dim) * 1e-5

    def _project_inputs(self, hidden_states):
        if self._ori_mode:
            if self.diag_init == "sq_style":
                self.ln_smax = _record_absmax(self.ln_smax.to(hidden_states.device), hidden_states)
            return (
                self.in_proj_qkv._ori_forward(hidden_states),
                self.in_proj_z._ori_forward(hidden_states),
                self.in_proj_b._ori_forward(hidden_states),
                self.in_proj_a._ori_forward(hidden_states),
            )
        hidden_states = self.ln_trans(hidden_states)
        return (
            self.in_proj_qkv(hidden_states, qa_trans=self.ln_trans),
            self.in_proj_z(hidden_states, qa_trans=self.ln_trans),
            self.in_proj_b(hidden_states, qa_trans=self.ln_trans),
            self.in_proj_a(hidden_states, qa_trans=self.ln_trans),
        )

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        hidden_states = qwen3_5.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1
        )
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states
        mixed_qkv, z, b, a = self._project_inputs(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if use_precomputed_states:
            core_attn_out, last_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=None,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
            )
        if cache_params is not None:
            cache_params.update_recurrent_state(last_state, self.layer_idx)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        if self._ori_mode:
            if self.diag_init == "sq_style":
                self.out_smax = _record_absmax(self.out_smax.to(core_attn_out.device), core_attn_out)
            return self.out_proj._ori_forward(core_attn_out)
        core_attn_out = self.out_trans(core_attn_out)
        return self.out_proj(core_attn_out, qa_trans=self.out_trans)

    def init_diag_scale(self, alpha=0.5):
        if self.diag_init == "one_style":
            return
        if self.diag_init != "sq_style":
            raise ValueError(f"Unsupported FlatQuant diag init: {self.diag_init}")
        in_wmax = torch.cat(
            [
                self.in_proj_qkv.linear.weight, self.in_proj_z.linear.weight,
                self.in_proj_b.linear.weight, self.in_proj_a.linear.weight,
            ],
            dim=0,
        ).abs().max(0)[0]
        out_wmax = self.out_proj.linear.weight.abs().max(0)[0]
        self.ln_trans.diag_scale.data = get_init_scale(in_wmax, self.ln_smax.to(in_wmax), alpha)
        self.out_trans.diag_scale.data = get_init_scale(out_wmax, self.out_smax.to(out_wmax), alpha)
        del self.ln_smax, self.out_smax
        self.diag_init = None

    def rep_matrix_only(self):
        self.ln_trans.to_eval_mode()
        self.out_trans.to_eval_mode()

    @torch.no_grad()
    def reparameterize(self):
        self.rep_matrix_only()
        for linear in (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a):
            linear.reparameterize(qa_trans=self.ln_trans)
        self.out_proj.reparameterize(qa_trans=self.out_trans)


def apply_flatquant_to_qwen3_5(args, model):
    if getattr(model.config, "model_type", "") != "qwen3_5_text":
        raise ValueError("HiF4 FlatQuant first version only supports Qwen3.5 Dense text models.")
    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn = HiF4FlatQuantQwen3_5GatedDeltaNet(args, layer.linear_attn)
        elif layer.layer_type == "full_attention":
            layer.self_attn = HiF4FlatQuantQwen3_5Attention(args, layer.self_attn)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer.layer_type}")
        layer.mlp = HiF4FlatQuantQwen3_5MLP(args, layer.mlp)
    return model
