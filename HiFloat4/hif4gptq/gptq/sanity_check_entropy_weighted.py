import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn


HIF4_ROOT = pathlib.Path(__file__).resolve().parents[2]
HIF4GPTQ_ROOT = HIF4_ROOT / "hif4gptq"
sys.path.insert(0, str(HIF4_ROOT))
sys.path.insert(0, str(HIF4GPTQ_ROOT))

from gptq.gptq_utils import (
    GPTQ,
    _LOCAL_IMPORTANCE_GROUPS,
    _compute_fp_token_entropy,
    _compute_layer_local_token_weights,
    _local_importance_group_for_linear,
    _normalize_entropy_importance,
)


class TinyLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return (self.linear(hidden_states),)


class TinyAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        mixed = self.q_proj(hidden_states) + self.k_proj(hidden_states) + self.v_proj(hidden_states)
        return self.o_proj(mixed)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = TinyAttention(hidden_size)
        self.mlp = TinyMLP(hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TinyCausalLM(nn.Module):
    def __init__(self, hidden_size=4, vocab_size=7, attribution_layers=False):
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", hidden_size=hidden_size)
        self.model = nn.Module()
        layer_cls = TinyDecoderLayer if attribution_layers else TinyLayer
        self.model.layers = nn.ModuleList([layer_cls(hidden_size), layer_cls(hidden_size)])
        self.model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


class EntropyWeightedGPTQSanityCheck(unittest.TestCase):
    def test_none_matches_original_hessian_update(self):
        layer = nn.Linear(4, 3, bias=False)
        block = GPTQ(layer)
        batches = [torch.randn(1, 3, 4), torch.randn(1, 3, 4)]

        expected_h = torch.zeros(4, 4)
        nsamples = 0
        for inp in batches:
            tmp = inp.shape[0]
            expected_h *= nsamples / (nsamples + tmp)
            nsamples += tmp
            x = inp.reshape(-1, inp.shape[-1]).t()
            x = math.sqrt(2 / nsamples) * x.float()
            expected_h += x.matmul(x.t())
            block.add_batch(inp, None)

        self.assertTrue(torch.equal(block.H, expected_h))

    def test_entropy_weighted_hessian_keeps_shape_and_alignment(self):
        layer = nn.Linear(4, 3, bias=False)
        block = GPTQ(layer)
        inp = torch.randn(2, 3, 4)
        token_weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        block.add_batch(inp, None, token_weights=token_weights)

        x = inp.reshape(-1, inp.shape[-1]).t().float()
        x = math.sqrt(2 / inp.shape[0]) * x
        x = x * torch.sqrt(token_weights.reshape(-1)).unsqueeze(0)
        expected_h = x.matmul(x.t())
        self.assertEqual(tuple(block.H.shape), (4, 4))
        self.assertEqual(token_weights.numel(), inp.shape[0] * inp.shape[1])
        self.assertTrue(torch.equal(block.H, expected_h))

    def test_entropy_normalization_masks_padding(self):
        entropy = torch.tensor(
            [
                [1.0, 2.0, 3.0, float("nan")],
                [4.0, float("nan"), float("nan"), float("nan")],
            ]
        )
        valid_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
            ]
        )

        for norm_mode in ("minmax", "zscore", "mean"):
            with self.subTest(norm_mode=norm_mode):
                token_weights = _normalize_entropy_importance(
                    entropy,
                    valid_mask,
                    alpha=1.0,
                    norm_mode=norm_mode,
                )

                self.assertEqual(tuple(token_weights.shape), tuple(valid_mask.shape))
                self.assertTrue(torch.all(token_weights[~valid_mask] == 0))
                self.assertTrue(torch.all(token_weights[valid_mask] >= 1))
                self.assertEqual(token_weights[0, -1].item(), 1.0)
                self.assertEqual(token_weights[1, 1].item(), 1.0)
                self.assertGreater(token_weights[0, 0].item(), token_weights[1, 0].item())

    def test_fp_entropy_aligns_with_next_token_positions(self):
        model = TinyCausalLM()
        inps = torch.randn(2, 5, 4)
        entropy = _compute_fp_token_entropy(
            model,
            model.model.layers,
            inps,
            layer_kwargs={},
            device=torch.device("cpu"),
            logits_chunk_size=2,
        )

        self.assertEqual(tuple(entropy.shape), (2, 5))
        self.assertTrue(torch.isfinite(entropy[:, :-1]).all())
        self.assertTrue(torch.isnan(entropy[:, -1]).all())

    def test_layer_local_entropy_grad_captures_four_groups(self):
        model = TinyCausalLM(attribution_layers=True)
        inps = torch.randn(3, 5, 4)
        valid_mask = torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, False, False],
                [True, True, True, True, False],
            ]
        )
        original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]

        layer_weights_batch_1 = _compute_layer_local_token_weights(
            model,
            model.model.layers,
            inps,
            layer_kwargs={},
            valid_token_mask=valid_mask,
            device=torch.device("cpu"),
            alpha=1.0,
            mean_normalize=True,
            batch_size=1,
        )
        layer_weights = _compute_layer_local_token_weights(
            model,
            model.model.layers,
            inps,
            layer_kwargs={},
            valid_token_mask=valid_mask,
            device=torch.device("cpu"),
            alpha=1.0,
            mean_normalize=True,
            batch_size=2,
        )

        self.assertEqual(len(layer_weights), 2)
        for weights in layer_weights:
            self.assertEqual(set(weights), set(_LOCAL_IMPORTANCE_GROUPS))
            for weight in weights.values():
                self.assertEqual(tuple(weight.shape), (3, 5))
                self.assertTrue(torch.all(weight[~valid_mask] == 0))
                self.assertAlmostEqual(weight[valid_mask].mean().item(), 1.0, places=6)
                self.assertTrue(torch.isfinite(weight).all())
                self.assertTrue(torch.all(weight >= 0))
        for weights_batch_1, weights_batch_2 in zip(layer_weights_batch_1, layer_weights):
            for group in _LOCAL_IMPORTANCE_GROUPS:
                self.assertTrue(
                    torch.allclose(
                        weights_batch_1[group],
                        weights_batch_2[group],
                        rtol=1e-5,
                        atol=1e-6,
                    )
                )
        self.assertEqual(
            [parameter.requires_grad for parameter in model.parameters()],
            original_requires_grad,
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_layer_local_linear_group_mapping(self):
        expected = {
            "self_attn.q_proj": "qkv",
            "self_attn.k_proj": "qkv",
            "self_attn.v_proj": "qkv",
            "self_attn.o_proj": "o",
            "linear_attn.in_proj_qkv": "qkv",
            "linear_attn.in_proj_z": "qkv",
            "linear_attn.in_proj_b": "qkv",
            "linear_attn.in_proj_a": "qkv",
            "linear_attn.out_proj": "o",
            "mlp.up_proj": "up_gate",
            "mlp.gate_proj": "up_gate",
            "mlp.down_proj": "down",
        }
        for name, group in expected.items():
            self.assertEqual(_local_importance_group_for_linear(name), group)


if __name__ == "__main__":
    unittest.main()
