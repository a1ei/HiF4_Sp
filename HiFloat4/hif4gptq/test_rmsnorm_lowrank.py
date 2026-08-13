import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
GPTQ_ROOT = ROOT / "hif4gptq"
for path in (ROOT, GPTQ_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rmsnorm_lowrank import (
    LowRankConfig,
    LowRankRMSNorm,
    _load_curvature_matrix_checkpoint,
    _register_activation_quant_ste_pre_hooks,
    _save_curvature_matrix_checkpoint,
    attach_lowrank_adapters,
    load_curvature_checkpoint,
    load_sidecar,
    reasoning_nll,
    save_curvature_checkpoint,
    save_sidecar,
    select_topk_gradients,
)


class DummyNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * self.weight


class DummyLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.input_layernorm = DummyNorm(hidden_size)
        self.post_attention_layernorm = DummyNorm(hidden_size)


class DummyModel(nn.Module):
    def __init__(self, hidden_size=8, layers=2):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_5_text",
            hidden_size=hidden_size,
            num_hidden_layers=layers,
        )
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList([DummyLayer(hidden_size) for _ in range(layers)])


class RMSNormLowRankTest(unittest.TestCase):
    def test_reasoning_nll_shift(self):
        logits = torch.full((1, 4, 5), -10.0)
        ids = torch.tensor([[0, 1, 2, 3]])
        mask = torch.tensor([[False, False, True, True]])
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0
        self.assertLess(float(reasoning_nll(logits, ids, mask)), 1e-5)

    def test_identity_initialization_and_projection(self):
        torch.manual_seed(0)
        hidden, sensitive_rank = 8, 3
        s, _ = torch.linalg.qr(torch.randn(hidden, sensitive_rank))
        config = LowRankConfig(rank=2, sensitive_rank=sensitive_rank)
        wrapper = LowRankRMSNorm(DummyNorm(hidden), config, s)
        x = torch.randn(2, 4, hidden)
        original = wrapper.norm(x)
        self.assertTrue(torch.equal(wrapper(x), original))
        wrapper.U.data.normal_()
        adapted, delta = wrapper.transform(original)
        self.assertEqual(adapted.shape, original.shape)
        leakage = torch.linalg.vector_norm(delta @ s)
        self.assertLess(float(leakage.detach()), 1e-5)

    def test_topk_keeps_original_vectors(self):
        gradient = torch.tensor([[[1.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 0.5]]])
        mask = torch.tensor([[False, True, True, True]])
        selected, valid_norms, selected_norms = select_topk_gradients(gradient, mask, 2)
        self.assertEqual(tuple(selected.shape), (2, 2))
        self.assertEqual(valid_norms.numel(), 3)
        self.assertEqual(set(selected_norms.tolist()), {2.0, 3.0})

    def test_a4_ste_keeps_lowrank_parameters_connected(self):
        torch.manual_seed(0)
        hidden = 64
        sensitive_rank = 3
        s, _ = torch.linalg.qr(torch.randn(hidden, sensitive_rank))
        wrapper = LowRankRMSNorm(
            DummyNorm(hidden),
            LowRankConfig(rank=2, sensitive_rank=sensitive_rank),
            s,
        )
        linear = nn.Linear(hidden, hidden, bias=False)
        for parameter in linear.parameters():
            parameter.requires_grad_(False)
        block = nn.Sequential(wrapper, linear)
        from hif4_gpu.quant_cy import QType

        handles = _register_activation_quant_ste_pre_hooks(
            block, QType("hifx4").dim(-1)
        )
        try:
            block(torch.randn(2, 4, hidden)).float().square().mean().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertIsNotNone(wrapper.U.grad)
        self.assertIsNotNone(wrapper.V.grad)
        self.assertTrue(torch.isfinite(wrapper.U.grad).all())
        self.assertTrue(torch.isfinite(wrapper.V.grad).all())

    def test_sidecar_roundtrip(self):
        config = LowRankConfig(rank=2, sensitive_rank=3)
        model = DummyModel()
        subspaces = {}
        for layer_idx in range(2):
            for name in ("input_layernorm", "post_attention_layernorm"):
                q, _ = torch.linalg.qr(torch.randn(8, 3))
                subspaces[f"layers.{layer_idx}.{name}"] = q
        wrappers = attach_lowrank_adapters(model, config, subspaces)
        for wrapper in wrappers.values():
            wrapper.U.data.normal_()
        with tempfile.TemporaryDirectory() as directory:
            path = str(pathlib.Path(directory) / "adapter.pt")
            save_sidecar(path, model, config, wrappers, {}, [])
            restored_model = DummyModel()
            restored, payload = load_sidecar(restored_model, path)
        self.assertEqual(payload["format"], "reasoning_safe_rmsnorm_lowrank_v1")
        for name in wrappers:
            self.assertTrue(torch.equal(wrappers[name].U, restored[name].U))
            self.assertTrue(torch.equal(wrappers[name].V, restored[name].V))
            self.assertTrue(torch.equal(wrappers[name].S, restored[name].S))

    def test_curvature_checkpoint_roundtrip(self):
        config = LowRankConfig(rank=2, sensitive_rank=3)
        model = DummyModel()
        subspaces = {}
        diagnostics = {}
        for layer_idx in range(2):
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                name = f"layers.{layer_idx}.{norm_name}"
                subspaces[name], _ = torch.linalg.qr(torch.randn(8, 3))
                diagnostics[name] = {"top_eigenvalues": [3.0, 2.0, 1.0]}
        calibration = {"model": "dummy", "nsamples": 2, "seqlen": 16}
        with tempfile.TemporaryDirectory() as directory:
            path = str(pathlib.Path(directory) / "curvature.pt")
            save_curvature_checkpoint(
                path, model, config, subspaces, diagnostics, calibration
            )
            restored, restored_diagnostics = load_curvature_checkpoint(
                path, model, config, calibration
            )
        self.assertEqual(set(restored), set(subspaces))
        self.assertEqual(restored_diagnostics, diagnostics)
        for name in subspaces:
            self.assertTrue(torch.equal(restored[name], subspaces[name]))

    def test_raw_curvature_matrix_checkpoint_roundtrip(self):
        model = DummyModel()
        config = LowRankConfig(curvature_tokens_per_sample=2)
        names = [
            f"layers.{layer_idx}.{norm_name}"
            for layer_idx in range(2)
            for norm_name in ("input_layernorm", "post_attention_layernorm")
        ]
        owners = {name: 0 for name in names}
        curvature = {name: torch.randn(8, 8).float() for name in names}
        stats = {
            name: torch.tensor([4, 2, 3, 2, 2, 1], dtype=torch.float64)
            for name in names
        }
        calibration = {"model": "dummy", "nsamples": 2}
        with tempfile.TemporaryDirectory() as directory:
            path = str(pathlib.Path(directory) / "raw-c")
            _save_curvature_matrix_checkpoint(
                path, model, config, calibration, owners, curvature, stats, 0, 1
            )
            restored_c, restored_stats = _load_curvature_matrix_checkpoint(
                path, model, config, calibration, owners, 0, 1
            )
        for name in names:
            self.assertTrue(torch.equal(restored_c[name], curvature[name]))
            self.assertTrue(torch.equal(restored_stats[name], stats[name]))


if __name__ == "__main__":
    unittest.main()
