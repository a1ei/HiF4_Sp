import copy
import logging
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from HiFloat4.lfq import backward_lfq_chunks, lfq_loss
from HiFloat4.omniquant.calibration import _weighted_mse, loss_mode, omniquant
from HiFloat4.omniquant.datautils import _slice_ids
from HiFloat4.omniquant.quantizer import UniformAffineQuantizer
from HiFloat4.hif4_gpu.quant_cy import QType, quant_dequant_float


class TestLFQ(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "HiF4 requires CUDA")
    def test_hif4_weight_quantizer_matches_native_and_has_lwc_gradient(self):
        torch.manual_seed(3)
        weight = torch.randn(4, 16, device="cuda", dtype=torch.float32)
        quantizer = UniformAffineQuantizer(
            n_bits=4, quant_format="hif4", lwc=False
        ).cuda()
        actual = quantizer(weight)
        expected = quant_dequant_float(
            weight.contiguous(), QType("hifx4").dim(-1), force_fp32=True
        )
        self.assertTrue(torch.equal(actual, expected))

        lwc_quantizer = UniformAffineQuantizer(
            n_bits=4, quant_format="hif4", lwc=True, shape=weight.shape
        ).cuda()
        lwc_quantizer(weight).square().mean().backward()
        for factor in (lwc_quantizer.upbound_factor, lwc_quantizer.lowbound_factor):
            self.assertIsNotNone(factor.grad)
            self.assertTrue(torch.isfinite(factor.grad).all())
            self.assertGreater(float(factor.grad.abs().sum().item()), 0.0)

    def test_entropy_weighted_mse(self):
        target = torch.zeros(1, 2, 2)
        output = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]], requires_grad=True)
        weights = torch.tensor([[3.0, 1.0]])
        unweighted = _weighted_mse(target, output)
        self.assertTrue(torch.equal(unweighted, nn.functional.mse_loss(output, target)))
        weighted = _weighted_mse(target, output, weights)
        self.assertTrue(torch.allclose(weighted, torch.tensor(3.0)))
        weighted.backward()
        self.assertIsNotNone(output.grad)

    def test_calibration_head_slice(self):
        input_ids = torch.arange(12).reshape(1, 12)
        sliced = _slice_ids(input_ids, 5, "head", 0, None)
        self.assertTrue(torch.equal(sliced, input_ids[:, :5]))

    def test_full_distribution_mask_and_chunking(self):
        torch.manual_seed(0)
        teacher = torch.randn(2, 5, 8)
        student = torch.randn(2, 5, 8, requires_grad=True)
        norm = nn.LayerNorm(8).requires_grad_(False)
        head = nn.Linear(8, 13, bias=False).requires_grad_(False)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        loss, metrics = lfq_loss(teacher, student, norm, head, mask, chunk_size=2)
        with torch.no_grad():
            teacher_prob = torch.softmax(head(norm(teacher)).float(), dim=-1)
        student_log_prob = torch.log_softmax(head(norm(student)).float(), dim=-1)
        reference = (-(teacher_prob * student_log_prob).sum(dim=-1) * mask).sum() / mask.sum()
        self.assertTrue(torch.allclose(loss, reference, atol=1e-6, rtol=1e-6))
        self.assertEqual(metrics.valid_tokens, 8)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(norm.weight.grad)
        self.assertIsNone(head.weight.grad)

    def test_chunked_gradient_matches_full_logits(self):
        torch.manual_seed(1)
        teacher = torch.randn(1, 7, 6)
        first = torch.randn(1, 7, 6, requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        norm = nn.LayerNorm(6).requires_grad_(False)
        head = nn.Linear(6, 11, bias=False).requires_grad_(False)
        mask = torch.ones(1, 7, dtype=torch.bool)
        backward_lfq_chunks(teacher, first, norm, head, mask, chunk_size=3)
        full, _ = lfq_loss(teacher, second, norm, head, mask, chunk_size=7)
        full.backward()
        self.assertTrue(torch.allclose(first.grad, second.grad, atol=1e-6, rtol=1e-5))


class TinyLM:
    def __init__(self, model, device):
        self.model = model
        self.device = device


def tiny_model():
    config = Qwen3_5TextConfig(
        vocab_size=37, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, linear_conv_kernel_dim=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"], use_cache=False,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
        patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
        out_hidden_size=32, num_position_embeddings=16,
    )
    wrapper_config = Qwen3_5Config(
        text_config=config, vision_config=vision_config,
        image_token_id=35, video_token_id=36,
    )
    return Qwen3_5ForConditionalGeneration(wrapper_config).eval()


def args_for(output_dir, lfq):
    return SimpleNamespace(
        nsamples=2, seqlen=8, batch_size=1, epochs=1, lfq=lfq,
        lfq_logits_chunk_size=3, let=True, lwc=True, let_lr=5e-3,
        lwc_lr=1e-2, lfq_lr=2e-3, wd=0.0, alpha=0.5, aug_loss=False, resume=None,
        output_dir=output_dir, abits=16, token_importance="none",
        entropy_alpha=1.0, entropy_norm="minmax", importance_alpha=1.0,
        importance_mean_normalize=True, importance_batch_size=1,
        weight_quant_params={"n_bits": 4, "symmetric": False,
            "dynamic_method": "per_channel", "group_size": None,
            "lwc": True, "disable_zero_point": False,
            "quant_format": "hif4" if lfq else "int4"},
        act_quant_params={"n_bits": 16, "symmetric": False,
            "dynamic_method": "per_token", "group_size": None,
            "lwc": False, "disable_zero_point": False},
    )


@unittest.skipUnless(torch.cuda.is_available(), "OmniQuant calibration requires CUDA")
class TestTinyQwenOmniQuant(unittest.TestCase):
    def _run(self, use_lfq, token_importance="none"):
        torch.manual_seed(2)
        model = tiny_model().to(dtype=torch.bfloat16)
        original = copy.deepcopy(model.state_dict())
        samples = [torch.randint(0, model.config.text_config.vocab_size, (1, 8)) for _ in range(2)]
        with tempfile.TemporaryDirectory() as output_dir:
            calibration_args = args_for(output_dir, use_lfq)
            calibration_args.token_importance = token_importance
            calibrated = omniquant(TinyLM(model, torch.device("cuda")), calibration_args, samples, logging.getLogger("tiny-omniquant"))
            checkpoint = torch.load(os.path.join(output_dir, "omni_parameters.pth"), map_location="cpu")
            self.assertEqual(set(checkpoint), {0, 1})
            self.assertTrue(any("bound_factor" in key for key in checkpoint[1]))
            self.assertTrue(any("smooth_scale" in key for key in checkpoint[1]))
            reloaded = tiny_model().to(dtype=torch.bfloat16)
            reloaded.load_state_dict(calibrated.state_dict(), strict=True)
            with torch.no_grad():
                logits = reloaded(samples[0]).logits
            self.assertEqual(logits.shape, (1, 8, model.config.text_config.vocab_size))
            self.assertTrue(torch.isfinite(logits).all())
            changed = any(not torch.equal(value.cpu(), calibrated.state_dict()[name].cpu()) for name, value in original.items() if name in calibrated.state_dict() and name.endswith("weight"))
            self.assertTrue(changed)

            resume_model = tiny_model().to(dtype=torch.bfloat16)
            resume_args = args_for(output_dir, use_lfq)
            resume_args.epochs = 0
            resume_args.resume = os.path.join(output_dir, "omni_parameters.pth")
            resumed = omniquant(
                TinyLM(resume_model, torch.device("cuda")), resume_args, samples,
                logging.getLogger("tiny-omniquant-resume"),
            )
            strict_reload = tiny_model().to(dtype=torch.bfloat16)
            strict_reload.load_state_dict(resumed.state_dict(), strict=True)

    def test_baseline_routes_all_layers_to_mse(self):
        self.assertEqual([loss_mode(i, 2, False) for i in range(2)], ["mse", "mse"])
        self._run(False)

    def test_entropy_modes_run_end_to_end(self):
        self._run(False, "entropy")
        self._run(False, "entropy_grad")

    def test_lfq_routes_only_last_layer(self):
        self.assertEqual([loss_mode(i, 2, True) for i in range(2)], ["mse", "lfq"])
        self._run(True)


if __name__ == "__main__":
    unittest.main()
