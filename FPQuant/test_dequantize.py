import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from FPQuant.dequantize import (
    convert_fpquant_nvfp4_checkpoint,
    dequantize_fpquant_nvfp4_weight,
)


class FPQuantDequantizeTest(unittest.TestCase):
    def test_dequantizes_and_folds_hadamard(self):
        nibbles = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
            dtype=torch.uint8,
        )
        qweight = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
        scale_bits = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn).view(torch.uint8)
        global_scale = torch.tensor([4.0], dtype=torch.bfloat16)
        hadamard = torch.eye(16, dtype=torch.bfloat16)

        result = dequantize_fpquant_nvfp4_weight(
            qweight,
            scale_bits,
            global_scale,
            hadamard,
            output_dtype=torch.float32,
        )
        expected = torch.tensor(
            [[0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 0, -0.25, -0.5, -0.75, -1, -1.5, -2, -3]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result, expected)

    def test_uses_transpose_to_fold_input_transform(self):
        nibbles = torch.tensor([[2, 4] + [2] * 14], dtype=torch.uint8)
        qweight = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
        scale_bits = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn).view(torch.uint8)
        global_scale = torch.tensor([1.0])
        hadamard = torch.eye(16)
        hadamard[[0, 1]] = hadamard[[1, 0]]

        result = dequantize_fpquant_nvfp4_weight(
            qweight,
            scale_bits,
            global_scale,
            hadamard,
            output_dtype=torch.bfloat16,
        )
        self.assertEqual(result.dtype, torch.bfloat16)
        expected = torch.tensor([[2.0, 1.0] + [1.0] * 14])
        torch.testing.assert_close(result.float(), expected)

    def test_converts_checkpoint_to_standard_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            config = {
                "model_type": "qwen3",
                "dtype": "bfloat16",
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "quantization_config": {
                    "quant_method": "fp_quant",
                    "forward_dtype": "nvfp4",
                    "store_master_weights": False,
                },
            }
            (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
            tokenizer_config = {
                "added_tokens_decoder": {
                    "151643": {"content": "<|endoftext|>"},
                    "151645": {"content": "<|im_end|>"},
                },
                "pad_token": "<|im_end|>",
            }
            (source / "tokenizer_config.json").write_text(
                json.dumps(tokenizer_config), encoding="utf-8"
            )
            (source / "special_tokens_map.json").write_text(
                json.dumps(
                    {
                        "eos_token": {"content": "<|im_end|>"},
                        "pad_token": {"content": "<|im_end|>"},
                    }
                ),
                encoding="utf-8",
            )
            (source / "generation_config.json").write_text(
                json.dumps({"bos_token_id": 151643, "eos_token_id": 151645}),
                encoding="utf-8",
            )

            base = "model.layers.0.test_proj."
            nibbles = torch.full((1, 16), 2, dtype=torch.uint8)
            tensors = {
                base + "qweight": nibbles[:, 0::2] | (nibbles[:, 1::2] << 4),
                base + "scales": torch.tensor(
                    [[1.0]], dtype=torch.float8_e4m3fn
                ).view(torch.uint8),
                base + "weight_global_scale": torch.tensor([1.0]),
                base + "act_global_scale": torch.tensor([1.0]),
                base + "forward_hadamard_matrix": torch.eye(16),
                base + "backward_hadamard_matrix": torch.eye(16),
                "model.norm.weight": torch.ones(16, dtype=torch.bfloat16),
            }
            shard_name = "model-00001-of-00001.safetensors"
            save_file(tensors, source / shard_name)
            index = {
                "metadata": {},
                "weight_map": {key: shard_name for key in tensors},
            }
            (source / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            convert_fpquant_nvfp4_checkpoint(
                source,
                output,
                output_dtype="bfloat16",
                max_shard_size=1024**2,
            )

            output_index = json.loads(
                (output / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            weight_key = base + "weight"
            self.assertIn(weight_key, output_index["weight_map"])
            self.assertNotIn(base + "qweight", output_index["weight_map"])
            converted = load_file(output / output_index["weight_map"][weight_key])
            torch.testing.assert_close(
                converted[weight_key].float(), torch.ones((1, 16))
            )
            output_config = json.loads(
                (output / "config.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("quantization_config", output_config)
            self.assertTrue(output_config["dequantization_config"]["hadamard_folded"])
            output_generation_config = json.loads(
                (output / "generation_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                output_generation_config["eos_token_id"], [151645, 151643]
            )
            self.assertEqual(output_generation_config["pad_token_id"], 151643)
            output_tokenizer_config = json.loads(
                (output / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                output_tokenizer_config["pad_token"], "<|endoftext|>"
            )
            output_special_tokens = json.loads(
                (output / "special_tokens_map.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                output_special_tokens["pad_token"]["content"], "<|endoftext|>"
            )
            self.assertEqual(
                (output / "chat_template.jinja").read_text(encoding="utf-8"),
                (Path(__file__).with_name("qwen3_chat_template.jinja")).read_text(
                    encoding="utf-8"
                ),
            )


if __name__ == "__main__":
    unittest.main()
