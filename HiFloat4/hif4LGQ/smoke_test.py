from __future__ import annotations

import io
import logging
import pathlib
import sys

import torch
import torch.nn as nn

HIFLOAT4_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(HIFLOAT4_ROOT) not in sys.path:
    sys.path.insert(0, str(HIFLOAT4_ROOT))

from hif4LGQ.calibration import ReasoningCalibrationSample, truncate_input_ids
from hif4LGQ.logic_tokens import LogicTokenMatcher
from hif4LGQ.optimizer import (
    ActivationStatistics,
    build_logic_gram,
    direct_logic_loss,
    gram_logic_loss,
    group_amax,
    optimize_linear_group,
    optimize_linear_weight,
    resolve_group_size,
    select_spectral_subspace,
)
from hif4LGQ.runner import (
    _capture_first_layer_inputs,
    _collect_layer_statistics,
    _forward_updated_layer,
    _layer_kwargs,
    _linear_input_groups,
)


class _ToyTokenizer:
    _vocabulary = {
        "therefore": [7],
        "as a result": [10, 11, 12],
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(self._vocabulary.get(text.strip().lower(), []))


def _test_sequences_and_logic_matching() -> None:
    token_ids = torch.arange(9)
    assert torch.equal(truncate_input_ids(token_ids, 0), token_ids)
    assert torch.equal(truncate_input_ids(token_ids, 4), token_ids[:4])

    matcher = LogicTokenMatcher(
        _ToyTokenizer(),
        ["therefore", "as a result"],
    )
    mask = matcher.match(torch.tensor([1, 10, 11, 12, 2, 7, 3]))
    expected = torch.tensor([False, True, True, True, False, True, False])
    assert torch.equal(mask, expected)


def _test_hessian_and_logic_loss() -> None:
    inputs = torch.tensor(
        [
            [
                [1.0, 0.0, 2.0],
                [0.5, 1.0, -1.0],
                [2.0, 1.0, 0.0],
                [-1.0, 2.0, 1.0],
                [1.5, -0.5, 0.5],
                [100.0, 100.0, 100.0],
            ]
        ]
    )
    valid_mask = torch.tensor([True, True, True, True, True, False])
    logic_mask = torch.tensor([False, True, False, True, False, True])
    statistics = ActivationStatistics()
    statistics.add(inputs, valid_mask, logic_mask)

    expected_valid = inputs[0, :5]
    expected_logic = inputs[0, [1, 3]]
    assert statistics.token_count == 5
    assert statistics.logic_token_count == 2
    assert torch.allclose(
        statistics.hessian,
        expected_valid.T @ expected_valid,
    )
    assert torch.equal(statistics.logic_chunks[0], expected_logic)

    raw_basis = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    basis = torch.linalg.qr(raw_basis.T).Q.T
    coefficients = torch.tensor(
        [
            [0.2, -0.1],
            [0.3, 0.4],
        ]
    )
    delta_weight = coefficients @ basis
    logic_gram = build_logic_gram(
        statistics.logic_chunks,
        basis,
        statistics.logic_token_count,
    )
    direct = direct_logic_loss(expected_logic, delta_weight)
    gram = gram_logic_loss(coefficients, logic_gram)
    assert torch.allclose(direct, gram, atol=1e-6, rtol=1e-6)


def _test_group_formats_and_adam_merge() -> None:
    assert resolve_group_size("hif4", 0) == 64
    assert resolve_group_size("hif4-1", 0) == 64
    assert resolve_group_size("nvfp4", 0) == 16

    weight = torch.tensor(
        [
            [3.0, 1.0, 2.0, 1.0],
            [4.0, 1.0, 5.0, 1.0],
        ]
    )
    assert group_amax(weight, group_size=2).shape == (2, 2)

    linear = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    original_weight = linear.weight.detach().clone()
    hessian = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    basis, spectrum = select_spectral_subspace(
        hessian,
        rank=2,
        mode="low",
        low_mid_start_quantile=0.1,
    )
    original_basis = basis.clone()
    logic_inputs = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    logic_gram = build_logic_gram([logic_inputs], basis, 1)
    log_stream = io.StringIO()
    logger = logging.getLogger("hif4lgq_smoke")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(log_stream))
    stats, delta = optimize_linear_weight(
        layer_name="toy",
        linear=linear,
        basis=basis,
        logic_gram=logic_gram,
        spectrum=spectrum,
        hessian_tokens=4,
        logic_tokens=1,
        group_size=2,
        steps=5,
        learning_rate=1e-2,
        lambda_logic=1.0,
        lambda_reg=1e-4,
        group_loss_mode="max",
        group_smooth_tau=1e-3,
        return_delta=True,
        log_interval=2,
        logger=logger,
    )

    assert linear.weight.grad is None
    assert basis.grad is None
    assert torch.equal(basis, original_basis)
    assert delta is not None
    assert torch.allclose(linear.weight - original_weight, delta)
    assert not torch.equal(linear.weight, original_weight)
    assert stats.hessian_tokens == 4
    assert stats.logic_tokens == 1
    assert torch.isfinite(torch.tensor(stats.final_total_loss))
    progress_log = log_stream.getvalue()
    assert "step=0/5" in progress_log
    assert "step=2/5" in progress_log
    assert "step=4/5" in progress_log
    assert "step=5/5" in progress_log

    joint_a = nn.Linear(4, 2, bias=False)
    joint_b = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        joint_a.weight.copy_(weight)
        joint_b.weight.copy_(
            torch.tensor(
                [
                    [2.0, 1.0, 3.0, 1.0],
                    [3.0, 1.0, 4.0, 1.0],
                    [4.0, 1.0, 5.0, 1.0],
                ]
            )
        )
    original_a = joint_a.weight.detach().clone()
    original_b = joint_b.weight.detach().clone()
    joint_log_stream = io.StringIO()
    joint_logger = logging.getLogger("hif4lgq_joint_smoke")
    joint_logger.handlers.clear()
    joint_logger.setLevel(logging.INFO)
    joint_logger.propagate = False
    joint_logger.addHandler(logging.StreamHandler(joint_log_stream))
    joint_results = optimize_linear_group(
        linears=[("joint.a", joint_a), ("joint.b", joint_b)],
        basis=basis,
        logic_gram=logic_gram,
        spectrum=spectrum,
        hessian_tokens=4,
        logic_tokens=1,
        group_size=2,
        steps=3,
        learning_rate=1e-2,
        lambda_logic=1.0,
        lambda_reg=1e-4,
        group_loss_mode="max",
        group_smooth_tau=1e-3,
        return_delta=False,
        log_interval=2,
        logger=joint_logger,
    )
    assert len(joint_results) == 2
    assert not torch.equal(joint_a.weight, original_a)
    assert not torch.equal(joint_b.weight, original_b)
    joint_log = joint_log_stream.getvalue()
    assert "(hif4LGQ/train-group) [joint.a,joint.b] step=2/3" in joint_log
    assert "joint.a step=2/3" in joint_log
    assert "joint.b step=2/3" in joint_log


def _assert_transformers_model_pipeline(model) -> None:
    samples = [
        ReasoningCalibrationSample(
            input_ids=torch.tensor([1, 3, 4, 5, 2]),
            logic_mask=torch.tensor([False, True, False, False, False]),
            source_index=0,
        ),
        ReasoningCalibrationSample(
            input_ids=torch.tensor([1, 6, 7, 8, 9, 10, 2]),
            logic_mask=torch.tensor(
                [False, False, True, True, False, False, False]
            ),
            source_index=1,
        ),
    ]
    hidden_states, layer_kwargs = _capture_first_layer_inputs(
        model,
        samples,
        torch.device("cpu"),
    )
    assert [hidden.shape[1] for hidden in hidden_states] == [5, 7]

    layer = model.model.layers[0]
    groups = _linear_input_groups(
        layer,
        layer_index=0,
        target_patterns=("*",),
        excluded_layers=[],
    )
    statistics = _collect_layer_statistics(
        model,
        layer,
        hidden_states,
        layer_kwargs,
        samples,
        groups,
        torch.device("cpu"),
    )
    assert groups
    assert all(item.token_count == 12 for item in statistics)
    assert all(item.logic_token_count == 3 for item in statistics)

    outputs = _forward_updated_layer(
        model,
        layer,
        hidden_states,
        layer_kwargs,
        torch.device("cpu"),
    )
    assert [output.shape[1] for output in outputs] == [5, 7]


def _test_transformers_layer_pipeline() -> None:
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )

    qwen3_config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
    )
    _assert_transformers_model_pipeline(Qwen3ForCausalLM(qwen3_config).eval())

    for layer_type in ("full_attention", "linear_attention"):
        qwen3_5_config = Qwen3_5TextConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=[layer_type],
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
        )
        model = AutoModelForCausalLM.from_config(qwen3_5_config).eval()
        _assert_transformers_model_pipeline(model)


def _test_qwen3_5_flash_attention_mask() -> None:
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )

    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["full_attention"],
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    hidden_states = torch.randn(1, 7, 8)
    base_mask = torch.ones(1, 7, dtype=torch.long)

    model.config._attn_implementation = "flash_attention_2"
    flash_kwargs = _layer_kwargs(
        model,
        model.model.layers[0],
        hidden_states,
        {"attention_mask": base_mask},
    )
    assert flash_kwargs["attention_mask"] is None

    model.config._attn_implementation = "eager"
    eager_kwargs = _layer_kwargs(
        model,
        model.model.layers[0],
        hidden_states,
        {"attention_mask": base_mask},
    )
    assert eager_kwargs["attention_mask"].shape == (1, 1, 7, 7)


def _test_default_guard() -> None:
    class EmptyArgs:
        pass

    assert not bool(getattr(EmptyArgs(), "hif4lgq", False))


def main() -> None:
    torch.manual_seed(0)
    _test_sequences_and_logic_matching()
    _test_hessian_and_logic_loss()
    _test_group_formats_and_adam_merge()
    _test_transformers_layer_pipeline()
    _test_qwen3_5_flash_attention_mask()
    _test_default_guard()
    print("hif4LGQ CPU smoke test passed.")


if __name__ == "__main__":
    main()
