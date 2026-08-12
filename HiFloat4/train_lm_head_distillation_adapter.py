#!/usr/bin/env python3
"""Distill selected response positions into an lm_head-only LoRA adapter."""

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from HiFloat4.train_end_think_adapter import (
    _dtype_from_name,
    _replace_student_linears_with_activation_quant,
    _single_token_id,
)


START_THINK_TEXT = "<think>"
END_THINK_TEXT = "</think>"

# Deliberately exclude generic words such as "and". They occur too often to
# identify a reasoning transition. Matching is token based and case aware.
LOGICAL_CONNECTIVES = (
    "but",
    "however",
    "although",
    "though",
    "yet",
    "nevertheless",
    "nonetheless",
    "still",
    "on the other hand",
    "in contrast",
    "instead",
    "conversely",
    "because",
    "since",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "as a result",
    "so",
    "due to",
    "moreover",
    "furthermore",
    "additionally",
    "in addition",
    "besides",
    "also",
    "first",
    "firstly",
    "second",
    "secondly",
    "third",
    "thirdly",
    "next",
    "then",
    "finally",
    "subsequently",
    "meanwhile",
    "for example",
    "for instance",
    "specifically",
    "in particular",
    "indeed",
    "namely",
    "in other words",
    "if",
    "otherwise",
    "assuming",
    "suppose",
    "given that",
    "similarly",
    "likewise",
    "overall",
    "in conclusion",
    "to conclude",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-vocabulary soft distillation for lm_head LoRA."
    )
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--teacher_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="simplescaling/s1K-1.1")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logits_chunk_size", type=int, default=256)
    parser.add_argument(
        "--position_scope",
        choices=["all_tokens", "connectives_end", "end_think"],
        required=True,
    )
    parser.add_argument(
        "--adapter_scope",
        choices=["full_lm_head", "end_think_row"],
        required=True,
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--student_fake_act_quant",
        choices=["none", "hif4", "hif4-1"],
        default="none",
    )
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def _connector_token_patterns(tokenizer):
    patterns = set()
    for phrase in LOGICAL_CONNECTIVES:
        variants = {
            phrase,
            phrase.capitalize(),
            " " + phrase,
            " " + phrase.capitalize(),
            "\n" + phrase,
            "\n" + phrase.capitalize(),
        }
        for variant in variants:
            ids = tuple(tokenizer.encode(variant, add_special_tokens=False))
            if ids:
                patterns.add(ids)
    return sorted(patterns, key=lambda ids: (-len(ids), ids))


def _find_pattern_targets(input_ids, patterns, target_start, target_stop):
    """Return target-token indices covered by matched connective token sequences."""
    matched = set()
    for pattern in patterns:
        width = len(pattern)
        last_start = target_stop - width + 1
        for start in range(target_start, last_start + 1):
            if tuple(input_ids[start : start + width]) == pattern:
                matched.update(range(start, start + width))
    return matched


def build_examples(args, tokenizer):
    start_id = _single_token_id(tokenizer, START_THINK_TEXT)
    end_id = _single_token_id(tokenizer, END_THINK_TEXT)
    patterns = _connector_token_patterns(tokenizer)
    dataset = load_dataset(args.dataset, split="train")
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    examples = []
    for source_index in indices:
        row = dataset[source_index]
        question = row.get("question")
        thinking = row.get("deepseek_thinking_trajectory")
        attempt = row.get("deepseek_attempt")
        if not all(isinstance(value, str) and value.strip() for value in (question, thinking, attempt)):
            raise ValueError(f"Dataset sample {source_index} has an invalid required field.")
        messages = [
            {"role": "user", "content": question.strip()},
            {
                "role": "assistant",
                "content": (
                    f"{START_THINK_TEXT}\n{thinking.strip()}\n"
                    f"{END_THINK_TEXT}\n\n{attempt.strip()}"
                ),
            },
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        input_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        if len(input_ids) > args.max_length:
            continue
        start_indices = [i for i, token_id in enumerate(input_ids) if token_id == start_id]
        end_indices = [i for i, token_id in enumerate(input_ids) if token_id == end_id]
        if len(start_indices) != 1 or len(end_indices) != 1:
            raise ValueError(
                f"Dataset sample {source_index} must contain exactly one think pair; "
                f"got starts={start_indices}, ends={end_indices}."
            )
        start_index, end_index = start_indices[0], end_indices[0]
        if start_index == 0 or end_index <= start_index + 1:
            raise ValueError(f"Dataset sample {source_index} has an invalid thinking span.")

        if args.position_scope == "all_tokens":
            # A hidden state at i predicts token i+1. Include every assistant
            # target from <think> through the final assistant token.
            positions = list(range(start_index - 1, len(input_ids) - 1))
        elif args.position_scope == "connectives_end":
            target_indices = _find_pattern_targets(
                input_ids,
                patterns,
                target_start=start_index + 1,
                target_stop=end_index - 1,
            )
            target_indices.add(end_index)
            positions = sorted(target_index - 1 for target_index in target_indices)
        else:
            positions = [end_index - 1]

        if not positions:
            raise RuntimeError(f"Dataset sample {source_index} selected no positions.")
        if args.position_scope != "all_tokens" and input_ids[end_index] != end_id:
            raise RuntimeError("The selected end-think target is incorrect.")
        examples.append(
            {
                "source_index": source_index,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "positions": torch.tensor(positions, dtype=torch.long),
            }
        )
        if len(examples) == args.nsamples:
            break

    if len(examples) != args.nsamples:
        raise ValueError(
            f"Only {len(examples)} complete samples fit max_length={args.max_length}; "
            f"{args.nsamples} are required."
        )
    counts = [example["positions"].numel() for example in examples]
    print(
        f"Prepared {len(examples)} samples for {args.position_scope}: "
        f"positions min={min(counts)} mean={sum(counts) / len(counts):.1f} "
        f"max={max(counts)} total={sum(counts)}."
    )
    return examples, end_id


def _load_model(path, args, student):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=_dtype_from_name(args.dtype),
        device_map={"": args.device},
        attn_implementation=args.attn_implementation,
        trust_remote_code=False,
    )
    if student:
        model = _replace_student_linears_with_activation_quant(
            model, args.student_fake_act_quant
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_hidden_states(args, examples):
    teacher = _load_model(args.teacher_model, args, student=False)
    student = _load_model(args.student_model, args, student=True)
    teacher_head = teacher.get_output_embeddings().weight.detach()
    student_head = student.get_output_embeddings().weight.detach()
    if teacher_head.shape != student_head.shape:
        raise ValueError(
            f"Teacher/student lm_head shapes differ: {teacher_head.shape} vs {student_head.shape}."
        )

    features = []
    with torch.inference_mode():
        for index, example in enumerate(examples):
            input_ids = example["input_ids"].unsqueeze(0).to(args.device)
            positions = example["positions"].to(args.device)
            teacher_output = teacher.model(
                input_ids=input_ids, use_cache=False, return_dict=True
            ).last_hidden_state[0]
            teacher_hidden = teacher_output.index_select(0, positions).to(
                device="cpu", dtype=torch.bfloat16
            )
            del teacher_output
            student_output = student.model(
                input_ids=input_ids, use_cache=False, return_dict=True
            ).last_hidden_state[0]
            student_hidden = student_output.index_select(0, positions).to(
                device="cpu", dtype=torch.bfloat16
            )
            del student_output, input_ids, positions
            features.append((teacher_hidden, student_hidden))
            print(f"Extracted hidden states: {index + 1}/{len(examples)}")

    # Keep only the two frozen output matrices on GPU during LoRA training.
    teacher_head = teacher_head.clone()
    student_head = student_head.clone()
    del teacher, student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features, teacher_head, student_head


def soft_cross_entropy(teacher_logits, student_logits):
    teacher_prob = torch.softmax(teacher_logits.float(), dim=-1).detach()
    student_log_prob = torch.log_softmax(student_logits.float(), dim=-1)
    return -(teacher_prob * student_log_prob).sum(dim=-1)


def _initialize_lora(args, hidden_size, vocab_size, device):
    if args.adapter_scope == "end_think_row":
        if args.position_scope != "end_think":
            raise ValueError("end_think_row requires --position_scope end_think.")
        if args.rank != 1:
            raise ValueError("end_think_row requires --rank 1.")
        lora_a = torch.nn.Parameter(torch.zeros(1, hidden_size, device=device))
        lora_b = None
    else:
        if args.rank < 1:
            raise ValueError("Full lm_head LoRA rank must be positive.")
        lora_a = torch.nn.Parameter(torch.empty(args.rank, hidden_size, device=device))
        lora_b = torch.nn.Parameter(torch.zeros(vocab_size, args.rank, device=device))
        torch.nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
    return lora_a, lora_b


def train_adapter(args, features, teacher_head, student_head, end_think_id):
    device = torch.device(args.device)
    hidden_size = student_head.shape[1]
    vocab_size = student_head.shape[0]
    lora_a, lora_b = _initialize_lora(args, hidden_size, vocab_size, device)
    parameters = [lora_a] if lora_b is None else [lora_a, lora_b]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    total_positions = sum(teacher_hidden.shape[0] for teacher_hidden, _ in features)
    if total_positions == 0:
        raise RuntimeError("No distillation positions were selected.")
    epoch_losses = []
    generator = random.Random(args.seed)

    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(features)))
        generator.shuffle(order)
        loss_sum = 0.0
        for sample_index in order:
            teacher_hidden, student_hidden = features[sample_index]
            for start in range(0, teacher_hidden.shape[0], args.logits_chunk_size):
                end = min(start + args.logits_chunk_size, teacher_hidden.shape[0])
                teacher_chunk = teacher_hidden[start:end].to(device)
                student_chunk = student_hidden[start:end].to(device)
                with torch.no_grad():
                    teacher_logits = F.linear(teacher_chunk, teacher_head)
                    student_base_logits = F.linear(student_chunk, student_head)
                projected = F.linear(student_chunk.float(), lora_a)
                if lora_b is None:
                    correction = torch.zeros(
                        end - start, vocab_size, dtype=torch.float32, device=device
                    )
                    correction[:, end_think_id] = projected[:, 0]
                else:
                    correction = F.linear(projected, lora_b)
                token_loss = soft_cross_entropy(
                    teacher_logits,
                    student_base_logits.float() + correction,
                )
                chunk_loss_sum = token_loss.sum()
                (chunk_loss_sum / total_positions).backward()
                loss_sum += chunk_loss_sum.detach().item()
                del teacher_chunk, student_chunk, teacher_logits, student_base_logits
                del projected, correction, token_loss, chunk_loss_sum
        optimizer.step()
        epoch_loss = loss_sum / total_positions
        epoch_losses.append(epoch_loss)
        grad_norms = [
            parameter.grad.float().norm().item()
            for parameter in parameters
            if parameter.grad is not None
        ]
        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={epoch_loss:.6f}, "
            f"grad_norm={math.sqrt(sum(value * value for value in grad_norms)):.6f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if lora_b is None:
        saved_b = torch.zeros(vocab_size, 1, dtype=torch.bfloat16)
        saved_b[end_think_id, 0] = 1
    else:
        saved_b = lora_b.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    save_file(
        {
            "base_model.model.lm_head.lora_A.weight": lora_a.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous(),
            "base_model.model.lm_head.lora_B.weight": saved_b,
        },
        output_dir / "adapter_model.safetensors",
    )
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": os.path.abspath(args.student_model),
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": args.rank,
        "lora_alpha": args.rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["lm_head"],
    }
    metrics = {
        "position_scope": args.position_scope,
        "adapter_scope": args.adapter_scope,
        "rank": args.rank,
        "student_fake_act_quant": args.student_fake_act_quant,
        "samples": len(features),
        "total_positions": total_positions,
        "epoch_losses": epoch_losses,
        "connectives": list(LOGICAL_CONNECTIVES)
        if args.position_scope == "connectives_end"
        else [],
    }
    with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as handle:
        json.dump(adapter_config, handle, indent=2, ensure_ascii=False)
    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved adapter to {output_dir}.")


def self_test():
    teacher = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    student = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]], requires_grad=True)
    loss = soft_cross_entropy(teacher, student).mean()
    expected = -(
        torch.softmax(teacher, dim=-1) * torch.log_softmax(student, dim=-1)
    ).sum(dim=-1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    if student.grad is None or not torch.isfinite(student.grad).all():
        raise RuntimeError("Soft distillation gradient test failed.")
    print("Self-test passed.")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.nsamples < 1 or args.max_length < 2:
        raise ValueError("nsamples and max_length must be positive.")
    if args.epochs < 1 or args.learning_rate <= 0 or args.logits_chunk_size < 1:
        raise ValueError("epochs, learning_rate and logits_chunk_size must be positive.")
    if Path(args.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, use_fast=False, trust_remote_code=False
    )
    examples, end_think_id = build_examples(args, tokenizer)
    features, teacher_head, student_head = extract_hidden_states(args, examples)
    train_adapter(args, features, teacher_head, student_head, end_think_id)


if __name__ == "__main__":
    main()
