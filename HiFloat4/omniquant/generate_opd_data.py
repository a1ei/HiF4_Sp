"""Generate compact student-forced OPD token corpora with RTN HiF4 W4A16."""

import argparse
import os
import random

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from .modeling import resolve_model_structure
from .quantizer import UniformAffineQuantizer


@torch.no_grad()
def _rtn_hif4_transformer(model, device):
    layers, _, _ = resolve_model_structure(model)
    for layer_index, layer in enumerate(layers):
        layer.to(device)
        for module in layer.modules():
            if not isinstance(module, nn.Linear):
                continue
            quantizer = UniformAffineQuantizer(
                n_bits=4, symmetric=False, dynamic_method="per_channel",
                group_size=None, lwc=False, disable_zero_point=False,
                quant_format="hif4",
            ).to(device)
            module.weight.copy_(quantizer(module.weight))
        layer.cpu()
        torch.cuda.empty_cache()
        print(f"RTN HiF4 layer {layer_index + 1}/{len(layers)}", flush=True)


def _selected_questions(nsamples, seed, tokenizer, seqlen):
    dataset = load_dataset("simplescaling/s1K-1.1", split="train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = []
    for index in indices:
        row = dataset[index]
        calibration_text = "\n\n".join(
            [row["question"], row["deepseek_thinking_trajectory"], row["deepseek_attempt"]]
        )
        if len(tokenizer(calibration_text, add_special_tokens=False).input_ids) < seqlen:
            continue
        selected.append((index, row["question"]))
        if len(selected) == nsamples:
            return selected
    raise RuntimeError(f"s1K produced only {len(selected)} eligible OPD prompts.")


def _merge(paths, output, nsamples, seqlen):
    rows = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        for index, ids, mask in zip(
            payload["indices"], payload["input_ids"], payload["attention_mask"]
        ):
            index = int(index)
            if index in rows:
                raise ValueError(f"Duplicate OPD sample index {index}.")
            rows[index] = (ids, mask)
    if set(rows) != set(range(nsamples)):
        raise ValueError(f"OPD shards contain {len(rows)} samples; expected {nsamples}.")
    input_ids = torch.stack([rows[i][0] for i in range(nsamples)])
    attention_mask = torch.stack([rows[i][1] for i in range(nsamples)]).bool()
    if input_ids.shape != (nsamples, seqlen):
        raise ValueError(f"Merged OPD shape is {tuple(input_ids.shape)}.")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, output)
    print(f"Saved {output}: shape={tuple(input_ids.shape)}, valid={attention_mask.sum().item()}")


def _generate_vllm(args, tokenizer):
    from vllm import LLM, SamplingParams

    selected = _selected_questions(args.nsamples, args.seed, tokenizer, args.seqlen)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        for _, question in selected
    ]
    engine = LLM(
        model=args.model, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, dtype=args.dtype,
        max_model_len=args.max_model_len, trust_remote_code=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.seqlen, seed=args.seed)
    outputs = engine.generate(prompts, params, use_tqdm=True)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((args.nsamples, args.seqlen), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((args.nsamples, args.seqlen), dtype=torch.bool)
    for index, request_output in enumerate(outputs):
        generated = torch.tensor(request_output.outputs[0].token_ids, dtype=torch.long)
        length = min(generated.numel(), args.seqlen)
        input_ids[index, :length] = generated[:length]
        attention_mask[index, :length] = True
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, args.output)
    print(f"Saved {args.output}: shape={tuple(input_ids.shape)}, valid={attention_mask.sum().item()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--merge_shards", nargs="*")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    args = parser.parse_args()
    if args.merge_shards is not None:
        _merge(args.merge_shards, args.output, args.nsamples, args.seqlen)
        return
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must be in [0, num_shards).")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.backend == "vllm":
        _generate_vllm(args, tokenizer)
        return
    config = AutoConfig.from_pretrained(args.model)
    auto_model = AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM
    model = auto_model.from_pretrained(args.model, torch_dtype=dtype, device_map="cpu")
    model.eval().requires_grad_(False)
    _rtn_hif4_transformer(model, device)
    model.to(device)

    shard_rows = []
    for order, (_, question) in enumerate(_selected_questions(args.nsamples, args.seed, tokenizer, args.seqlen)):
        if order % args.num_shards != args.shard_id:
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        generated = model.generate(
            **encoded, max_new_tokens=args.seqlen, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )[0, encoded.input_ids.shape[1] :].cpu()
        length = min(generated.numel(), args.seqlen)
        ids = torch.full((args.seqlen,), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros(args.seqlen, dtype=torch.bool)
        ids[:length] = generated[:length]
        mask[:length] = True
        shard_rows.append((order, ids, mask))
        print(f"shard={args.shard_id} sample={order} generated={length}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "indices": [row[0] for row in shard_rows],
            "input_ids": torch.stack([row[1] for row in shard_rows]),
            "attention_mask": torch.stack([row[2] for row in shard_rows]),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
