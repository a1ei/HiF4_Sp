"""Calibration data loaders."""

import random

from datasets import load_dataset


def _slice_ids(input_ids, seqlen, slice_mode, slice_offset, rng):
    token_count = input_ids.shape[1]
    required = slice_offset + seqlen if slice_mode == "offset" else seqlen
    if token_count < required:
        return None
    if slice_mode == "head":
        start = 0
    elif slice_mode == "tail":
        start = token_count - seqlen
    elif slice_mode == "offset":
        start = slice_offset
    elif slice_mode == "random":
        start = rng.randint(0, token_count - seqlen)
    else:
        raise ValueError(f"Unsupported calibration slice mode: {slice_mode}")
    return input_ids[:, start : start + seqlen]


def get_s1k(nsamples, seed, seqlen, tokenizer, slice_mode="random", slice_offset=0):
    dataset = load_dataset("simplescaling/s1K-1.1", split="train")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    samples = []
    for index in indices:
        row = dataset[index]
        text = "\n\n".join(
            [row["question"], row["deepseek_thinking_trajectory"], row["deepseek_attempt"]]
        )
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        sample = _slice_ids(input_ids, seqlen, slice_mode, slice_offset, rng)
        if sample is None:
            continue
        samples.append(sample)
        if len(samples) == nsamples:
            return samples
    raise RuntimeError(f"s1K-1.1 produced only {len(samples)} eligible calibration samples.")


def get_loaders(name, nsamples, seed, seqlen, tokenizer, slice_mode="random", slice_offset=0):
    if name != "s1k-1.1":
        raise NotImplementedError("The first release only supports s1k-1.1 calibration.")
    if slice_offset < 0:
        raise ValueError("Calibration slice offset must be non-negative.")
    if slice_mode != "offset" and slice_offset != 0:
        raise ValueError("Calibration slice offset is only valid in offset mode.")
    return get_s1k(nsamples, seed, seqlen, tokenizer, slice_mode, slice_offset)
