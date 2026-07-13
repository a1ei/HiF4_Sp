import random

import datasets
import torch
import transformers
from datasets import load_dataset


def _get_tokenizer(model_name, hf_token=None):
    if hf_token is None:
        return transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=hf_token)


def get_wikitext2(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    tokenizer = _get_tokenizer(model, hf_token)
    split = "test" if eval_mode else "train"
    data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(data["text"])
    encoded = tokenizer(text, return_tensors="pt")

    if eval_mode:
        return encoded

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = encoded.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_ptb(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    tokenizer = _get_tokenizer(model, hf_token)
    split = "test" if eval_mode else "train"
    data = datasets.load_dataset("ptb_text_only", "penn_treebank", split=split)
    text = " ".join(data["sentence"])
    encoded = tokenizer(text, return_tensors="pt")

    if eval_mode:
        return encoded

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = encoded.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_c4(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    tokenizer = _get_tokenizer(model, hf_token)

    if eval_mode:
        val_data = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        encoded = tokenizer(" ".join(val_data[:1100]["text"]), return_tensors="pt")
        encoded.input_ids = encoded.input_ids[:, : 256 * seqlen]
        return encoded

    train_data = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(train_data) - 1)
            encoded = tokenizer(train_data[idx]["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = encoded.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_s1k_1_1(
    nsamples,
    seed,
    seqlen,
    model,
    hf_token=None,
    eval_mode=False,
    slice_mode="random",
    slice_offset=0,
):
    if eval_mode:
        raise ValueError("s1k-1.1 only supports calibration mode; it has no evaluation split here.")
    if nsamples <= 0:
        raise ValueError("cal_nsamples must be greater than 0.")
    if seqlen <= 0:
        raise ValueError("cal_seqlen must be greater than 0.")
    if slice_mode not in {"random", "head", "tail", "offset"}:
        raise ValueError(f"Unsupported calibration slice mode: {slice_mode}")
    if slice_offset < 0:
        raise ValueError("cal_slice_offset must be greater than or equal to 0.")
    if slice_mode != "offset" and slice_offset != 0:
        raise ValueError("cal_slice_offset can only be non-zero when cal_slice_mode=offset.")

    tokenizer = _get_tokenizer(model, hf_token)
    train_data = load_dataset("simplescaling/s1K-1.1", split="train")
    rng = random.Random(seed)
    indices = list(range(len(train_data)))
    rng.shuffle(indices)

    trainloader = []
    for idx in indices:
        question = train_data[idx].get("question")
        thinking_trajectory = train_data[idx].get("deepseek_thinking_trajectory")
        attempt = train_data[idx].get("deepseek_attempt")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"s1k-1.1 sample {idx} has an empty or invalid question field.")
        if not isinstance(thinking_trajectory, str) or not thinking_trajectory.strip():
            raise ValueError(
                f"s1k-1.1 sample {idx} has an empty or invalid deepseek_thinking_trajectory field."
            )
        if not isinstance(attempt, str) or not attempt.strip():
            raise ValueError(f"s1k-1.1 sample {idx} has an empty or invalid deepseek_attempt field.")

        calibration_text = f"{question}\n\n{thinking_trajectory}\n\n{attempt}"
        input_ids = tokenizer(
            calibration_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        token_count = input_ids.shape[1]
        required_tokens = slice_offset + seqlen if slice_mode == "offset" else seqlen
        if token_count < required_tokens:
            continue

        if slice_mode == "head":
            start = 0
        elif slice_mode == "tail":
            start = token_count - seqlen
        elif slice_mode == "offset":
            start = slice_offset
        else:
            start = rng.randint(0, token_count - seqlen)

        inp = input_ids[:, start : start + seqlen]
        if inp.shape != (1, seqlen):
            raise RuntimeError(
                f"s1k-1.1 sample {idx} produced invalid calibration shape {tuple(inp.shape)}."
            )
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        if len(trainloader) == nsamples:
            return trainloader

    raise ValueError(
        "s1k-1.1 does not contain enough eligible samples: "
        f"requested={nsamples}, collected={len(trainloader)}, seqlen={seqlen}, "
        f"slice_mode={slice_mode}, slice_offset={slice_offset}."
    )


def get_wikitext2_test(seed, seqlen, model):
    del seed
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    testenc = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
    return testenc


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    hf_token=None,
    eval_mode=False,
    slice_mode="random",
    slice_offset=0,
):
    if name == "s1k-1.1":
        return get_s1k_1_1(
            nsamples,
            seed,
            seqlen,
            model,
            hf_token,
            eval_mode,
            slice_mode,
            slice_offset,
        )
    if slice_mode != "random" or slice_offset != 0:
        raise ValueError("Calibration slice controls currently only support cal_dataset=s1k-1.1.")
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model, hf_token, eval_mode)
    raise NotImplementedError(f"Unsupported calibration dataset: {name}")
