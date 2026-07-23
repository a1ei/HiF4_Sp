import ast
import random
import re

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


def _validate_sequence_calibration_args(dataset_name, nsamples, seqlen, eval_mode, slice_mode, slice_offset):
    if eval_mode:
        raise ValueError(f"{dataset_name} only supports calibration mode; it has no evaluation split here.")
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


def _slice_calibration_ids(input_ids, seqlen, slice_mode, slice_offset, rng):
    token_count = input_ids.shape[1]
    required_tokens = slice_offset + seqlen if slice_mode == "offset" else seqlen
    if token_count < required_tokens:
        return None

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
        raise RuntimeError(f"Calibration sample produced invalid shape {tuple(inp.shape)}.")
    tar = inp.clone()
    tar[:, :-1] = -100
    return inp, tar


def _make_causal_lm_sample(input_ids):
    tar = input_ids.clone()
    tar[:, :-1] = -100
    return input_ids, tar


def _normalize_dedup_text(text):
    return re.sub(r"\s+", " ", text.strip())


_TACO_TRAIN_PARQUET_FILES = [
    f"hf://datasets/BAAI/TACO/ALL/train-{idx:05d}-of-00009.parquet"
    for idx in range(9)
]
_TACO_TEXT_SEPARATOR = "\n\n<|endoftext|>\n\n"


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
    _validate_sequence_calibration_args(
        "s1k-1.1",
        nsamples,
        seqlen,
        eval_mode,
        slice_mode,
        slice_offset,
    )

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
        sample = _slice_calibration_ids(input_ids, seqlen, slice_mode, slice_offset, rng)
        if sample is None:
            continue
        trainloader.append(sample)
        if len(trainloader) == nsamples:
            return trainloader

    raise ValueError(
        "s1k-1.1 does not contain enough eligible samples: "
        f"requested={nsamples}, collected={len(trainloader)}, seqlen={seqlen}, "
        f"slice_mode={slice_mode}, slice_offset={slice_offset}."
    )


def _get_livecodebench_questions_for_dedup():
    livecodebench = load_dataset("lighteval/code_generation_lite", "v6", split="test")
    questions = set()
    for idx, row in enumerate(livecodebench):
        question = row.get("question_content")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"LiveCodeBench v6 sample {idx} has an empty question_content field.")
        questions.add(_normalize_dedup_text(question))
    return questions


def _format_taco_calibration_text(row, idx):
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"TACO train sample {idx} has an empty or invalid question field.")

    text = f"Question:\n{question}\n\n"
    starter_code = row.get("starter_code")
    if isinstance(starter_code, str) and starter_code.strip():
        text += "Starter code:\n"
        text += f"```python\n{starter_code}\n```\n"
    first_solution = _get_taco_first_solution(row, idx)
    if first_solution is None:
        return None
    text += "\nSolution:\n"
    text += f"```python\n{first_solution}\n```\n"
    return text


def _get_taco_first_solution(row, idx):
    solutions = row.get("solutions")
    if not isinstance(solutions, str) or not solutions.strip():
        return None
    try:
        parsed_solutions = ast.literal_eval(solutions)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"TACO train sample {idx} has an invalid solutions field.") from exc
    if not isinstance(parsed_solutions, list):
        raise ValueError(f"TACO train sample {idx} solutions field is not a list.")
    if not parsed_solutions:
        return None
    first_solution = parsed_solutions[0]
    if not isinstance(first_solution, str) or not first_solution.strip():
        return None
    return first_solution


def _pack_taco_question_start_samples(calibration_texts, tokenizer, nsamples, seqlen):
    trainloader = []
    next_text_idx = 0
    for sample_idx in range(nsamples):
        token_chunks = []
        token_count = 0
        while token_count < seqlen and next_text_idx < len(calibration_texts):
            text = calibration_texts[next_text_idx]
            if token_chunks:
                text = _TACO_TEXT_SEPARATOR + text
            input_ids = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            token_chunks.append(input_ids)
            token_count += input_ids.shape[1]
            next_text_idx += 1
        if token_count < seqlen:
            raise ValueError(
                "TACO train does not contain enough eligible text for question-start packing: "
                f"requested={nsamples}, collected={sample_idx}, cal_seqlen={seqlen}, "
                f"remaining_tokens={token_count}."
            )

        sample_ids = torch.cat(token_chunks, dim=1)[:, :seqlen]
        if sample_ids.shape != (1, seqlen):
            raise RuntimeError(f"TACO calibration sample produced invalid shape {tuple(sample_ids.shape)}.")
        trainloader.append(_make_causal_lm_sample(sample_ids))
    return trainloader


def get_taco(
    nsamples,
    seed,
    seqlen,
    model,
    hf_token=None,
    eval_mode=False,
    slice_mode="random",
    slice_offset=0,
):
    _validate_sequence_calibration_args(
        "taco",
        nsamples,
        seqlen,
        eval_mode,
        slice_mode,
        slice_offset,
    )
    if slice_mode != "head" or slice_offset != 0:
        raise ValueError("TACO question-start calibration requires cal_slice_mode=head and cal_slice_offset=0.")

    tokenizer = _get_tokenizer(model, hf_token)
    train_data = load_dataset("parquet", data_files={"train": _TACO_TRAIN_PARQUET_FILES}, split="train")
    livecodebench_questions = _get_livecodebench_questions_for_dedup()
    indices = list(range(len(train_data)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    calibration_texts = []
    for idx in indices:
        question = train_data[idx].get("question")
        if isinstance(question, str) and _normalize_dedup_text(question) in livecodebench_questions:
            continue

        calibration_text = _format_taco_calibration_text(train_data[idx], idx)
        if calibration_text is None:
            continue
        calibration_texts.append(calibration_text)

    if not calibration_texts:
        raise ValueError("TACO train has no eligible samples after LiveCodeBench v6 deduplication.")

    return _pack_taco_question_start_samples(calibration_texts, tokenizer, nsamples, seqlen)


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
    if name == "taco":
        return get_taco(
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
        raise ValueError("Calibration slice controls currently only support cal_dataset=s1k-1.1 or taco.")
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model, hf_token, eval_mode)
    raise NotImplementedError(f"Unsupported calibration dataset: {name}")
