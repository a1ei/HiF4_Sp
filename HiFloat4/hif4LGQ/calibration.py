from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .logic_tokens import LogicTokenMatcher


@dataclass(frozen=True)
class ReasoningCalibrationSample:
    input_ids: torch.Tensor
    logic_mask: torch.Tensor
    source_index: int

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.numel())


def truncate_input_ids(input_ids: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if input_ids.ndim != 1:
        raise ValueError(f"Expected 1D input_ids, got {tuple(input_ids.shape)}.")
    if sequence_length < 0:
        raise ValueError("lgq_calib_seq_len must be greater than or equal to 0.")
    if sequence_length == 0:
        return input_ids
    return input_ids[:sequence_length]


def _reasoning_text(row: dict, index: int) -> str:
    field_names = (
        "question",
        "deepseek_thinking_trajectory",
        "deepseek_attempt",
    )
    fields: list[str] = []
    for field_name in field_names:
        value = row.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"s1k-1.1 sample {index} has an empty or invalid {field_name} field."
            )
        fields.append(value.strip())
    return "\n\n".join(fields)


def load_reasoning_calibration_samples(
    tokenizer,
    matcher: LogicTokenMatcher,
    dataset_name: str,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> list[ReasoningCalibrationSample]:
    if dataset_name != "s1k-1.1":
        raise ValueError("hif4LGQ currently supports only --cal_dataset=s1k-1.1.")
    if num_samples <= 0:
        raise ValueError("cal_nsamples must be greater than 0 for hif4LGQ.")
    if sequence_length < 0:
        raise ValueError("lgq_calib_seq_len must be greater than or equal to 0.")

    from datasets import load_dataset

    dataset = load_dataset("simplescaling/s1K-1.1", split="train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    samples: list[ReasoningCalibrationSample] = []
    for index in indices:
        text = _reasoning_text(dataset[index], index)
        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].to(dtype=torch.long, device="cpu")
        input_ids = truncate_input_ids(input_ids, sequence_length).contiguous()
        if input_ids.numel() == 0:
            continue
        logic_mask = matcher.match(input_ids)
        samples.append(
            ReasoningCalibrationSample(
                input_ids=input_ids,
                logic_mask=logic_mask,
                source_index=index,
            )
        )
        if len(samples) == num_samples:
            break

    if len(samples) != num_samples:
        raise ValueError(
            "s1k-1.1 does not contain enough hif4LGQ calibration samples: "
            f"requested={num_samples}, collected={len(samples)}."
        )
    total_logic_tokens = sum(int(sample.logic_mask.sum().item()) for sample in samples)
    if total_logic_tokens == 0:
        raise ValueError(
            "No logic keyword token positions were found in the selected calibration data."
        )
    return samples
