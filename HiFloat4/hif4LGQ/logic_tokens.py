from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch


DEFAULT_KEYWORDS_PATH = Path(__file__).with_name("default_logic_keywords.json")


def load_logic_keywords(path: str | None = None) -> list[str]:
    config_path = Path(path) if path else DEFAULT_KEYWORDS_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = []
        for category, category_values in payload.items():
            if not isinstance(category_values, list):
                raise ValueError(
                    f"Logic keyword category {category!r} must contain a JSON list."
                )
            values.extend(category_values)
    else:
        raise ValueError("Logic keyword config must be a JSON list or object of lists.")

    keywords: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid logic keyword: {value!r}")
        keyword = value.strip()
        normalized = keyword.casefold()
        if normalized not in seen:
            seen.add(normalized)
            keywords.append(keyword)
    if not keywords:
        raise ValueError("Logic keyword config is empty.")
    return keywords


def _keyword_text_variants(keyword: str) -> set[str]:
    case_variants = {
        keyword,
        keyword.lower(),
        keyword.capitalize(),
        keyword.upper(),
    }
    variants: set[str] = set()
    for value in case_variants:
        variants.update({value, f" {value}", f"\n{value}", f"\n\n{value}"})
    return variants


class LogicTokenMatcher:
    """Match configurable words and phrases as tokenizer ID subsequences."""

    def __init__(self, tokenizer, keywords: list[str]):
        patterns: set[tuple[int, ...]] = set()
        for keyword in keywords:
            for variant in _keyword_text_variants(keyword):
                token_ids = tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    patterns.add(tuple(int(token_id) for token_id in token_ids))
        if not patterns:
            raise ValueError("No token patterns were produced from the logic keywords.")

        by_first: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        for pattern in patterns:
            by_first[pattern[0]].append(pattern)
        for candidates in by_first.values():
            candidates.sort(key=len, reverse=True)

        self.keywords = tuple(keywords)
        self.patterns = tuple(sorted(patterns))
        self._by_first = dict(by_first)

    def match(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError(
                f"Logic token matching expects 1D input_ids, got {tuple(input_ids.shape)}."
            )

        ids = [int(token_id) for token_id in input_ids.tolist()]
        mask = torch.zeros(len(ids), dtype=torch.bool)
        for start, token_id in enumerate(ids):
            for pattern in self._by_first.get(token_id, ()):
                end = start + len(pattern)
                if end <= len(ids) and tuple(ids[start:end]) == pattern:
                    mask[start:end] = True
        return mask
