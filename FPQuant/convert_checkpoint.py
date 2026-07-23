#!/usr/bin/env python3
"""CLI for exporting FP-Quant NVFP4 checkpoints as ordinary HF models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FPQuant.dequantize import (  # noqa: E402
    DEFAULT_MAX_SHARD_SIZE,
    convert_fpquant_nvfp4_checkpoint,
    estimate_converted_checkpoint_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dequantize FP-Quant NVFP4 weights and fold Hadamard transforms."
    )
    parser.add_argument(
        "--input_model",
        required=True,
        help="Local model directory or an already-cached Hugging Face model ID.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--output_dtype",
        required=True,
        choices=["bfloat16", "float32"],
        help="Storage dtype for the formerly-NVFP4 Linear weights.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max_shard_size_gb",
        type=float,
        default=DEFAULT_MAX_SHARD_SIZE / 1024**3,
    )
    return parser.parse_args()


def _resolve_input_model(value: str) -> Path:
    local_path = Path(value).expanduser()
    if local_path.is_dir():
        return local_path.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(value, local_files_only=True)).resolve()


def main() -> None:
    args = parse_args()
    input_path = _resolve_input_model(args.input_model)
    max_shard_size = int(args.max_shard_size_gb * 1024**3)
    if max_shard_size <= 0:
        raise ValueError("--max_shard_size_gb must be positive")

    estimate = estimate_converted_checkpoint_size(input_path, args.output_dtype)
    print(
        f"Input: {input_path}\n"
        f"Output dtype: {args.output_dtype}\n"
        f"Estimated size: {estimate.total_bytes / 1024**3:.2f} GiB"
    )
    output_path = convert_fpquant_nvfp4_checkpoint(
        input_dir=input_path,
        output_dir=args.output_dir,
        output_dtype=args.output_dtype,
        overwrite=args.overwrite,
        max_shard_size=max_shard_size,
    )
    print(f"Converted checkpoint written to: {output_path}")


if __name__ == "__main__":
    main()
