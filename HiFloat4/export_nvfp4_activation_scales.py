#!/usr/bin/env python3
"""Export vLLM NVFP4 activation scale sidecar for an existing HF checkpoint."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

HIF4_ROOT = pathlib.Path(__file__).resolve().parent
HIF4GPTQ_ROOT = HIF4_ROOT / "hif4gptq"
REPO_ROOT = HIF4_ROOT.parent
for path in (REPO_ROOT, HIF4_ROOT, HIF4GPTQ_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from hif4smoothquant.smoothquant_utils import (  # noqa: E402
    collect_nvfp4_activation_scales,
    save_nvfp4_activation_scales,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    vv = str(v).lower()
    if vv in {"yes", "true", "t", "y", "1"}:
        return True
    if vv in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def torch_dtype_from_arg(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and save nvfp4_activation_scales.safetensors."
    )
    parser.add_argument("--model_path", required=True, help="Existing HF checkpoint directory")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for nvfp4_activation_scales.safetensors. Default: model_path",
    )
    parser.add_argument(
        "--cal_dataset",
        default="s1k-1.1",
        choices=["wikitext2", "ptb", "c4", "s1k-1.1", "taco"],
    )
    parser.add_argument("--cal_nsamples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cal_seqlen", type=int, default=4096)
    parser.add_argument(
        "--cal_slice_mode",
        default="head",
        choices=["random", "head", "tail", "offset"],
    )
    parser.add_argument("--cal_slice_offset", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trust-remote-code", type=str2bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cal_nsamples <= 0:
        raise ValueError("--cal_nsamples must be greater than 0.")
    if args.cal_seqlen <= 0:
        raise ValueError("--cal_seqlen must be greater than 0.")
    if args.cal_slice_offset < 0:
        raise ValueError("--cal_slice_offset must be greater than or equal to 0.")
    if args.cal_slice_mode != "offset" and args.cal_slice_offset != 0:
        raise ValueError("--cal_slice_offset can only be non-zero with offset mode.")

    from transformers import AutoModelForCausalLM
    import brq.calib as calib

    dtype = torch_dtype_from_arg(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    trainloader = calib.get_loaders(
        args.cal_dataset,
        nsamples=args.cal_nsamples,
        seed=args.seed,
        seqlen=args.cal_seqlen,
        model=args.model_path,
        eval_mode=False,
        slice_mode=args.cal_slice_mode,
        slice_offset=args.cal_slice_offset,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("NVFP4 activation scale export requires CUDA.")

    scales = collect_nvfp4_activation_scales(model, trainloader, device, args)
    output_dir = args.output_dir or args.model_path
    path = save_nvfp4_activation_scales(scales, output_dir)
    print(f"Saved {len(scales)} NVFP4 activation scales to: {path}")


if __name__ == "__main__":
    main()
