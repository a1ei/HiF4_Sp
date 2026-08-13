#!/usr/bin/env python3
"""Distributed entry for reasoning-safe Qwen3.5 RMSNorm adaptation."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pathlib
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

HIF4_ROOT = pathlib.Path(__file__).resolve().parent
HIF4GPTQ_ROOT = HIF4_ROOT / "hif4gptq"
for path in (HIF4_ROOT, HIF4GPTQ_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from brq.calib import get_s1k_reasoning
from gptq import gptq_utils
from rmsnorm_lowrank import (
    LowRankConfig,
    clone_reference_layers,
    evaluate_function_preservation,
    extract_sensitive_subspaces,
    load_curvature_checkpoint,
    load_sidecar,
    optimize_blocks,
    save_curvature_checkpoint,
    save_sidecar,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def optional_float(value):
    if str(value).lower() in {"none", "null", "off"}:
        return None
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", type=str2bool, default=False)
    parser.add_argument(
        "--local-files-only",
        type=str2bool,
        default=True,
        help="Load the base model/tokenizer only from local files or the HF cache.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-rmsnorm-lowrank", type=str2bool, default=True)
    parser.add_argument(
        "--lowrank-mode",
        choices=["baseline", "naive_lowrank", "reasoning_projected_lowrank"],
        default="reasoning_projected_lowrank",
    )
    parser.add_argument("--lowrank-rank", type=int, default=4)
    parser.add_argument("--sensitive-rank", type=int, default=64)
    parser.add_argument("--curvature-tokens-per-sample", type=int, default=32)
    parser.add_argument(
        "--curvature-token-selection", choices=["grad_norm_topk"], default="grad_norm_topk"
    )
    parser.add_argument("--lowrank-epochs", type=int, default=10)
    parser.add_argument("--lowrank-lr", type=float, default=1e-3)
    parser.add_argument("--lowrank-eta", type=float, default=0.05)
    parser.add_argument("--max-relative-delta", type=optional_float, default=None)
    parser.add_argument("--lowrank-init-std", type=float, default=0.02)
    parser.add_argument("--lowrank-save-path", default=None)
    parser.add_argument("--lowrank-load-path", default=None)
    parser.add_argument(
        "--curvature-checkpoint-path",
        default=None,
        help="Save/load the completed sensitive-subspace phase for exact restart.",
    )
    parser.add_argument(
        "--curvature-matrix-checkpoint-path",
        default=None,
        help="Save/load sharded raw C=sum(G^T G) matrices before eigendecomposition.",
    )
    parser.add_argument("--evaluate-function-preservation", type=str2bool, default=False)
    parser.add_argument("--cal-dataset", choices=["s1k-1.1"], default="s1k-1.1")
    parser.add_argument("--cal-nsamples", type=int, default=512)
    parser.add_argument("--cal-seqlen", type=int, default=4096)
    parser.add_argument(
        "--cal-slice-mode", choices=["head", "tail", "offset", "random"], default="head"
    )
    parser.add_argument("--cal-slice-offset", type=int, default=0)
    parser.add_argument("--gptq", type=str2bool, default=True)
    parser.add_argument(
        "--gptq-load-path",
        default=None,
        help="Load an existing HiF4 GPTQ checkpoint and skip GPTQ quantization.",
    )
    parser.add_argument("--hif4a", type=str2bool, default=True)
    parser.add_argument("--hif4-weight-format", choices=["hif4", "hif4-1", "nvfp4"], default="hif4")
    parser.add_argument("--act-quant-format", choices=["hif4", "hif4-1", "nvfp4"], default="hif4")
    parser.add_argument("--gptq-percdamp", type=float, default=0.01)
    parser.add_argument("--gptq-calib-batch-size", type=int, default=1)
    parser.add_argument("--block-size-linear", type=int, default=64)
    parser.add_argument("--exclude-layers", nargs="*", default=["lm_head"])
    return parser.parse_args()


def _qtype(format_name: str) -> str:
    return {"hif4": "hifx4", "hif4-1": "hifx4_1", "nvfp4": "nvf4"}[format_name]


def _init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed RMSNorm adaptation requires CUDA/NCCL.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Rank 0 performs the deliberately unchanged single-GPU GPTQ pass while
        # peers wait.  Its duration can exceed the default process-group timeout.
        dist.init_process_group("nccl", timeout=timedelta(hours=24))
        return dist.get_rank(), world_size, local_rank, torch.device(f"cuda:{local_rank}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


def _broadcast_model(model, device: torch.device) -> None:
    if not dist.is_initialized():
        return
    # GPTQ leaves decoder blocks on CPU. NCCL therefore uses one explicit GPU
    # staging tensor at a time instead of silently selecting another backend.
    for tensor in list(model.parameters()) + list(model.buffers()):
        staging = tensor.detach().to(device)
        dist.broadcast(staging, src=0)
        if dist.get_rank() != 0:
            tensor.data.copy_(staging.to(device=tensor.device, dtype=tensor.dtype))
        del staging


def _validate_args(args, world_size):
    if not args.enable_rmsnorm_lowrank:
        raise ValueError("This independent entry requires --enable-rmsnorm-lowrank true.")
    if args.lowrank_load_path is not None and not args.evaluate_function_preservation:
        raise ValueError("--lowrank-load-path requires --evaluate-function-preservation true.")
    if args.evaluate_function_preservation and args.lowrank_load_path is None:
        raise ValueError("Function-preservation evaluation requires --lowrank-load-path.")
    if not args.evaluate_function_preservation and args.lowrank_save_path is None:
        raise ValueError("A training run requires --lowrank-save-path.")
    if not args.gptq or not args.hif4a:
        raise ValueError("The experiment requires --gptq true and --hif4a true.")
    if args.gptq_load_path is not None and not pathlib.Path(args.gptq_load_path).is_dir():
        raise ValueError(f"GPTQ model directory does not exist: {args.gptq_load_path}")
    if args.cal_nsamples % world_size:
        raise ValueError("cal_nsamples must be divisible by distributed world size.")
    if args.cal_seqlen != 4096:
        raise ValueError("The first experiment requires cal_seqlen=4096.")
    if args.cal_slice_mode != "offset" and args.cal_slice_offset != 0:
        raise ValueError("cal_slice_offset is only valid with cal_slice_mode=offset.")


def main():
    args = parse_args()
    rank, world_size, _local_rank, device = _init_distributed()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s rank={rank} %(levelname)s %(message)s",
    )
    _validate_args(args, world_size)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = LowRankConfig(
        mode=args.lowrank_mode,
        rank=args.lowrank_rank,
        sensitive_rank=args.sensitive_rank,
        curvature_tokens_per_sample=args.curvature_tokens_per_sample,
        curvature_token_selection=args.curvature_token_selection,
        epochs=args.lowrank_epochs,
        lr=args.lowrank_lr,
        eta=args.lowrank_eta,
        max_relative_delta=args.max_relative_delta,
        init_std=args.lowrank_init_std,
    )

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    if getattr(model.config, "model_type", "") not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("This entry only supports Qwen3.5 4B.")
    # The first version deliberately locks the advertised 4B architecture.
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 3_000_000_000 <= parameter_count <= 6_000_000_000:
        raise ValueError(f"Expected a Qwen3.5 4B-class model, got {parameter_count:,} parameters.")
    config.validate(model.config.hidden_size)

    all_samples = get_s1k_reasoning(
        args.cal_nsamples,
        args.seed,
        args.cal_seqlen,
        args.model,
        slice_mode=args.cal_slice_mode,
        slice_offset=args.cal_slice_offset,
    )
    local_samples = all_samples[rank::world_size]
    if len(local_samples) * world_size != len(all_samples):
        raise RuntimeError("Distributed calibration shards are not equal-sized.")

    if args.evaluate_function_preservation:
        adapted_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
        )
        load_sidecar(adapted_model, args.lowrank_load_path)
        metrics = evaluate_function_preservation(model, adapted_model, local_samples, device)
        if dist.is_initialized():
            count = float(metrics.pop("reasoning_token_count"))
            keys = sorted(metrics)
            totals = torch.tensor([metrics[key] * count for key in keys] + [count], device=device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            global_count = float(totals[-1])
            metrics = {
                key: float(totals[index] / global_count) for index, key in enumerate(keys)
            }
            metrics["reasoning_token_count"] = int(global_count)
        if rank == 0:
            logging.info("BF16 function-preservation metrics: %s", metrics)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    calibration_signature = {
        "model": str(args.model),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "dataset": args.cal_dataset,
        "nsamples": args.cal_nsamples,
        "seqlen": args.cal_seqlen,
        "slice_mode": args.cal_slice_mode,
        "slice_offset": args.cal_slice_offset,
        "seed": args.seed,
    }
    curvature_checkpoint_path = args.curvature_checkpoint_path
    if curvature_checkpoint_path is None and args.lowrank_save_path is not None:
        curvature_checkpoint_path = args.lowrank_save_path + ".curvature.pt"

    curvature_backward_ran = False
    if (
        config.mode == "reasoning_projected_lowrank"
        and curvature_checkpoint_path is not None
        and pathlib.Path(curvature_checkpoint_path).is_file()
    ):
        subspaces, curvature_diagnostics = load_curvature_checkpoint(
            curvature_checkpoint_path, model, config, calibration_signature
        )
        logging.info(
            "Loaded completed curvature checkpoint from %s; skipping curvature backward.",
            curvature_checkpoint_path,
        )
    else:
        curvature_samples = all_samples if world_size > 1 else local_samples
        curvature_matrix_checkpoint_path = args.curvature_matrix_checkpoint_path
        if curvature_matrix_checkpoint_path is None and curvature_checkpoint_path is not None:
            curvature_matrix_checkpoint_path = curvature_checkpoint_path + ".C"
        subspaces, curvature_diagnostics, curvature_backward_ran = extract_sensitive_subspaces(
            model,
            curvature_samples,
            config,
            device,
            matrix_checkpoint_path=curvature_matrix_checkpoint_path,
            calibration=calibration_signature,
        )
        if config.mode == "reasoning_projected_lowrank" and curvature_checkpoint_path is not None:
            if rank == 0:
                save_curvature_checkpoint(
                    curvature_checkpoint_path,
                    model,
                    config,
                    subspaces,
                    curvature_diagnostics,
                    calibration_signature,
                )
                logging.info("Saved completed curvature checkpoint to %s", curvature_checkpoint_path)
            if dist.is_initialized():
                dist.barrier()

    if world_size > 1 and config.mode == "reasoning_projected_lowrank" and curvature_backward_ran:
        # FSDP mutates parameters into shards. Discard that model at the phase
        # boundary and reload pristine BF16 weights for reference block outputs.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        logging.info("Reloading pristine BF16 model after FSDP curvature extraction.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
        )
    # Snapshot pristine BF16 blocks before existing GPTQ weights overwrite them.
    reference_layers = clone_reference_layers(model)

    if args.gptq_load_path is not None:
        # Load on every rank because each rank independently optimizes its data
        # shard. This replaces only model weights; the pristine BF16 reference
        # blocks and sensitive subspaces were captured before this point.
        from main import _load_quantized_model

        logging.info("Loading existing GPTQ W4 weights from %s; skipping GPTQ.", args.gptq_load_path)
        _load_quantized_model(model, args.gptq_load_path)
    elif rank == 0:
        logging.info("Starting unchanged GPTQ W4 on rank 0 with A4 disabled.")
        args.hif4_weight_qtype = _qtype(args.hif4_weight_format)
        args.act_quant_qtype = _qtype(args.act_quant_format)
        args.token_importance = "none"
        args.entropy_direction = "low"
        args.cal_nsamples = len(all_samples)
        args.cal_seqlen = all_samples[0]["input_ids"].shape[1]
        original_hif4a = args.hif4a
        args.hif4a = False
        try:
            gptq_loader = [sample["input_ids"] for sample in all_samples]
            gptq_utils.gptq_fwrd(model, gptq_loader, device, args)
        finally:
            args.hif4a = original_hif4a
    if dist.is_initialized() and args.gptq_load_path is None:
        dist.barrier()
        _broadcast_model(model, device)
        dist.barrier()

    wrappers, optimization_diagnostics = optimize_blocks(
        model,
        reference_layers,
        local_samples,
        subspaces,
        config,
        device,
        act_qtype=_qtype(args.act_quant_format),
        calib_batch_size=1,
    )
    if rank == 0:
        save_sidecar(
            args.lowrank_save_path,
            model,
            config,
            wrappers,
            curvature_diagnostics,
            optimization_diagnostics,
        )
        logging.info("Saved RMSNorm low-rank sidecar to %s", args.lowrank_save_path)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
