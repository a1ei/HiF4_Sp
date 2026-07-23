import argparse
import json
import logging
import os
import pathlib
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

HIF4_ROOT = pathlib.Path(__file__).resolve().parent
HIF4GPTQ_ROOT = pathlib.Path(__file__).resolve().parent / "hif4gptq"
if str(HIF4_ROOT) not in sys.path:
    sys.path.append(str(HIF4_ROOT))
if str(HIF4GPTQ_ROOT) not in sys.path:
    sys.path.append(str(HIF4GPTQ_ROOT))

from hif4_gpu.quant_cy import QType, quant_dequant_float
from hif4_gpu.quant_cy.layers.QLinear2 import QLinear2


def str2bool(v):
    if isinstance(v, bool):
        return v
    vv = str(v).lower()
    if vv in {"yes", "true", "t", "y", "1"}:
        return True
    if vv in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str2path(v):
    if v is None or str(v).lower() in {"none"}:
        return None
    return str(v)


def configure_logging(log_dir: str = "hif4_logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger("hif4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _torch_dtype_from_arg(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _first_input_device(device_map):
    for _, dev in device_map.items():
        if isinstance(dev, int):
            return torch.device(f"cuda:{dev}")
        if isinstance(dev, str) and dev.startswith("cuda"):
            return torch.device(dev)
    return torch.device("cpu")


def _quant_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _no_split_module_classes(model):
    model_type = getattr(model.config, "model_type", "")
    mapping = {
        "llama": ["LlamaDecoderLayer"],
        "qwen3": ["Qwen3DecoderLayer"],
        "qwen3_5_text": ["Qwen3_5DecoderLayer"],
    }
    return mapping.get(model_type, ["LlamaDecoderLayer", "Qwen3DecoderLayer", "Qwen3_5DecoderLayer"])


def _save_quantized_model(model, path: str, tokenizer=None, args=None) -> None:
    os.makedirs(path, exist_ok=True)
    safe_serialization = bool(
        getattr(args, "safe_serialization", False) if args is not None else False
    )
    model.save_pretrained(
        path,
        safe_serialization=safe_serialization,
        max_shard_size="5GB",
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    nvfp4_activation_scales = getattr(model, "_nvfp4_activation_scales", None)
    if nvfp4_activation_scales:
        from hif4smoothquant.smoothquant_utils import save_nvfp4_activation_scales

        save_nvfp4_activation_scales(nvfp4_activation_scales, path)
    if args is not None:
        with open(os.path.join(path, "quantization_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    logging.info("Saved quantized model to %s", path)


def _load_state_dict_file(path: str):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu")


def _load_sharded_state_dict(model, index_file: str, path: str) -> None:
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    for shard_file in shard_files:
        shard_path = os.path.join(path, shard_file)
        model.load_state_dict(_load_state_dict_file(shard_path), strict=False)


def _load_quantized_model(model, path: str) -> None:
    safetensors_index = os.path.join(path, "model.safetensors.index.json")
    pytorch_index = os.path.join(path, "pytorch_model.bin.index.json")
    safetensors_file = os.path.join(path, "model.safetensors")
    pytorch_file = os.path.join(path, "pytorch_model.bin")

    if os.path.exists(safetensors_index):
        _load_sharded_state_dict(model, safetensors_index, path)
        logging.info("Loaded sharded quantized model weights from %s", path)
        return

    if os.path.exists(pytorch_index):
        _load_sharded_state_dict(model, pytorch_index, path)
        logging.info("Loaded sharded quantized model weights from %s", path)
        return

    if os.path.exists(safetensors_file):
        from safetensors.torch import load_file

        model.load_state_dict(load_file(safetensors_file), strict=False)
        logging.info("Loaded quantized model weights from %s", safetensors_file)
        return

    if os.path.exists(pytorch_file):
        model.load_state_dict(torch.load(pytorch_file, map_location="cpu"), strict=False)
        logging.info("Loaded quantized model weights from %s", pytorch_file)
        return

    raise FileNotFoundError(f"No supported weight files found under: {path}")


def _is_excluded_layer(name: str, exclude_layers: list[str]) -> bool:
    return name in exclude_layers


def _quant_format_qtype(quant_format: str) -> str:
    mapping = {
        "hif4": "hifx4",
        "hif4-1": "hifx4_1",
        "nvfp4": "nvf4",
    }
    if quant_format not in mapping:
        raise ValueError(f"Unsupported quant format: {quant_format}")
    return mapping[quant_format]


def _hif4_weight_qtype(weight_format: str) -> str:
    return _quant_format_qtype(weight_format)


@torch.no_grad()
def hif4_rtn_quant(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    qparams = QType(args.hif4_weight_qtype).dim(-1)
    quant_device = _quant_device()
    if quant_device.type != "cuda":
        raise RuntimeError("HiF4 RTN quantization requires CUDA because quant_dequant_float uses a CUDA kernel.")

    quantized_layers = 0

    for name, module in model.named_modules():
        if _is_excluded_layer(name, args.exclude_layers):
            logging.info("(HiF4 RTN) Excluding layer: %s", name)
            continue
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            weight_device = weight.device
            if weight_device == quant_device:
                quant_input = weight.contiguous()
            else:
                quant_input = weight.to(device=quant_device).contiguous()
            quant_weight = quant_dequant_float(quant_input, qparams, force_fp32=True)
            if torch.any(torch.isnan(quant_weight)):
                raise ValueError(f"NaN in HiF4 RTN quantized weights: {name}")
            module.weight.data = quant_weight.to(dtype=weight.dtype, device=weight.device).contiguous()
            if weight_device.type == "cpu":
                del quant_input, quant_weight
                torch.cuda.empty_cache()
            quantized_layers += 1

    logging.info("Applied HiF4 RTN fake quantization to %s Linear layers.", quantized_layers)
    return model


def replace_linear_with_hif4_activation_quant(module: nn.Module, args: argparse.Namespace) -> nn.Module:
    act_qtype_name = getattr(args, "act_quant_qtype", _quant_format_qtype(args.act_quant_format))
    act_qtype = QType(act_qtype_name)

    if isinstance(module, nn.Linear):
        if _is_excluded_layer("", args.exclude_layers):
            return module
        new_module = QLinear2(module.in_features, module.out_features, module.bias is not None)
        new_module.transfer(module)
        new_module.assign_qparams(act_qtype)
        new_module.assign_input_qparams(act_qtype)
        new_module.set_quant_grad(False)
        new_module._fast_forward = not args.disable_fast_forward
        return new_module

    module_dict = dict(module.named_modules())
    replaced_layers = 0
    for name, child in list(module.named_modules()):
        if not name:
            continue
        if _is_excluded_layer(name, args.exclude_layers):
            logging.info("(HiF4 activation) Excluding layer: %s", name)
            continue
        if isinstance(child, nn.Linear):
            new_module = QLinear2(child.in_features, child.out_features, child.bias is not None)
            new_module.transfer(child)
            new_module.assign_qparams(act_qtype)
            new_module.assign_input_qparams(act_qtype)
            new_module.set_quant_grad(False)
            new_module._fast_forward = not args.disable_fast_forward
            parent_name = ".".join(name.split(".")[:-1])
            parent_module = module_dict[parent_name]
            setattr(parent_module, name.split(".")[-1], new_module)
            replaced_layers += 1

    logging.info(
        "Replaced %s Linear layers with %s activation-only QLinear2.",
        replaced_layers,
        args.act_quant_format,
    )
    return module


def distribute_model_for_eval(model, logger: logging.Logger) -> torch.device:
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    if not torch.cuda.is_available():
        logger.info("CUDA is unavailable, evaluating on CPU.")
        return torch.device("cpu")

    n_gpus = torch.cuda.device_count()
    if n_gpus == 1:
        model.to("cuda:0")
        logger.info("Single GPU detected, evaluating on cuda:0.")
        return torch.device("cuda:0")

    no_split = _no_split_module_classes(model)
    logger.info("Multi-GPU detected (%s), dispatching model with accelerate.", n_gpus)

    max_memory = get_balanced_memory(model, no_split_module_classes=no_split)
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split)
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )
    input_device = _first_input_device(device_map)
    logger.info("Model dispatched. Input device: %s", input_device)
    return input_device


def arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B", help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--trust-remote-code", type=str2bool, default=False)

    parser.add_argument("--ppl_tasks", nargs="+", default=["wikitext2"])
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["arc_challenge", "arc_easy", "boolq", "openbookqa", "piqa", "winogrande", "hellaswag",]
    )
    parser.add_argument("--test_zero_task", action="store_true")
    parser.add_argument(
        "--save_only",
        action="store_true",
        help="Save the transformed model and stop before PPL/zero-shot evaluation.",
    )
    parser.add_argument(
        "--safe_serialization",
        type=str2bool,
        default=False,
        help="Save model shards as safetensors instead of PyTorch bin files.",
    )

    parser.add_argument("--hif4w", type=str2bool, default=False, help="Enable one-shot HiF4 weight fake quantization")
    parser.add_argument(
        "--hif4_weight_format",
        type=str,
        default="hif4",
        choices=["hif4", "hif4-1", "nvfp4"],
        help="HiF4 weight fake quant format for RTN/GPTQ.",
    )
    parser.add_argument("--hif4a", type=str2bool, default=False, help="Enable HiF4 input activation fake quantization")
    parser.add_argument(
        "--act_quant_format",
        type=str,
        default="hif4",
        choices=["hif4", "hif4-1", "nvfp4"],
        help="Input activation fake quant format used when --hif4a is true.",
    )
    parser.add_argument("--exclude-layers", nargs="*", default=["lm_head"], help="Exact layer names to skip")
    parser.add_argument("--disable-fast-forward", action="store_true", help="Disable QLinear2 fast-forward path")

    parser.add_argument("--gptq", type=str2bool, default=False)
    parser.add_argument("--gptq_percdamp", type=float, default=0.01)
    parser.add_argument(
        "--cal_dataset",
        type=str,
        default="c4",
        choices=["wikitext2", "ptb", "c4", "s1k-1.1", "taco"],
    )
    parser.add_argument("--cal_nsamples", type=int, default=512)
    parser.add_argument("--cal_seqlen", type=int, default=512)
    parser.add_argument(
        "--cal_slice_mode",
        type=str,
        default="random",
        choices=["random", "head", "tail", "offset"],
        help="s1k-1.1 calibration slice mode within each question-plus-DeepSeek-response sequence.",
    )
    parser.add_argument(
        "--cal_slice_offset",
        type=int,
        default=0,
        help="Zero-based token offset used only when --cal_slice_mode=offset.",
    )
    parser.add_argument("--gptq_load_path", type=str2path, default=None)
    parser.add_argument("--gptq_save_path", type=str2path, default=None)
    parser.add_argument("--block_size_linear", type=int, default=64)
    parser.add_argument(
        "--token_importance",
        type=str,
        default="none",
        choices=["none", "entropy", "entropy_grad"],
        help="Optional GPTQ token-level Hessian weighting method.",
    )
    parser.add_argument("--entropy_alpha", type=float, default=1.0)
    parser.add_argument(
        "--entropy_norm",
        type=str,
        default="minmax",
        choices=["minmax", "zscore", "mean"],
    )
    parser.add_argument("--importance_alpha", type=float, default=1.0)
    parser.add_argument("--importance_mean_normalize", type=str2bool, default=True)
    parser.add_argument(
        "--importance_batch_size",
        type=int,
        default=1,
        help="Batch size for the entropy_grad FP attribution pass.",
    )
    parser.add_argument("--smoothquant", type=str2bool, default=False)
    parser.add_argument("--smoothquant_alpha", type=float, default=0.5)
    parser.add_argument(
        "--smoothquant_scale_only",
        type=str2bool,
        default=False,
        help="Only calibrate/apply SmoothQuant scales; skip SmoothQuant weight quantization.",
    )
    parser.add_argument(
        "--save_nvfp4_activation_scales",
        type=str2bool,
        default=True,
        help="Save nvfp4_activation_scales.safetensors for vLLM NVFP4 fake activation when using NVFP4-related quantization.",
    )
    parser.add_argument("--awq", type=str2bool, default=False)
    parser.add_argument("--awq_n_grid", type=int, default=20)
    parser.add_argument("--magr", type=str2bool, default=False)
    parser.add_argument("--magr_cd_iter", type=int, default=1)
    parser.add_argument("--magr_alpha", type=float, default=0.001)
    parser.add_argument("--magr_alpha_groupwise", type=float, default=0.0001)
    parser.add_argument("--magr_preprocess_iter", type=int, default=200)
    parser.add_argument("--flatquant", type=str2bool, default=False)
    parser.add_argument("--flatquant_epochs", type=int, default=15)
    parser.add_argument("--flatquant_cali_bsz", type=int, default=1)
    parser.add_argument("--flatquant_lr", type=float, default=1e-5)
    parser.add_argument("--flatquant_cali_trans", type=str2bool, default=True)
    parser.add_argument("--flatquant_add_diag", type=str2bool, default=True)
    parser.add_argument("--flatquant_lwc", type=str2bool, default=True)
    parser.add_argument("--flatquant_lac", type=str2bool, default=True)
    parser.add_argument("--flatquant_direct_inv", type=str2bool, default=True)
    parser.add_argument("--flatquant_diag_init", type=str, default="sq_style", choices=["sq_style", "one_style"])
    parser.add_argument("--flatquant_diag_alpha", type=float, default=0.3)
    parser.add_argument("--flatquant_matrix_path", type=str2path, default=None)

    return parser.parse_args() if interactive else parser.parse_args("")


def run_main(args: argparse.Namespace, logger: logging.Logger) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from utils import data_utils, eval_utils

    logger.info("Running with args: %s", vars(args))
    set_seed(args.seed)
    args.hif4_weight_qtype = _hif4_weight_qtype(args.hif4_weight_format)
    args.act_quant_qtype = _quant_format_qtype(args.act_quant_format)
    if args.save_only and not args.gptq_save_path:
        raise ValueError("--save_only requires --gptq_save_path.")
    if args.smoothquant_scale_only and not args.smoothquant:
        raise ValueError("--smoothquant_scale_only requires --smoothquant true.")
    if args.hif4_weight_qtype == "hifx4_1" and (args.gptq or args.magr) and args.block_size_linear != 64:
        raise ValueError("hif4-1 GPTQ/MagR requires --block_size_linear 64.")
    enabled_weight_methods = sum(bool(x) for x in (args.gptq, args.smoothquant, args.awq, args.magr, args.flatquant))
    if enabled_weight_methods > 1:
        raise ValueError("--gptq, --smoothquant, --awq, --magr, and --flatquant cannot be enabled at the same time.")
    if args.entropy_alpha < 0:
        raise ValueError("--entropy_alpha must be greater than or equal to 0.")
    if args.importance_alpha < 0:
        raise ValueError("--importance_alpha must be greater than or equal to 0.")
    if args.importance_batch_size <= 0:
        raise ValueError("--importance_batch_size must be greater than 0.")
    if args.token_importance != "none" and (not args.gptq or args.gptq_load_path):
        raise ValueError("--token_importance is only supported when running new GPTQ quantization.")
    needs_calibration = (args.gptq and not args.gptq_load_path) or any(
        (args.smoothquant, args.awq, args.magr, args.flatquant)
    )
    if needs_calibration:
        if args.cal_nsamples <= 0:
            raise ValueError("--cal_nsamples must be greater than 0.")
        if args.cal_seqlen <= 0:
            raise ValueError("--cal_seqlen must be greater than 0.")
        if args.token_importance in {"entropy", "entropy_grad"} and args.cal_seqlen < 2:
            raise ValueError("Entropy-based token importance requires --cal_seqlen to be at least 2.")
        if args.cal_slice_offset < 0:
            raise ValueError("--cal_slice_offset must be greater than or equal to 0.")
        if args.cal_slice_mode != "offset" and args.cal_slice_offset != 0:
            raise ValueError("--cal_slice_offset can only be non-zero when --cal_slice_mode=offset.")
        if args.cal_dataset not in {"s1k-1.1", "taco"} and (
            args.cal_slice_mode != "random" or args.cal_slice_offset != 0
        ):
            raise ValueError("Calibration slice controls currently only support --cal_dataset=s1k-1.1 or taco.")
        if args.flatquant and args.cal_nsamples % args.flatquant_cali_bsz != 0:
            raise ValueError("--cal_nsamples must be divisible by --flatquant_cali_bsz.")

    dtype = _torch_dtype_from_arg(args.dtype)
    load_device_map = "cpu"
    if args.hif4w and not args.gptq and not args.smoothquant and not args.awq and not args.magr and not args.flatquant:
        quant_device = _quant_device()
        if quant_device.type != "cuda":
            raise RuntimeError("HiF4 RTN quantization requires CUDA.")
        load_device_map = {"": str(quant_device)}
        logger.info("Loading model directly on %s for HiF4 RTN quantization.", quant_device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=load_device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=args.trust_remote_code)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.gptq:
        if args.hif4w:
            logger.info("Both --hif4w and --gptq are enabled; GPTQ controls weight quantization and HiF4 RTN is skipped.")
        if args.gptq_load_path:
            _load_quantized_model(model, args.gptq_load_path)
        else:
            from gptq import gptq_utils
            import brq.calib as calib

            logger.info("Quantizing model weights with HiFloat4 GPTQ.")
            trainloader = calib.get_loaders(
                args.cal_dataset,
                nsamples=args.cal_nsamples,
                seqlen=args.cal_seqlen,
                model=args.model,
                eval_mode=False,
                slice_mode=args.cal_slice_mode,
                slice_offset=args.cal_slice_offset,
            )
            gptq_utils.gptq_fwrd(model, trainloader, _quant_device(), args)

        if args.gptq_save_path:
            _save_quantized_model(model, args.gptq_save_path, tokenizer, args)
    elif args.smoothquant:
        if args.hif4w and args.smoothquant_scale_only:
            logger.info("Applying %s RTN before SmoothQuant scale-only.", args.hif4_weight_format)
            model = hif4_rtn_quant(model, args)
        elif args.hif4w:
            logger.info("Both --hif4w and --smoothquant are enabled; SmoothQuant controls weight quantization and HiF4 RTN is skipped.")

        from hif4smoothquant import smoothquant_utils
        import brq.calib as calib

        if args.smoothquant_scale_only:
            logger.info("Calibrating and applying SmoothQuant scales without final weight quantization.")
        else:
            logger.info("Quantizing model weights with HiFloat4 SmoothQuant.")
        trainloader = calib.get_loaders(
            args.cal_dataset,
            nsamples=args.cal_nsamples,
            seqlen=args.cal_seqlen,
            model=args.model,
            eval_mode=False,
            slice_mode=args.cal_slice_mode,
            slice_offset=args.cal_slice_offset,
        )
        smoothquant_utils.smoothquant_fwrd(model, trainloader, _quant_device(), args)

        if args.gptq_save_path:
            _save_quantized_model(model, args.gptq_save_path, tokenizer, args)
    elif args.awq:
        if args.hif4w:
            logger.info("Both --hif4w and --awq are enabled; AWQ controls weight quantization and HiF4 RTN is skipped.")

        from hif4awq import awq_utils
        import brq.calib as calib

        logger.info("Quantizing model weights with HiFloat4 AWQ.")
        trainloader = calib.get_loaders(
            args.cal_dataset,
            nsamples=args.cal_nsamples,
            seqlen=args.cal_seqlen,
            model=args.model,
            eval_mode=False,
            slice_mode=args.cal_slice_mode,
            slice_offset=args.cal_slice_offset,
        )
        awq_utils.awq_fwrd(model, trainloader, _quant_device(), args)

        if args.gptq_save_path:
            _save_quantized_model(model, args.gptq_save_path, tokenizer, args)
    elif args.magr:
        if args.hif4w:
            logger.info("Both --hif4w and --magr are enabled; MagR controls weight quantization and HiF4 RTN is skipped.")

        from hif4magr import magr_fwrd
        import brq.calib as calib

        logger.info("Quantizing model weights with HiFloat4 MagR.")
        trainloader = calib.get_loaders(
            args.cal_dataset,
            nsamples=args.cal_nsamples,
            seqlen=args.cal_seqlen,
            model=args.model,
            eval_mode=False,
            slice_mode=args.cal_slice_mode,
            slice_offset=args.cal_slice_offset,
        )
        magr_fwrd(model, trainloader, _quant_device(), args)

        if args.gptq_save_path:
            _save_quantized_model(model, args.gptq_save_path, tokenizer, args)
    elif args.flatquant:
        if args.hif4w:
            logger.info("Both --hif4w and --flatquant are enabled; FlatQuant controls weight quantization and HiF4 RTN is skipped.")

        from hif4flatquant import flatquant_fwrd, save_hif4_flatquant_model
        import brq.calib as calib

        logger.info("Quantizing model weights and Linear inputs with HiFloat4 FlatQuant.")
        trainloader = calib.get_loaders(
            args.cal_dataset,
            nsamples=args.cal_nsamples,
            seqlen=args.cal_seqlen,
            model=args.model,
            eval_mode=False,
            slice_mode=args.cal_slice_mode,
            slice_offset=args.cal_slice_offset,
        )
        flatquant_fwrd(model, trainloader, _quant_device(), args)
        save_hif4_flatquant_model(model, tokenizer, args.gptq_save_path, args)
    elif args.hif4w:
        logger.info("Quantizing model weights with one-shot HiF4 RTN.")
        model = hif4_rtn_quant(model, args)
        if args.gptq_save_path:
            _save_quantized_model(model, args.gptq_save_path, tokenizer, args)

    if args.hif4a:
        logger.info(
            "Replacing Linear layers with %s activation-only QLinear2.",
            args.act_quant_format,
        )
        model = replace_linear_with_hif4_activation_quant(model, args)

    if args.save_only:
        logger.info("Save-only mode complete; skipping PPL and zero-shot evaluation.")
        return

    dataset = data_utils.get_dataset(args.ppl_tasks[0])
    test_loader = data_utils.prepare_test_dataloader(dataset=dataset["test"], tokenizer=tokenizer, batch_size=1)

    input_device = distribute_model_for_eval(model, logger)
    model._hif4_input_device = input_device

    logger.info("Starting PPL evaluation...")
    ppl = eval_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logger.info("PPL: %.4f", ppl)

    if args.test_zero_task:
        logger.info("Starting zero-shot evaluation...")
        eval_utils.eval_zero_shot_task(model, tokenizer, args.tasks, logger)


if __name__ == "__main__":
    cli_args = arg_parser()
    cli_logger = configure_logging()
    run_main(cli_args, cli_logger)
