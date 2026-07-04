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


def _save_quantized_model(model, path: str, tokenizer=None) -> None:
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=False, max_shard_size="5GB")
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
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


def _hif4_weight_qtype(weight_format: str) -> str:
    mapping = {
        "hif4": "hifx4",
        "hif4-1": "hifx4_1",
    }
    if weight_format not in mapping:
        raise ValueError(f"Unsupported hif4 weight format: {weight_format}")
    return mapping[weight_format]


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
    hif4_qtype = QType("hifx4")

    if isinstance(module, nn.Linear):
        if _is_excluded_layer("", args.exclude_layers):
            return module
        new_module = QLinear2(module.in_features, module.out_features, module.bias is not None)
        new_module.transfer(module)
        new_module.assign_qparams(hif4_qtype)
        new_module.assign_input_qparams(hif4_qtype)
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
            new_module.assign_qparams(hif4_qtype)
            new_module.assign_input_qparams(hif4_qtype)
            new_module.set_quant_grad(False)
            new_module._fast_forward = not args.disable_fast_forward
            parent_name = ".".join(name.split(".")[:-1])
            parent_module = module_dict[parent_name]
            setattr(parent_module, name.split(".")[-1], new_module)
            replaced_layers += 1

    logging.info("Replaced %s Linear layers with HiF4 activation-only QLinear2.", replaced_layers)
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

    parser.add_argument("--hif4w", type=str2bool, default=False, help="Enable one-shot HiF4 weight fake quantization")
    parser.add_argument(
        "--hif4_weight_format",
        type=str,
        default="hif4",
        choices=["hif4", "hif4-1"],
        help="HiF4 weight fake quant format for RTN/GPTQ.",
    )
    parser.add_argument("--hif4a", type=str2bool, default=False, help="Enable HiF4 input activation fake quantization")
    parser.add_argument("--exclude-layers", nargs="*", default=["lm_head"], help="Exact layer names to skip")
    parser.add_argument("--disable-fast-forward", action="store_true", help="Disable QLinear2 fast-forward path")

    parser.add_argument("--gptq", type=str2bool, default=False)
    parser.add_argument("--gptq_percdamp", type=float, default=0.01)
    parser.add_argument(
        "--gptq_cal_dataset",
        type=str,
        default="c4",
        choices=["wikitext2", "ptb", "c4"],
    )
    parser.add_argument("--gptq_cal_nsamples", type=int, default=512)
    parser.add_argument("--gptq_cal_seqlen", type=int, default=512)
    parser.add_argument("--gptq_load_path", type=str2path, default=None)
    parser.add_argument("--gptq_save_path", type=str2path, default=None)
    parser.add_argument("--block_size_linear", type=int, default=64)
    parser.add_argument("--smoothquant", type=str2bool, default=False)
    parser.add_argument("--smoothquant_alpha", type=float, default=0.5)
    parser.add_argument("--awq", type=str2bool, default=False)
    parser.add_argument("--awq_n_grid", type=int, default=20)
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
    if args.hif4_weight_qtype == "hifx4_1" and args.gptq and args.block_size_linear != 64:
        raise ValueError("hif4-1 GPTQ requires --block_size_linear 64.")
    enabled_weight_methods = sum(bool(x) for x in (args.gptq, args.smoothquant, args.awq, args.flatquant))
    if enabled_weight_methods > 1:
        raise ValueError("--gptq, --smoothquant, --awq, and --flatquant cannot be enabled at the same time.")

    dtype = _torch_dtype_from_arg(args.dtype)
    load_device_map = "cpu"
    if args.hif4w and not args.gptq and not args.smoothquant and not args.awq and not args.flatquant:
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
                args.gptq_cal_dataset,
                nsamples=args.gptq_cal_nsamples,
                seqlen=args.gptq_cal_seqlen,
                model=args.model,
                eval_mode=False,
            )
            gptq_utils.gptq_fwrd(model, trainloader, _quant_device(), args)

        if args.gptq_save_path:
            _save_quantized_model(model, args.gptq_save_path)
    elif args.smoothquant:
        if args.hif4w:
            logger.info("Both --hif4w and --smoothquant are enabled; SmoothQuant controls weight quantization and HiF4 RTN is skipped.")

        from hif4smoothquant import smoothquant_utils
        import brq.calib as calib

        logger.info("Quantizing model weights with HiFloat4 SmoothQuant.")
        trainloader = calib.get_loaders(
            args.gptq_cal_dataset,
            nsamples=args.gptq_cal_nsamples,
            seqlen=args.gptq_cal_seqlen,
            model=args.model,
            eval_mode=False,
        )
        smoothquant_utils.smoothquant_fwrd(model, trainloader, _quant_device(), args)

        if args.gptq_save_path:
            _save_quantized_model(model, args.gptq_save_path, tokenizer)
    elif args.awq:
        if args.hif4w:
            logger.info("Both --hif4w and --awq are enabled; AWQ controls weight quantization and HiF4 RTN is skipped.")

        from hif4awq import awq_utils
        import brq.calib as calib

        logger.info("Quantizing model weights with HiFloat4 AWQ.")
        trainloader = calib.get_loaders(
            args.gptq_cal_dataset,
            nsamples=args.gptq_cal_nsamples,
            seqlen=args.gptq_cal_seqlen,
            model=args.model,
            eval_mode=False,
        )
        awq_utils.awq_fwrd(model, trainloader, _quant_device(), args)

        if args.gptq_save_path:
            _save_quantized_model(model, args.gptq_save_path, tokenizer)
    elif args.flatquant:
        if args.hif4w:
            logger.info("Both --hif4w and --flatquant are enabled; FlatQuant controls weight quantization and HiF4 RTN is skipped.")

        from hif4flatquant import flatquant_fwrd, save_hif4_flatquant_model
        import brq.calib as calib

        logger.info("Quantizing model weights and Linear inputs with HiFloat4 FlatQuant.")
        trainloader = calib.get_loaders(
            args.gptq_cal_dataset,
            nsamples=args.gptq_cal_nsamples,
            seqlen=args.gptq_cal_seqlen,
            model=args.model,
            eval_mode=False,
        )
        flatquant_fwrd(model, trainloader, _quant_device(), args)
        save_hif4_flatquant_model(model, tokenizer, args.gptq_save_path, args)
    elif args.hif4w:
        logger.info("Quantizing model weights with one-shot HiF4 RTN.")
        model = hif4_rtn_quant(model, args)
        if args.gptq_save_path:
            _save_quantized_model(model, args.gptq_save_path)

    if args.hif4a:
        logger.info("Replacing Linear layers with HiF4 activation-only QLinear2.")
        model = replace_linear_with_hif4_activation_quant(model, args)

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
