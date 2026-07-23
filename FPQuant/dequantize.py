"""Export FP-Quant NVFP4 weights after dequantization and Hadamard folding."""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INDEX_FILE = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_SIZE = 5 * 1024**3
NVFP4_GROUP_SIZE = 16

_FP4_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

_OUTPUT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class ConversionEstimate:
    checkpoint_bytes: int
    auxiliary_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.checkpoint_bytes + self.auxiliary_bytes


def _decode_e4m3_scales(scales: torch.Tensor) -> torch.Tensor:
    if scales.dtype == torch.uint8:
        return scales.contiguous().view(torch.float8_e4m3fn).to(torch.float32)
    if scales.dtype == torch.float8_e4m3fn:
        return scales.to(torch.float32)
    raise TypeError(
        "FP-Quant NVFP4 scales must contain E4M3 bit patterns as uint8 or "
        f"float8_e4m3fn, got {scales.dtype}"
    )


@torch.no_grad()
def dequantize_fpquant_nvfp4_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    weight_global_scale: torch.Tensor,
    forward_hadamard_matrix: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one FP-Quant NVFP4 Linear and fold its input transform."""
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported output dtype: {output_dtype}")
    if qweight.dtype != torch.uint8 or qweight.ndim != 2:
        raise TypeError(
            "qweight must be a 2D uint8 tensor, got "
            f"shape={tuple(qweight.shape)}, dtype={qweight.dtype}"
        )
    if scales.ndim != 2:
        raise ValueError(f"scales must be 2D, got shape={tuple(scales.shape)}")
    if weight_global_scale.numel() != 1:
        raise ValueError(
            "weight_global_scale must be scalar, got "
            f"shape={tuple(weight_global_scale.shape)}"
        )
    if (
        forward_hadamard_matrix.ndim != 2
        or forward_hadamard_matrix.shape[0]
        != forward_hadamard_matrix.shape[1]
    ):
        raise ValueError(
            "forward_hadamard_matrix must be square, got "
            f"shape={tuple(forward_hadamard_matrix.shape)}"
        )

    out_features, packed_in_features = qweight.shape
    in_features = packed_in_features * 2
    expected_scale_shape = (out_features, in_features // NVFP4_GROUP_SIZE)
    if in_features % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by {NVFP4_GROUP_SIZE}"
        )
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scales must have shape {expected_scale_shape}, got {tuple(scales.shape)}"
        )

    hadamard_size = forward_hadamard_matrix.shape[0]
    if in_features % hadamard_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by Hadamard size {hadamard_size}"
        )

    global_scale = weight_global_scale.reshape(()).to(torch.float32)
    if not torch.isfinite(global_scale) or global_scale <= 0:
        raise ValueError(f"Invalid weight_global_scale: {global_scale.item()}")

    packed = qweight.contiguous()
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(out_features, in_features)
    fp4 = _FP4_E2M1_VALUES.to(qweight.device)[nibbles.long()]

    group_scales = _decode_e4m3_scales(scales).to(qweight.device)
    rotated = (
        fp4.reshape(out_features, -1, NVFP4_GROUP_SIZE)
        * group_scales.unsqueeze(-1)
        / global_scale
    ).reshape(out_features, in_features)

    hadamard = forward_hadamard_matrix.to(device=qweight.device, dtype=torch.float32)
    folded = (
        rotated.reshape(out_features, -1, hadamard_size)
        .matmul(hadamard.transpose(0, 1))
        .reshape(out_features, in_features)
    )
    if torch.any(~torch.isfinite(folded)):
        raise ValueError("Dequantized FP-Quant weight contains non-finite values")
    return folded.to(output_dtype).contiguous()


def convert_fpquant_nvfp4_checkpoint(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    output_dtype: str,
    overwrite: bool = False,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> Path:
    """Convert an FP-Quant NVFP4 checkpoint into ordinary HF weights."""
    if output_dtype not in _OUTPUT_DTYPES:
        raise ValueError(
            f"output_dtype must be one of {sorted(_OUTPUT_DTYPES)}, got {output_dtype}"
        )
    if max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive")

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    _validate_paths(input_path, output_path, overwrite)
    quant_config = _validate_fpquant_config(input_path)
    index = _load_index(input_path)
    bases = _validate_weight_groups(index["weight_map"])
    estimate = estimate_converted_checkpoint_size(input_path, output_dtype)
    _check_free_space(output_path, estimate)

    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    try:
        _copy_auxiliary_files(input_path, output_path)
        _write_converted_shards(
            input_path=input_path,
            output_path=output_path,
            weight_map=index["weight_map"],
            quantized_bases=bases,
            output_dtype=_OUTPUT_DTYPES[output_dtype],
            max_shard_size=max_shard_size,
        )
        _rewrite_config(output_path, quant_config, output_dtype)
        _normalize_qwen3_generation_config(output_path)
    except Exception:
        shutil.rmtree(output_path, ignore_errors=True)
        raise

    return output_path


def estimate_converted_checkpoint_size(
    input_dir: str | os.PathLike[str],
    output_dtype: str,
) -> ConversionEstimate:
    if output_dtype not in _OUTPUT_DTYPES:
        raise ValueError(
            f"output_dtype must be one of {sorted(_OUTPUT_DTYPES)}, got {output_dtype}"
        )
    input_path = Path(input_dir).resolve()
    index = _load_index(input_path)
    weight_map = index["weight_map"]
    bases = _quantized_bases(weight_map)
    removed = _removed_quantized_keys(bases)
    output_element_size = torch.empty((), dtype=_OUTPUT_DTYPES[output_dtype]).element_size()

    checkpoint_bytes = 0
    for filename in sorted(set(weight_map.values())):
        header = _read_safetensors_header(input_path / filename)
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(".qweight"):
                checkpoint_bytes += _numel(info["shape"]) * 2 * output_element_size
            elif key in removed:
                continue
            else:
                checkpoint_bytes += _numel(info["shape"]) * _dtype_nbytes(info["dtype"])

    auxiliary_bytes = sum(
        path.stat().st_size
        for path in input_path.iterdir()
        if not _is_checkpoint_file(path) and (path.is_file() or path.is_symlink())
    )
    return ConversionEstimate(checkpoint_bytes, auxiliary_bytes)


def _validate_paths(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_path}. Use --overwrite to replace it."
        )
    try:
        output_path.relative_to(input_path)
    except ValueError:
        return
    raise ValueError("output_dir must not be inside input_dir")


def _validate_fpquant_config(input_path: Path) -> dict:
    config_path = input_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError("config.json has no quantization_config")
    if quant_config.get("quant_method") != "fp_quant":
        raise ValueError(
            "Expected quant_method=fp_quant, got "
            f"{quant_config.get('quant_method')}"
        )
    if quant_config.get("forward_dtype") != "nvfp4":
        raise ValueError(
            "Expected forward_dtype=nvfp4, got "
            f"{quant_config.get('forward_dtype')}"
        )
    if quant_config.get("store_master_weights") is not False:
        raise ValueError("Only finalized checkpoints with store_master_weights=false are supported")
    return quant_config


def _load_index(input_path: Path) -> dict:
    index_path = input_path / INDEX_FILE
    if not index_path.is_file():
        single_file = input_path / "model.safetensors"
        if not single_file.is_file():
            raise FileNotFoundError(f"Missing {INDEX_FILE}: {index_path}")
        header = _read_safetensors_header(single_file)
        return {
            "weight_map": {
                key: single_file.name for key in header if key != "__metadata__"
            }
        }
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"{INDEX_FILE} has no weight_map")
    return index


def _quantized_bases(weight_map: dict[str, str]) -> set[str]:
    return {
        key[: -len("qweight")]
        for key in weight_map
        if key.endswith(".qweight")
    }


def _validate_weight_groups(weight_map: dict[str, str]) -> set[str]:
    bases = _quantized_bases(weight_map)
    if not bases:
        raise ValueError("No FP-Quant qweight tensors found")
    required_suffixes = (
        "qweight",
        "scales",
        "weight_global_scale",
        "forward_hadamard_matrix",
    )
    for base in sorted(bases):
        for suffix in required_suffixes:
            key = base + suffix
            if key not in weight_map:
                raise ValueError(f"Missing required tensor: {key}")
    return bases


def _removed_quantized_keys(bases: set[str]) -> set[str]:
    return {
        base + suffix
        for base in bases
        for suffix in (
            "qweight",
            "scales",
            "weight_global_scale",
            "act_global_scale",
            "forward_hadamard_matrix",
            "backward_hadamard_matrix",
        )
    }


def _write_converted_shards(
    input_path: Path,
    output_path: Path,
    weight_map: dict[str, str],
    quantized_bases: set[str],
    output_dtype: torch.dtype,
    max_shard_size: int,
) -> None:
    removed = _removed_quantized_keys(quantized_bases)
    output_weight_map: dict[str, str] = {}
    output_total_size = 0
    output_shard: dict[str, torch.Tensor] = {}
    output_shard_bytes = 0
    output_shard_index = 1

    def flush() -> None:
        nonlocal output_shard, output_shard_bytes, output_shard_index
        if not output_shard:
            return
        filename = f"model-{output_shard_index:05d}-of-00000.safetensors"
        save_file(output_shard, output_path / filename, metadata={"format": "pt"})
        for tensor_name in output_shard:
            output_weight_map[tensor_name] = filename
        output_shard = {}
        output_shard_bytes = 0
        output_shard_index += 1

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        nonlocal output_shard_bytes, output_total_size
        tensor = tensor.contiguous()
        tensor_bytes = tensor.numel() * tensor.element_size()
        if output_shard and output_shard_bytes + tensor_bytes > max_shard_size:
            flush()
        output_shard[name] = tensor
        output_shard_bytes += tensor_bytes
        output_total_size += tensor_bytes

    for filename in sorted(set(weight_map.values())):
        with safe_open(input_path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith(".qweight"):
                    base = key[: -len("qweight")]
                    weight = dequantize_fpquant_nvfp4_weight(
                        qweight=handle.get_tensor(key),
                        scales=_get_tensor(
                            input_path, weight_map, base + "scales", filename, handle
                        ),
                        weight_global_scale=_get_tensor(
                            input_path,
                            weight_map,
                            base + "weight_global_scale",
                            filename,
                            handle,
                        ),
                        forward_hadamard_matrix=_get_tensor(
                            input_path,
                            weight_map,
                            base + "forward_hadamard_matrix",
                            filename,
                            handle,
                        ),
                        output_dtype=output_dtype,
                    )
                    add_tensor(base + "weight", weight)
                elif key in removed:
                    continue
                else:
                    add_tensor(key, handle.get_tensor(key))

    flush()
    total_shards = output_shard_index - 1
    renamed_weight_map = _rename_output_shards(
        output_path, output_weight_map, total_shards
    )
    with (output_path / INDEX_FILE).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {"total_size": output_total_size},
                "weight_map": renamed_weight_map,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")


def _get_tensor(
    input_path: Path,
    weight_map: dict[str, str],
    key: str,
    current_filename: str,
    current_handle,
) -> torch.Tensor:
    filename = weight_map[key]
    if filename == current_filename:
        return current_handle.get_tensor(key)
    with safe_open(input_path / filename, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _rename_output_shards(
    output_path: Path, weight_map: dict[str, str], total_shards: int
) -> dict[str, str]:
    renamed: dict[str, str] = {}
    for old_index in range(1, total_shards + 1):
        old_name = f"model-{old_index:05d}-of-00000.safetensors"
        new_name = f"model-{old_index:05d}-of-{total_shards:05d}.safetensors"
        (output_path / old_name).rename(output_path / new_name)
        for key, filename in weight_map.items():
            if filename == old_name:
                renamed[key] = new_name
    return renamed


def _copy_auxiliary_files(input_path: Path, output_path: Path) -> None:
    for src in input_path.iterdir():
        if _is_checkpoint_file(src):
            continue
        dst = output_path / src.name
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=False)
        else:
            shutil.copy2(src, dst, follow_symlinks=True)


def _rewrite_config(
    output_path: Path,
    source_quant_config: dict,
    output_dtype: str,
) -> None:
    config_path = output_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    if "torch_dtype" in config:
        config["torch_dtype"] = "bfloat16"
    config["dequantization_config"] = {
        "source_quant_method": source_quant_config["quant_method"],
        "source_weight_dtype": source_quant_config["forward_dtype"],
        "linear_weight_dtype": output_dtype,
        "hadamard_folded": True,
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _normalize_qwen3_generation_config(output_path: Path) -> None:
    config_path = output_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_type") != "qwen3":
        return

    generation_path = output_path / "generation_config.json"
    tokenizer_path = output_path / "tokenizer_config.json"
    if not generation_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(
            "Qwen3 export requires generation_config.json and tokenizer_config.json"
        )

    bos_token_id = config.get("bos_token_id")
    model_eos_token_id = config.get("eos_token_id")
    if not isinstance(bos_token_id, int) or not isinstance(model_eos_token_id, int):
        raise ValueError("Qwen3 config must contain integer BOS and EOS token IDs")

    with tokenizer_path.open("r", encoding="utf-8") as handle:
        tokenizer_config = json.load(handle)
    bos_token = tokenizer_config.get("added_tokens_decoder", {}).get(
        str(bos_token_id), {}
    )
    if bos_token.get("content") != "<|endoftext|>":
        raise ValueError(
            f"Qwen3 BOS token {bos_token_id} is not the expected <|endoftext|> token"
        )
    tokenizer_config["pad_token"] = "<|endoftext|>"
    with tokenizer_path.open("w", encoding="utf-8") as handle:
        json.dump(tokenizer_config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    special_tokens_path = output_path / "special_tokens_map.json"
    if special_tokens_path.is_file():
        with special_tokens_path.open("r", encoding="utf-8") as handle:
            special_tokens = json.load(handle)
        special_tokens["pad_token"] = bos_token
        with special_tokens_path.open("w", encoding="utf-8") as handle:
            json.dump(special_tokens, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    with generation_path.open("r", encoding="utf-8") as handle:
        generation_config = json.load(handle)
    configured_eos = generation_config.get("eos_token_id", model_eos_token_id)
    eos_token_ids = (
        [configured_eos] if isinstance(configured_eos, int) else list(configured_eos)
    )
    if not all(isinstance(token_id, int) for token_id in eos_token_ids):
        raise TypeError("generation_config eos_token_id must contain integers")

    generation_config["eos_token_id"] = list(
        dict.fromkeys([model_eos_token_id, *eos_token_ids, bos_token_id])
    )
    generation_config["pad_token_id"] = bos_token_id
    with generation_path.open("w", encoding="utf-8") as handle:
        json.dump(generation_config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    chat_template_source = Path(__file__).with_name("qwen3_chat_template.jinja")
    if not chat_template_source.is_file():
        raise FileNotFoundError(f"Missing Qwen3 chat template: {chat_template_source}")
    shutil.copy2(chat_template_source, output_path / "chat_template.jinja")


def _check_free_space(output_path: Path, estimate: ConversionEstimate) -> None:
    parent = output_path.parent
    while not parent.exists():
        parent = parent.parent
    free_bytes = shutil.disk_usage(parent).free
    required_bytes = estimate.total_bytes + 1024**3
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Not enough disk space: "
            f"required={required_bytes / 1024**3:.2f} GiB, "
            f"available={free_bytes / 1024**3:.2f} GiB"
        )


def _is_checkpoint_file(path: Path) -> bool:
    return path.name == INDEX_FILE or path.suffix in {".safetensors", ".bin"}


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_len))


def _numel(shape: list[int]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _dtype_nbytes(dtype: str) -> int:
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E4M3FN": 1,
        "F8_E5M2": 1,
        "F16": 2,
        "BF16": 2,
        "I16": 2,
        "U16": 2,
        "F32": 4,
        "I32": 4,
        "U32": 4,
        "F64": 8,
        "I64": 8,
        "U64": 8,
    }
    normalized = dtype.upper()
    if normalized not in sizes:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return sizes[normalized]
