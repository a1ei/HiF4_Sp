from __future__ import annotations

import fnmatch
import inspect
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .calibration import (
    ReasoningCalibrationSample,
    load_reasoning_calibration_samples,
)
from .logic_tokens import LogicTokenMatcher, load_logic_keywords
from .optimizer import (
    ActivationStatistics,
    build_logic_gram,
    optimize_linear_group,
    resolve_group_size,
    select_spectral_subspace,
)


_SUPPORTED_MODEL_TYPES = {"llama", "qwen3", "qwen3_5_text"}


@dataclass(frozen=True)
class LGQConfig:
    calibration_sequence_length: int
    subspace_mode: str
    subspace_rank: int
    low_mid_start_quantile: float
    steps: int
    log_interval: int
    learning_rate: float
    lambda_logic: float
    lambda_reg: float
    group_size: int
    group_loss: str
    group_smooth_tau: float
    target_patterns: tuple[str, ...]
    artifact_dir: str | None
    save_mode: str

    @classmethod
    def from_args(cls, args) -> "LGQConfig":
        return cls(
            calibration_sequence_length=args.lgq_calib_seq_len,
            subspace_mode=args.lgq_subspace_mode,
            subspace_rank=args.lgq_subspace_rank,
            low_mid_start_quantile=args.lgq_low_mid_start_quantile,
            steps=args.lgq_steps,
            log_interval=args.lgq_log_interval,
            learning_rate=args.lgq_lr,
            lambda_logic=args.lgq_lambda_logic,
            lambda_reg=args.lgq_lambda_reg,
            group_size=resolve_group_size(
                args.hif4_weight_format,
                args.lgq_group_size,
            ),
            group_loss=args.lgq_group_loss,
            group_smooth_tau=args.lgq_group_smooth_tau,
            target_patterns=tuple(args.lgq_target_patterns),
            artifact_dir=args.lgq_artifact_dir,
            save_mode=args.lgq_save_mode,
        )


class _CaptureComplete(RuntimeError):
    pass


def _move_to_device(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }
    return value


def _move_to_cpu(value: Any):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_move_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    return value


def _filter_forward_kwargs(module: nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(module.forward)
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    accepted = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in accepted}


def _extract_hidden_states(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _make_causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(hidden_states):
        raise TypeError("LGQ causal mask requires floating-point hidden states.")
    batch_size, sequence_length, _ = hidden_states.shape
    mask = torch.full(
        (sequence_length, sequence_length),
        torch.finfo(hidden_states.dtype).min,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, sequence_length, sequence_length).expand(
        batch_size,
        1,
        sequence_length,
        sequence_length,
    )


def _layer_kwargs(
    model,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    base_kwargs: dict[str, Any],
) -> dict[str, Any]:
    kwargs = dict(base_kwargs)
    if getattr(model.config, "model_type", "") == "qwen3_5_text":
        layer_type = getattr(layer, "layer_type", None)
        if layer_type == "linear_attention":
            kwargs["attention_mask"] = None
        elif layer_type == "full_attention":
            attention_backend = getattr(
                model.config,
                "_attn_implementation",
                None,
            )
            if attention_backend == "flash_attention_2":
                # Calibration samples are unpadded. FlashAttention applies
                # causality internally and only accepts a 2D padding mask.
                kwargs["attention_mask"] = None
            else:
                kwargs["attention_mask"] = _make_causal_mask(hidden_states)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer_type!r}.")
    kwargs["use_cache"] = False
    kwargs["past_key_values"] = None
    return _filter_forward_kwargs(layer, kwargs)


@torch.no_grad()
def _run_layer(
    model,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    base_kwargs: dict[str, Any],
) -> torch.Tensor:
    kwargs = _layer_kwargs(model, layer, hidden_states, base_kwargs)
    kwargs = _move_to_device(kwargs, hidden_states.device)
    return _extract_hidden_states(layer(hidden_states, **kwargs))


class _FirstLayerCapture(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.hidden_states: torch.Tensor | None = None
        self.layer_kwargs: dict[str, Any] | None = None

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        self.hidden_states = hidden_states.detach().cpu()
        self.layer_kwargs = _move_to_cpu(kwargs)
        raise _CaptureComplete


@torch.no_grad()
def _capture_first_layer_inputs(
    model,
    samples: list[ReasoningCalibrationSample],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    core = getattr(model, "model", None)
    if core is None or not hasattr(core, "layers") or not hasattr(core, "embed_tokens"):
        raise NotImplementedError(
            "hif4LGQ requires a decoder-only model with model.layers and "
            "model.embed_tokens."
        )

    layers = core.layers
    if len(layers) == 0:
        raise ValueError("hif4LGQ cannot run on a model with no decoder layers.")

    moved_modules: list[tuple[str, nn.Module]] = []
    for name in ("embed_tokens", "norm", "rotary_emb"):
        module = getattr(core, name, None)
        if isinstance(module, nn.Module):
            setattr(core, name, module.to(device))
            moved_modules.append((name, module))

    original_first_layer = layers[0]
    catcher = _FirstLayerCapture(original_first_layer.to(device))
    layers[0] = catcher
    hidden_states: list[torch.Tensor] = []
    layer_kwargs: list[dict[str, Any]] = []
    try:
        for sample in samples:
            catcher.hidden_states = None
            catcher.layer_kwargs = None
            attention_mask = torch.ones(
                (1, sample.sequence_length),
                dtype=torch.long,
                device=device,
            )
            try:
                model(
                    input_ids=sample.input_ids.unsqueeze(0).to(device),
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            except _CaptureComplete:
                pass
            else:
                raise RuntimeError("LGQ first-layer capture did not stop model forward.")

            if catcher.hidden_states is None or catcher.layer_kwargs is None:
                raise RuntimeError("LGQ first-layer capture produced no activations.")
            hidden_states.append(catcher.hidden_states)
            layer_kwargs.append(catcher.layer_kwargs)
    finally:
        layers[0] = original_first_layer.cpu()
        for name, module in moved_modules:
            setattr(core, name, module.cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return hidden_states, layer_kwargs


def _known_input_groups(layer: nn.Module) -> list[list[str]]:
    mlp_groups = [
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    if getattr(layer, "layer_type", None) == "linear_attention":
        return [
            [
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.in_proj_b",
                "linear_attn.in_proj_a",
            ],
            ["linear_attn.out_proj"],
            *mlp_groups,
        ]
    return [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        *mlp_groups,
    ]


def _matches_target(
    local_name: str,
    full_name: str,
    target_patterns: tuple[str, ...],
    excluded_layers: list[str],
) -> bool:
    if local_name in excluded_layers or full_name in excluded_layers:
        return False
    return any(
        fnmatch.fnmatchcase(local_name, pattern)
        or fnmatch.fnmatchcase(full_name, pattern)
        for pattern in target_patterns
    )


def _linear_input_groups(
    layer: nn.Module,
    layer_index: int,
    target_patterns: tuple[str, ...],
    excluded_layers: list[str],
) -> list[list[tuple[str, nn.Linear]]]:
    all_linears = {
        name: module
        for name, module in layer.named_modules()
        if name and isinstance(module, nn.Linear)
    }
    selected = {
        name: module
        for name, module in all_linears.items()
        if _matches_target(
            name,
            f"model.layers.{layer_index}.{name}",
            target_patterns,
            excluded_layers,
        )
    }

    groups: list[list[tuple[str, nn.Linear]]] = []
    assigned: set[str] = set()
    for candidate_names in _known_input_groups(layer):
        group = [
            (name, selected[name])
            for name in candidate_names
            if name in selected
        ]
        if group:
            groups.append(group)
            assigned.update(name for name, _ in group)

    for name, module in selected.items():
        if name not in assigned:
            groups.append([(name, module)])
    return groups


class _ArtifactWriter:
    def __init__(
        self,
        config: LGQConfig,
        args,
        keywords: list[str],
    ) -> None:
        self.root = Path(config.artifact_dir) if config.artifact_dir else None
        self.save_delta = config.save_mode in {"delta", "both"}
        self.save_weights = config.save_mode in {"weights", "both"}
        self.delta_manifest: dict[str, str] = {}
        if self.root is None:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        if self.save_delta:
            (self.root / "delta").mkdir(parents=True, exist_ok=True)
        metadata = {
            "lgq_config": asdict(config),
            "command_args": vars(args),
            "logic_keywords": keywords,
        }
        (self.root / "config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        (self.root / "metrics.jsonl").write_text("", encoding="utf-8")

    def write_stats(self, stats) -> None:
        if self.root is None:
            return
        with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stats.to_dict(), sort_keys=True) + "\n")

    def write_delta(self, layer_name: str, delta: torch.Tensor | None) -> None:
        if not self.save_delta:
            return
        if self.root is None or delta is None:
            raise RuntimeError("LGQ delta saving was requested without delta data.")
        from safetensors.torch import save_file

        relative_path = Path("delta") / f"{layer_name}.safetensors"
        save_file(
            {f"{layer_name}.weight": delta.contiguous()},
            str(self.root / relative_path),
        )
        self.delta_manifest[f"{layer_name}.weight"] = str(relative_path)

    def finish(self, model, tokenizer, safe_serialization: bool) -> None:
        if self.root is None:
            return
        if self.save_delta:
            (self.root / "delta_manifest.json").write_text(
                json.dumps(self.delta_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if self.save_weights:
            output_dir = self.root / "optimized_model"
            model.save_pretrained(
                output_dir,
                safe_serialization=safe_serialization,
                max_shard_size="5GB",
            )
            tokenizer.save_pretrained(output_dir)


def _log_stats(logger: logging.Logger, stats) -> None:
    logger.info(
        "(hif4LGQ) %s H_tokens=%d logic_tokens=%d group=%d "
        "spectrum=[%.6e, %.6e] selected=[%.6e, %.6e] "
        "loss(group %.6e->%.6e logic=%.6e reg=%.6e total=%.6e) "
        "norm(C=%.6e delta=%.6e) "
        "group_amax(mean %.6e->%.6e, max %.6e->%.6e)",
        stats.layer_name,
        stats.hessian_tokens,
        stats.logic_tokens,
        stats.group_size,
        stats.spectrum_min,
        stats.spectrum_max,
        stats.selected_spectrum_min,
        stats.selected_spectrum_max,
        stats.initial_group_loss,
        stats.final_group_loss,
        stats.final_logic_loss,
        stats.final_reg_loss,
        stats.final_total_loss,
        stats.c_frobenius_norm,
        stats.delta_frobenius_norm,
        stats.initial_group_amax_mean,
        stats.final_group_amax_mean,
        stats.initial_group_amax_max,
        stats.final_group_amax_max,
    )


@torch.no_grad()
def _collect_layer_statistics(
    model,
    layer: nn.Module,
    hidden_states: list[torch.Tensor],
    layer_kwargs: list[dict[str, Any]],
    samples: list[ReasoningCalibrationSample],
    groups: list[list[tuple[str, nn.Linear]]],
    device: torch.device,
) -> list[ActivationStatistics]:
    statistics = [ActivationStatistics() for _ in groups]
    active_masks: dict[str, torch.Tensor | None] = {
        "valid": None,
        "logic": None,
    }
    handles = []

    for group, group_stats in zip(groups, statistics):
        representative = group[0][1]

        def hook(module, inputs, current_stats=group_stats):
            del module
            if not inputs:
                raise RuntimeError("LGQ Linear pre-hook received no positional input.")
            if active_masks["valid"] is None or active_masks["logic"] is None:
                raise RuntimeError("LGQ token masks were not set before layer forward.")
            current_stats.add(
                inputs[0],
                active_masks["valid"],
                active_masks["logic"],
            )

        handles.append(representative.register_forward_pre_hook(hook))

    try:
        for hidden, kwargs, sample in zip(hidden_states, layer_kwargs, samples):
            layer_input = hidden.to(device)
            active_masks["valid"] = torch.ones(
                sample.sequence_length,
                dtype=torch.bool,
                device=device,
            )
            active_masks["logic"] = sample.logic_mask.to(device)
            _run_layer(model, layer, layer_input, kwargs)
    finally:
        for handle in handles:
            handle.remove()

    expected_tokens = sum(sample.sequence_length for sample in samples)
    for group, group_stats in zip(groups, statistics):
        if group_stats.token_count != expected_tokens:
            names = [name for name, _ in group]
            raise RuntimeError(
                "LGQ representative Linear was not called exactly once per sample: "
                f"group={names}, expected_tokens={expected_tokens}, "
                f"collected_tokens={group_stats.token_count}."
            )
        if group_stats.logic_token_count == 0:
            names = [name for name, _ in group]
            raise ValueError(
                f"LGQ found no logic token activations for Linear group {names}."
            )
        group_stats.offload_hessian()
    return statistics


@torch.no_grad()
def _forward_updated_layer(
    model,
    layer: nn.Module,
    hidden_states: list[torch.Tensor],
    layer_kwargs: list[dict[str, Any]],
    device: torch.device,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for hidden, kwargs in zip(hidden_states, layer_kwargs):
        output = _run_layer(model, layer, hidden.to(device), kwargs)
        outputs.append(output.detach().cpu())
    return outputs


def run_hif4lgq(
    model,
    tokenizer,
    device: torch.device,
    args,
    logger: logging.Logger | None = None,
) -> None:
    logger = logger or logging.getLogger("hif4")
    model_type = getattr(model.config, "model_type", "")
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"hif4LGQ does not support model type {model_type!r}; "
            f"supported={sorted(_SUPPORTED_MODEL_TYPES)}."
        )
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise NotImplementedError("hif4LGQ requires model.model.layers.")

    config = LGQConfig.from_args(args)
    keywords = load_logic_keywords(args.lgq_logic_keywords_path)
    matcher = LogicTokenMatcher(tokenizer, keywords)
    samples = load_reasoning_calibration_samples(
        tokenizer=tokenizer,
        matcher=matcher,
        dataset_name=args.cal_dataset,
        num_samples=args.cal_nsamples,
        sequence_length=config.calibration_sequence_length,
        seed=args.seed,
    )
    total_tokens = sum(sample.sequence_length for sample in samples)
    total_logic_tokens = sum(
        int(sample.logic_mask.sum().item())
        for sample in samples
    )
    logger.info(
        "(hif4LGQ) Loaded %d reasoning samples: valid_tokens=%d logic_tokens=%d "
        "calib_seq_len=%d.",
        len(samples),
        total_tokens,
        total_logic_tokens,
        config.calibration_sequence_length,
    )

    artifact_writer = _ArtifactWriter(config, args, keywords)
    previous_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    model.eval()

    try:
        hidden_states, layer_kwargs = _capture_first_layer_inputs(
            model,
            samples,
            device,
        )
        layers = model.model.layers
        for layer_index in range(len(layers)):
            layer = layers[layer_index].to(device)
            groups = _linear_input_groups(
                layer,
                layer_index,
                config.target_patterns,
                args.exclude_layers,
            )
            if not groups:
                logger.info(
                    "(hif4LGQ) Skipping model.layers.%d: no target Linear matched.",
                    layer_index,
                )
                hidden_states = _forward_updated_layer(
                    model,
                    layer,
                    hidden_states,
                    layer_kwargs,
                    device,
                )
                layers[layer_index] = layer.cpu()
                continue

            logger.info(
                "(hif4LGQ) Processing model.layers.%d with %d Linear input groups.",
                layer_index,
                len(groups),
            )
            statistics = _collect_layer_statistics(
                model,
                layer,
                hidden_states,
                layer_kwargs,
                samples,
                groups,
                device,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for group, group_stats in zip(groups, statistics):
                if group_stats.hessian is None:
                    raise RuntimeError("LGQ Hessian disappeared before optimization.")
                hessian = group_stats.hessian.to(device)
                basis, spectrum = select_spectral_subspace(
                    hessian,
                    config.subspace_rank,
                    config.subspace_mode,
                    config.low_mid_start_quantile,
                )
                logic_gram = build_logic_gram(
                    group_stats.logic_chunks,
                    basis,
                    group_stats.logic_token_count,
                )

                named_linears = [
                    (f"model.layers.{layer_index}.{local_name}", linear)
                    for local_name, linear in group
                ]
                optimization_results = optimize_linear_group(
                    linears=named_linears,
                    basis=basis,
                    logic_gram=logic_gram,
                    spectrum=spectrum,
                    hessian_tokens=group_stats.token_count,
                    logic_tokens=group_stats.logic_token_count,
                    group_size=config.group_size,
                    steps=config.steps,
                    learning_rate=config.learning_rate,
                    lambda_logic=config.lambda_logic,
                    lambda_reg=config.lambda_reg,
                    group_loss_mode=config.group_loss,
                    group_smooth_tau=config.group_smooth_tau,
                    return_delta=artifact_writer.save_delta,
                    log_interval=config.log_interval,
                    logger=logger,
                )
                for (full_name, _), (stats, delta) in zip(
                    named_linears,
                    optimization_results,
                    strict=True,
                ):
                    _log_stats(logger, stats)
                    artifact_writer.write_stats(stats)
                    artifact_writer.write_delta(full_name, delta)

                group_stats.free()
                del hessian, basis, logic_gram
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            hidden_states = _forward_updated_layer(
                model,
                layer,
                hidden_states,
                layer_kwargs,
                device,
            )
            layers[layer_index] = layer.cpu()
            del statistics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache

    artifact_writer.finish(
        model,
        tokenizer,
        safe_serialization=bool(args.safe_serialization),
    )
    logger.info("(hif4LGQ) Optimization complete.")
