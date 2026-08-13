"""Reasoning-safe low-rank RMSNorm activation adaptation for Qwen3.5.

The module is independent of GPTQ.  It consumes the W4 model produced by the
existing GPTQ entry and uses the existing HiF4 activation fake quantizer.
"""

from __future__ import annotations

import copy
import contextlib
import functools
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from gptq.gptq_utils import (
    _capture_calibration_inputs,
    _get_layers,
    _layer_kwargs_for_current_layer,
    _run_layer,
)
from hif4_gpu.quant_cy import QType, quant_func


NORM_NAMES = ("input_layernorm", "post_attention_layernorm")
MODES = ("baseline", "naive_lowrank", "reasoning_projected_lowrank")
CURVATURE_CHECKPOINT_FORMAT = "reasoning_safe_rmsnorm_curvature_v1"
CURVATURE_MATRIX_CHECKPOINT_FORMAT = "reasoning_safe_rmsnorm_curvature_matrix_v1"


@dataclass
class LowRankConfig:
    mode: str = "reasoning_projected_lowrank"
    rank: int = 4
    sensitive_rank: int = 64
    curvature_tokens_per_sample: int = 32
    curvature_token_selection: str = "grad_norm_topk"
    epochs: int = 10
    lr: float = 1e-3
    eta: float = 0.05
    max_relative_delta: float | None = None
    init_std: float = 0.02

    def validate(self, hidden_size: int | None = None) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unsupported low-rank mode: {self.mode}")
        if self.rank <= 0 or self.sensitive_rank <= 0:
            raise ValueError("low-rank and sensitive ranks must be positive.")
        if hidden_size is not None and self.sensitive_rank > hidden_size:
            raise ValueError("sensitive_rank cannot exceed hidden_size.")
        if self.curvature_tokens_per_sample <= 0:
            raise ValueError("curvature_tokens_per_sample must be positive.")
        if self.curvature_token_selection != "grad_norm_topk":
            raise ValueError("Only grad_norm_topk curvature selection is supported.")
        if self.epochs <= 0 or self.lr <= 0 or self.eta < 0 or self.init_std <= 0:
            raise ValueError("Invalid low-rank optimization hyperparameters.")
        if self.max_relative_delta is not None and self.max_relative_delta <= 0:
            raise ValueError("max_relative_delta must be positive when set.")


class LowRankRMSNorm(nn.Module):
    """Frozen RMSNorm followed by an optional rank-r residual branch."""

    def __init__(
        self,
        norm: nn.Module,
        config: LowRankConfig,
        sensitive_subspace: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(norm, "weight") or norm.weight.ndim != 1:
            raise TypeError("LowRankRMSNorm requires a one-dimensional RMSNorm weight.")
        self.norm = norm
        self.config = config
        hidden_size = norm.weight.numel()
        config.validate(hidden_size)
        for parameter in self.norm.parameters():
            parameter.requires_grad_(False)

        self.U = nn.Parameter(torch.zeros(hidden_size, config.rank))
        self.V = nn.Parameter(torch.empty(hidden_size, config.rank))
        nn.init.normal_(self.V, mean=0.0, std=config.init_std)
        if sensitive_subspace is None:
            sensitive_subspace = torch.empty(hidden_size, 0)
        if sensitive_subspace.ndim != 2 or sensitive_subspace.shape[0] != hidden_size:
            raise ValueError("Sensitive subspace must have shape [hidden_size, sensitive_rank].")
        if config.mode == "reasoning_projected_lowrank" and (
            sensitive_subspace.shape[1] != config.sensitive_rank
        ):
            raise ValueError("Projected mode requires exactly sensitive_rank columns in S.")
        self.register_buffer("S", sensitive_subspace.detach().float(), persistent=True)
        self.enabled = config.mode != "baseline"
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def safe_u(self) -> torch.Tensor:
        if self.config.mode != "reasoning_projected_lowrank":
            return self.U
        s = self.S.to(device=self.U.device, dtype=self.U.dtype)
        return self.U - s @ (s.transpose(0, 1) @ self.U)

    def transform(self, activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return activation, torch.zeros_like(activation)
        # Keep the small trainable parameters in FP32 for AdamW, but execute the
        # activation-sized matrix products in the model activation dtype.
        safe_u = self.safe_u().to(dtype=activation.dtype)
        v = self.V.to(device=activation.device, dtype=activation.dtype)
        delta = (activation @ v) @ safe_u.transpose(0, 1)
        delta = delta * self.config.eta
        if self.config.max_relative_delta is not None:
            eps = torch.finfo(delta.dtype).eps
            activation_norm = torch.linalg.vector_norm(activation, dim=-1, keepdim=True)
            delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            limit = self.config.max_relative_delta * activation_norm
            scale = torch.minimum(torch.ones_like(delta_norm), limit / (delta_norm + eps))
            delta = delta * scale
        return activation + delta, delta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activation = self.norm(hidden_states)
        output, delta = self.transform(activation)
        eps = 1e-12
        with torch.no_grad():
            relative = torch.linalg.vector_norm(delta.float()) / (
                torch.linalg.vector_norm(activation.float()) + eps
            )
            if self.S.shape[1] and torch.count_nonzero(delta).item():
                projected = delta.float() @ self.S.to(delta.device)
                leakage = torch.linalg.vector_norm(projected) / (
                    torch.linalg.vector_norm(delta.float()) + eps
                )
            else:
                leakage = torch.zeros((), device=activation.device)
            self.last_diagnostics = {
                "relative_activation_change": relative.detach(),
                "sensitive_space_leakage": leakage.detach(),
            }
        return output


def reasoning_nll(logits: torch.Tensor, input_ids: torch.Tensor, reasoning_mask: torch.Tensor) -> torch.Tensor:
    """Mean teacher-forced NLL over reasoning labels with causal shifting."""
    if logits.ndim != 3 or input_ids.shape != reasoning_mask.shape:
        raise ValueError("Expected logits [B,L,V] and matching input_ids/reasoning_mask [B,L].")
    if logits.shape[:2] != input_ids.shape or input_ids.shape[1] < 2:
        raise ValueError("Logit and label sequence shapes do not match.")
    label_mask = reasoning_mask[:, 1:].bool()
    if not label_mask.any():
        raise ValueError("The sample has no causally predictable reasoning labels.")
    token_loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view_as(label_mask)
    return token_loss[label_mask].mean()


def _target_norms(model: nn.Module) -> dict[str, nn.Module]:
    if getattr(model.config, "model_type", "") not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("Reasoning-safe RMSNorm adaptation only supports Qwen3.5.")
    targets = {}
    for layer_idx, layer in enumerate(_get_layers(model)):
        for norm_name in NORM_NAMES:
            norm = getattr(layer, norm_name, None)
            if norm is None:
                raise ValueError(f"Layer {layer_idx} has no {norm_name}.")
            targets[f"layers.{layer_idx}.{norm_name}"] = norm
    return targets


def select_topk_gradients(
    gradient: torch.Tensor, reasoning_mask: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select original gradient vectors by per-token FP32 L2 norm."""
    grad = gradient.detach().float()
    if grad.ndim == 3:
        if grad.shape[0] != 1:
            raise ValueError("Curvature extraction requires per-rank batch size 1.")
        grad = grad[0]
    mask = reasoning_mask.reshape(-1).to(device=grad.device, dtype=torch.bool)
    if grad.ndim != 2 or mask.numel() != grad.shape[0]:
        raise ValueError("Gradient and reasoning mask sequence dimensions differ.")
    valid = grad[mask]
    if valid.shape[0] == 0:
        raise ValueError("No reasoning-position activation gradients were found.")
    norms = torch.linalg.vector_norm(valid, dim=-1)
    count = min(k, norms.numel())
    indices = torch.topk(norms, k=count, largest=True, sorted=False).indices
    return valid[indices], norms, norms[indices]


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_cpu(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not _distributed():
        return tensor
    reduced = tensor.to(device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.cpu()


def _curvature_matrix_manifest(path: str) -> Path:
    return Path(path) / "manifest.pt"


def _save_curvature_matrix_checkpoint(
    path: str,
    model: nn.Module,
    config: LowRankConfig,
    calibration: dict[str, object],
    target_owners: dict[str, int],
    curvature: dict[str, torch.Tensor],
    stats: dict[str, torch.Tensor],
    rank: int,
    world_size: int,
) -> None:
    """Save each rank's raw C=sum(G^T G) shard, then publish one manifest."""
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shard_path = checkpoint_dir / f"rank-{rank:05d}.pt"
    temporary_shard = checkpoint_dir / f"rank-{rank:05d}.pt.tmp"
    torch.save(
        {
            "format": CURVATURE_MATRIX_CHECKPOINT_FORMAT,
            "rank": rank,
            "curvature": curvature,
            "stats": stats,
        },
        temporary_shard,
    )
    os.replace(temporary_shard, shard_path)
    if _distributed():
        dist.barrier()
    if rank == 0:
        manifest = {
            "format": CURVATURE_MATRIX_CHECKPOINT_FORMAT,
            "model_type": model.config.model_type,
            "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "world_size": world_size,
            "curvature_config": {
                "curvature_tokens_per_sample": config.curvature_tokens_per_sample,
                "curvature_token_selection": config.curvature_token_selection,
            },
            "calibration": dict(calibration),
            "target_owners": target_owners,
            "shards": [f"rank-{index:05d}.pt" for index in range(world_size)],
        }
        manifest_path = _curvature_matrix_manifest(path)
        temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        torch.save(manifest, temporary_manifest)
        os.replace(temporary_manifest, manifest_path)
    if _distributed():
        dist.barrier()


def _load_curvature_matrix_checkpoint(
    path: str,
    model: nn.Module,
    config: LowRankConfig,
    calibration: dict[str, object],
    target_owners: dict[str, int],
    rank: int,
    world_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load and strictly validate this rank's raw curvature matrices."""
    manifest = torch.load(
        _curvature_matrix_manifest(path), map_location="cpu", weights_only=False
    )
    if manifest.get("format") != CURVATURE_MATRIX_CHECKPOINT_FORMAT:
        raise ValueError("Unsupported RMSNorm curvature-matrix checkpoint format.")
    for field in ("model_type", "hidden_size", "num_hidden_layers"):
        if manifest.get(field) != getattr(model.config, field):
            raise ValueError(f"Curvature-matrix checkpoint {field} does not match the model.")
    if manifest.get("world_size") != world_size:
        raise ValueError("Curvature-matrix checkpoint world size does not match this run.")
    expected_config = {
        "curvature_tokens_per_sample": config.curvature_tokens_per_sample,
        "curvature_token_selection": config.curvature_token_selection,
    }
    if manifest.get("curvature_config") != expected_config:
        raise ValueError("Curvature-matrix checkpoint extraction parameters do not match.")
    if manifest.get("calibration") != calibration:
        raise ValueError("Curvature-matrix checkpoint calibration parameters do not match.")
    if manifest.get("target_owners") != target_owners:
        raise ValueError("Curvature-matrix checkpoint RMSNorm ownership does not match.")
    expected_shards = [f"rank-{index:05d}.pt" for index in range(world_size)]
    if manifest.get("shards") != expected_shards:
        raise ValueError("Curvature-matrix checkpoint shard list is invalid.")
    checkpoint_dir = Path(path)
    for shard_name in expected_shards:
        if not (checkpoint_dir / shard_name).is_file():
            raise ValueError(f"Missing curvature-matrix shard: {shard_name}")

    shard = torch.load(
        checkpoint_dir / expected_shards[rank], map_location="cpu", weights_only=False
    )
    if shard.get("format") != CURVATURE_MATRIX_CHECKPOINT_FORMAT or shard.get("rank") != rank:
        raise ValueError(f"Invalid curvature-matrix shard for rank {rank}.")
    curvature = shard.get("curvature")
    stats = shard.get("stats")
    expected_names = {name for name, owner in target_owners.items() if owner == rank}
    if not isinstance(curvature, dict) or set(curvature) != expected_names:
        raise ValueError(f"Curvature matrices for rank {rank} do not match expected RMSNorms.")
    if not isinstance(stats, dict) or set(stats) != expected_names:
        raise ValueError(f"Curvature statistics for rank {rank} do not match expected RMSNorms.")
    expected_shape = (model.config.hidden_size, model.config.hidden_size)
    for name in expected_names:
        c = curvature[name]
        stat = stats[name]
        if not torch.is_tensor(c) or c.dtype != torch.float32 or c.shape != expected_shape:
            raise ValueError(f"Raw curvature matrix {name} has invalid dtype or shape.")
        if not torch.isfinite(c).all():
            raise ValueError(f"Raw curvature matrix {name} contains non-finite values.")
        if not torch.is_tensor(stat) or stat.dtype != torch.float64 or stat.shape != (6,):
            raise ValueError(f"Curvature statistics {name} have invalid dtype or shape.")
    return curvature, stats


def extract_sensitive_subspaces(
    model: nn.Module,
    samples: Iterable[dict[str, torch.Tensor]],
    config: LowRankConfig,
    device: torch.device,
    matrix_checkpoint_path: str | None = None,
    calibration: dict[str, object] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, object]], bool]:
    """Extract empirical-Fisher-style RMSNorm activation subspaces."""
    config.validate(model.config.hidden_size)
    if config.mode != "reasoning_projected_lowrank":
        return {}, {}, False
    if any(isinstance(module, LowRankRMSNorm) for module in model.modules()):
        raise RuntimeError("Sensitive subspaces must be extracted from the original BF16 model.")

    distributed_fsdp = _distributed()
    if distributed_fsdp and device.type != "cuda":
        raise RuntimeError("FSDP curvature extraction requires CUDA.")
    targets = _target_norms(model)
    hidden_size = model.config.hidden_size
    rank = dist.get_rank() if distributed_fsdp else 0
    world_size = dist.get_world_size() if distributed_fsdp else 1
    target_owners = {
        name: index % world_size for index, name in enumerate(targets)
    }
    owned_targets = {
        name: module for name, module in targets.items() if target_owners[name] == rank
    }
    calibration = dict(calibration or {})
    matrix_checkpoint_loaded = bool(
        matrix_checkpoint_path is not None
        and _curvature_matrix_manifest(matrix_checkpoint_path).is_file()
    )
    if matrix_checkpoint_loaded:
        curvature, stats = _load_curvature_matrix_checkpoint(
            matrix_checkpoint_path,
            model,
            config,
            calibration,
            target_owners,
            rank,
            world_size,
        )
        logging.info(
            "Loaded raw curvature C checkpoint shard for rank %d from %s; "
            "skipping curvature forward/backward.",
            rank,
            matrix_checkpoint_path,
        )
    else:
        curvature = {
            name: torch.zeros(hidden_size, hidden_size, dtype=torch.float32)
            for name in owned_targets
        }
        # valid, selected, valid norm sum/max, selected norm sum/min
        stats = {
            name: torch.tensor([0, 0, 0, 0, 0, float("inf")], dtype=torch.float64)
            for name in owned_targets
        }

    if not matrix_checkpoint_loaded:
        _accumulate_curvature_matrices(
            model,
            samples,
            config,
            device,
            distributed_fsdp,
            owned_targets,
            curvature,
            stats,
            rank,
            world_size,
            len(targets),
        )
        if matrix_checkpoint_path is not None:
            _save_curvature_matrix_checkpoint(
                matrix_checkpoint_path,
                model,
                config,
                calibration,
                target_owners,
                curvature,
                stats,
                rank,
                world_size,
            )
            logging.info(
                "Saved raw curvature C checkpoint shard for rank %d to %s.",
                rank,
                matrix_checkpoint_path,
            )

    subspaces = {}
    diagnostics = {}
    for name in targets:
        owner = target_owners[name]
        if rank == owner:
            c = curvature[name]
            stat = stats[name]
            selected_count = int(stat[1].item())
            if selected_count == 0:
                raise RuntimeError(f"No selected gradients for {name}.")
            eigenvalues, eigenvectors = torch.linalg.eigh(c)
            top_values = eigenvalues[-config.sensitive_rank :].flip(0).contiguous()
            s = eigenvectors[:, -config.sensitive_rank :].flip(1).contiguous()
            orthogonality = torch.linalg.matrix_norm(
                s.transpose(0, 1) @ s - torch.eye(config.sensitive_rank), ord="fro"
            )
            total = eigenvalues.clamp_min(0).sum()
            energy = top_values.clamp_min(0).sum() / (total + 1e-12)
            diag = {
                "number_valid_reasoning_tokens": int(stat[0]),
                "number_selected_tokens": selected_count,
                "mean_grad_l2_valid": float(stat[2] / stat[0]),
                "max_grad_l2_valid": float(stat[3]),
                "mean_grad_l2_selected": float(stat[4] / stat[1]),
                "min_grad_l2_selected": float(stat[5]),
                "top_eigenvalues": top_values.tolist(),
                "top_energy_ratio": float(energy),
                "orthogonality_error": float(orthogonality),
            }
        else:
            s = torch.empty(hidden_size, config.sensitive_rank)
            diag = None
        if distributed_fsdp:
            s_device = s.to(device)
            dist.broadcast(s_device, src=owner)
            s = s_device.cpu()
            diag_container = [diag]
            dist.broadcast_object_list(diag_container, src=owner, device=device)
            diag = diag_container[0]
        subspaces[name] = s
        if rank == 0:
            diagnostics[name] = diag
        if rank == owner:
            del c
    if distributed_fsdp:
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return subspaces, diagnostics, not matrix_checkpoint_loaded


def _accumulate_curvature_matrices(
    model: nn.Module,
    samples: Iterable[dict[str, torch.Tensor]],
    config: LowRankConfig,
    device: torch.device,
    distributed_fsdp: bool,
    owned_targets: dict[str, nn.Module],
    curvature: dict[str, torch.Tensor],
    stats: dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    target_count: int,
) -> None:
    """Run the expensive forward/backward phase and accumulate raw C matrices."""
    nonzero_dropout = [
        (name, module.p)
        for name, module in model.named_modules()
        if isinstance(module, nn.Dropout) and module.p != 0
    ]
    if nonzero_dropout:
        raise RuntimeError(
            "Curvature checkpointing requires training mode, but the model has "
            f"non-zero dropout modules: {nonzero_dropout}"
        )
    # Transformers' GradientCheckpointingLayer only checkpoints when
    # ``module.training`` is true. Qwen3.5-4B has zero dropout, so training mode
    # changes no numerical operation and only activates recomputation.
    model.train()
    model.config.use_cache = False
    # FSDP needs gradient participation to schedule its backward all-gathers
    # and resharding. There is deliberately no optimizer for these parameters,
    # so the original BF16 weights remain frozen despite requires_grad=True.
    for parameter in model.parameters():
        parameter.requires_grad_(distributed_fsdp)
    if not getattr(model, "supports_gradient_checkpointing", False):
        raise RuntimeError("Sensitive-subspace extraction requires gradient checkpointing support.")
    # A 4096-token full-model backward does not fit on a 24 GiB 3090 when all
    # block activations are retained. Non-reentrant checkpointing recomputes
    # block forwards during backward and still permits RMSNorm output hooks.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if not all(
        getattr(layer, "gradient_checkpointing", False) and layer.training
        for layer in _get_layers(model)
    ):
        raise RuntimeError("Gradient checkpointing is not active on every decoder layer.")
    logging.info(
        "Enabled active non-reentrant activation checkpointing for curvature extraction "
        "(training mode with zero dropout)."
    )
    current_reasoning_mask: torch.Tensor | None = None
    received_gradients: set[str] = set()
    handles = []

    def capture(name):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"{name} RMSNorm output is not a Tensor.")
            output.requires_grad_(True)

            def accumulate_gradient(gradient):
                if name in received_gradients:
                    return gradient
                if current_reasoning_mask is None:
                    raise RuntimeError("Reasoning mask is unavailable in the RMSNorm gradient hook.")
                selected, valid_norms, selected_norms = select_topk_gradients(
                    gradient,
                    current_reasoning_mask,
                    config.curvature_tokens_per_sample,
                )
                selected_cpu = selected.cpu()
                curvature[name].add_(selected_cpu.transpose(0, 1) @ selected_cpu)
                row = stats[name]
                row[0] += valid_norms.numel()
                row[1] += selected_norms.numel()
                row[2] += valid_norms.double().sum().cpu()
                row[3] = max(row[3], valid_norms.double().max().cpu())
                row[4] += selected_norms.double().sum().cpu()
                row[5] = min(row[5], selected_norms.double().min().cpu())
                received_gradients.add(name)
                return gradient

            output.register_hook(accumulate_gradient)
            return output
        return hook

    for name, norm in owned_targets.items():
        handles.append(norm.register_forward_hook(capture(name)))

    if distributed_fsdp:
        from torch.distributed.fsdp import (
            BackwardPrefetch,
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        layer_class = type(_get_layers(model)[0])
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_class},
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_POST,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=device,
            sync_module_states=False,
            forward_prefetch=False,
            limit_all_gathers=True,
            use_orig_params=True,
        )
        logging.info(
            "Enabled FSDP FULL_SHARD across %d ranks; this rank owns %d/%d curvature matrices.",
            world_size,
            len(owned_targets),
            target_count,
        )
        saved_tensor_context_factory = contextlib.nullcontext
    else:
        model.to(device)
        logging.info("Enabled saved-tensor CPU offload for single-GPU curvature backward.")
        saved_tensor_context_factory = (
            lambda: torch.autograd.graph.save_on_cpu(pin_memory=True)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )

    try:
        for sample_idx, sample in enumerate(samples):
            received_gradients.clear()
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            current_reasoning_mask = sample["reasoning_mask"].to(device)
            with torch.enable_grad(), saved_tensor_context_factory():
                output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                loss = reasoning_nll(output.logits, input_ids, current_reasoning_mask)
                loss.backward()
            missing = set(owned_targets) - received_gradients
            if missing:
                raise RuntimeError(f"Missing RMSNorm activation gradients: {sorted(missing)}")
            model.zero_grad(set_to_none=True)
            del output, loss, input_ids, attention_mask
            current_reasoning_mask = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if rank == 0:
                logging.info("FSDP curvature sample %d complete.", sample_idx)
    finally:
        for handle in handles:
            handle.remove()

    if not distributed_fsdp:
        model.gradient_checkpointing_disable()
        model.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def attach_lowrank_adapters(
    model: nn.Module,
    config: LowRankConfig,
    subspaces: dict[str, torch.Tensor] | None = None,
) -> dict[str, LowRankRMSNorm]:
    subspaces = subspaces or {}
    wrappers = {}
    for layer_idx, layer in enumerate(_get_layers(model)):
        for norm_name in NORM_NAMES:
            key = f"layers.{layer_idx}.{norm_name}"
            current = getattr(layer, norm_name)
            if isinstance(current, LowRankRMSNorm):
                raise RuntimeError(f"Adapter already attached to {key}.")
            s = subspaces.get(key)
            wrapper = LowRankRMSNorm(current, config, s)
            setattr(layer, norm_name, wrapper)
            wrappers[key] = wrapper
    return wrappers


def detach_lowrank_adapters(model: nn.Module) -> None:
    for layer in _get_layers(model):
        for name in NORM_NAMES:
            module = getattr(layer, name)
            if isinstance(module, LowRankRMSNorm):
                setattr(layer, name, module.norm)


def _mean_wrapper_diagnostics(wrappers: dict[str, LowRankRMSNorm]) -> dict[str, float]:
    result = {}
    for name, wrapper in wrappers.items():
        for metric, value in wrapper.last_diagnostics.items():
            result[f"{name}.{metric}"] = float(value)
    return result


def _register_activation_quant_ste_pre_hooks(layer: nn.Module, qparams: QType) -> list:
    """Apply the existing HiF4 A4 forward with identity STE for adapter training."""
    handles = []

    def quantize_linear_input(_module, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("Low-rank A4 training requires a Tensor Linear input.")
        quantized_input = quant_func(
            args[0].contiguous(), qparams, force_fp32=True
        )
        return (quantized_input, *args[1:])

    for module in layer.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(quantize_linear_input))
    return handles


def optimize_blocks(
    model: nn.Module,
    reference_layers: list[nn.Module],
    samples: list[dict[str, torch.Tensor]],
    subspaces: dict[str, torch.Tensor],
    config: LowRankConfig,
    device: torch.device,
    act_qtype: str,
    calib_batch_size: int = 1,
) -> tuple[dict[str, LowRankRMSNorm], list[dict[str, object]]]:
    """Sequential block reconstruction with synchronized data parallel grads."""
    if calib_batch_size != 1:
        raise ValueError("The first implementation requires low-rank batch size 1.")
    if _distributed() and len(samples) == 0:
        raise ValueError("Every distributed rank must own at least one sample.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = _get_layers(model)
    plain_loader = [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in samples]
    q_inps, layer_kwargs, _ = _capture_calibration_inputs(
        model, layers, plain_loader, device, len(samples), samples[0]["input_ids"].shape[1], next(model.parameters()).dtype
    )
    fp_inps = q_inps.clone()
    q_outs = torch.empty_like(q_inps)
    fp_outs = torch.empty_like(fp_inps)
    all_wrappers: dict[str, LowRankRMSNorm] = {}
    diagnostics = []
    act_qparams = QType(act_qtype).dim(-1)
    world_size = dist.get_world_size() if _distributed() else 1

    for layer_idx, (q_layer, reference_layer) in enumerate(zip(layers, reference_layers)):
        reference_layer = reference_layer.to(device).eval()
        with torch.no_grad():
            for sample_idx in range(len(samples)):
                x = fp_inps[sample_idx : sample_idx + 1].to(device)
                kwargs = _layer_kwargs_for_current_layer(model, reference_layer, x, layer_kwargs)
                fp_outs[sample_idx].copy_(_run_layer(reference_layer, x, kwargs)[0].cpu())
        reference_layer.cpu()

        block_subspaces = {
            f"layers.{layer_idx}.{name}": subspaces[f"layers.{layer_idx}.{name}"]
            for name in NORM_NAMES if f"layers.{layer_idx}.{name}" in subspaces
        }
        wrappers = {}
        for norm_name in NORM_NAMES:
            key = f"layers.{layer_idx}.{norm_name}"
            wrapper = LowRankRMSNorm(getattr(q_layer, norm_name), config, block_subspaces.get(key))
            setattr(q_layer, norm_name, wrapper)
            wrappers[key] = wrapper
            all_wrappers[key] = wrapper
        q_layer.to(device)
        quant_handles = _register_activation_quant_ste_pre_hooks(q_layer, act_qparams)

        named_trainable = {
            f"{name}.{parameter_name}": parameter
            for name, wrapper in wrappers.items()
            for parameter_name, parameter in (("U", wrapper.U), ("V", wrapper.V))
        }
        trainable = list(named_trainable.values())
        optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.0) if config.mode != "baseline" else None
        before_loss = 0.0
        last_loss = None
        try:
            with torch.no_grad():
                for sample_idx in range(len(samples)):
                    x = q_inps[sample_idx : sample_idx + 1].to(device)
                    kwargs = _layer_kwargs_for_current_layer(model, q_layer, x, layer_kwargs)
                    initial_output = _run_layer(q_layer, x, kwargs)
                    initial_target = fp_outs[sample_idx : sample_idx + 1].to(device)
                    initial_loss = (
                        (initial_output.float() - initial_target.float()).square().sum()
                        / (initial_target.float().square().sum() + 1e-12)
                    )
                    before_loss += float(initial_loss)
            before_loss /= len(samples)
            if _distributed():
                reduced_before = torch.tensor(before_loss, device=device)
                dist.all_reduce(reduced_before, op=dist.ReduceOp.SUM)
                before_loss = float(reduced_before / world_size)
            epoch_range = range(config.epochs) if optimizer is not None else range(1)
            for epoch in epoch_range:
                epoch_loss = 0.0
                for sample_idx in range(len(samples)):
                    optimizer.zero_grad(set_to_none=True) if optimizer is not None else None
                    x = q_inps[sample_idx : sample_idx + 1].to(device)
                    kwargs = _layer_kwargs_for_current_layer(model, q_layer, x, layer_kwargs)
                    with torch.set_grad_enabled(optimizer is not None):
                        q_output = _run_layer(q_layer, x, kwargs)
                        target = fp_outs[sample_idx : sample_idx + 1].to(device)
                        loss = (q_output.float() - target.float()).square().sum() / (
                            target.float().square().sum() + 1e-12
                        )
                    if optimizer is not None:
                        loss.backward()
                        missing_gradients = [
                            name for name, parameter in named_trainable.items()
                            if parameter.grad is None
                        ]
                        if missing_gradients:
                            raise RuntimeError(
                                "Low-rank parameters are disconnected from the loss: "
                                f"{missing_gradients}"
                            )
                        for parameter in trainable:
                            if not torch.isfinite(parameter.grad).all():
                                raise RuntimeError("A low-rank parameter received a non-finite gradient.")
                            if _distributed():
                                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                                parameter.grad.div_(world_size)
                        optimizer.step()
                    epoch_loss += float(loss.detach())
                last_loss = epoch_loss / len(samples)
                logging.info("Low-rank block=%d epoch=%d loss=%.8f", layer_idx, epoch, last_loss)

            final_loss_sum = 0.0
            with torch.no_grad():
                for sample_idx in range(len(samples)):
                    x = q_inps[sample_idx : sample_idx + 1].to(device)
                    kwargs = _layer_kwargs_for_current_layer(model, q_layer, x, layer_kwargs)
                    final_output = _run_layer(q_layer, x, kwargs)
                    q_outs[sample_idx].copy_(final_output[0].cpu())
                    final_target = fp_outs[sample_idx : sample_idx + 1].to(device)
                    final_loss_sum += float(
                        (final_output.float() - final_target.float()).square().sum()
                        / (final_target.float().square().sum() + 1e-12)
                    )
            last_loss = final_loss_sum / len(samples)
            if _distributed():
                reduced_after = torch.tensor(last_loss, device=device)
                dist.all_reduce(reduced_after, op=dist.ReduceOp.SUM)
                last_loss = float(reduced_after / world_size)
            block_diag = {
                "layer": layer_idx,
                "loss_before": before_loss,
                "loss_after": last_loss,
                "adapters": _mean_wrapper_diagnostics(wrappers),
            }
            for key, wrapper in wrappers.items():
                block_diag[f"{key}.U_grad_norm"] = float(wrapper.U.grad.float().norm()) if wrapper.U.grad is not None else 0.0
                block_diag[f"{key}.V_grad_norm"] = float(wrapper.V.grad.float().norm()) if wrapper.V.grad is not None else 0.0
                block_diag[f"{key}.U_norm"] = float(wrapper.U.detach().float().norm())
                block_diag[f"{key}.V_norm"] = float(wrapper.V.detach().float().norm())
            diagnostics.append(block_diag)
        finally:
            for handle in quant_handles:
                handle.remove()
        q_layer.cpu()
        fp_inps, fp_outs = fp_outs, fp_inps
        q_inps, q_outs = q_outs, q_inps
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_wrappers, diagnostics


def adapter_state(wrappers: dict[str, LowRankRMSNorm]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "S": wrapper.S.detach().cpu(),
            "U": wrapper.U.detach().cpu(),
            "V": wrapper.V.detach().cpu(),
        }
        for name, wrapper in wrappers.items()
    }


def _expected_norm_names(model: nn.Module) -> set[str]:
    return {
        f"layers.{layer_idx}.{norm_name}"
        for layer_idx in range(model.config.num_hidden_layers)
        for norm_name in NORM_NAMES
    }


def save_curvature_checkpoint(
    path: str,
    model: nn.Module,
    config: LowRankConfig,
    subspaces: dict[str, torch.Tensor],
    curvature_diagnostics: dict,
    calibration: dict[str, object],
) -> None:
    """Atomically persist the expensive curvature phase before optimization."""
    # The just-finished FSDP pass may have wrapped decoder blocks in-place, so
    # validate names from the locked Qwen3.5 architecture metadata here.
    expected_names = _expected_norm_names(model)
    if set(subspaces) != expected_names:
        raise ValueError("Curvature checkpoint is missing RMSNorm sensitive subspaces.")
    expected_shape = (model.config.hidden_size, config.sensitive_rank)
    for name, tensor in subspaces.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Sensitive subspace {name} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}."
            )
    payload = {
        "format": CURVATURE_CHECKPOINT_FORMAT,
        "model_type": model.config.model_type,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "curvature_config": {
            "sensitive_rank": config.sensitive_rank,
            "curvature_tokens_per_sample": config.curvature_tokens_per_sample,
            "curvature_token_selection": config.curvature_token_selection,
        },
        "calibration": dict(calibration),
        "subspaces": {name: tensor.detach().float().cpu() for name, tensor in subspaces.items()},
        "curvature_diagnostics": curvature_diagnostics,
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def load_curvature_checkpoint(
    path: str,
    model: nn.Module,
    config: LowRankConfig,
    calibration: dict[str, object],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load only an exactly matching completed curvature checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CURVATURE_CHECKPOINT_FORMAT:
        raise ValueError("Unsupported RMSNorm curvature checkpoint format.")
    for field in ("model_type", "hidden_size", "num_hidden_layers"):
        if payload.get(field) != getattr(model.config, field):
            raise ValueError(f"Curvature checkpoint {field} does not match the model.")
    expected_curvature_config = {
        "sensitive_rank": config.sensitive_rank,
        "curvature_tokens_per_sample": config.curvature_tokens_per_sample,
        "curvature_token_selection": config.curvature_token_selection,
    }
    if payload.get("curvature_config") != expected_curvature_config:
        raise ValueError("Curvature checkpoint extraction parameters do not match this run.")
    if payload.get("calibration") != calibration:
        raise ValueError("Curvature checkpoint calibration data parameters do not match this run.")

    subspaces = payload.get("subspaces")
    if not isinstance(subspaces, dict):
        raise ValueError("Curvature checkpoint does not contain sensitive subspaces.")
    expected_names = _expected_norm_names(model)
    if set(subspaces) != expected_names:
        raise ValueError("Curvature checkpoint RMSNorm names do not match the model.")
    expected_shape = (model.config.hidden_size, config.sensitive_rank)
    for name, tensor in subspaces.items():
        if not torch.is_tensor(tensor) or tensor.shape != expected_shape:
            raise ValueError(
                f"Curvature checkpoint tensor {name} does not have shape {expected_shape}."
            )
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"Curvature checkpoint tensor {name} is not finite FP32 data.")
    diagnostics = payload.get("curvature_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != expected_names:
        raise ValueError("Curvature checkpoint diagnostics do not match the model.")
    return subspaces, diagnostics


def save_sidecar(
    path: str,
    model: nn.Module,
    config: LowRankConfig,
    wrappers: dict[str, LowRankRMSNorm],
    curvature_diagnostics: dict,
    optimization_diagnostics: list,
    function_preservation: dict | None = None,
) -> None:
    payload = {
        "format": "reasoning_safe_rmsnorm_lowrank_v1",
        "model_type": model.config.model_type,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "config": asdict(config),
        "adapters": adapter_state(wrappers),
        "curvature_diagnostics": curvature_diagnostics,
        "optimization_diagnostics": optimization_diagnostics,
        "function_preservation": function_preservation or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_sidecar(model: nn.Module, path: str) -> tuple[dict[str, LowRankRMSNorm], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "reasoning_safe_rmsnorm_lowrank_v1":
        raise ValueError("Unsupported RMSNorm adapter checkpoint format.")
    for field in ("model_type", "hidden_size", "num_hidden_layers"):
        if payload[field] != getattr(model.config, field):
            raise ValueError(f"Adapter {field} does not match the model.")
    config = LowRankConfig(**payload["config"])
    subspaces = {name: value["S"] for name, value in payload["adapters"].items()}
    wrappers = attach_lowrank_adapters(model, config, subspaces)
    if set(wrappers) != set(payload["adapters"]):
        raise ValueError("Adapter layer names do not match the model.")
    for name, wrapper in wrappers.items():
        state = payload["adapters"][name]
        if state["U"].shape != wrapper.U.shape or state["V"].shape != wrapper.V.shape:
            raise ValueError(f"Adapter rank/hidden shape mismatch for {name}.")
        wrapper.U.data.copy_(state["U"])
        wrapper.V.data.copy_(state["V"])
    return wrappers, payload


@torch.no_grad()
def evaluate_function_preservation(
    original_model: nn.Module,
    adapted_model: nn.Module,
    samples: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Compare original BF16 and BF16+adapter on the same reasoning labels."""
    original_model.to(device).eval()
    adapted_model.to(device).eval()
    kl_sum = nll_original = nll_adapted = agreement_sum = 0.0
    token_count = 0
    for sample in samples:
        ids = sample["input_ids"].to(device)
        attention = sample["attention_mask"].to(device)
        mask = sample["reasoning_mask"].to(device)[:, 1:]
        original_logits = original_model(ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
        adapted_logits = adapted_model(ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
        original_selected = original_logits[mask]
        adapted_selected = adapted_logits[mask]
        labels = ids[:, 1:][mask]
        original_logp = F.log_softmax(original_selected, dim=-1)
        adapted_logp = F.log_softmax(adapted_selected, dim=-1)
        original_p = original_logp.exp()
        current_count = labels.numel()
        kl_sum += float((original_p * (original_logp - adapted_logp)).sum(dim=-1).sum())
        nll_original += float(F.nll_loss(original_logp, labels, reduction="sum"))
        nll_adapted += float(F.nll_loss(adapted_logp, labels, reduction="sum"))
        agreement_sum += float((original_selected.argmax(-1) == adapted_selected.argmax(-1)).sum())
        token_count += current_count
    if token_count == 0:
        raise ValueError("Function-preservation evaluation received no samples.")
    return {
        "next_token_kl": kl_sum / token_count,
        "original_reasoning_nll": nll_original / token_count,
        "adapted_reasoning_nll": nll_adapted / token_count,
        "top1_agreement": agreement_sum / token_count,
        "reasoning_token_count": token_count,
    }


def clone_reference_layers(model: nn.Module) -> list[nn.Module]:
    """Snapshot original BF16 blocks before GPTQ overwrites their weights."""
    return [copy.deepcopy(layer).cpu() for layer in _get_layers(model)]


def diagnostics_to_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
