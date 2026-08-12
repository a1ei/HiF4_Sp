import inspect
import logging
import math
import pathlib
import sys

import torch
import torch.nn as nn
import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from hif4_gpu.quant_cy import QType, quant_dequant_float


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _filter_kwargs_for_callable(callable_obj, kwargs):
    signature = inspect.signature(callable_obj)
    params = signature.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs
    valid_keys = set(signature.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def _extract_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _run_layer(layer, hidden_states, layer_kwargs):
    call_kwargs = _filter_kwargs_for_callable(layer.forward, layer_kwargs)
    call_kwargs = _move_to_device(call_kwargs, hidden_states.device)
    output = layer(hidden_states, **call_kwargs)
    return _extract_hidden(output)


def _register_activation_quant_pre_hooks(layer, qparams):
    """Quantize-dequantize every Linear input during GPTQ calibration.

    Forward pre-hooks keep the GPTQ Linear discovery/grouping unchanged. GPTQ
    forward hooks therefore observe the same quantized inputs used by the
    Linear computation, and that block output propagates to the next block.
    """
    handles = []

    def quantize_linear_input(_, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError(
                "GPTQ activation quantization requires a Tensor Linear input."
            )
        quantized_input = quant_dequant_float(
            args[0].contiguous(), qparams, force_fp32=True
        )
        return (quantized_input, *args[1:])

    for module in layer.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(quantize_linear_input))
    return handles


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    raise NotImplementedError("Only decoder-only models with model.layers are supported.")


def _is_qwen3_5_text_model(model):
    return getattr(model.config, "model_type", "") in {"qwen3_5", "qwen3_5_text"}


def _get_final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "norm")
    ):
        return model.model.language_model.norm
    return None


def _get_quant_groups(model, layer=None):
    model_type = getattr(model.config, "model_type", "")
    if model_type not in {"llama", "qwen3", "qwen3_5", "qwen3_5_text"}:
        raise NotImplementedError(f"Model type {model_type} is out of scope. Supported: llama, qwen3, qwen3_5.")

    mlp_groups = [
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]

    if model_type == "qwen3_5_text":
        if layer is None:
            raise ValueError("Qwen3.5 GPTQ requires a concrete decoder layer to select quantization groups.")

        layer_type = getattr(layer, "layer_type", None)
        if layer_type == "linear_attention":
            return [
                ["linear_attn.in_proj_qkv"],
                ["linear_attn.in_proj_z"],
                ["linear_attn.in_proj_b"],
                ["linear_attn.in_proj_a"],
                ["linear_attn.out_proj"],
                *mlp_groups,
            ]
        if layer_type == "full_attention":
            return [
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                ["self_attn.o_proj"],
                *mlp_groups,
            ]
        raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer_type}")

    return [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        *mlp_groups,
    ]


def _validate_no_padding_attention_mask(batch) -> None:
    if not isinstance(batch, dict) or "attention_mask" not in batch:
        return

    attention_mask = batch["attention_mask"]
    if torch.is_tensor(attention_mask) and not torch.all(attention_mask == 1):
        raise ValueError("Qwen3.5 GPTQ calibration does not support padded attention_mask yet.")


def _make_causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(hidden_states):
        raise TypeError("Causal mask requires floating-point hidden states.")

    batch_size, seq_len, _ = hidden_states.shape
    min_value = torch.finfo(hidden_states.dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=hidden_states.dtype, device=hidden_states.device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


def _layer_kwargs_for_current_layer(model, layer, hidden_states, base_layer_kwargs):
    layer_kwargs = dict(base_layer_kwargs)
    if not _is_qwen3_5_text_model(model):
        return layer_kwargs

    layer_type = getattr(layer, "layer_type", None)
    if layer_type == "linear_attention":
        layer_kwargs["attention_mask"] = None
    elif layer_type == "full_attention":
        layer_kwargs["attention_mask"] = _make_causal_mask(hidden_states)
    else:
        raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer_type}")
    return layer_kwargs


def _capture_calibration_inputs(model, layers, dataloader, device, max_samples, seqlen, dtype):
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    inps = torch.zeros((max_samples, seqlen, model.config.hidden_size), dtype=dtype, device="cpu")
    valid_token_mask = torch.ones((max_samples, seqlen), dtype=torch.bool, device="cpu")
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            idx = cache["i"]
            if idx < max_samples:
                inps[idx].copy_(inp[0].detach().cpu())
            cache["i"] += 1
            for key, val in kwargs.items():
                cache[key] = val
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache["i"] >= max_samples:
            break

        if _is_qwen3_5_text_model(model):
            _validate_no_padding_attention_mask(batch)

        if isinstance(batch, (list, tuple)):
            input_ids = batch[0]
            attention_mask = None
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
        else:
            input_ids = batch
            attention_mask = None

        if input_ids.shape[0] != 1 or input_ids.shape[1] != seqlen:
            raise ValueError(
                "GPTQ calibration currently requires batches shaped "
                f"[1, {seqlen}], got {tuple(input_ids.shape)}."
            )
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "Calibration attention_mask must match input_ids shape, got "
                    f"{tuple(attention_mask.shape)} and {tuple(input_ids.shape)}."
                )
            valid_token_mask[cache["i"]].copy_(attention_mask[0].detach().bool().cpu())

        model_kwargs = {}
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.to(device)
        try:
            model(input_ids.to(device), **model_kwargs)
        except ValueError:
            pass

    layers[0] = layers[0].module
    nsamples = min(cache["i"], max_samples)
    if nsamples == 0:
        raise RuntimeError("Calibration dataloader produced zero samples.")

    layers[0] = layers[0].cpu()
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_kwargs = {k: v for k, v in cache.items() if k != "i"}
    return inps[:nsamples], layer_kwargs, valid_token_mask[:nsamples]


@torch.no_grad()
def _compute_fp_token_entropy(model, layers, inps, layer_kwargs, device, logits_chunk_size=128):
    """Run a layer-wise FP forward and return next-token entropy on CPU."""
    logging.info("Computing FP next-token entropy before GPTQ quantization.")
    fp_inps = inps
    fp_outs = torch.zeros_like(fp_inps)

    for i in tqdm.tqdm(range(len(layers)), desc="(GPTQ Entropy FP) Layers"):
        layer = layers[i].to(device)
        for j in range(fp_inps.shape[0]):
            layer_input = fp_inps[j].unsqueeze(0).to(device)
            current_layer_kwargs = _layer_kwargs_for_current_layer(
                model, layer, layer_input, layer_kwargs
            )
            fp_outs[j].copy_(_run_layer(layer, layer_input, current_layer_kwargs)[0].cpu())

        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fp_inps, fp_outs = fp_outs, fp_inps

    norm = _get_final_norm(model)
    output_head = model.get_output_embeddings()
    if output_head is None:
        raise RuntimeError("Entropy-weighted GPTQ requires a causal LM output embedding layer.")
    if norm is not None:
        norm = norm.to(device)
    output_head = output_head.to(device)

    nsamples, seqlen, _ = fp_inps.shape
    entropy = torch.full((nsamples, seqlen), float("nan"), dtype=torch.float32, device="cpu")
    for j in tqdm.tqdm(range(nsamples), desc="(GPTQ Entropy FP) Logits"):
        for start in range(0, max(seqlen - 1, 0), logits_chunk_size):
            end = min(start + logits_chunk_size, seqlen - 1)
            hidden = fp_inps[j, start:end].to(device)
            if norm is not None:
                hidden = norm(hidden)
            logits = output_head(hidden).float()
            probs = torch.softmax(logits, dim=-1)
            chunk_entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
            entropy[j, start:end].copy_(chunk_entropy.cpu())
            del hidden, logits, probs, chunk_entropy

    output_head = output_head.cpu()
    if norm is not None:
        norm = norm.cpu()
    del fp_inps, fp_outs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return entropy


def _normalize_entropy_importance(
    entropy, valid_token_mask, alpha, norm_mode, entropy_direction="low"
):
    if entropy.shape != valid_token_mask.shape:
        raise ValueError(
            f"Entropy shape {tuple(entropy.shape)} must match token mask shape "
            f"{tuple(valid_token_mask.shape)}."
        )
    if entropy.shape[1] < 2:
        raise ValueError("Entropy-weighted GPTQ requires calibration sequences with at least 2 tokens.")

    entropy_valid_mask = valid_token_mask.clone()
    entropy_valid_mask[:, -1] = False
    entropy_valid_mask[:, :-1] &= valid_token_mask[:, 1:]
    entropy_valid_mask &= torch.isfinite(entropy)
    if not entropy_valid_mask.any():
        raise ValueError("No valid next-token entropy positions were found in calibration data.")

    values = entropy[entropy_valid_mask]
    eps = 1e-12
    if norm_mode == "minmax":
        value_range = values.max() - values.min()
        normalized = (values - values.min()) / (value_range + eps) if value_range > 0 else torch.zeros_like(values)
    elif norm_mode == "zscore":
        std = values.std(unbiased=False)
        if std > 0:
            normalized = (values - values.mean()) / std
            normalized = normalized - normalized.min()
        else:
            normalized = torch.zeros_like(values)
    elif norm_mode == "mean":
        mean = values.mean()
        normalized = values / (mean + eps) if mean > 0 else torch.zeros_like(values)
    else:
        raise ValueError(f"Unsupported entropy normalization mode: {norm_mode}")

    if entropy_direction == "low":
        normalized = normalized.max() - normalized
    elif entropy_direction != "high":
        raise ValueError(f"Unsupported entropy direction: {entropy_direction}")
    token_weights = torch.ones_like(entropy, dtype=torch.float32)
    token_weights[entropy_valid_mask] = 1.0 + alpha * normalized
    token_weights[~valid_token_mask] = 0.0
    if not torch.isfinite(token_weights).all() or torch.any(token_weights < 0):
        raise ValueError("Entropy importance produced non-finite or negative token weights.")

    valid_weights = token_weights[valid_token_mask]
    logging.info(
        "Token entropy stats: mean=%.6f std=%.6f min=%.6f max=%.6f",
        values.mean().item(),
        values.std(unbiased=False).item(),
        values.min().item(),
        values.max().item(),
    )
    logging.info(
        "Token lambda stats: mean=%.6f std=%.6f min=%.6f max=%.6f",
        valid_weights.mean().item(),
        valid_weights.std(unbiased=False).item(),
        valid_weights.min().item(),
        valid_weights.max().item(),
    )
    return token_weights


_LOCAL_IMPORTANCE_GROUPS = ("qkv", "o", "up_gate", "down")


def _local_importance_group_for_linear(name):
    if name in {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
    }:
        return "qkv"
    if name in {"self_attn.o_proj", "linear_attn.out_proj"}:
        return "o"
    if name in {"mlp.up_proj", "mlp.gate_proj"}:
        return "up_gate"
    if name == "mlp.down_proj":
        return "down"
    raise ValueError(f"No local token-importance group is defined for linear: {name}")


def _local_importance_representatives(layer):
    full = find_qlayers(layer, layers=[nn.Linear])
    qkv_name = "linear_attn.in_proj_qkv" if getattr(layer, "layer_type", None) == "linear_attention" else "self_attn.q_proj"
    o_name = "linear_attn.out_proj" if getattr(layer, "layer_type", None) == "linear_attention" else "self_attn.o_proj"
    names = {
        "qkv": qkv_name,
        "o": o_name,
        "up_gate": "mlp.up_proj",
        "down": "mlp.down_proj",
    }
    missing = [name for name in names.values() if name not in full]
    if missing:
        raise ValueError(f"entropy_grad cannot find required linear modules: {missing}")
    return {group: full[name] for group, name in names.items()}


def _activation_gradient_importance(activation, gradient, mode="entropy_grad"):
    if activation.shape != gradient.shape:
        raise RuntimeError(
            f"Activation shape {tuple(activation.shape)} does not match gradient shape {tuple(gradient.shape)}."
        )
    if mode == "entropy_grad":
        signal = activation.float() * gradient.float()
    elif mode == "entropy_grad_norm":
        signal = gradient.float()
    else:
        raise ValueError(f"Unsupported layer-local importance mode: {mode}")
    importance = torch.linalg.vector_norm(signal, dim=-1)
    if importance.ndim == 1:
        importance = importance.unsqueeze(0)
    if importance.ndim != 2:
        raise RuntimeError(
            "Layer-local token importance must have shape [batch, seqlen], got "
            f"{tuple(importance.shape)}."
        )
    if not torch.isfinite(importance).all() or torch.any(importance < 0):
        raise RuntimeError("Layer-local token importance contains NaN, Inf, or negative values.")
    return importance


class _LayerLocalAttribution(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layer,
        model,
        layer_kwargs,
        device,
        representatives,
        importance_store,
        layer_idx,
        sample_start,
        importance_mode,
    ):
        ctx.layer = layer
        ctx.model = model
        ctx.layer_kwargs = layer_kwargs
        ctx.device = device
        ctx.representatives = representatives
        ctx.importance_store = importance_store
        ctx.layer_idx = layer_idx
        ctx.sample_start = sample_start
        ctx.importance_mode = importance_mode
        ctx.save_for_backward(hidden_states.detach())

        layer = layer.to(device)
        try:
            layer_input = hidden_states.to(device)
            current_layer_kwargs = _layer_kwargs_for_current_layer(
                model, layer, layer_input, layer_kwargs
            )
            output = _run_layer(layer, layer_input, current_layer_kwargs)
            output_cpu = output.detach().cpu()
        finally:
            layer.cpu()
        return output_cpu

    @staticmethod
    def backward(ctx, grad_output):
        (hidden_states,) = ctx.saved_tensors
        layer = ctx.layer.to(ctx.device)
        captured = {}
        handles = []

        def capture_input(group):
            def hook(_, inputs, __):
                captured[group] = inputs[0]

            return hook

        for group, module in ctx.representatives.items():
            handles.append(module.register_forward_hook(capture_input(group)))

        try:
            with torch.enable_grad():
                layer_input = hidden_states.to(ctx.device).detach().requires_grad_(True)
                current_layer_kwargs = _layer_kwargs_for_current_layer(
                    ctx.model, layer, layer_input, ctx.layer_kwargs
                )
                output = _run_layer(layer, layer_input, current_layer_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        missing = [group for group in _LOCAL_IMPORTANCE_GROUPS if group not in captured]
        if missing:
            raise RuntimeError(
                f"Layer {ctx.layer_idx} did not capture local importance inputs for groups: {missing}"
            )

        targets = [layer_input] + [captured[group] for group in _LOCAL_IMPORTANCE_GROUPS]
        gradients = torch.autograd.grad(
            output,
            targets,
            grad_outputs=grad_output.to(ctx.device),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        grad_input = gradients[0].detach().cpu()

        for group, activation, gradient in zip(
            _LOCAL_IMPORTANCE_GROUPS,
            targets[1:],
            gradients[1:],
        ):
            importance = _activation_gradient_importance(
                activation.detach(), gradient.detach(), ctx.importance_mode
            )
            sample_end = ctx.sample_start + importance.shape[0]
            if sample_end > len(ctx.importance_store[ctx.layer_idx][group]):
                raise RuntimeError(
                    f"Layer {ctx.layer_idx} group {group} attribution batch exceeds calibration samples."
                )
            for batch_idx in range(importance.shape[0]):
                ctx.importance_store[ctx.layer_idx][group][ctx.sample_start + batch_idx] = (
                    importance[batch_idx : batch_idx + 1].cpu()
                )

        del layer_input, output, targets, gradients, captured
        layer.cpu()
        return grad_input, None, None, None, None, None, None, None, None, None


def _build_entropy_seed(
    normalized_hidden,
    output_head,
    valid_token_mask,
    logits_chunk_size=128,
    entropy_direction="low",
):
    if entropy_direction not in {"low", "high"}:
        raise ValueError(f"Unsupported entropy direction: {entropy_direction}")
    batch_size, seqlen, _ = normalized_hidden.shape
    entropy = torch.full(
        (batch_size, seqlen),
        float("nan"),
        dtype=torch.float32,
        device=normalized_hidden.device,
    )
    with torch.no_grad():
        for start in range(0, seqlen - 1, logits_chunk_size):
            end = min(start + logits_chunk_size, seqlen - 1)
            logits = output_head(normalized_hidden[:, start:end].detach()).float()
            probs = torch.softmax(logits, dim=-1)
            entropy[:, start:end] = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
            del logits, probs

    seed_mask = valid_token_mask.clone()
    seed_mask[:, -1] = False
    seed_mask[:, :-1] &= valid_token_mask[:, 1:]
    seed_mask &= torch.isfinite(entropy)
    if not seed_mask.any():
        raise RuntimeError("entropy_grad found no valid next-token positions in the current batch.")

    seed = torch.zeros_like(entropy)
    for batch_idx in range(batch_size):
        sample_mask = seed_mask[batch_idx]
        if not sample_mask.any():
            raise RuntimeError(
                f"entropy_grad found no valid next-token positions in batch item {batch_idx}."
            )
        values = entropy[batch_idx][sample_mask]
        value_range = values.max() - values.min()
        normalized = (
            (values - values.min()) / (value_range + 1e-12)
            if value_range > 0
            else torch.zeros_like(values)
        )
        if value_range <= 0:
            normalized = torch.ones_like(values)
        elif entropy_direction == "low":
            normalized = 1.0 - normalized
        elif entropy_direction != "high":
            raise ValueError(f"Unsupported entropy direction: {entropy_direction}")
        seed[batch_idx][sample_mask] = normalized
    seed = seed.detach()
    if not torch.isfinite(seed).all() or torch.any(seed < 0):
        raise RuntimeError("Entropy seed contains NaN, Inf, or negative values.")
    return seed


def _build_low_entropy_seed(
    normalized_hidden, output_head, valid_token_mask, logits_chunk_size=128
):
    return _build_entropy_seed(
        normalized_hidden,
        output_head,
        valid_token_mask,
        logits_chunk_size=logits_chunk_size,
        entropy_direction="low",
    )


def _margin_anchor_loss(normalized_hidden, output_head, seed, logits_chunk_size=128):
    loss_sum = torch.zeros(
        (normalized_hidden.shape[0],),
        dtype=torch.float32,
        device=normalized_hidden.device,
    )
    seqlen = normalized_hidden.shape[1]
    for start in range(0, seqlen - 1, logits_chunk_size):
        end = min(start + logits_chunk_size, seqlen - 1)
        logits = output_head(normalized_hidden[:, start:end]).float()
        top2 = torch.topk(logits, k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        loss_sum = loss_sum - (seed[:, start:end] * margin).sum(dim=1)
        del logits, top2, margin
    valid_seed_sum = seed.sum(dim=1)
    if torch.any(valid_seed_sum <= 0):
        raise RuntimeError("Each calibration sample must have a low-entropy seed sum greater than 0.")
    # Sum the independent per-sample objectives so each sample keeps the same
    # gradient scale regardless of attribution batch size.
    return (loss_sum / valid_seed_sum).sum()


def _normalize_layer_local_importance(raw_importance, valid_token_mask, alpha, mean_normalize):
    layer_weights = []
    for layer_idx, layer_values in enumerate(raw_importance):
        group_weights = {}
        for group in _LOCAL_IMPORTANCE_GROUPS:
            if any(value is None for value in layer_values[group]):
                raise RuntimeError(f"Layer {layer_idx} group {group} has missing token importance.")
            importance = torch.cat(layer_values[group], dim=0).float()
            if importance.shape != valid_token_mask.shape:
                raise RuntimeError(
                    f"Layer {layer_idx} group {group} importance shape {tuple(importance.shape)} "
                    f"does not match token mask shape {tuple(valid_token_mask.shape)}."
                )
            values = importance[valid_token_mask]
            value_range = values.max() - values.min()
            normalized = (values - values.min()) / (value_range + 1e-12) if value_range > 0 else torch.zeros_like(values)
            weight = torch.zeros_like(importance)
            weight[valid_token_mask] = 1.0 + alpha * normalized
            if mean_normalize:
                valid_mean = weight[valid_token_mask].mean()
                if not torch.isfinite(valid_mean) or valid_mean <= 0:
                    raise RuntimeError(f"Layer {layer_idx} group {group} has invalid weight mean.")
                weight[valid_token_mask] /= valid_mean
            if not torch.isfinite(weight).all() or torch.any(weight < 0):
                raise RuntimeError(f"Layer {layer_idx} group {group} has invalid token weights.")

            valid_weight = weight[valid_token_mask]
            logging.info(
                "Layer %d %s importance shape=%s mean=%.6f std=%.6f min=%.6f max=%.6f",
                layer_idx,
                group,
                tuple(importance.shape),
                values.mean().item(),
                values.std(unbiased=False).item(),
                values.min().item(),
                values.max().item(),
            )
            logging.info(
                "Layer %d %s weight shape=%s mean=%.6f std=%.6f min=%.6f max=%.6f",
                layer_idx,
                group,
                tuple(weight.shape),
                valid_weight.mean().item(),
                valid_weight.std(unbiased=False).item(),
                valid_weight.min().item(),
                valid_weight.max().item(),
            )
            group_weights[group] = weight
        layer_weights.append(group_weights)
    return layer_weights


def _compute_layer_local_token_weights(
    model,
    layers,
    inps,
    layer_kwargs,
    valid_token_mask,
    device,
    alpha,
    mean_normalize,
    batch_size=1,
    importance_mode="entropy_grad",
    entropy_direction="low",
):
    if importance_mode not in {"entropy_grad", "entropy_grad_norm"}:
        raise ValueError(f"Unsupported layer-local importance mode: {importance_mode}")
    logging.info("Computing layer-local %s token importance.", importance_mode)
    nsamples = inps.shape[0]
    if batch_size <= 0:
        raise ValueError("entropy_grad importance batch size must be greater than 0.")
    batch_size = min(batch_size, nsamples)
    logging.info(
        "entropy_grad attribution batch size: %d (%d samples, %d batches).",
        batch_size,
        nsamples,
        math.ceil(nsamples / batch_size),
    )
    raw_importance = [
        {group: [None] * nsamples for group in _LOCAL_IMPORTANCE_GROUPS}
        for _ in range(len(layers))
    ]
    requires_grad_state = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)

    norm = _get_final_norm(model)
    output_head = model.get_output_embeddings()
    if output_head is None:
        raise RuntimeError("entropy_grad requires a causal LM output embedding layer.")
    if norm is not None:
        norm = norm.to(device)
    output_head = output_head.to(device)

    try:
        sample_starts = range(0, nsamples, batch_size)
        for sample_start in tqdm.tqdm(
            sample_starts,
            total=math.ceil(nsamples / batch_size),
            desc=f"(GPTQ {importance_mode}) Batches",
        ):
            sample_end = min(sample_start + batch_size, nsamples)
            with torch.enable_grad():
                hidden = inps[sample_start:sample_end].detach().requires_grad_(True)
                for layer_idx, layer in enumerate(layers):
                    representatives = _local_importance_representatives(layer)
                    hidden = _LayerLocalAttribution.apply(
                        hidden,
                        layer,
                        model,
                        layer_kwargs,
                        device,
                        representatives,
                        raw_importance,
                        layer_idx,
                        sample_start,
                        importance_mode,
                    )

                hidden_device = hidden.to(device)
                normalized_hidden = norm(hidden_device) if norm is not None else hidden_device
                sample_mask = valid_token_mask[sample_start:sample_end].to(device)
                current_batch_size = sample_end - sample_start
                logits_chunk_size = max(1, 128 // current_batch_size)
                seed = _build_entropy_seed(
                    normalized_hidden,
                    output_head,
                    sample_mask,
                    logits_chunk_size=logits_chunk_size,
                    entropy_direction=entropy_direction,
                )
                loss_anchor = _margin_anchor_loss(
                    normalized_hidden,
                    output_head,
                    seed,
                    logits_chunk_size=logits_chunk_size,
                )
                loss_anchor.backward()

            del hidden, hidden_device, normalized_hidden, sample_mask, seed, loss_anchor
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        output_head.cpu()
        if norm is not None:
            norm.cpu()
        for layer in layers:
            layer.cpu()
        model.zero_grad(set_to_none=True)
        for parameter, requires_grad in zip(model.parameters(), requires_grad_state):
            parameter.requires_grad_(requires_grad)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _normalize_layer_local_importance(
        raw_importance,
        valid_token_mask,
        alpha=alpha,
        mean_normalize=mean_normalize,
    )


def find_qlayers(module, layers=None, name=""):
    if layers is None:
        layers = [nn.Linear]
    if type(module) in layers:
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        child_name = name + "." + name1 if name else name1
        res.update(find_qlayers(child, layers=layers, name=child_name))
    return res


class WeightHiFxQuantizer(nn.Module):
    def __init__(self, qtype="hifx4"):
        super().__init__()
        self.qparams = QType(qtype)
        self._cached_scale = None
        self._cached_width = 0
        self._cached_col = 0

    @staticmethod
    def _bf16_round(x):
        return x.to(torch.bfloat16).to(torch.float32)

    @staticmethod
    def _e6m2_round(x):
        e_sf = torch.floor(torch.log2(x))
        return torch.round(x * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)

    def _quantize_with_scale(self, x, scale):
        x_fp32 = x.float()
        scale = scale.float().clamp(min=2 ** (-48))
        sign = torch.sign(x_fp32)
        mant = torch.abs(x_fp32) / scale
        mant = torch.floor(mant * 2 ** (self.qparams.man_bits - 1) + 0.5)
        mant = mant / 2 ** (self.qparams.man_bits - 1)
        mant = torch.clamp(mant, max=2 - 2 ** (-self.qparams.man_bits + 1))
        return (sign * mant * scale).to(x.dtype)

    def _extract_hifx4_1_group_scale(self, x):
        x = x.float()
        orig_cols = x.shape[-1]
        block = self.qparams.blk_size * self.qparams.blk_outer_size
        pad_cols = (block - orig_cols % block) % block
        if pad_cols > 0:
            x = torch.nn.functional.pad(x, (0, pad_cols), value=0.0)

        x_group = x.unflatten(-1, (-1, 64))
        max_abs = torch.max(torch.abs(x_group), dim=-1, keepdim=True)[0]
        max_mant = 2 - 2 ** (-self.qparams.man_bits + 1)
        scale_factor = max_abs * self._bf16_round(torch.ones_like(max_abs) / max_mant)
        scale_factor = self._bf16_round(scale_factor).clip(min=2 ** (-48), max=49152)
        scale_factor = self._e6m2_round(scale_factor)

        full_scale = scale_factor.expand_as(x_group)
        full_scale = full_scale.flatten(-2, -1)[..., :orig_cols].contiguous()
        return full_scale

    def _extract_group_scale(self, x):
        if self.qparams.desc == "hifx4_1":
            return self._extract_hifx4_1_group_scale(x)

        x = x.float()
        orig_cols = x.shape[-1]
        block = self.qparams.blk_size * self.qparams.blk_outer_size
        pad_cols = (block - orig_cols % block) % block
        if pad_cols > 0:
            x = torch.nn.functional.pad(x, (0, pad_cols), value=0.0)

        x_group = x.unflatten(-1, (-1, 8, 2, 4))
        x_unsigned = torch.abs(x_group)

        max_lv3 = torch.max(x_unsigned, dim=-1, keepdim=True)[0]
        max_lv2 = torch.max(max_lv3, dim=-2, keepdim=True)[0]
        max_lv1 = torch.max(max_lv2, dim=-3, keepdim=True)[0]

        div7 = self._bf16_round(torch.ones_like(max_lv1) / 7.0)
        scale_factor = max_lv1 * div7
        scale_factor = self._bf16_round(scale_factor).clip(min=2 ** (-48), max=49152)

        e_sf = torch.floor(torch.log2(scale_factor))
        mant_sf = scale_factor / torch.exp2(e_sf) * 2 ** 7
        scale_factor = torch.round(mant_sf) / 2 ** 7 * torch.exp2(e_sf)

        scale_factor = self._e6m2_round(scale_factor)

        rec_sf = self._bf16_round(1.0 / scale_factor)
        scale_lv2 = torch.exp2(torch.floor((max_lv2 * rec_sf).clip(0, 4) / 4))
        scale_lv3 = torch.exp2(torch.floor((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2))

        full_scale = (scale_factor * scale_lv2 * scale_lv3).expand_as(x_group)
        full_scale = full_scale.flatten(-4, -1)[..., :orig_cols].contiguous()
        return full_scale

    def forward(self, x, block_size=1):
        del block_size
        if not x.is_contiguous():
            x = x.contiguous()

        if self._cached_scale is not None and x.ndim == 2 and x.shape[0] == 1:
            if self._cached_col >= self._cached_width:
                raise RuntimeError("WeightHiFxQuantizer cached group columns are exhausted.")
            scale = self._cached_scale[:, self._cached_col].view_as(x)
            qx = self._quantize_with_scale(x, scale)
            self._cached_col += 1
            return qx

        qp_in = self.qparams.dim(-1)
        qx = quant_dequant_float(x, qp_in, force_fp32=True)
        return qx.to(x.dtype)

    def find_params(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        self._cached_scale = self._extract_group_scale(x)
        self._cached_width = x.shape[-1]
        self._cached_col = 0

    def ready(self):
        return True


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        w = layer.weight.data.clone()
        self.columns = w.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, token_weights=None):
        del out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        if token_weights is not None:
            token_weights = token_weights.reshape(-1)
            if token_weights.numel() != inp.shape[1]:
                raise ValueError(
                    f"Token importance shape {tuple(token_weights.shape)} does not match "
                    f"captured activation token count {inp.shape[1]}."
                )
            token_weights = token_weights.to(device=inp.device, dtype=inp.dtype)
            if not torch.isfinite(token_weights).all() or torch.any(token_weights < 0):
                raise ValueError("GPTQ token importance weights must be finite and non-negative.")
            inp = inp * torch.sqrt(token_weights).unsqueeze(0)
        self.H += inp.matmul(inp.t())
        if self.H.shape != (self.columns, self.columns):
            raise RuntimeError(
                f"GPTQ Hessian shape changed unexpectedly: {tuple(self.H.shape)}."
            )

    def fasterquant(self, blocksize=128, groupsize=-1, percdamp=0.01):
        W = self.layer.weight.data.clone().float()
        if groupsize == -1:
            self.quantizer.find_params(W)
        elif not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Q = torch.zeros_like(W)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)])

                q = self.quantizer(w.unsqueeze(0)).flatten()
                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if torch.cuda.is_available() and self.layer.weight.is_cuda:
            torch.cuda.synchronize(self.layer.weight.device)

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            raise ValueError("NaN in quantized weights")

    def free(self):
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    logging.info("----- HiFloat4 GPTQ Quantization -----")
    device = torch.device(dev)
    weight_qtype = getattr(args, "hif4_weight_qtype", "hifx4")
    use_act_quant = bool(getattr(args, "hif4a", False))
    act_qtype = getattr(args, "act_quant_qtype", "hifx4")
    act_qparams = QType(act_qtype).dim(-1) if use_act_quant else None
    if weight_qtype == "hifx4_1" and getattr(args, "block_size_linear", 64) != 64:
        raise ValueError("hif4-1 GPTQ requires --block_size_linear 64.")
    logging.info(
        "GPTQ calibration activation mode: %s",
        act_qtype if use_act_quant else "A16/BF16 (disabled)",
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = _get_layers(model)
    dtype = next(iter(model.parameters())).dtype
    max_samples = args.cal_nsamples
    token_importance = getattr(args, "token_importance", "none")
    entropy_direction = getattr(args, "entropy_direction", "low")
    inps, layer_kwargs, valid_token_mask = _capture_calibration_inputs(
        model,
        layers,
        dataloader,
        device,
        max_samples,
        args.cal_seqlen,
        dtype,
    )
    nsamples = inps.shape[0]
    calib_batch_size = min(getattr(args, "gptq_calib_batch_size", 1), nsamples)
    if calib_batch_size <= 0:
        raise ValueError("GPTQ calibration batch size must be greater than 0.")
    logging.info(
        "GPTQ calibration batch size: %d (%d samples, %d batches).",
        calib_batch_size,
        nsamples,
        math.ceil(nsamples / calib_batch_size),
    )
    token_weights = None
    layer_local_weights = None

    if token_importance == "entropy":
        entropy = _compute_fp_token_entropy(model, layers, inps, layer_kwargs, device)
        token_weights = _normalize_entropy_importance(
            entropy,
            valid_token_mask,
            alpha=args.entropy_alpha,
            norm_mode=args.entropy_norm,
            entropy_direction=entropy_direction,
        )
        del entropy, inps

        inps, layer_kwargs, recaptured_mask = _capture_calibration_inputs(
            model,
            layers,
            dataloader,
            device,
            max_samples,
            args.cal_seqlen,
            dtype,
        )
        if inps.shape[0] != nsamples or not torch.equal(valid_token_mask, recaptured_mask):
            raise RuntimeError("Calibration data changed between entropy prepass and GPTQ capture.")
        del recaptured_mask
    elif token_importance in {"entropy_grad", "entropy_grad_norm"}:
        layer_local_weights = _compute_layer_local_token_weights(
            model,
            layers,
            inps,
            layer_kwargs,
            valid_token_mask,
            device,
            alpha=args.importance_alpha,
            mean_normalize=args.importance_mean_normalize,
            batch_size=args.importance_batch_size,
            importance_mode=token_importance,
            entropy_direction=entropy_direction,
        )
    elif token_importance != "none":
        raise ValueError(f"Unsupported GPTQ token importance mode: {token_importance}")

    outs = torch.zeros_like(inps)
    active_token_weights = None

    for i in tqdm.tqdm(range(len(layers)), desc="(GPTQ Quant.) Layers"):
        layer = layers[i].to(device)
        act_quant_handles = (
            _register_activation_quant_pre_hooks(layer, act_qparams)
            if act_qparams is not None
            else []
        )
        logging.info(
            "GPTQ layer %d token-importance Hessian mode: %s; activation calibration: %s",
            i,
            token_importance,
            act_qtype if use_act_quant else "disabled",
        )
        full = find_qlayers(layer, layers=[nn.Linear])
        quant_groups = _get_quant_groups(model, layer)

        for names in quant_groups:
            subset = {name: full[name] for name in names if name in full}
            if not subset:
                continue

            group_token_weights = token_weights
            if layer_local_weights is not None:
                importance_groups = {
                    _local_importance_group_for_linear(name) for name in subset
                }
                if len(importance_groups) != 1:
                    raise RuntimeError(
                        f"GPTQ quantization group mixes token-importance groups: {sorted(subset)}"
                    )
                importance_group = importance_groups.pop()
                group_token_weights = layer_local_weights[i][importance_group]

            gptq_blocks = {}
            for name, sub_layer in subset.items():
                if "lm_head" in name:
                    continue
                gptq_blocks[name] = GPTQ(sub_layer)
                gptq_blocks[name].quantizer = WeightHiFxQuantizer(qtype=weight_qtype)

            if not gptq_blocks:
                continue

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq_blocks[name].add_batch(
                        inp[0].data,
                        out.data,
                        token_weights=active_token_weights,
                    )

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in gptq_blocks]

            for start in range(0, nsamples, calib_batch_size):
                end = min(start + calib_batch_size, nsamples)
                active_token_weights = (
                    group_token_weights[start:end].to(device)
                    if group_token_weights is not None
                    else None
                )
                layer_input = inps[start:end].to(device)
                current_layer_kwargs = _layer_kwargs_for_current_layer(model, layer, layer_input, layer_kwargs)
                outs[start:end].copy_(_run_layer(layer, layer_input, current_layer_kwargs)[0].cpu())

            for handle in handles:
                handle.remove()

            for block in gptq_blocks.values():
                block.fasterquant(
                    percdamp=args.gptq_percdamp,
                    groupsize=getattr(args, "block_size_linear", 64),
                )
                block.free()

        for start in range(0, nsamples, calib_batch_size):
            end = min(start + calib_batch_size, nsamples)
            layer_input = inps[start:end].to(device)
            current_layer_kwargs = _layer_kwargs_for_current_layer(model, layer, layer_input, layer_kwargs)
            outs[start:end].copy_(_run_layer(layer, layer_input, current_layer_kwargs)[0].cpu())

        for handle in act_quant_handles:
            handle.remove()

        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    logging.info("----- HiFloat4 GPTQ Quantization Done -----")
