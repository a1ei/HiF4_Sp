"""Block-output token attribution used only by OmniQuant reconstruction."""

import logging
import math

import torch
import tqdm

from HiFloat4.hif4gptq.gptq.gptq_utils import (
    _build_entropy_seed,
    _get_final_norm,
    _layer_kwargs_for_current_layer,
    _margin_anchor_loss,
    _run_layer,
)


def _output_importance(output, gradient, mode):
    if output.shape != gradient.shape:
        raise RuntimeError("Block output and output gradient shapes do not match.")
    if mode == "entropy_grad":
        signal = output.float() * gradient.float()
    elif mode == "entropy_grad_norm":
        signal = gradient.float()
    else:
        raise ValueError(f"Unsupported OmniQuant output importance mode: {mode}")
    importance = torch.linalg.vector_norm(signal, dim=-1)
    if importance.ndim == 1:
        importance = importance.unsqueeze(0)
    if importance.ndim != 2 or not torch.isfinite(importance).all():
        raise RuntimeError("OmniQuant output importance must be finite [batch, sequence].")
    return importance


class _BlockOutputAttribution(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, hidden_states, layer, model, layer_kwargs, device,
        importance_store, layer_idx, sample_start, importance_mode,
    ):
        ctx.layer = layer
        ctx.model = model
        ctx.layer_kwargs = layer_kwargs
        ctx.device = device
        ctx.importance_store = importance_store
        ctx.layer_idx = layer_idx
        ctx.sample_start = sample_start
        ctx.importance_mode = importance_mode
        ctx.save_for_backward(hidden_states.detach())
        layer = layer.to(device)
        try:
            layer_input = hidden_states.to(device)
            kwargs = _layer_kwargs_for_current_layer(model, layer, layer_input, layer_kwargs)
            output_cpu = _run_layer(layer, layer_input, kwargs).detach().cpu()
        finally:
            layer.cpu()
        return output_cpu

    @staticmethod
    def backward(ctx, grad_output):
        (hidden_states,) = ctx.saved_tensors
        layer = ctx.layer.to(ctx.device)
        with torch.enable_grad():
            layer_input = hidden_states.to(ctx.device).detach().requires_grad_(True)
            kwargs = _layer_kwargs_for_current_layer(
                ctx.model, layer, layer_input, ctx.layer_kwargs
            )
            output = _run_layer(layer, layer_input, kwargs)
        device_gradient = grad_output.to(ctx.device)
        grad_input = torch.autograd.grad(
            output, layer_input, grad_outputs=device_gradient,
            retain_graph=False, create_graph=False,
        )[0]
        importance = _output_importance(output.detach(), device_gradient, ctx.importance_mode)
        for batch_idx in range(importance.shape[0]):
            ctx.importance_store[ctx.layer_idx][ctx.sample_start + batch_idx] = (
                importance[batch_idx : batch_idx + 1].cpu()
            )
        layer.cpu()
        return grad_input.detach().cpu(), None, None, None, None, None, None, None, None


def compute_output_token_weights(
    model, layers, inputs, layer_kwargs, valid_token_mask, device,
    alpha, mean_normalize, batch_size=1, importance_mode="entropy_grad",
    entropy_direction="low",
):
    """Return per-layer weights from each block output y, not projection inputs x."""
    if importance_mode not in {"entropy_grad", "entropy_grad_norm"}:
        raise ValueError(f"Unsupported OmniQuant output importance mode: {importance_mode}")
    nsamples = inputs.shape[0]
    batch_size = min(batch_size, nsamples)
    raw_importance = [[None] * nsamples for _ in layers]
    requires_grad_state = [parameter.requires_grad for parameter in model.parameters()]
    model.requires_grad_(False)
    norm = _get_final_norm(model)
    output_head = model.get_output_embeddings()
    if output_head is None:
        raise RuntimeError("Output-gradient OmniQuant requires an LM Head.")
    if norm is not None:
        norm.to(device)
    output_head.to(device)
    try:
        for sample_start in tqdm.tqdm(
            range(0, nsamples, batch_size),
            total=math.ceil(nsamples / batch_size),
            desc=f"(OmniQuant output-y {importance_mode}) Batches",
        ):
            sample_end = min(sample_start + batch_size, nsamples)
            with torch.enable_grad():
                hidden = inputs[sample_start:sample_end].detach().requires_grad_(True)
                for layer_idx, layer in enumerate(layers):
                    hidden = _BlockOutputAttribution.apply(
                        hidden, layer, model, layer_kwargs, device, raw_importance,
                        layer_idx, sample_start, importance_mode,
                    )
                hidden_device = hidden.to(device)
                normalized_hidden = norm(hidden_device) if norm is not None else hidden_device
                sample_mask = valid_token_mask[sample_start:sample_end].to(device)
                chunk_size = max(1, 128 // (sample_end - sample_start))
                seed = _build_entropy_seed(
                    normalized_hidden,
                    output_head,
                    sample_mask,
                    logits_chunk_size=chunk_size,
                    entropy_direction=entropy_direction,
                )
                anchor = _margin_anchor_loss(
                    normalized_hidden, output_head, seed, logits_chunk_size=chunk_size
                )
                anchor.backward()
            del hidden, hidden_device, normalized_hidden, sample_mask, seed, anchor
            torch.cuda.empty_cache()
    finally:
        output_head.cpu()
        if norm is not None:
            norm.cpu()
        for layer in layers:
            layer.cpu()
        for parameter, requires_grad in zip(model.parameters(), requires_grad_state):
            parameter.requires_grad_(requires_grad)

    weights = []
    for layer_idx, samples in enumerate(raw_importance):
        if any(sample is None for sample in samples):
            raise RuntimeError(f"Layer {layer_idx} has missing output attribution.")
        importance = torch.cat(samples, dim=0).float()
        values = importance[valid_token_mask]
        value_range = values.max() - values.min()
        normalized = (
            (values - values.min()) / (value_range + 1e-12)
            if value_range > 0 else torch.zeros_like(values)
        )
        weight = torch.zeros_like(importance)
        weight[valid_token_mask] = 1.0 + alpha * normalized
        if mean_normalize:
            weight[valid_token_mask] /= weight[valid_token_mask].mean()
        if not torch.isfinite(weight).all() or torch.any(weight < 0):
            raise RuntimeError(f"Layer {layer_idx} has invalid output-gradient weights.")
        logging.info(
            "Layer %d output-y %s weights mean=%.6f std=%.6f min=%.6f max=%.6f",
            layer_idx, importance_mode, weight[valid_token_mask].mean().item(),
            weight[valid_token_mask].std(unbiased=False).item(),
            weight[valid_token_mask].min().item(), weight[valid_token_mask].max().item(),
        )
        weights.append(weight)
    return weights
