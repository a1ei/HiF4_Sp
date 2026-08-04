"""Qwen3.5 model structure and block-forward helpers."""

import inspect

import torch
import torch.nn as nn


def get_text_model(model):
    model_type = getattr(model.config, "model_type", None)
    if model_type == "qwen3_5_text":
        return model.model
    if model_type == "qwen3_5" and hasattr(model.model, "language_model"):
        return model.model.language_model
    raise NotImplementedError("Only Qwen3.5 dense text backbones are supported.")


def resolve_model_structure(model):
    text_model = get_text_model(model)
    if not hasattr(text_model, "layers"):
        raise AttributeError("Qwen3.5 transformer blocks were not found.")
    if not hasattr(text_model, "norm") or not hasattr(model, "lm_head"):
        raise AttributeError("Qwen3.5 final norm or LM Head was not found.")
    return text_model.layers, text_model.norm, model.lm_head


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    return value


def _expand_batch(value, batch_size):
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1 and batch_size > 1:
        return value.expand(batch_size, *value.shape[1:])
    if isinstance(value, tuple):
        return tuple(_expand_batch(item, batch_size) for item in value)
    return value


def layer_kwargs(layer, cached_kwargs, hidden_states):
    accepted = inspect.signature(layer.forward).parameters
    result = {}
    for name, value in cached_kwargs.items():
        if name in accepted and name not in {"hidden_states", "past_key_values", "cache_params"}:
            result[name] = _move(_expand_batch(value, hidden_states.shape[0]), hidden_states.device)
    if layer.layer_type == "linear_attention":
        result["attention_mask"] = None
    return result


def run_layer(layer, hidden_states, cached_kwargs):
    output = layer(hidden_states, **layer_kwargs(layer, cached_kwargs, hidden_states))
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def capture_first_layer_inputs(model, dataloader, nsamples, seqlen, device, dtype):
    layers, _, _ = resolve_model_structure(model)
    text_model = get_text_model(model)
    text_model.embed_tokens.to(device)
    text_model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)
    inputs = torch.zeros((nsamples, seqlen, text_model.config.hidden_size), dtype=dtype, device="cpu")
    masks = torch.ones((nsamples, seqlen), dtype=torch.bool, device="cpu")
    cache = {"index": 0, "kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            index = cache["index"]
            inputs[index].copy_(hidden_states[0].detach().cpu())
            cache["kwargs"] = {key: _detach_to_cpu(value) for key, value in kwargs.items()}
            cache["index"] += 1
            raise ValueError("capture complete")

    layers[0] = Catcher(layers[0])
    try:
        for batch in dataloader:
            if cache["index"] >= nsamples:
                break
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
            elif isinstance(batch, (tuple, list)):
                input_ids, attention_mask = batch[0], None
            else:
                input_ids, attention_mask = batch, None
            if input_ids.shape != (1, seqlen):
                raise ValueError(f"Calibration sample must have shape [1, {seqlen}].")
            if attention_mask is not None:
                if attention_mask.shape != input_ids.shape:
                    raise ValueError("Calibration attention mask must match input_ids.")
                masks[cache["index"]].copy_(attention_mask[0].bool())
            try:
                model(input_ids=input_ids.to(device), attention_mask=None if attention_mask is None else attention_mask.to(device))
            except ValueError as error:
                if str(error) != "capture complete":
                    raise
    finally:
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        text_model.embed_tokens.cpu()
        text_model.rotary_emb.cpu()
    count = cache["index"]
    if count != nsamples:
        raise RuntimeError(f"Expected {nsamples} calibration samples, captured {count}.")
    return inputs, masks, cache["kwargs"]
