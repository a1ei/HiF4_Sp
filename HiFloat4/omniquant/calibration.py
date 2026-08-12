"""Layer-wise Qwen3.5 OmniQuant calibration."""

import copy
import gc
import math
import os
from contextlib import nullcontext

import torch
import torch.nn as nn

from HiFloat4.lfq import backward_lfq_chunks
from HiFloat4.hif4gptq.gptq.gptq_utils import (
    _compute_fp_token_entropy,
    _normalize_entropy_importance,
)

from .let import (
    clear_temporary,
    initialize_let,
    let_parameters,
    lwc_parameters,
    omni_parameters,
    omni_state_dict,
    prepare_temporary,
    remove_let_parameters,
    smooth_and_quant_inplace,
)
from .modeling import capture_first_layer_inputs, get_text_model, resolve_model_structure, run_layer
from .modules import materialize_linears, replace_linears, set_quant_state
from .output_importance import compute_output_token_weights


def loss_mode(layer_index, number_of_layers, use_lfq):
    return "lfq" if use_lfq and layer_index == number_of_layers - 1 else "mse"


def _weighted_mse(target, output, token_weights=None):
    if token_weights is None:
        return nn.functional.mse_loss(output, target)
    token_error = (output - target).float().pow(2).mean(dim=-1)
    weights = token_weights.to(device=token_error.device, dtype=token_error.dtype)
    if weights.shape != token_error.shape:
        raise ValueError(
            f"OmniQuant token weight shape {tuple(weights.shape)} does not match "
            f"reconstruction error shape {tuple(token_error.shape)}."
        )
    weight_sum = weights.sum()
    if not torch.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("OmniQuant token weights must have a positive finite sum.")
    return (token_error * weights).sum() / weight_sum


def _grad_norm(parameters):
    gradients = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return float(torch.stack(gradients).norm().item())


def _optimizer(args, layer, mode):
    groups = []
    let_params = let_parameters(layer) if args.let else []
    lwc_params = lwc_parameters(layer) if args.lwc else []
    let_lr = args.lfq_lr if mode == "lfq" else args.let_lr
    lwc_lr = args.lfq_lr if mode == "lfq" else args.lwc_lr
    if let_params:
        groups.append({"params": let_params, "lr": let_lr})
    if lwc_params:
        groups.append({"params": lwc_params, "lr": lwc_lr})
    if not groups:
        raise ValueError("OmniQuant has no trainable LWC or LET parameters.")
    return torch.optim.AdamW(groups, weight_decay=args.wd)


@torch.no_grad()
def _forward_samples(layer, inputs, cached_kwargs, device, dtype):
    outputs = torch.empty_like(inputs)
    for index in range(inputs.shape[0]):
        hidden = inputs[index : index + 1].to(device=device, dtype=dtype)
        outputs[index].copy_(run_layer(layer, hidden, cached_kwargs)[0].detach().cpu())
    return outputs


def _assert_frozen_head(final_norm, lm_head):
    if any(parameter.grad is not None for parameter in final_norm.parameters()):
        raise RuntimeError("LFQ final norm unexpectedly received gradients.")
    if any(parameter.grad is not None for parameter in lm_head.parameters()):
        raise RuntimeError("LFQ LM Head unexpectedly received gradients.")


def omniquant(lm, args, dataloader, logger):
    model = lm.model
    device = lm.device
    layers, final_norm, lm_head = resolve_model_structure(model)
    text_config = get_text_model(model).config
    original_use_cache = text_config.use_cache
    text_config.use_cache = False
    dtype = next(model.parameters()).dtype
    fp_inputs, valid_token_mask, cached_kwargs = capture_first_layer_inputs(
        model, dataloader, args.nsamples, args.seqlen, device, dtype
    )
    quant_inputs = fp_inputs.clone()
    opd_fp_inputs = opd_quant_inputs = opd_valid_token_mask = opd_cached_kwargs = None
    if getattr(args, "opd_dataloader", None) is not None:
        opd_fp_inputs, opd_valid_token_mask, opd_cached_kwargs = capture_first_layer_inputs(
            model, args.opd_dataloader, args.nsamples, args.seqlen, device, dtype
        )
        opd_quant_inputs = opd_fp_inputs.clone()
        logger.info(
            "OPD student-forced corpus loaded: %s samples, %s valid tokens.",
            args.nsamples, int(opd_valid_token_mask.sum().item()),
        )
    token_importance = getattr(args, "token_importance", "none")
    entropy_direction = getattr(args, "entropy_direction", "low")
    token_weights = None
    layer_local_weights = None
    if args.epochs > 0 and token_importance == "entropy":
        entropy = _compute_fp_token_entropy(
            model, layers, fp_inputs, cached_kwargs, device
        )
        token_weights = _normalize_entropy_importance(
            entropy,
            valid_token_mask,
            alpha=args.entropy_alpha,
            norm_mode=args.entropy_norm,
            entropy_direction=entropy_direction,
        )
        del entropy
    elif args.epochs > 0 and token_importance in {"entropy_grad", "entropy_grad_norm"}:
        layer_local_weights = compute_output_token_weights(
            model,
            layers,
            fp_inputs,
            cached_kwargs,
            valid_token_mask,
            device,
            alpha=args.importance_alpha,
            mean_normalize=args.importance_mean_normalize,
            batch_size=args.importance_batch_size,
            importance_mode=token_importance,
            entropy_direction=entropy_direction,
        )
    elif token_importance != "none" and token_importance not in {"entropy", "entropy_grad", "entropy_grad_norm"}:
        raise ValueError(f"Unsupported OmniQuant token importance mode: {token_importance}")
    logger.info("OmniQuant token-importance reconstruction mode: %s", token_importance)
    omni_states = torch.load(args.resume, map_location="cpu") if args.resume else {}
    amp_context = (
        nullcontext
        if dtype == torch.float32
        else lambda: torch.amp.autocast(device_type="cuda", dtype=dtype)
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    for layer_index in range(len(layers)):
        mode = loss_mode(layer_index, len(layers), args.lfq)
        if mode == "lfq" and opd_fp_inputs is not None:
            fp_inputs = opd_fp_inputs
            quant_inputs = opd_quant_inputs
            valid_token_mask = opd_valid_token_mask
            cached_kwargs = opd_cached_kwargs
            logger.info("Final layer switched to OPD student-forced calibration tokens.")
        layer_token_weights = token_weights
        if layer_local_weights is not None:
            layer_token_weights = layer_local_weights[layer_index]
        logger.info("=== Start quantize layer %s (%s) ===", layer_index, mode.upper())
        layer = layers[layer_index].to(device)
        with torch.no_grad(), amp_context():
            fp_outputs = _forward_samples(layer, fp_inputs, cached_kwargs, device, dtype)
            opd_fp_outputs = (
                _forward_samples(layer, opd_fp_inputs, opd_cached_kwargs, device, dtype)
                if opd_fp_inputs is not None and mode == "mse"
                else None
            )
            fp_outputs_from_quant = (
                _forward_samples(layer, quant_inputs, cached_kwargs, device, dtype)
                if args.aug_loss and mode == "mse"
                else None
            )

        quant_layer = copy.deepcopy(layer)
        replace_linears(quant_layer, args.weight_quant_params, args.act_quant_params)
        quant_layer.to(device)
        set_quant_state(quant_layer, weight_quant=False, act_quant=args.abits < 16)
        for parameter in quant_layer.parameters():
            parameter.requires_grad = False

        if args.let:
            with amp_context():
                initialize_let(
                    quant_layer,
                    fp_inputs.to(device=device, dtype=dtype),
                    lambda current_layer, hidden: run_layer(current_layer, hidden, cached_kwargs),
                    args.alpha,
                )
        for parameter in omni_parameters(quant_layer):
            parameter.requires_grad = True
        if args.resume:
            quant_layer.load_state_dict(omni_states[layer_index], strict=False)

        if mode == "lfq":
            final_norm.to(device)
            lm_head.to(device)
            final_norm.requires_grad_(False)
            lm_head.requires_grad_(False)

        if args.epochs > 0:
            optimizer = _optimizer(args, quant_layer, mode)
            trainable = omni_parameters(quant_layer)
            steps = math.ceil(args.nsamples / args.batch_size)
            for epoch in range(args.epochs):
                epoch_loss = 0.0
                epoch_agreement = 0.0
                epoch_tokens = 0
                epoch_grad_norm = 0.0
                for step in range(steps):
                    start = step * args.batch_size
                    end = min(start + args.batch_size, args.nsamples)
                    optimizer.zero_grad(set_to_none=True)
                    prepare_temporary(quant_layer, args.let)
                    with amp_context():
                        student_input = quant_inputs[start:end].to(device=device, dtype=dtype)
                        quant_output = run_layer(quant_layer, student_input, cached_kwargs)
                        if mode == "mse":
                            target = fp_outputs[start:end].to(device=device, dtype=quant_output.dtype)
                            loss = _weighted_mse(
                                target, quant_output, layer_token_weights[start:end]
                                if layer_token_weights is not None else None
                            )
                            if fp_outputs_from_quant is not None:
                                second_target = fp_outputs_from_quant[start:end].to(
                                    device=device, dtype=quant_output.dtype
                                )
                                loss = loss + _weighted_mse(
                                    second_target, quant_output, layer_token_weights[start:end]
                                    if layer_token_weights is not None else None
                                )
                    if mode == "mse":
                        loss.backward()
                        epoch_loss += float(loss.detach().item())
                        epoch_tokens += end - start
                    else:
                        metrics = backward_lfq_chunks(
                            fp_outputs[start:end].to(device=device, dtype=dtype),
                            quant_output,
                            final_norm,
                            lm_head,
                            valid_token_mask[start:end],
                            args.lfq_logits_chunk_size,
                        )
                        epoch_loss += metrics.cross_entropy * metrics.valid_tokens
                        epoch_agreement += metrics.top1_agreement * metrics.valid_tokens
                        epoch_tokens += metrics.valid_tokens
                        _assert_frozen_head(final_norm, lm_head)
                    norm = _grad_norm(trainable)
                    if not math.isfinite(norm) or norm == 0.0:
                        raise RuntimeError(
                            f"Layer {layer_index} has invalid OmniQuant gradient norm: {norm}."
                        )
                    epoch_grad_norm += norm
                    optimizer.step()
                    clear_temporary(quant_layer)

                if mode == "mse":
                    logger.info(
                        "layer %s epoch %s MSE %.8e grad_norm %.8e",
                        layer_index,
                        epoch,
                        epoch_loss / steps,
                        epoch_grad_norm / steps,
                    )
                else:
                    logger.info(
                        "layer %s epoch %s LFQ cross-entropy %.8f FP/Q top-1 agreement %.8f "
                        "valid tokens %s grad_norm %.8e",
                        layer_index,
                        epoch,
                        epoch_loss / epoch_tokens,
                        epoch_agreement / epoch_tokens,
                        epoch_tokens,
                        epoch_grad_norm / steps,
                    )
            del optimizer

        omni_states[layer_index] = omni_state_dict(quant_layer)
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(omni_states, os.path.join(args.output_dir, "omni_parameters.pth"))
        smooth_and_quant_inplace(quant_layer, args.let)
        remove_let_parameters(quant_layer)
        materialize_linears(quant_layer)
        quant_layer.requires_grad_(False)
        layers[layer_index] = quant_layer
        with torch.no_grad(), amp_context():
            quant_inputs = _forward_samples(quant_layer, quant_inputs, cached_kwargs, device, dtype)
            if opd_quant_inputs is not None and mode == "mse":
                opd_quant_inputs = _forward_samples(
                    quant_layer, opd_quant_inputs, opd_cached_kwargs, device, dtype
                )
        fp_inputs = fp_outputs
        if opd_fp_outputs is not None:
            opd_fp_inputs = opd_fp_outputs
        layers[layer_index] = quant_layer.cpu()
        if mode == "lfq":
            final_norm.cpu()
            lm_head.cpu()
        del layer, quant_layer, fp_outputs, fp_outputs_from_quant, opd_fp_outputs
        torch.cuda.empty_cache()

    text_config.use_cache = original_use_cache
    del fp_inputs, quant_inputs
    gc.collect()
    torch.cuda.empty_cache()
    return model
