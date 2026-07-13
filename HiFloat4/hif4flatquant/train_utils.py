import functools
import logging
import os
import pathlib
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import tqdm

HIF4_ROOT = pathlib.Path(__file__).resolve().parents[1]
HIF4GPTQ_ROOT = HIF4_ROOT / "hif4gptq"
if str(HIF4GPTQ_ROOT) not in sys.path:
    sys.path.append(str(HIF4GPTQ_ROOT))

from gptq.gptq_utils import (
    _get_layers,
    _is_qwen3_5_text_model,
    _layer_kwargs_for_current_layer,
    _run_layer,
    _validate_no_padding_attention_mask,
)

from .flat_utils import load_flat_matrices, reparameterize_model, save_flat_matrices
from .function_utils import get_n_set_parameters_byname, get_paras_dict_by_name, set_require_grad_all
from .qwen3_5_utils import apply_flatquant_to_qwen3_5


def _token_mixer(layer):
    return layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn


def _set_ori_mode(layer, enabled):
    _token_mixer(layer)._ori_mode = enabled
    layer.mlp._ori_mode = enabled


def _capture_first_layer_inputs(model, dataloader, device, args):
    layers = _get_layers(model)
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype
    max_samples = args.cal_nsamples
    inps = torch.zeros(
        (max_samples, args.cal_seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            idx = cache["i"]
            if idx < max_samples:
                inps[idx] = inp
            cache["i"] += 1
            cache.update(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache["i"] >= max_samples:
            break
        if _is_qwen3_5_text_model(model):
            _validate_no_padding_attention_mask(batch)
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0]
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
        else:
            input_ids = batch
        try:
            model(input_ids.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    nsamples = min(cache["i"], max_samples)
    if nsamples == 0:
        raise RuntimeError("FlatQuant calibration dataloader produced zero samples.")
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    return layers, inps[:nsamples], {k: v for k, v in cache.items() if k != "i"}, nsamples


def _trainable_groups(args, layer):
    groups = []
    if args.flatquant_cali_trans:
        groups.append({"params": get_n_set_parameters_byname(layer, ["trans.linear"]), "lr": args.flatquant_lr})
    if args.flatquant_add_diag:
        groups.append({"params": get_n_set_parameters_byname(layer, ["trans.diag_scale"]), "lr": args.flatquant_lr})
    if args.flatquant_lwc:
        groups.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_w"]), "lr": args.flatquant_lr * 10})
    if args.flatquant_lac:
        groups.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_a"]), "lr": args.flatquant_lr * 10})
    groups = [group for group in groups if group["params"]]
    if not groups:
        raise ValueError("FlatQuant has no trainable calibration parameters.")
    return groups


def _save_flat_parameters(model, path, upto):
    params = {}
    for idx in range(upto + 1):
        params[idx] = get_paras_dict_by_name(
            model.model.layers[idx],
            required_names=["trans.linear", "trans.diag_scale", "clip_factor_w", "clip_factor_a"],
        )
    torch.save(params, os.path.join(path, "flat_parameters.pth"))


def cali_flat_quant(args, model, dataloader, device):
    layers, fp_inps, layer_kwargs, nsamples = _capture_first_layer_inputs(model, dataloader, device, args)
    if nsamples % args.flatquant_cali_bsz != 0:
        raise ValueError("--cal_nsamples must be divisible by --flatquant_cali_bsz.")
    fp_outs = torch.zeros_like(fp_inps)
    steps_per_epoch = nsamples // args.flatquant_cali_bsz
    loss_func = nn.MSELoss()
    traincast = nullcontext if args.dtype == "float32" else functools.partial(
        torch.amp.autocast, device_type="cuda", dtype=torch.bfloat16
    )
    for idx in tqdm.tqdm(range(len(layers)), desc="(FlatQuant Calib.) Layers"):
        logging.info("========= FlatQuant Layer %s =========", idx)
        layer = layers[idx].to(device)
        _set_ori_mode(layer, True)
        with torch.no_grad():
            for sample_idx in range(nsamples):
                inp = fp_inps[sample_idx].unsqueeze(0)
                kwargs = _layer_kwargs_for_current_layer(model, layer, inp, layer_kwargs)
                fp_outs[sample_idx] = _run_layer(layer, inp, kwargs)
        _set_ori_mode(layer, False)
        if args.flatquant_add_diag:
            _token_mixer(layer).init_diag_scale(alpha=args.flatquant_diag_alpha)
            layer.mlp.init_diag_scale(alpha=args.flatquant_diag_alpha)
        set_require_grad_all(layer, False)
        optimizer = torch.optim.AdamW(_trainable_groups(args, layer))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.flatquant_epochs * steps_per_epoch,
            eta_min=args.flatquant_lr * 1e-3,
        )
        for epoch in range(args.flatquant_epochs):
            mse = 0.0
            start = time.time()
            with traincast():
                for batch_idx in range(steps_per_epoch):
                    begin = batch_idx * args.flatquant_cali_bsz
                    end = begin + args.flatquant_cali_bsz
                    inp = fp_inps[begin:end].detach()
                    kwargs = _layer_kwargs_for_current_layer(model, layer, inp, layer_kwargs)
                    quant_out = _run_layer(layer, inp, kwargs)
                    loss = loss_func(fp_outs[begin:end].detach(), quant_out)
                    mse += loss.detach().float().item()
                    loss = loss / loss.detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            logging.info(
                "FlatQuant layer %s epoch %s lr %.8f time %.2fs mse %.8f",
                idx, epoch, scheduler.get_last_lr()[0], time.time() - start, mse,
            )
        layers[idx] = layer.cpu()
        _save_flat_parameters(model, args.gptq_save_path, idx)
        fp_inps, fp_outs = fp_outs, fp_inps
        del layer
        torch.cuda.empty_cache()


def flatquant_fwrd(model, dataloader, dev, args):
    logging.info("----- HiFloat4 FlatQuant Weight + Activation Quantization -----")
    device = torch.device(dev)
    if device.type != "cuda":
        raise RuntimeError("HiF4 FlatQuant requires CUDA.")
    if not args.gptq_save_path:
        raise ValueError("--gptq_save_path is required for FlatQuant.")
    os.makedirs(args.gptq_save_path, exist_ok=True)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        apply_flatquant_to_qwen3_5(args, model)
        if args.flatquant_matrix_path:
            load_flat_matrices(model, args.flatquant_matrix_path)
        else:
            cali_flat_quant(args, model, dataloader, device)
        save_flat_matrices(model, args.gptq_save_path)
        reparameterize_model(model, device=device)
    finally:
        model.config.use_cache = use_cache
    logging.info("----- HiFloat4 FlatQuant Quantization Done -----")
