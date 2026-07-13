import logging
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
import tqdm

HIF4_ROOT = pathlib.Path(__file__).resolve().parents[1]
HIF4GPTQ_ROOT = HIF4_ROOT / "hif4gptq"
if str(HIF4_ROOT) not in sys.path:
    sys.path.append(str(HIF4_ROOT))
if str(HIF4GPTQ_ROOT) not in sys.path:
    sys.path.append(str(HIF4GPTQ_ROOT))

from gptq.gptq_utils import (
    WeightHiFxQuantizer,
    _get_layers,
    _get_quant_groups,
    _is_qwen3_5_text_model,
    _layer_kwargs_for_current_layer,
    _run_layer,
    _validate_no_padding_attention_mask,
    find_qlayers,
)

from .magr import W_proximal_preprocess_groupwise_xtx, W_proximal_preprocess_xtx


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class MagRGPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        del out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.XtX += inp.float().matmul(inp.float().t())
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def _quantize_column(self, w):
        return self.quantizer(w.unsqueeze(0)).flatten()

    def _run_magr_preprocess(self, W, groupsize, alpha, alpha_groupwise, n_iter):
        if self.nsamples == 0:
            raise RuntimeError("MagR requires calibration inputs before quantization.")
        if groupsize != -1:
            logging.info("MagR proximal preprocessing: per group.")
            return W_proximal_preprocess_groupwise_xtx(
                W,
                self.XtX,
                alpha=alpha_groupwise,
                n_iter=n_iter,
                group_size=groupsize,
            )

        logging.info("MagR proximal preprocessing: per layer.")
        return W_proximal_preprocess_xtx(
            W,
            self.XtX,
            alpha=alpha,
            n_iter=n_iter,
        )

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        magr=True,
        CD_iter=1,
        magr_alpha=0.001,
        magr_alpha_groupwise=0.0001,
        magr_preprocess_iter=200,
    ):
        W = self.layer.weight.data.clone().float()
        W_orig = W.clone()

        if magr_preprocess_iter < 0:
            raise ValueError("--magr_preprocess_iter must be >= 0.")

        if magr and magr_preprocess_iter > 0:
            W = self._run_magr_preprocess(
                W,
                groupsize,
                magr_alpha,
                magr_alpha_groupwise,
                magr_preprocess_iter,
            )
        elif magr:
            logging.info("MagR proximal preprocessing skipped.")

        tick = time.time()

        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if groupsize == -1:
            self.quantizer.find_params(W)

        Losses = torch.zeros_like(W)
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
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)])

                q = self._quantize_column(w)
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if torch.cuda.is_available() and self.layer.weight.is_cuda:
            torch.cuda.synchronize(self.layer.weight.device)

        logging.info("MagR GPTQ time %.2f", time.time() - tick)
        logging.info("MagR GPTQ error %s", torch.sum(Losses).item())

        del H, Hinv, W1, Q1, Err1, Losses1, Hinv1

        if CD_iter > 0:
            Q = self._coordinate_descent_refine(Q, W_orig, groupsize, CD_iter)

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            raise ValueError("NaN in MagR quantized weights.")

    def _coordinate_descent_refine(self, Q, W_orig, groupsize, CD_iter):
        logging.info("MagR COMQ coordinate descent iterations: %s", CD_iter)
        Q_CD = Q.reshape(self.layer.weight.shape).clone()

        self.XtX = self.XtX.to(self.dev)
        diag_Sigma = torch.diagonal(self.XtX, 0)
        diag_Sigma += 0.75 * torch.mean(torch.diagonal(self.XtX))
        norm_Sigma = torch.div(self.XtX, diag_Sigma + 0.1)

        P = torch.matmul(W_orig, norm_Sigma)

        norm_Sigma.fill_diagonal_(0)
        norm_Sigma = norm_Sigma.t()

        for _ in range(CD_iter):
            if groupsize == -1:
                self.quantizer.find_params(Q_CD)

            delta_Q = Q_CD.clone().t()
            P_hat = torch.matmul(norm_Sigma, delta_Q).t()

            for j in range(self.columns):
                u = P[:, j] - P_hat[:, j]

                if j > 0:
                    u += torch.matmul(norm_Sigma[j, :j], delta_Q[:j, :])

                if groupsize != -1 and j % groupsize == 0:
                    self.quantizer.find_params(Q[:, j : (j + groupsize)])

                u = self._quantize_column(u)
                Q_CD[:, j] = u
                delta_Q[j, :] -= u

        return Q_CD

    def free(self):
        self.H = None
        self.XtX = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def magr_fwrd(model, dataloader, dev, args):
    logging.info("----- HiFloat4 MagR Weight Quantization -----")
    device = torch.device(dev)
    if device.type != "cuda":
        raise RuntimeError("HiF4 MagR requires CUDA because HiF4 quantization uses a CUDA kernel.")

    weight_qtype = getattr(args, "hif4_weight_qtype", "hifx4")
    if weight_qtype == "hifx4_1" and getattr(args, "block_size_linear", 64) != 64:
        raise ValueError("hif4-1 MagR requires --block_size_linear 64.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = _get_layers(model)

    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    max_samples = args.gptq_cal_nsamples
    inps = torch.zeros(
        (max_samples, args.gptq_cal_seqlen, model.config.hidden_size),
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
        raise RuntimeError("Calibration dataloader produced zero samples.")

    inps = inps[:nsamples]
    outs = torch.zeros_like(inps)

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
    groupsize = getattr(args, "block_size_linear", 64)
    CD_iter = getattr(args, "magr_cd_iter", 1)
    magr_alpha = getattr(args, "magr_alpha", 0.001)
    magr_alpha_groupwise = getattr(args, "magr_alpha_groupwise", 0.0001)
    magr_preprocess_iter = getattr(args, "magr_preprocess_iter", 200)
    exclude_layers = getattr(args, "exclude_layers", ["lm_head"])

    for i in tqdm.tqdm(range(len(layers)), desc="(MagR Quant.) Layers"):
        layer = layers[i].to(device)
        full = find_qlayers(layer, layers=[nn.Linear])
        quant_groups = _get_quant_groups(model, layer)

        for names in quant_groups:
            subset = {
                name: full[name]
                for name in names
                if name in full and name not in exclude_layers and "lm_head" not in name
            }
            if not subset:
                continue

            magr_blocks = {}
            for name, sub_layer in subset.items():
                magr_blocks[name] = MagRGPTQ(sub_layer)
                magr_blocks[name].quantizer = WeightHiFxQuantizer(qtype=weight_qtype)

            def add_batch(name):
                def tmp(_, inp, out):
                    magr_blocks[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in magr_blocks]

            for j in range(nsamples):
                layer_input = inps[j].unsqueeze(0)
                current_layer_kwargs = _layer_kwargs_for_current_layer(
                    model, layer, layer_input, layer_kwargs
                )
                outs[j] = _run_layer(layer, layer_input, current_layer_kwargs)

            for handle in handles:
                handle.remove()

            for name, block in magr_blocks.items():
                logging.info("MagR quantizing layer %s.%s", i, name)
                block.fasterquant(
                    percdamp=args.gptq_percdamp,
                    groupsize=groupsize,
                    magr=True,
                    CD_iter=CD_iter,
                    magr_alpha=magr_alpha,
                    magr_alpha_groupwise=magr_alpha_groupwise,
                    magr_preprocess_iter=magr_preprocess_iter,
                )
                block.free()

        for j in range(nsamples):
            layer_input = inps[j].unsqueeze(0)
            current_layer_kwargs = _layer_kwargs_for_current_layer(
                model, layer, layer_input, layer_kwargs
            )
            outs[j] = _run_layer(layer, layer_input, current_layer_kwargs)

        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    logging.info("----- HiFloat4 MagR Weight Quantization Done -----")
