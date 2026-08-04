"""Uniform affine fake quantization in the style of official OmniQuant."""

import math

import torch
import torch.nn as nn
from HiFloat4.hif4_gpu.quant_cy import QType, quant_dequant_float


CLIPMIN = 1e-5


def round_ste(x):
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits=8,
        symmetric=False,
        dynamic_method="per_channel",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
        quant_format="int4",
    ):
        super().__init__()
        if not 2 <= n_bits <= 16:
            raise ValueError("bitwidth not supported")
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.dynamic_method = dynamic_method
        self.group_size = group_size
        self.disable_zero_point = disable_zero_point
        if quant_format not in {"int4", "hif4"}:
            raise ValueError(f"Unsupported weight quantization format: {quant_format}")
        if quant_format == "hif4" and n_bits != 4:
            raise ValueError("HiF4 weight quantization requires n_bits=4.")
        if quant_format == "hif4" and group_size is not None:
            raise ValueError("HiF4 uses native per-row scaling and does not accept group_size.")
        self.quant_format = quant_format
        self.hif4_qparams = QType("hifx4").dim(-1) if quant_format == "hif4" else None
        self.qmin = -(2 ** (n_bits - 1)) if disable_zero_point else 0
        self.qmax = 2 ** (n_bits - 1) - 1 if disable_zero_point else 2**n_bits - 1
        self.deficiency = 0
        if lwc:
            if shape is None:
                raise ValueError("LWC requires a weight shape.")
            if group_size:
                rows = shape[0] * math.ceil(shape[1] / group_size)
                self.deficiency = (-shape[1]) % group_size
            else:
                rows = shape[0]
            self.upbound_factor = nn.Parameter(torch.full((rows, 1), 4.0))
            self.lowbound_factor = nn.Parameter(torch.full((rows, 1), 4.0))
        self.lwc = lwc
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.n_bits >= 16:
            return x
        original_shape = x.shape
        if self.group_size:
            if x.ndim != 2:
                raise ValueError("Group quantization only supports 2D weights.")
            if self.deficiency:
                x = torch.cat([x, x.new_zeros((x.shape[0], self.deficiency))], dim=1)
            padded_shape = x.shape
            x = x.reshape(-1, self.group_size)
        else:
            padded_shape = original_shape

        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        if self.quant_format == "hif4":
            clipped = torch.minimum(torch.maximum(x, xmin), xmax) if self.lwc else x
            quantized = quant_dequant_float(
                clipped.contiguous(), self.hif4_qparams, force_fp32=True
            ).to(dtype=clipped.dtype)
            out = clipped + (quantized - clipped).detach()
            return out.reshape(original_shape)
        if self.symmetric:
            abs_max = torch.maximum(xmax.abs(), xmin.abs())
            scale = (abs_max / (2 ** (self.n_bits - 1) - 1)).clamp(CLIPMIN, 1e4)
            zero = None if self.disable_zero_point else torch.full_like(scale, 2 ** (self.n_bits - 1) - 1)
        else:
            scale = ((xmax - xmin) / (2**self.n_bits - 1)).clamp(CLIPMIN, 1e4)
            zero = None if self.disable_zero_point else (-xmin / scale).clamp(-1e4, 1e4).round()
        x_int = round_ste(x / scale)
        if zero is not None:
            x_int = x_int + zero
        x_int = x_int.clamp(self.qmin, self.qmax)
        if zero is not None:
            x_int = x_int - zero
        out = x_int * scale
        if self.group_size:
            out = out.reshape(padded_shape)
            if self.deficiency:
                out = out[:, :-self.deficiency]
        return out.reshape(original_shape)
