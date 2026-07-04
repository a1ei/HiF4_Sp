import math

import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.hif4_fake import (
    hif4_fake_quantize_hifx4,
    hif4_fake_quantize_hifx4_1,
)


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            return a - b, a + b
        a += 1


def kronecker_matmul(x, matrix_left, matrix_right):
    init_shape = x.shape
    x = x.reshape(-1, matrix_left.shape[0], matrix_right.shape[0])
    x = torch.matmul(x, matrix_right)
    x = torch.matmul(matrix_left.t(), x)
    return x.reshape(init_shape)


class EvalDecomposeTransMatrix(nn.Module):
    def __init__(self, size, add_diag, use_diag):
        super().__init__()
        left_size, right_size = get_decompose_dim(size)
        self.matrix_left = nn.Parameter(torch.eye(left_size), requires_grad=False)
        self.matrix_right = nn.Parameter(torch.eye(right_size), requires_grad=False)
        self.matrix_left_inv = nn.Parameter(torch.eye(left_size), requires_grad=False)
        self.matrix_right_inv = nn.Parameter(torch.eye(right_size), requires_grad=False)
        self.add_diag = add_diag
        self.use_diag = use_diag
        if add_diag:
            self.diag_scale = nn.Parameter(torch.ones(size), requires_grad=False)

    def forward(self, inp):
        if self.add_diag and self.use_diag:
            inp = inp * self.diag_scale.to(inp)
        return kronecker_matmul(inp, self.matrix_left.to(inp), self.matrix_right.to(inp))


class Hif4ActivationQuantizer(nn.Module):
    def __init__(self, lac, qtype):
        super().__init__()
        self.lac = lac
        self.qtype = qtype
        if qtype not in {"hifx4", "hifx4_1"}:
            raise ValueError(f"Unsupported HiF4 FlatQuant activation qtype: {qtype}")
        if lac:
            self.clip_factor_a_max = nn.Parameter(torch.ones(1) * 4.0, requires_grad=False)
            self.clip_factor_a_min = nn.Parameter(torch.ones(1) * 4.0, requires_grad=False)

    def forward(self, x):
        if self.lac:
            reshaped = x.reshape(-1, x.shape[-1])
            xmax = reshaped.amax(1, keepdim=True).clamp(min=0)
            xmin = reshaped.amin(1, keepdim=True).clamp(max=0)
            xmax = xmax * torch.sigmoid(self.clip_factor_a_max)
            xmin = xmin * torch.sigmoid(self.clip_factor_a_min)
            x = torch.clamp(reshaped, min=xmin, max=xmax).reshape_as(x)
        if self.qtype == "hifx4_1":
            return hif4_fake_quantize_hifx4_1(x)
        return hif4_fake_quantize_hifx4(x)
