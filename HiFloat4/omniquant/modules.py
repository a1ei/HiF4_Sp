"""Fake-quantized linear modules used during OmniQuant calibration."""

import torch.nn as nn
import torch.nn.functional as F

from .quantizer import UniformAffineQuantizer


class QuantLinear(nn.Module):
    def __init__(self, linear, weight_quant_params, act_quant_params):
        super().__init__()
        self.register_buffer("weight", linear.weight.detach().clone())
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().clone())
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight_quantizer = UniformAffineQuantizer(shape=self.weight.shape, **weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.use_weight_quant = False
        self.use_act_quant = False
        self.use_temporary_parameter = False

    def forward(self, x):
        if self.use_temporary_parameter:
            weight, bias = self.temp_weight, self.temp_bias
        elif self.use_weight_quant:
            weight, bias = self.weight_quantizer(self.weight), self.bias
        else:
            weight, bias = self.weight, self.bias
        if self.use_act_quant:
            x = self.act_quantizer(x)
        return F.linear(x, weight, bias)

    def set_quant_state(self, weight_quant=False, act_quant=False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


def replace_linears(module, weight_quant_params, act_quant_params):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child, weight_quant_params, act_quant_params))
        else:
            replace_linears(child, weight_quant_params, act_quant_params)
    return module


def iter_quant_linears(module):
    for child in module.modules():
        if isinstance(child, QuantLinear):
            yield child


def set_quant_state(module, weight_quant=False, act_quant=False):
    for linear in iter_quant_linears(module):
        linear.set_quant_state(weight_quant, act_quant)


def materialize_linears(module):
    for name, child in list(module.named_children()):
        if isinstance(child, QuantLinear):
            linear = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
            linear = linear.to(device=child.weight.device, dtype=child.weight.dtype)
            linear.weight.data.copy_(child.weight)
            if child.bias is not None:
                linear.bias.data.copy_(child.bias)
            linear.requires_grad_(False)
            setattr(module, name, linear)
        else:
            materialize_linears(child)
    return module
