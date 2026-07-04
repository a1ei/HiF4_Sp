import torch
import torch.nn as nn
import torch.nn.functional as F

from hif4_gpu.quant_cy import QType, quant_dequant_float, quant_func

from .flat_utils import kronecker_matmul


class HiF4ActivationQuantizer(nn.Module):
    def __init__(self, lac=False, qtype="hifx4"):
        super().__init__()
        self.lac = lac
        if lac:
            self.sigmoid = nn.Sigmoid()
            self.clip_factor_a_max = nn.Parameter(torch.ones(1) * 4.0)
            self.clip_factor_a_min = nn.Parameter(torch.ones(1) * 4.0)
        self.enable = True
        self.qparams = QType(qtype).dim(-1)

    def apply_aclip(self, x):
        reshaped = x.reshape(-1, x.shape[-1])
        xmax = reshaped.amax(1, keepdim=True).clamp(min=0)
        xmin = reshaped.amin(1, keepdim=True).clamp(max=0)
        xmax = xmax * self.sigmoid(self.clip_factor_a_max)
        xmin = xmin * self.sigmoid(self.clip_factor_a_min)
        return torch.clamp(reshaped, min=xmin, max=xmax).reshape_as(x)

    def forward(self, x):
        if not self.enable:
            return x
        if self.lac:
            x = self.apply_aclip(x)
        return quant_func(x.contiguous(), self.qparams, force_fp32=True)


class HiF4FlatQuantizedLinear(nn.Module):
    def __init__(self, args, linear: nn.Linear):
        super().__init__()
        self.args = args
        self.linear = linear
        weight_qtype = getattr(args, "hif4_weight_qtype", "hifx4")
        self.act_quantizer = HiF4ActivationQuantizer(lac=args.flatquant_lac, qtype=weight_qtype)
        self.lwc = args.flatquant_lwc
        if self.lwc:
            self.clip_factor_w_max = nn.Parameter(torch.ones((linear.weight.shape[0], 1)) * 4.0)
            self.clip_factor_w_min = nn.Parameter(torch.ones((linear.weight.shape[0], 1)) * 4.0)
            self.sigmoid = nn.Sigmoid()
        self.qparams = QType(weight_qtype).dim(-1)
        self._eval_mode = False

    def apply_wclip(self, weight):
        wmin = weight.min(1, keepdim=True)[0] * self.sigmoid(self.clip_factor_w_min)
        wmax = weight.max(1, keepdim=True)[0] * self.sigmoid(self.clip_factor_w_max)
        return torch.clamp(weight, min=wmin, max=wmax)

    def apply_trans(self, weight, qa_trans):
        if isinstance(qa_trans, list):
            return kronecker_matmul(weight, qa_trans[0].to(weight), qa_trans[1].to(weight))
        return qa_trans(weight, inv_t=True)

    def quantize_activation(self, hidden_states):
        return self.act_quantizer(hidden_states)

    def _ori_forward(self, hidden_states):
        return self.linear(hidden_states)

    def _train_forward(self, hidden_states, qa_trans=None, quantize_activation=True):
        weight = self.linear.weight.data
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        weight = quant_func(weight.contiguous(), self.qparams, force_fp32=True)
        if quantize_activation:
            hidden_states = self.quantize_activation(hidden_states)
        hidden_states = hidden_states.to(dtype=weight.dtype)
        return F.linear(hidden_states, weight, self.linear.bias)

    def forward(self, hidden_states, qa_trans=None, quantize_activation=True):
        if self._eval_mode:
            if quantize_activation:
                hidden_states = self.quantize_activation(hidden_states)
            hidden_states = hidden_states.to(dtype=self.linear.weight.dtype)
            return self.linear(hidden_states)
        return self._train_forward(hidden_states, qa_trans=qa_trans, quantize_activation=quantize_activation)

    @torch.no_grad()
    def reparameterize(self, qa_trans=None):
        weight = self.linear.weight.data.to(torch.float64)
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        weight = quant_dequant_float(weight.contiguous(), self.qparams, force_fp32=True)
        if torch.any(~torch.isfinite(weight)):
            raise ValueError("FlatQuant produced non-finite HiF4 weight.")
        self.linear.weight.data = weight.to(self.linear.weight.dtype).contiguous()
        self._eval_mode = True
