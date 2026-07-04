import math
from collections import OrderedDict

import numpy as np
import torch
from scipy.linalg import qr


def get_init_scale(w_smax, x_smax, alpha=0.5):
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-5)


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


def get_random_orthg(size):
    h = np.random.randn(size, size)
    q, r = qr(h)
    return torch.from_numpy(q @ np.diag(np.sign(np.diag(r))))


def get_init_weight(dim):
    return get_random_orthg(dim)


def get_inverse(matrix):
    dtype = matrix.dtype
    return matrix.double().inverse().to(dtype)


def get_n_set_parameters_byname(model, required_names):
    params = []
    for required_name in required_names:
        for name, param in model.named_parameters():
            if required_name in name:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return params


def get_paras_dict_by_name(model, required_names, destination=None, prefix=""):
    if destination is None:
        destination = OrderedDict()
    for required_name in required_names:
        for name, param in model.named_parameters():
            if required_name in name:
                destination[prefix + name] = param.detach().cpu()
    return destination


def set_require_grad_all(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
    return model
