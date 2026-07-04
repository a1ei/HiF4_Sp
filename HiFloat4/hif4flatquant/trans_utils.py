import torch
import torch.nn as nn

from .flat_utils import kronecker_matmul
from .function_utils import get_init_weight, get_inverse


class SVDSingleTransMatrix(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_u = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_u.weight.data = get_init_weight(size).to(self.linear_u.weight)
        self.linear_u = nn.utils.parametrizations.orthogonal(
            self.linear_u, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_v = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_v.weight.data = get_init_weight(size).to(self.linear_v.weight)
        self.linear_v = nn.utils.parametrizations.orthogonal(
            self.linear_v, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_diag = nn.Parameter(torch.ones(size, dtype=torch.float32))
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        init_shape = inp.shape
        matrix = self.get_matrix(inv_t=inv_t).to(inp)
        return inp.reshape(-1, matrix.shape[0]).matmul(matrix).reshape(init_shape)

    def get_matrix(self, inv_t=False):
        if self._eval_mode:
            return self.matrix_inv_t if inv_t else self.matrix
        linear_diag = 1 / self.linear_diag if inv_t else self.linear_diag
        return self.linear_u.weight @ torch.diag(linear_diag) @ self.linear_v.weight.t()

    def to_eval_mode(self):
        if self._eval_mode:
            return
        self.matrix = nn.Parameter(self.get_matrix(), requires_grad=False)
        self.matrix_inv_t = nn.Parameter(self.get_matrix(inv_t=True), requires_grad=False)
        self._eval_mode = True
        del self.linear_u, self.linear_diag, self.linear_v


class SVDDecomposeTransMatrix(nn.Module):
    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None):
        super().__init__()
        self.linear_u_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_u_left.weight.data = get_init_weight(left_size).to(self.linear_u_left.weight)
        self.linear_u_left = nn.utils.parametrizations.orthogonal(
            self.linear_u_left, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_v_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_v_left.weight.data = get_init_weight(left_size).to(self.linear_v_left.weight)
        self.linear_v_left = nn.utils.parametrizations.orthogonal(
            self.linear_v_left, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_diag_left = nn.Parameter(torch.ones(left_size, dtype=torch.float32))
        self.linear_u_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_u_right.weight.data = get_init_weight(right_size).to(self.linear_u_right.weight)
        self.linear_u_right = nn.utils.parametrizations.orthogonal(
            self.linear_u_right, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_v_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_v_right.weight.data = get_init_weight(right_size).to(self.linear_v_right.weight)
        self.linear_v_right = nn.utils.parametrizations.orthogonal(
            self.linear_v_right, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_diag_right = nn.Parameter(torch.ones(right_size, dtype=torch.float32))
        self.add_diag = add_diag
        self.use_diag = True
        if add_diag:
            diag = torch.ones(left_size * right_size, dtype=torch.float32) if diag_init_para is None else diag_init_para
            self.diag_scale = nn.Parameter(diag)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        if self.add_diag and self.use_diag: 
            inp = inp / self.diag_scale.to(inp) if inv_t else inp * self.diag_scale.to(inp)
        if self._eval_mode:
            left = self.matrix_left_inv if inv_t else self.matrix_left
            right = self.matrix_right_inv if inv_t else self.matrix_right
        else:
            diag_left = 1 / self.linear_diag_left if inv_t else self.linear_diag_left
            diag_right = 1 / self.linear_diag_right if inv_t else self.linear_diag_right
            left = self.linear_u_left.weight @ torch.diag(diag_left) @ self.linear_v_left.weight.t()
            right = self.linear_u_right.weight @ torch.diag(diag_right) @ self.linear_v_right.weight.t()
        return kronecker_matmul(inp, left.to(inp), right.to(inp))

    def to_eval_mode(self):
        if self._eval_mode:
            return
        self.matrix_left = nn.Parameter(
            self.linear_u_left.weight @ torch.diag(self.linear_diag_left) @ self.linear_v_left.weight.t(),
            requires_grad=False,
        )
        self.matrix_right = nn.Parameter(
            self.linear_u_right.weight @ torch.diag(self.linear_diag_right) @ self.linear_v_right.weight.t(),
            requires_grad=False,
        )
        self.matrix_left_inv = nn.Parameter(
            self.linear_u_left.weight @ torch.diag(1 / self.linear_diag_left) @ self.linear_v_left.weight.t(),
            requires_grad=False,
        )
        self.matrix_right_inv = nn.Parameter(
            self.linear_u_right.weight @ torch.diag(1 / self.linear_diag_right) @ self.linear_v_right.weight.t(),
            requires_grad=False,
        )
        del self.linear_u_left, self.linear_diag_left, self.linear_v_left
        del self.linear_u_right, self.linear_diag_right, self.linear_v_right
        self._eval_mode = True


class InvSingleTransMatrix(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear.weight.data = get_init_weight(size).to(self.linear.weight)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        init_shape = inp.shape
        matrix = self.get_matrix(inv_t=inv_t).to(inp)
        return inp.reshape(-1, matrix.shape[0]).matmul(matrix).reshape(init_shape)

    def get_matrix(self, inv_t=False):
        if self._eval_mode:
            return self.matrix_inv_t if inv_t else self.matrix
        return get_inverse(self.linear.weight).t() if inv_t else self.linear.weight

    def to_eval_mode(self):
        if self._eval_mode:
            return
        self.matrix = nn.Parameter(self.linear.weight, requires_grad=False)
        self.matrix_inv_t = nn.Parameter(get_inverse(self.linear.weight).t(), requires_grad=False)
        del self.linear
        self._eval_mode = True


class InvDecomposeTransMatrix(nn.Module):
    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None):
        super().__init__()
        self.linear_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_left.weight.data = get_init_weight(left_size).to(self.linear_left.weight)
        self.linear_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_right.weight.data = get_init_weight(right_size).to(self.linear_right.weight)
        self.add_diag = add_diag
        self.use_diag = True
        if add_diag:
            diag = torch.ones(left_size * right_size, dtype=torch.float32) if diag_init_para is None else diag_init_para
            self.diag_scale = nn.Parameter(diag)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        if self.add_diag and self.use_diag:
            inp = inp / self.diag_scale.to(inp) if inv_t else inp * self.diag_scale.to(inp)
        if self._eval_mode:
            left = self.matrix_left_inv if inv_t else self.matrix_left
            right = self.matrix_right_inv if inv_t else self.matrix_right
        else:
            left = get_inverse(self.linear_left.weight).t() if inv_t else self.linear_left.weight
            right = get_inverse(self.linear_right.weight).t() if inv_t else self.linear_right.weight
        return kronecker_matmul(inp, left.to(inp), right.to(inp))

    def to_eval_mode(self):
        if self._eval_mode:
            return
        self.matrix_left = nn.Parameter(self.linear_left.weight, requires_grad=False)
        self.matrix_right = nn.Parameter(self.linear_right.weight, requires_grad=False)
        self.matrix_left_inv = nn.Parameter(get_inverse(self.linear_left.weight).t(), requires_grad=False)
        self.matrix_right_inv = nn.Parameter(get_inverse(self.linear_right.weight).t(), requires_grad=False)
        del self.linear_left, self.linear_right
        self._eval_mode = True
