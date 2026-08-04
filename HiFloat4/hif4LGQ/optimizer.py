from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationStatistics:
    """Accumulate a dense Hessian and logic-position activations."""

    def __init__(self) -> None:
        self.hessian: torch.Tensor | None = None
        self.logic_chunks: list[torch.Tensor] = []
        self.token_count = 0
        self.logic_token_count = 0
        self.input_features: int | None = None

    @torch.no_grad()
    def add(
        self,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        logic_mask: torch.Tensor,
    ) -> None:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        if inputs.ndim != 3 or inputs.shape[0] != 1:
            raise ValueError(
                "hif4LGQ Linear inputs must have shape [1, sequence, features], "
                f"got {tuple(inputs.shape)}."
            )

        sequence_length = inputs.shape[1]
        valid_mask = valid_mask.reshape(-1).to(dtype=torch.bool, device=inputs.device)
        logic_mask = logic_mask.reshape(-1).to(dtype=torch.bool, device=inputs.device)
        if valid_mask.numel() != sequence_length or logic_mask.numel() != sequence_length:
            raise ValueError(
                "LGQ token masks must match the Linear input sequence length: "
                f"sequence={sequence_length}, valid={valid_mask.numel()}, "
                f"logic={logic_mask.numel()}."
            )

        features = inputs.shape[-1]
        if self.input_features is None:
            self.input_features = features
            self.hessian = torch.zeros(
                (features, features),
                dtype=torch.float32,
                device=inputs.device,
            )
        elif self.input_features != features:
            raise ValueError(
                f"LGQ input width changed from {self.input_features} to {features}."
            )

        valid_inputs = inputs[0, valid_mask].float()
        if valid_inputs.numel():
            self.hessian.addmm_(valid_inputs.T, valid_inputs)
            self.token_count += int(valid_inputs.shape[0])

        selected_logic = valid_mask & logic_mask
        logic_inputs = inputs[0, selected_logic]
        if logic_inputs.numel():
            logic_inputs = logic_inputs.detach().to(dtype=torch.float32, device="cpu")
            self.logic_chunks.append(logic_inputs.contiguous())
            self.logic_token_count += int(logic_inputs.shape[0])

    def offload_hessian(self) -> None:
        if self.hessian is None:
            raise RuntimeError("No Hessian was collected.")
        self.hessian = self.hessian.cpu()

    def free(self) -> None:
        self.hessian = None
        self.logic_chunks.clear()


@dataclass(frozen=True)
class SpectrumInfo:
    minimum: float
    maximum: float
    selected_minimum: float
    selected_maximum: float
    selected_start: int
    selected_end: int


@dataclass(frozen=True)
class LinearOptimizationStats:
    layer_name: str
    hessian_tokens: int
    logic_tokens: int
    group_size: int
    spectrum_min: float
    spectrum_max: float
    selected_spectrum_min: float
    selected_spectrum_max: float
    selected_start: int
    selected_end: int
    initial_group_loss: float
    final_group_loss: float
    final_logic_loss: float
    final_reg_loss: float
    final_total_loss: float
    c_frobenius_norm: float
    delta_frobenius_norm: float
    initial_group_amax_mean: float
    initial_group_amax_max: float
    final_group_amax_mean: float
    final_group_amax_max: float

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_group_size(weight_format: str, configured_group_size: int) -> int:
    if configured_group_size < 0:
        raise ValueError("lgq_group_size must be greater than or equal to 0.")
    if configured_group_size:
        return configured_group_size
    mapping = {
        "hif4": 64,
        "hif4-1": 64,
        "nvfp4": 16,
    }
    try:
        return mapping[weight_format]
    except KeyError as exc:
        raise ValueError(
            f"No automatic hif4LGQ group size for weight format {weight_format!r}."
        ) from exc


def _reshape_weight_groups(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D Linear weight, got {tuple(weight.shape)}.")
    if group_size <= 0:
        raise ValueError("LGQ group size must be positive.")
    pad_columns = (-weight.shape[1]) % group_size
    if pad_columns:
        weight = F.pad(weight, (0, pad_columns), value=0.0)
    return weight.reshape(weight.shape[0], -1, group_size)


def group_amax(
    weight: torch.Tensor,
    group_size: int,
    loss_mode: str = "max",
    smooth_tau: float = 1e-3,
) -> torch.Tensor:
    groups = _reshape_weight_groups(weight, group_size)
    if loss_mode == "max":
        return groups.abs().amax(dim=-1)
    if loss_mode == "logsumexp":
        if smooth_tau <= 0:
            raise ValueError("lgq_group_smooth_tau must be greater than 0.")
        return smooth_tau * torch.logsumexp(groups.abs() / smooth_tau, dim=-1)
    raise ValueError(f"Unsupported LGQ group loss mode: {loss_mode}")


def group_loss(
    weight: torch.Tensor,
    group_size: int,
    loss_mode: str = "max",
    smooth_tau: float = 1e-3,
) -> torch.Tensor:
    return group_amax(weight, group_size, loss_mode, smooth_tau).square().mean()


def select_spectral_subspace(
    hessian: torch.Tensor,
    rank: int,
    mode: str,
    low_mid_start_quantile: float,
) -> tuple[torch.Tensor, SpectrumInfo]:
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"Hessian must be square, got {tuple(hessian.shape)}.")
    if rank <= 0 or rank > hessian.shape[0]:
        raise ValueError(
            f"lgq_subspace_rank must be in [1, {hessian.shape[0]}], got {rank}."
        )
    if not torch.isfinite(hessian).all():
        raise ValueError("LGQ Hessian contains NaN or Inf.")

    symmetric_hessian = (hessian.float() + hessian.float().T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_hessian)
    dimension = eigenvalues.numel()
    if mode == "low":
        start = 0
    elif mode == "low_mid":
        if not 0 <= low_mid_start_quantile < 1:
            raise ValueError("lgq_low_mid_start_quantile must be in [0, 1).")
        start = int(math.floor(low_mid_start_quantile * dimension))
    else:
        raise ValueError(f"Unsupported LGQ subspace mode: {mode}")
    end = start + rank
    if end > dimension:
        raise ValueError(
            "Selected LGQ spectral band exceeds the Hessian dimension: "
            f"start={start}, rank={rank}, dimension={dimension}."
        )

    selected = eigenvalues[start:end]
    basis = eigenvectors[:, start:end].T.contiguous()
    info = SpectrumInfo(
        minimum=float(eigenvalues[0].item()),
        maximum=float(eigenvalues[-1].item()),
        selected_minimum=float(selected[0].item()),
        selected_maximum=float(selected[-1].item()),
        selected_start=start,
        selected_end=end,
    )
    return basis, info


@torch.no_grad()
def build_logic_gram(
    logic_chunks: list[torch.Tensor],
    basis: torch.Tensor,
    logic_token_count: int,
) -> torch.Tensor:
    if logic_token_count <= 0:
        raise ValueError("A target Linear has no matched logic token positions.")
    gram = torch.zeros(
        (basis.shape[0], basis.shape[0]),
        dtype=torch.float32,
        device=basis.device,
    )
    counted = 0
    for chunk in logic_chunks:
        chunk = chunk.to(device=basis.device, dtype=torch.float32)
        projection = chunk @ basis.T
        gram.addmm_(projection.T, projection)
        counted += int(chunk.shape[0])
    if counted != logic_token_count:
        raise RuntimeError(
            f"LGQ logic token count changed: expected={logic_token_count}, got={counted}."
        )
    return gram / logic_token_count


def direct_logic_loss(
    logic_inputs: torch.Tensor,
    delta_weight: torch.Tensor,
) -> torch.Tensor:
    if logic_inputs.shape[0] == 0:
        raise ValueError("Direct LGQ logic loss requires at least one logic token.")
    return (logic_inputs.float() @ delta_weight.float().T).square().sum() / logic_inputs.shape[0]


def gram_logic_loss(coefficients: torch.Tensor, logic_gram: torch.Tensor) -> torch.Tensor:
    return ((coefficients @ logic_gram) * coefficients).sum()


@dataclass
class _LinearOptimizationState:
    layer_name: str
    linear: nn.Linear
    original_weight: torch.Tensor
    coefficients: nn.Parameter
    initial_amax: torch.Tensor
    initial_group_loss: torch.Tensor


def optimize_linear_group(
    linears: list[tuple[str, nn.Linear]],
    basis: torch.Tensor,
    logic_gram: torch.Tensor,
    spectrum: SpectrumInfo,
    hessian_tokens: int,
    logic_tokens: int,
    group_size: int,
    steps: int,
    learning_rate: float,
    lambda_logic: float,
    lambda_reg: float,
    group_loss_mode: str,
    group_smooth_tau: float,
    return_delta: bool,
    log_interval: int = 0,
    logger: logging.Logger | None = None,
) -> list[tuple[LinearOptimizationStats, torch.Tensor | None]]:
    if not linears:
        raise ValueError("LGQ joint optimization requires at least one Linear.")
    if steps <= 0:
        raise ValueError("lgq_steps must be greater than 0.")
    if learning_rate <= 0:
        raise ValueError("lgq_lr must be greater than 0.")
    if lambda_logic < 0 or lambda_reg < 0:
        raise ValueError("LGQ loss weights must be non-negative.")
    if log_interval < 0:
        raise ValueError("lgq_log_interval must be greater than or equal to 0.")

    device = linears[0][1].weight.device
    basis = basis.to(device=device, dtype=torch.float32)
    logic_gram = logic_gram.to(device=device, dtype=torch.float32)
    states: list[_LinearOptimizationState] = []
    for layer_name, linear in linears:
        if linear.weight.device != device:
            raise ValueError("All jointly optimized LGQ Linears must be on one device.")
        if linear.in_features != basis.shape[1]:
            raise ValueError(
                f"LGQ basis width {basis.shape[1]} does not match "
                f"{layer_name} input width {linear.in_features}."
            )
        original_weight = linear.weight.detach().to(dtype=torch.float32).clone()
        coefficients = nn.Parameter(
            torch.zeros(
                (linear.out_features, basis.shape[0]),
                dtype=torch.float32,
                device=device,
            )
        )
        with torch.no_grad():
            initial_amax = group_amax(
                original_weight,
                group_size,
                group_loss_mode,
                group_smooth_tau,
            )
            initial_group_loss = initial_amax.square().mean()
        states.append(
            _LinearOptimizationState(
                layer_name=layer_name,
                linear=linear,
                original_weight=original_weight,
                coefficients=coefficients,
                initial_amax=initial_amax,
                initial_group_loss=initial_group_loss,
            )
        )

    optimizer = torch.optim.Adam(
        [state.coefficients for state in states],
        lr=learning_rate,
    )
    group_label = ",".join(state.layer_name for state in states)
    if logger is not None and log_interval:
        initial_joint_total = sum(
            state.initial_group_loss.item()
            for state in states
        )
        logger.info(
            "(hif4LGQ/train-group) [%s] step=0/%d joint_total=%.6e",
            group_label,
            steps,
            initial_joint_total,
        )
        for state in states:
            logger.info(
                "(hif4LGQ/train) %s step=0/%d "
                "loss(group=%.6e logic=0.000000e+00 weighted_logic=0.000000e+00 "
                "reg=0.000000e+00 weighted_reg=0.000000e+00 total=%.6e) "
                "norm(C=0.000000e+00 delta=0.000000e+00 grad=0.000000e+00)",
                state.layer_name,
                steps,
                state.initial_group_loss.item(),
                state.initial_group_loss.item(),
            )

    with torch.enable_grad():
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            module_totals = []
            for state in states:
                delta_weight = state.coefficients @ basis
                adjusted_weight = state.original_weight + delta_weight
                current_group = group_loss(
                    adjusted_weight,
                    group_size,
                    group_loss_mode,
                    group_smooth_tau,
                )
                current_logic = gram_logic_loss(
                    state.coefficients,
                    logic_gram,
                )
                current_reg = state.coefficients.square().sum()
                module_totals.append(
                    current_group
                    + lambda_logic * current_logic
                    + lambda_reg * current_reg
                )
            joint_total = torch.stack(module_totals).sum()
            if not torch.isfinite(joint_total):
                raise ValueError(
                    f"Non-finite joint hif4LGQ loss for [{group_label}]."
                )
            joint_total.backward()
            should_log = (
                logger is not None
                and log_interval > 0
                and (step % log_interval == 0 or step == steps)
            )
            grad_norms = {
                state.layer_name: (
                    torch.linalg.vector_norm(state.coefficients.grad).item()
                    if should_log and state.coefficients.grad is not None
                    else 0.0
                )
                for state in states
            }
            optimizer.step()

            if should_log:
                logged_modules = []
                with torch.no_grad():
                    for state in states:
                        logged_delta = state.coefficients @ basis
                        logged_group = group_loss(
                            state.original_weight + logged_delta,
                            group_size,
                            group_loss_mode,
                            group_smooth_tau,
                        )
                        logged_logic = gram_logic_loss(
                            state.coefficients,
                            logic_gram,
                        )
                        logged_reg = state.coefficients.square().sum()
                        weighted_logic = lambda_logic * logged_logic
                        weighted_reg = lambda_reg * logged_reg
                        logged_total = (
                            logged_group + weighted_logic + weighted_reg
                        )
                        logged_modules.append(
                            (
                                state,
                                logged_delta,
                                logged_group,
                                logged_logic,
                                weighted_logic,
                                logged_reg,
                                weighted_reg,
                                logged_total,
                            )
                        )
                    logged_joint_total = sum(
                        item[-1].item()
                        for item in logged_modules
                    )
                    logger.info(
                        "(hif4LGQ/train-group) [%s] step=%d/%d joint_total=%.6e",
                        group_label,
                        step,
                        steps,
                        logged_joint_total,
                    )
                    for (
                        state,
                        logged_delta,
                        logged_group,
                        logged_logic,
                        weighted_logic,
                        logged_reg,
                        weighted_reg,
                        logged_total,
                    ) in logged_modules:
                        logger.info(
                            "(hif4LGQ/train) %s step=%d/%d "
                            "loss(group=%.6e logic=%.6e weighted_logic=%.6e "
                            "reg=%.6e weighted_reg=%.6e total=%.6e) "
                            "norm(C=%.6e delta=%.6e grad=%.6e)",
                            state.layer_name,
                            step,
                            steps,
                            logged_group.item(),
                            logged_logic.item(),
                            weighted_logic.item(),
                            logged_reg.item(),
                            weighted_reg.item(),
                            logged_total.item(),
                            torch.linalg.vector_norm(
                                state.coefficients
                            ).item(),
                            torch.linalg.vector_norm(logged_delta).item(),
                            grad_norms[state.layer_name],
                        )

    results = []
    with torch.no_grad():
        for state in states:
            candidate_delta = state.coefficients @ basis
            adjusted_weight = state.original_weight + candidate_delta
            final_group = group_loss(
                adjusted_weight,
                group_size,
                group_loss_mode,
                group_smooth_tau,
            )
            final_logic = gram_logic_loss(state.coefficients, logic_gram)
            final_reg = state.coefficients.square().sum()
            final_total = (
                final_group
                + lambda_logic * final_logic
                + lambda_reg * final_reg
            )
            merged_weight = adjusted_weight.to(state.linear.weight.dtype)
            state.linear.weight.data.copy_(merged_weight)
            final_amax = group_amax(
                merged_weight.float(),
                group_size,
                group_loss_mode,
                group_smooth_tau,
            )
            delta_norm = torch.linalg.vector_norm(candidate_delta)
            c_norm = torch.linalg.vector_norm(state.coefficients)

            stats = LinearOptimizationStats(
                layer_name=state.layer_name,
                hessian_tokens=hessian_tokens,
                logic_tokens=logic_tokens,
                group_size=group_size,
                spectrum_min=spectrum.minimum,
                spectrum_max=spectrum.maximum,
                selected_spectrum_min=spectrum.selected_minimum,
                selected_spectrum_max=spectrum.selected_maximum,
                selected_start=spectrum.selected_start,
                selected_end=spectrum.selected_end,
                initial_group_loss=float(
                    state.initial_group_loss.item()
                ),
                final_group_loss=float(final_group.item()),
                final_logic_loss=float(final_logic.item()),
                final_reg_loss=float(final_reg.item()),
                final_total_loss=float(final_total.item()),
                c_frobenius_norm=float(c_norm.item()),
                delta_frobenius_norm=float(delta_norm.item()),
                initial_group_amax_mean=float(
                    state.initial_amax.mean().item()
                ),
                initial_group_amax_max=float(
                    state.initial_amax.max().item()
                ),
                final_group_amax_mean=float(final_amax.mean().item()),
                final_group_amax_max=float(final_amax.max().item()),
            )
            delta_cpu = (
                candidate_delta.cpu().contiguous()
                if return_delta
                else None
            )
            results.append((stats, delta_cpu))
    return results


def optimize_linear_weight(
    layer_name: str,
    linear: nn.Linear,
    basis: torch.Tensor,
    logic_gram: torch.Tensor,
    spectrum: SpectrumInfo,
    hessian_tokens: int,
    logic_tokens: int,
    group_size: int,
    steps: int,
    learning_rate: float,
    lambda_logic: float,
    lambda_reg: float,
    group_loss_mode: str,
    group_smooth_tau: float,
    return_delta: bool,
    log_interval: int = 0,
    logger: logging.Logger | None = None,
) -> tuple[LinearOptimizationStats, torch.Tensor | None]:
    return optimize_linear_group(
        linears=[(layer_name, linear)],
        basis=basis,
        logic_gram=logic_gram,
        spectrum=spectrum,
        hessian_tokens=hessian_tokens,
        logic_tokens=logic_tokens,
        group_size=group_size,
        steps=steps,
        learning_rate=learning_rate,
        lambda_logic=lambda_logic,
        lambda_reg=lambda_reg,
        group_loss_mode=group_loss_mode,
        group_smooth_tau=group_smooth_tau,
        return_delta=return_delta,
        log_interval=log_interval,
        logger=logger,
    )[0]
