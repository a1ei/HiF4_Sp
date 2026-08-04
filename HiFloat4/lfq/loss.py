"""Full-vocabulary soft-label distillation used by LFQ calibration."""

from dataclasses import dataclass

import torch


@dataclass
class LFQMetrics:
    cross_entropy: float
    top1_agreement: float
    valid_tokens: int


def _validate_inputs(teacher_hidden, student_hidden, valid_token_mask, chunk_size):
    if teacher_hidden.shape != student_hidden.shape:
        raise ValueError("Teacher and student hidden states must have identical shapes.")
    if valid_token_mask.shape != teacher_hidden.shape[:2]:
        raise ValueError("valid_token_mask must have shape [batch, sequence].")
    if chunk_size <= 0:
        raise ValueError("LFQ logits chunk size must be positive.")
    valid_tokens = int(valid_token_mask.sum().item())
    if valid_tokens == 0:
        raise ValueError("LFQ requires at least one valid token.")
    return valid_tokens


def _chunk_loss(teacher_hidden, student_hidden, final_norm, lm_head, mask):
    with torch.no_grad():
        fp_logits = lm_head(final_norm(teacher_hidden)).float()
        teacher_prob = torch.softmax(fp_logits, dim=-1)
        teacher_top1 = fp_logits.argmax(dim=-1)

    q_logits = lm_head(final_norm(student_hidden)).float()
    student_log_prob = torch.log_softmax(q_logits, dim=-1)
    token_loss = -(teacher_prob * student_log_prob).sum(dim=-1)
    loss_sum = (token_loss * mask).sum()
    agreement = ((q_logits.argmax(dim=-1) == teacher_top1) & mask).sum()
    return loss_sum, agreement


def lfq_loss(
    teacher_hidden,
    student_hidden,
    final_norm,
    lm_head,
    valid_token_mask,
    chunk_size=128,
):
    """Return the exact masked LFQ objective while chunking logits by sequence."""
    valid_token_mask = valid_token_mask.to(device=student_hidden.device, dtype=torch.bool)
    valid_tokens = _validate_inputs(teacher_hidden, student_hidden, valid_token_mask, chunk_size)
    loss = student_hidden.new_zeros((), dtype=torch.float32)
    agreement = 0
    for start in range(0, student_hidden.shape[1], chunk_size):
        end = min(start + chunk_size, student_hidden.shape[1])
        chunk_loss, chunk_agreement = _chunk_loss(
            teacher_hidden[:, start:end],
            student_hidden[:, start:end],
            final_norm,
            lm_head,
            valid_token_mask[:, start:end],
        )
        loss = loss + chunk_loss / valid_tokens
        agreement += int(chunk_agreement.detach().item())
    metrics = LFQMetrics(
        cross_entropy=float(loss.detach().item()),
        top1_agreement=agreement / valid_tokens,
        valid_tokens=valid_tokens,
    )
    return loss, metrics


def backward_lfq_chunks(
    teacher_hidden,
    student_hidden,
    final_norm,
    lm_head,
    valid_token_mask,
    chunk_size=128,
):
    """Backpropagate the exact LFQ loss without retaining full-sequence logits."""
    valid_token_mask = valid_token_mask.to(device=student_hidden.device, dtype=torch.bool)
    valid_tokens = _validate_inputs(teacher_hidden, student_hidden, valid_token_mask, chunk_size)
    spans = [
        (start, min(start + chunk_size, student_hidden.shape[1]))
        for start in range(0, student_hidden.shape[1], chunk_size)
    ]
    loss_sum = 0.0
    agreement = 0
    for chunk_idx, (start, end) in enumerate(spans):
        chunk_loss, chunk_agreement = _chunk_loss(
            teacher_hidden[:, start:end],
            student_hidden[:, start:end],
            final_norm,
            lm_head,
            valid_token_mask[:, start:end],
        )
        normalized_loss = chunk_loss / valid_tokens
        normalized_loss.backward(retain_graph=chunk_idx + 1 < len(spans))
        loss_sum += float(chunk_loss.detach().item())
        agreement += int(chunk_agreement.detach().item())
        del chunk_loss, normalized_loss
    return LFQMetrics(
        cross_entropy=loss_sum / valid_tokens,
        top1_agreement=agreement / valid_tokens,
        valid_tokens=valid_tokens,
    )
