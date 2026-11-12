"""Symbolic-slot assignment regularization."""
from __future__ import annotations

import torch


def assignment_entropy_loss(assignments: torch.Tensor) -> torch.Tensor:
    """Mean per-position entropy of symbolic-slot assignments.

    Encourages each token position to use a small number of slots rather than
    spreading attention uniformly over the slot bank. This is a structural
    prior for the differentiable slot extractor, not a supervision signal.

    Args:
        assignments: Softmax slot weights of shape [batch, sequence, num_slots].

    Returns:
        A scalar tensor (nats per position).
    """
    if assignments.ndim != 3:
        raise ValueError("assignments must have shape [batch, sequence, num_slots]")
    entropy = -(assignments * torch.log(assignments + 1e-9)).sum(dim=-1)
    return entropy.mean()
