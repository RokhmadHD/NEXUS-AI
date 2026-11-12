"""Optimizer construction and learning-rate scheduling utilities."""
from __future__ import annotations

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> Optimizer:
    """Create an AdamW optimizer for all trainable model parameters.

    Args:
        model: The model whose parameters will be optimized.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay coefficient.

    Returns:
        An ``AdamW`` optimizer instance.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    return AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)


def build_warmup_scheduler(optimizer: Optimizer, warmup_steps: int) -> LambdaLR:
    """Linear warmup from 0 to the base learning rate over ``warmup_steps``.

    After warmup completes, the learning rate stays constant. ``warmup_steps``
    of 0 disables warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps (must be non-negative).

    Returns:
        A ``LambdaLR`` scheduler.
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    def scale(step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return LambdaLR(optimizer, scale)
