"""Causal language-modeling loss with ignored padding labels."""
from __future__ import annotations

import torch

DEFAULT_IGNORE_INDEX = -100


def causal_language_modeling_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> tuple[torch.Tensor, int]:
    """Compute mean next-token cross-entropy over non-ignored positions.

    Args:
        logits: Tensor of shape [batch, sequence, vocab_size].
        labels: Tensor of shape [batch, sequence] with ``ignore_index`` padding.
        ignore_index: Label value excluded from the loss.

    Returns:
        A tuple of ``(loss, counted_tokens)``. ``loss`` is a scalar tensor and
        ``counted_tokens`` is the number of positions included in the mean.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [batch, sequence]")

    counted = int((labels != ignore_index).sum().item())
    if counted == 0:
        raise ValueError("labels contain no supervised positions")
    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return token_losses, counted
