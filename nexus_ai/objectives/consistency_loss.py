"""Consistency loss between fused and latent representations."""
from __future__ import annotations

import torch


def fused_latent_consistency_loss(
    fused_states: torch.Tensor,
    latent_state: torch.Tensor,
) -> torch.Tensor:
    """Mean cosine distance between fused states and latent states.

    Encourages the decoder input (fused states) to stay aligned with the
    refined latent state, so the gated fusion supports the recurrent state
    instead of discarding it.

    Args:
        fused_states: Tensor of shape [batch, sequence, dim].
        latent_state: Tensor of shape [batch, sequence, dim].

    Returns:
        A scalar tensor in ``[0, 2]``.
    """
    if fused_states.shape != latent_state.shape:
        raise ValueError("fused_states and latent_state must have matching shapes")
    similarity = torch.nn.functional.cosine_similarity(
        fused_states, latent_state, dim=-1
    )
    return (1.0 - similarity).mean()
