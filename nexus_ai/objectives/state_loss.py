"""Latent-state alignment loss.

Pulls the refined latent state toward the encoded neural representation so
that recurrent refinement adapts the encoder states instead of drifting away
from them. This is a structural regularizer, not a supervised state target.
"""
from __future__ import annotations

import torch


def latent_state_alignment_loss(
    latent_state: torch.Tensor,
    encoded_states: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between refined latent states and encoder states.

    Args:
        latent_state: Tensor of shape [batch, sequence, dim].
        encoded_states: Tensor of shape [batch, sequence, dim].

    Returns:
        A scalar tensor.
    """
    if latent_state.shape != encoded_states.shape:
        raise ValueError("latent_state and encoded_states must have matching shapes")
    return torch.nn.functional.mse_loss(latent_state, encoded_states)
