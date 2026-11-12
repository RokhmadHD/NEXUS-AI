"""Training objectives for NEXUS-AI models."""

from .consistency_loss import fused_latent_consistency_loss
from .language_loss import DEFAULT_IGNORE_INDEX, causal_language_modeling_loss
from .state_loss import latent_state_alignment_loss
from .symbolic_loss import assignment_entropy_loss
from .total_loss import LossBreakdown, TotalLoss

__all__ = [
    "DEFAULT_IGNORE_INDEX",
    "LossBreakdown",
    "TotalLoss",
    "assignment_entropy_loss",
    "causal_language_modeling_loss",
    "fused_latent_consistency_loss",
    "latent_state_alignment_loss",
]
