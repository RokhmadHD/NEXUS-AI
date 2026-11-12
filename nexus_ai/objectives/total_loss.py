"""Weighted combination of the NEXUS-HSR training objectives."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .consistency_loss import fused_latent_consistency_loss
from .language_loss import DEFAULT_IGNORE_INDEX, causal_language_modeling_loss
from .state_loss import latent_state_alignment_loss
from .symbolic_loss import assignment_entropy_loss


@dataclass
class LossBreakdown:
    """Per-objective loss values from a single ``compute_total_loss`` call."""

    total: torch.Tensor
    language: torch.Tensor
    state: torch.Tensor | None = None
    consistency: torch.Tensor | None = None
    symbolic: torch.Tensor | None = None
    counted_tokens: int = 0


@dataclass
class TotalLoss:
    """Weighted multi-objective loss for models exposing internal states.

    The language weight must be positive; auxiliary weights may be zero to
    disable an objective. Weights are applied as-is (no normalization), so
    changing one weight changes the effective scale of the total loss.
    """

    language_weight: float = 1.0
    state_weight: float = 0.1
    consistency_weight: float = 0.1
    symbolic_weight: float = 0.01
    ignore_index: int = DEFAULT_IGNORE_INDEX

    def __post_init__(self) -> None:
        if self.language_weight <= 0:
            raise ValueError("language_weight must be positive")
        if min(self.state_weight, self.consistency_weight, self.symbolic_weight) < 0:
            raise ValueError("auxiliary weights must be non-negative")

    def compute(
        self,
        model_output: dict,
        labels: torch.Tensor,
    ) -> LossBreakdown:
        """Compute the weighted loss from a ``NexusHSR`` output dictionary.

        Args:
            model_output: Dict containing ``logits`` plus internal states
                (``latent_state``, ``encoded_states``, ``fused_states``,
                ``symbolic_assignments``).
            labels: Tensor of shape [batch, sequence] with ``ignore_index``.

        Returns:
            A ``LossBreakdown`` with the weighted total and each component.
        """
        missing = {
            "latent_state",
            "encoded_states",
            "fused_states",
            "symbolic_assignments",
        } - set(model_output)
        if missing:
            raise KeyError(f"model_output is missing internal states: {sorted(missing)}")

        language_loss, counted = causal_language_modeling_loss(
            model_output["logits"], labels, self.ignore_index
        )
        total = self.language_weight * language_loss

        state_loss = latent_state_alignment_loss(
            model_output["latent_state"], model_output["encoded_states"]
        )
        consistency_loss = fused_latent_consistency_loss(
            model_output["fused_states"], model_output["latent_state"]
        )
        symbolic_loss = assignment_entropy_loss(model_output["symbolic_assignments"])

        if self.state_weight > 0:
            total = total + self.state_weight * state_loss
        if self.consistency_weight > 0:
            total = total + self.consistency_weight * consistency_loss
        if self.symbolic_weight > 0:
            total = total + self.symbolic_weight * symbolic_loss

        return LossBreakdown(
            total=total,
            language=language_loss,
            state=state_loss,
            consistency=consistency_loss,
            symbolic=symbolic_loss,
            counted_tokens=counted,
        )
