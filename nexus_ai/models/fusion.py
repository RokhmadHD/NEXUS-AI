"""Gated fusion of neural and recurrent latent representations."""
import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, encoded: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([encoded, latent], dim=-1)))
        return self.norm(gate * latent + (1.0 - gate) * encoded)
