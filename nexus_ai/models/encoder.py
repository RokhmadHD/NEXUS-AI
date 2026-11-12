"""Causal Transformer representation encoder."""
import torch
from torch import nn


class CausalEncoder(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        return self.norm(self.layers(x, mask=causal_mask))
