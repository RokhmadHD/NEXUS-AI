"""Causal Transformer decoder head for next-token prediction."""
import torch
from torch import nn


class CausalDecoder(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int, ffn_dim: int,
                 vocab_size: int, dropout: float = 0.0):
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
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, fused_states: torch.Tensor) -> torch.Tensor:
        seq_len = fused_states.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=fused_states.device), diagonal=1
        )
        return self.output(self.norm(self.layers(fused_states, mask=causal_mask)))
