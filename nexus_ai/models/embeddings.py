"""Token and positional embeddings for NEXUS-HSR."""
import torch
from torch import nn


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embedding_dim)
        self.position = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        return self.dropout(self.token(input_ids) + self.position(positions))
