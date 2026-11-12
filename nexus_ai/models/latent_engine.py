"""Recurrent latent-state refinement using a shared GRU cell."""
import torch
from torch import nn


class LatentStateEngine(nn.Module):
    def __init__(self, embedding_dim: int, iterations: int = 2):
        super().__init__()
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.iterations = iterations
        self.input_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.cell = nn.GRUCell(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, encoded: torch.Tensor, symbolic_context: torch.Tensor):
        batch_size, seq_len, hidden_dim = encoded.shape
        recurrent = encoded
        projected = torch.tanh(self.input_projection(torch.cat([encoded, symbolic_context], dim=-1)))
        for _ in range(self.iterations):
            recurrent = self.cell(
                projected.reshape(batch_size * seq_len, hidden_dim),
                recurrent.reshape(batch_size * seq_len, hidden_dim),
            ).reshape(batch_size, seq_len, hidden_dim)
        return self.norm(recurrent)
