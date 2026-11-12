"""A lightweight differentiable slot-based symbolic feature extractor."""
import torch
from torch import nn


class SymbolicEngine(nn.Module):
    def __init__(self, embedding_dim: int, num_slots: int):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(num_slots, embedding_dim) * 0.02)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.scale = embedding_dim ** -0.5

    def forward(self, token_states: torch.Tensor):
        # Each position reads only its own causal token state and a learned slot bank.
        q = self.query(token_states)
        slot_bank = self.slots.unsqueeze(0).expand(token_states.size(0), -1, -1)
        k = self.key(slot_bank)
        v = self.value(slot_bank)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        assignments = torch.softmax(scores, dim=-1)
        context = torch.matmul(assignments, v)
        return context, assignments
