"""Parameter-matched causal Transformer baseline for NEXUS-HSR experiments."""
import torch
from torch import nn


class TransformerBaseline(nn.Module):
    """A plain causal Transformer with a parameter budget matched to NexusHSR.

    The baseline intentionally excludes the symbolic, recurrent latent, and
    gated-fusion modules. It retains token/position embeddings, a causal
    Transformer encoder stack, final normalization, and an LM head.
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        context_length: int = 256,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 721,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length exceeds context_length={self.context_length}"
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=input_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask)
        return self.head(self.norm(x))

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

