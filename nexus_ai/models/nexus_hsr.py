"""NEXUS-HSR Tiny v0.1: hybrid causal representation and state model."""
from torch import nn

from .decoder import CausalDecoder
from .embeddings import TokenPositionEmbedding
from .encoder import CausalEncoder
from .fusion import GatedFusion
from .latent_engine import LatentStateEngine
from .symbolic_engine import SymbolicEngine


class NexusHSR(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2048,
        context_length: int = 256,
        embedding_dim: int = 128,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        ffn_dim: int = 512,
        latent_iterations: int = 2,
        symbolic_slots: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.context_length = context_length
        self.embeddings = TokenPositionEmbedding(vocab_size, context_length, embedding_dim, dropout)
        self.encoder = CausalEncoder(embedding_dim, num_heads, encoder_layers, ffn_dim, dropout)
        self.symbolic = SymbolicEngine(embedding_dim, symbolic_slots)
        self.latent = LatentStateEngine(embedding_dim, latent_iterations)
        self.fusion = GatedFusion(embedding_dim)
        self.decoder = CausalDecoder(
            embedding_dim, num_heads, decoder_layers, ffn_dim, vocab_size, dropout
        )

    def forward(self, input_ids):
        if input_ids.size(1) > self.context_length:
            raise ValueError(f"sequence length exceeds context_length={self.context_length}")
        embedded = self.embeddings(input_ids)
        encoded = self.encoder(embedded)
        symbolic_context, assignments = self.symbolic(encoded)
        latent_state = self.latent(encoded, symbolic_context)
        fused = self.fusion(encoded, latent_state)
        logits = self.decoder(fused)
        return {
            "logits": logits,
            "encoded_states": encoded,
            "symbolic_context": symbolic_context,
            "symbolic_assignments": assignments,
            "latent_state": latent_state,
            "fused_states": fused,
        }

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
