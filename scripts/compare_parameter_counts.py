"""Verify the EXP-003A parameter budget for NEXUS-HSR and its baseline.

Vocabulary size is derived from the fitted tokenizer per project policy;
both models use the EXP-003A architecture dimensions.
"""
import json
import sys

from nexus_ai.data import VocabularyTokenizer
from nexus_ai.models import NexusHSR, TransformerBaseline

DEFAULT_TOKENIZER_PATH = "datasets/processed/v0.1/tokenizer.json"

NEXUS_CONFIG = {
    "context_length": 256,
    "embedding_dim": 128,
    "num_heads": 4,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "ffn_dim": 512,
    "latent_iterations": 2,
    "symbolic_slots": 16,
    "dropout": 0.0,
}

BASELINE_CONFIG = {
    "context_length": 256,
    "embedding_dim": 128,
    "num_heads": 4,
    "num_layers": 4,
    "ffn_dim": 723,
    "dropout": 0.0,
}


def main() -> None:
    tokenizer_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOKENIZER_PATH
    vocab_size = VocabularyTokenizer.load(tokenizer_path).vocab_size

    nexus = NexusHSR(vocab_size=vocab_size, **NEXUS_CONFIG)
    baseline = TransformerBaseline(vocab_size=vocab_size, **BASELINE_CONFIG)

    nexus_count = nexus.count_parameters()
    baseline_count = baseline.count_parameters()
    difference = abs(nexus_count - baseline_count)
    difference_percent = difference / nexus_count * 100

    report = {
        "experiment": "EXP-003A",
        "tokenizer_path": tokenizer_path,
        "vocab_size": vocab_size,
        "nexus_hsr_parameters": nexus_count,
        "transformer_baseline_parameters": baseline_count,
        "absolute_difference": difference,
        "difference_percent": round(difference_percent, 6),
        "tolerance_percent": 0.1,
        "parameter_matched": difference_percent <= 0.1,
    }
    print(json.dumps(report, indent=2))

    if not report["parameter_matched"]:
        raise SystemExit("Parameter budget mismatch exceeds 0.1%")


if __name__ == "__main__":
    main()
