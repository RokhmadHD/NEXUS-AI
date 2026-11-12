"""Evaluate a trained checkpoint on a dataset split."""
import argparse
import json

import torch

from nexus_ai.data import VocabularyTokenizer
from nexus_ai.data.dataloader import create_dataloader
from nexus_ai.models import NexusHSR, TransformerBaseline
from nexus_ai.training import Trainer, load_checkpoint

MODEL_CONFIGS = {
    "nexus": {
        "context_length": 256,
        "embedding_dim": 128,
        "num_heads": 4,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "ffn_dim": 512,
        "latent_iterations": 2,
        "symbolic_slots": 16,
        "dropout": 0.0,
    },
    "baseline": {
        "context_length": 256,
        "embedding_dim": 128,
        "num_heads": 4,
        "num_layers": 4,
        "ffn_dim": 723,
        "dropout": 0.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("nexus", "baseline"), default="nexus")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="datasets/processed/v0.1/tokenizer.json")
    parser.add_argument("--data", default="datasets/synthetic/v0.1/validation.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"device '{args.device}' requested but CUDA is not available")

    tokenizer = VocabularyTokenizer.load(args.tokenizer)
    config = dict(MODEL_CONFIGS[args.model])
    config["vocab_size"] = tokenizer.vocab_size
    model = NexusHSR(**config) if args.model == "nexus" else TransformerBaseline(**config)
    metadata = load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.to(args.device)

    trainer = Trainer(model, device=args.device)
    loader = create_dataloader(
        args.data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
    )
    metrics = trainer.evaluate(loader)
    print(json.dumps({
        "model": args.model,
        "checkpoint": args.checkpoint,
        "checkpoint_metadata": metadata,
        "data": args.data,
        "metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
