"""Train NEXUS-HSR or the Transformer baseline on synthetic v0.1 data.

CPU-first defaults: pass --device only to override; keep epochs small locally.
"""
import argparse
import json

import torch

from nexus_ai.data import VocabularyTokenizer
from nexus_ai.data.dataloader import create_dataloader
from nexus_ai.training import Trainer

from nexus_ai.models import NexusHSR, TransformerBaseline  # isort:skip

CONFIGS = {
    "nexus": {
        "vocab_size": 54,
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
        "vocab_size": 54,
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
    parser.add_argument("--train", default="datasets/synthetic/v0.1/train.jsonl")
    parser.add_argument("--validation", default="datasets/synthetic/v0.1/validation.jsonl")
    parser.add_argument("--tokenizer", default="datasets/processed/v0.1/tokenizer.json")
    parser.add_argument("--output", default="experiments/exp001")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override model dropout (default: config 0.0).")
    parser.add_argument("--loss-weights-json", default=None,
                        help="JSON dict overriding TotalLoss weights (nexus only).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="Cap batches per epoch for quick smoke runs.")
    args = parser.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"device '{args.device}' requested but CUDA is not available")

    tokenizer = VocabularyTokenizer.load(args.tokenizer)
    config = dict(CONFIGS[args.model])
    config["vocab_size"] = tokenizer.vocab_size
    if args.dropout is not None:
        if args.dropout < 0 or args.dropout > 1:
            raise SystemExit("--dropout must be in [0, 1]")
        config["dropout"] = args.dropout
    loss_weights = {}
    if args.loss_weights_json:
        if args.model != "nexus":
            raise SystemExit("--loss-weights-json applies only to the nexus model")
        loss_weights = json.loads(args.loss_weights_json)

    train_loader = create_dataloader(
        args.train, tokenizer, batch_size=args.batch_size,
        max_length=args.max_length, shuffle=True, num_workers=args.num_workers,
    )
    validation_loader = create_dataloader(
        args.validation, tokenizer, batch_size=args.batch_size,
        max_length=args.max_length, num_workers=args.num_workers,
    )
    steps_per_epoch = len(train_loader)
    if args.limit_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.limit_batches)

    model = NexusHSR(**config) if args.model == "nexus" else TransformerBaseline(**config)
    trainer = Trainer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=max(1, int(steps_per_epoch * args.warmup_ratio * args.epochs)),
        seed=args.seed,
        device=args.device,
        **loss_weights,
    )

    trainer.fit(
        train_loader,
        validation_loader,
        epochs=args.epochs,
        output_dir=args.output,
        max_steps_per_epoch=args.limit_batches,
    )

    final = trainer.evaluate(validation_loader)
    summary = {
        "model": args.model,
        "parameters": model.count_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "seed": args.seed,
        "device": str(trainer.device),
        "epochs": args.epochs,
        "dropout": config["dropout"],
        "loss_weights": loss_weights,
        "final_validation": final,
    }
    with open(f"{args.output}/summary.json", "w", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
