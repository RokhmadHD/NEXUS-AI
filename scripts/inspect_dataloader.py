"""Smoke-test the NEXUS-AI train/validation/test data pipeline."""
import argparse
import json

from nexus_ai.data.dataloader import create_dataloader
from nexus_ai.data.tokenizer import VocabularyTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/synthetic/v0.1")
    parser.add_argument("--tokenizer", default="datasets/processed/v0.1/tokenizer.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    tokenizer = VocabularyTokenizer.load(args.tokenizer)
    report = {"vocab_size": tokenizer.vocab_size, "splits": {}}
    for split in ("train", "validation", "test"):
        loader = create_dataloader(
            f"{args.data_dir}/{split}.jsonl",
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shuffle=(split == "train"),
        )
        batch = next(iter(loader), None)
        report["splits"][split] = {
            "samples": len(loader.dataset),
            "batches": len(loader),
            "first_batch_shape": list(batch["input_ids"].shape) if batch else None,
            "labels_have_ignored_padding": bool((batch["labels"] == -100).any()) if batch else False,
        }
        if batch is None:
            raise SystemExit(f"{split} split is empty")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
