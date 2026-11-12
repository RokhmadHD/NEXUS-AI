"""Fit a deterministic vocabulary using only the training split."""
import argparse
import json

from nexus_ai.data.tokenizer import VocabularyTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="datasets/synthetic/v0.1/train.jsonl")
    parser.add_argument("--output", default="datasets/processed/v0.1/tokenizer.json")
    parser.add_argument("--max-vocab-size", type=int, default=2048)
    args = parser.parse_args()

    texts = []
    with open(args.train, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            texts.append(VocabularyTokenizer.record_text(record))

    tokenizer = VocabularyTokenizer(max_vocab_size=args.max_vocab_size).fit(texts)
    tokenizer.save(args.output)
    print(json.dumps({
        "output": args.output,
        "training_records": len(texts),
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": 4,
    }, indent=2))


if __name__ == "__main__":
    main()
