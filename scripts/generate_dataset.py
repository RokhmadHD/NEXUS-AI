"""Generate NEXUS-AI synthetic world data."""
import argparse
import json
from nexus_ai.data.generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/synthetic/v0.1")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", choices=("en", "id"), default="en")
    args = parser.parse_args()
    counts = generate_dataset(args.output, args.count, args.seed, args.language)
    print(json.dumps({"output": args.output, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
