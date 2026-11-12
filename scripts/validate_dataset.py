"""Validate NEXUS-AI synthetic JSONL files."""
import argparse
import json
import sys
from nexus_ai.data.validator import validate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    reports = [validate_jsonl(path) for path in args.files]
    print(json.dumps(reports, indent=2))
    if any(not report["valid"] for report in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
