"""Deterministic synthetic quantity-transfer data for NEXUS-AI."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "nexus.synthetic.v0.1"
POOLS = {
    "train": (("Alice", "Ben", "Clara", "Daniel", "Eva", "Farid"),
              ("books", "cards", "marbles", "stickers")),
    "validation": (("Mira", "Noah", "Olive", "Pavel"), ("coins", "blocks", "shells")),
    "test": (("Zuri", "Kai", "Lena", "Omar"), ("beads", "crystals", "tokens")),
}
TEMPLATES = {
    "en": (
        "{giver} has {before} {item}. {giver} gives {amount} {item} to {receiver}.",
        "{giver} starts with {before} {item} and transfers {amount} to {receiver}.",
    ),
    "id": (
        "{giver} memiliki {before} {item}. {giver} memberikan {amount} {item} kepada {receiver}.",
        "{giver} mempunyai {before} {item}, lalu memberikan {amount} kepada {receiver}.",
    ),
}


def generate_sample(index: int, seed: int = 42, split: str = "train",
                    language: str = "en") -> dict[str, Any]:
    """Generate one sample; same arguments always produce the same record."""
    if split not in POOLS:
        raise ValueError(f"Unsupported split: {split}")
    if language not in TEMPLATES:
        raise ValueError("language must be 'en' or 'id'")
    if index < 0:
        raise ValueError("index must be non-negative")

    rng = random.Random(seed + index * 1_000_003)
    people, items = POOLS[split]
    giver, receiver = rng.sample(people, 2)
    item = rng.choice(items)
    before = rng.randint(2, 20)
    amount = rng.randint(1, before - 1)
    remaining = before - amount

    context = rng.choice(TEMPLATES[language]).format(
        giver=giver, receiver=receiver, before=before, amount=amount, item=item
    )
    if language == "en":
        query = f"How many {item} does {giver} have after the transfer?"
        answer = f"{giver} has {remaining} {item} after the transfer."
    else:
        query = f"Berapa {item} yang dimiliki {giver} setelah transfer?"
        answer = f"{giver} memiliki {remaining} {item} setelah transfer."

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"nexus-v0.1-{seed}-{split}-{index:07d}",
        "task": "state_transition",
        "language": language,
        "split": split,
        "context": context,
        "query": query,
        "symbolic_target": {
            "entities": [
                {"id": "giver", "type": "person", "name": giver},
                {"id": "receiver", "type": "person", "name": receiver},
                {"id": "item", "type": "object", "name": item},
            ],
            "relations": [
                {"type": "owns_before", "owner": giver, "item": item, "quantity": before},
                {"type": "owns_after", "owner": giver, "item": item, "quantity": remaining},
                {"type": "owns_after", "owner": receiver, "item": item, "quantity": amount},
            ],
            "events": [{
                "type": "transfer", "source": giver, "target": receiver,
                "item": item, "quantity": amount,
            }],
        },
        "state_before": {giver: before, receiver: 0},
        "state_after": {giver: remaining, receiver: amount},
        "answer": answer,
        "metadata": {
            "generator": "nexus_world_v0.1",
            "seed": seed,
            "sample_index": index,
            "split_strategy": "disjoint_entity_and_item_pools",
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def generate_dataset(output_dir: str | Path, count: int = 1000, seed: int = 42,
                     language: str = "en") -> dict[str, int]:
    """Write train/validation/test JSONL files and a reproducibility manifest."""
    if count < 3:
        raise ValueError("count must be at least 3")
    if language not in TEMPLATES:
        raise ValueError("language must be 'en' or 'id'")

    train_n = int(count * 0.8)
    valid_n = int(count * 0.1)
    counts = {"train": train_n, "validation": valid_n,
              "test": count - train_n - valid_n}
    root = Path(output_dir)
    for split, size in counts.items():
        _write_jsonl(root / f"{split}.jsonl", [
            generate_sample(i, seed=seed, split=split, language=language)
            for i in range(size)
        ])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "nexus_world_v0.1",
        "seed": seed,
        "language": language,
        "total_samples": count,
        "split_counts": counts,
        "split_strategy": "disjoint_entity_and_item_pools",
        "limitations": [
            "Single transfer event and single item per sample.",
            "Templates are shared across splits; this is not template-held-out evaluation.",
            "Synthetic results do not establish real-world language understanding.",
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return counts
