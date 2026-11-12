"""Deterministic synthetic quantity-transfer data for NEXUS-AI (v0.3).

Compared to v0.1/v0.2, the generator supports multi-event transfer chains
(1..max_events sequential transfers between the same pair), richer template
variety, larger disjoint entity/item pools, and wider quantity ranges.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "nexus.synthetic.v0.3"
MAX_EVENTS = 8
POOLS = {
    "train": (
        ("Alice", "Ben", "Clara", "Daniel", "Eva", "Farid", "Gita", "Hana", "Ivan", "Julia"),
        ("books", "cards", "marbles", "stickers", "pencils", "buttons"),
    ),
    "validation": (
        ("Mira", "Noah", "Olive", "Pavel", "Quinn"),
        ("coins", "blocks", "shells", "ribbons"),
    ),
    "test": (
        ("Zuri", "Kai", "Lena", "Omar", "Sari"),
        ("beads", "crystals", "tokens", "gems"),
    ),
}
TEMPLATES = {
    "en": (
        "{giver} has {before} {item}. {giver} gives {amount} {item} to {receiver}.",
        "{giver} starts with {before} {item} and transfers {amount} to {receiver}.",
        "{giver} is holding {before} {item}; {giver} hands {amount} over to {receiver}.",
        "In the room, {giver} counts {before} {item} and then passes {amount} to {receiver}.",
        "{giver} owns {before} {item}. A transfer of {amount} {item} to {receiver} happens.",
        "Before anything happens, {giver} has {before} {item}; then {giver} sends {amount} to {receiver}.",
    ),
    "id": (
        "{giver} memiliki {before} {item}. {giver} memberikan {amount} {item} kepada {receiver}.",
        "{giver} mempunyai {before} {item}, lalu memberikan {amount} kepada {receiver}.",
        "{giver} sedang memegang {before} {item}; {giver} menyerahkan {amount} kepada {receiver}.",
        "Di dalam ruangan, {giver} menghitung {before} {item} kemudian menyerahkan {amount} kepada {receiver}.",
        "{giver} memiliki {before} {item}. Terjadi transfer sebanyak {amount} {item} kepada {receiver}.",
        "Awalnya {giver} mempunyai {before} {item}; setelah itu {giver} mengirim {amount} kepada {receiver}.",
    ),
}


def _transfer_word(count: int, language: str) -> str:
    if language == "en":
        return "transfer" if count == 1 else "transfers"
    return "transfer" if count == 1 else "transfer-trans"


def generate_sample(index: int, seed: int = 42, split: str = "train",
                    language: str = "en", max_events: int = 1) -> dict[str, Any]:
    """Generate one sample; same arguments always produce the same record.

    ``max_events`` controls the upper bound of sequential transfer events in a
    sample; the actual count is drawn per sample, so single-event records are
    still produced when ``max_events`` is 1.
    """
    if split not in POOLS:
        raise ValueError(f"Unsupported split: {split}")
    if language not in TEMPLATES:
        raise ValueError("language must be 'en' or 'id'")
    if index < 0:
        raise ValueError("index must be non-negative")
    if max_events < 1 or max_events > MAX_EVENTS:
        raise ValueError(f"max_events must be between 1 and {MAX_EVENTS}")

    rng = random.Random(seed + index * 1_000_003)
    people, items = POOLS[split]
    giver, receiver = rng.sample(people, 2)
    item = rng.choice(items)
    before = rng.randint(2, 50)

    events: list[dict[str, Any]] = []
    state = {giver: before, receiver: 0}
    event_count = rng.randint(1, max_events)
    for _ in range(event_count):
        if state[giver] < 2:
            break
        amount = rng.randint(1, state[giver] - 1)
        state[giver] -= amount
        state[receiver] += amount
        events.append({
            "type": "transfer", "source": giver, "target": receiver,
            "item": item, "quantity": amount,
        })

    remaining = state[giver]
    context = rng.choice(TEMPLATES[language]).format(
        giver=giver, receiver=receiver, before=before,
        amount=events[0]["quantity"], item=item
    )
    transfer_word = _transfer_word(len(events), language)
    if language == "en":
        query = f"How many {item} does {giver} have after the {transfer_word}?"
        answer = f"{giver} has {remaining} {item} after the {transfer_word}."
    else:
        query = f"Berapa {item} yang dimiliki {giver} setelah {transfer_word}?"
        answer = f"{giver} memiliki {remaining} {item} setelah {transfer_word}."

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"nexus-v0.3-{seed}-{split}-{index:07d}",
        "task": "state_transition",
        "language": language,
        "split": split,
        "context": context,
        "query": query,
        "symbolic_target": {
            "entities": [
                {"id": "giver", "type": "person", "name": giver},
                {"id": "receiver", "type": "person", "name": receiver},
                {"item": item, "type": "object", "name": item},
            ],
            "relations": [
                {"type": "owns_before", "owner": giver, "item": item, "quantity": before},
                {"type": "owns_after", "owner": giver, "item": item, "quantity": remaining},
                {"type": "owns_after", "owner": receiver, "item": item,
                 "quantity": state[receiver]},
            ],
            "events": events,
        },
        "state_before": {giver: before, receiver: 0},
        "state_after": dict(state),
        "answer": answer,
        "metadata": {
            "generator": "nexus_world_v0.3",
            "seed": seed,
            "sample_index": index,
            "event_count": len(events),
            "split_strategy": "disjoint_entity_and_item_pools",
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def generate_dataset(output_dir: str | Path, count: int = 1000, seed: int = 42,
                     language: str = "en", max_events: int = 1) -> dict[str, int]:
    """Write train/validation/test JSONL files and a reproducibility manifest."""
    if count < 3:
        raise ValueError("count must be at least 3")
    if language not in TEMPLATES:
        raise ValueError("language must be 'en' or 'id'")
    if max_events < 1 or max_events > MAX_EVENTS:
        raise ValueError(f"max_events must be between 1 and {MAX_EVENTS}")

    train_n = int(count * 0.8)
    valid_n = int(count * 0.1)
    counts = {"train": train_n, "validation": valid_n,
              "test": count - train_n - valid_n}
    root = Path(output_dir)
    for split, size in counts.items():
        _write_jsonl(root / f"{split}.jsonl", [
            generate_sample(i, seed=seed, split=split, language=language,
                            max_events=max_events)
            for i in range(size)
        ])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "nexus_world_v0.3",
        "seed": seed,
        "language": language,
        "total_samples": count,
        "split_counts": counts,
        "max_events": max_events,
        "split_strategy": "disjoint_entity_and_item_pools",
        "limitations": [
            "Transfers occur between a fixed giver/receiver pair per sample.",
            "Only the final state is queried; intermediate states are not asked.",
            "Templates are shared across splits; this is not template-held-out evaluation.",
            "Synthetic results do not establish real-world language understanding.",
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return counts
