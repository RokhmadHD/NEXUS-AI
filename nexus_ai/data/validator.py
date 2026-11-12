"""Structural and simulator-ground-truth validation for synthetic JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus_ai.data.generator import SCHEMA_VERSION


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version", "sample_id", "task", "language", "split", "context",
        "query", "symbolic_target", "state_before", "state_after", "answer", "metadata",
    }
    missing = sorted(required - record.keys())
    if missing:
        return [f"missing fields: {', '.join(missing)}"]

    if record["schema_version"] != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if record["task"] != "state_transition":
        errors.append("unsupported task")
    if record["split"] not in {"train", "validation", "test"}:
        errors.append("invalid split")
    if record["language"] not in {"en", "id"}:
        errors.append("invalid language")
    if not all(isinstance(record[k], str) and record[k].strip()
               for k in ("sample_id", "context", "query", "answer")):
        errors.append("text fields must be non-empty strings")

    target = record["symbolic_target"]
    if not isinstance(target, dict):
        return errors + ["symbolic_target must be an object"]
    events = target.get("events")
    if not isinstance(events, list) or len(events) != 1:
        return errors + ["expected exactly one event"]

    event = events[0]
    giver, receiver = event.get("source"), event.get("target")
    amount = event.get("quantity")
    before, after = record["state_before"], record["state_after"]
    if not isinstance(before, dict) or not isinstance(after, dict):
        return errors + ["states must be objects"]
    if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
        return errors + ["event quantity must be a positive integer"]
    if giver == receiver or giver not in before or receiver not in before:
        return errors + ["event participants are invalid or missing"]
    if set(before) != set(after):
        errors.append("state keys differ before and after")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
               for v in list(before.values()) + list(after.values())):
        return errors + ["state quantities must be non-negative integers"]
    if errors:
        return errors
    if before[giver] < amount:
        errors.append("transfer exceeds source quantity")
    expected = dict(before)
    expected[giver] -= amount
    expected[receiver] += amount
    if expected != after:
        errors.append("state_after does not match simulated transfer")
    return errors


def validate_jsonl(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    with file_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                errors.append(f"line {line_no}: blank line")
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_no}: record must be an object")
                continue
            count += 1
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str):
                if sample_id in seen:
                    errors.append(f"line {line_no}: duplicate sample_id {sample_id}")
                seen.add(sample_id)
            errors.extend(f"line {line_no}: {e}" for e in validate_record(record))
    return {"path": str(file_path), "records": count, "errors": errors,
            "valid": not errors}
