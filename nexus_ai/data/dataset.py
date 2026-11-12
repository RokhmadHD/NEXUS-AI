"""JSONL-backed causal language-modeling dataset."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import VocabularyTokenizer


class StateTransitionDataset(Dataset):
    """Create next-token examples from context + question + answer text."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: VocabularyTokenizer,
        max_length: int = 256,
    ):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"line {line_number}: expected a JSON object")
                token_ids = tokenizer.encode(tokenizer.record_text(record))
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                    token_ids[-1] = tokenizer.eos_id
                if len(token_ids) >= 2:
                    self.samples.append({
                        "input_ids": torch.tensor(token_ids[:-1], dtype=torch.long),
                        "labels": torch.tensor(token_ids[1:], dtype=torch.long),
                        "sample_id": record.get("sample_id", f"line-{line_number}"),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]
