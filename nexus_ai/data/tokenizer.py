"""Deterministic train-split-only vocabulary tokenizer for NEXUS-AI."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
TOKEN_PATTERN = re.compile(r"<[^>]+>|[\w]+|[^\w\s]", re.UNICODE)


class VocabularyTokenizer:
    """A reproducible word/punctuation tokenizer; fit on train records only."""

    def __init__(self, vocab: list[str] | None = None, max_vocab_size: int = 2048):
        if max_vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("max_vocab_size must fit all special tokens")
        self.max_vocab_size = max_vocab_size
        self.vocab = list(SPECIAL_TOKENS) if vocab is None else list(vocab)
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        if len(self.vocab) != len(set(self.vocab)):
            raise ValueError("vocabulary contains duplicate tokens")
        if tuple(self.vocab[:4]) != SPECIAL_TOKENS:
            raise ValueError("vocabulary must begin with the four special tokens")
        if len(self.vocab) > self.max_vocab_size:
            raise ValueError("vocabulary exceeds max_vocab_size")
        self.token_to_id = {token: index for index, token in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    @staticmethod
    def record_text(record: dict) -> str:
        required = ("context", "query", "answer")
        if any(not isinstance(record.get(key), str) for key in required):
            raise ValueError("record must contain string context, query, and answer")
        return (
            f"context: {record['context']} "
            f"question: {record['query']} "
            f"answer: {record['answer']}"
        )

    def fit(self, texts: Iterable[str]) -> "VocabularyTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize(text))
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        available = self.max_vocab_size - len(SPECIAL_TOKENS)
        self.vocab = list(SPECIAL_TOKENS) + [
            token for token, _ in ranked[:available] if token not in SPECIAL_TOKENS
        ]
        self._rebuild_indices()
        return self

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in token_ids:
            token = (
                self.vocab[token_id]
                if isinstance(token_id, int) and 0 <= token_id < len(self.vocab)
                else "<unk>"
            )
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        text = ""
        for token in tokens:
            if not text:
                text = token
            elif re.fullmatch(r"[^\w\s]", token, re.UNICODE):
                text += token
            else:
                text += " " + token
        return text

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "nexus_vocab_v0.1",
            "max_vocab_size": self.max_vocab_size,
            "vocab": self.vocab,
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "nexus_vocab_v0.1":
            raise ValueError("unsupported tokenizer format")
        return cls(vocab=payload["vocab"], max_vocab_size=payload["max_vocab_size"])
