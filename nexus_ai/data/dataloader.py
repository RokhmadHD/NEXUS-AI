"""Batching utilities for causal language-modeling datasets."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .dataset import StateTransitionDataset
from .tokenizer import VocabularyTokenizer


def collate_causal_lm(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch must not be empty")
    max_length = max(sample["input_ids"].numel() for sample in batch)
    input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)

    for row, sample in enumerate(batch):
        inputs, targets = sample["input_ids"], sample["labels"]
        length = min(inputs.numel(), targets.numel())
        input_ids[row, :length] = inputs[:length]
        labels[row, :length] = targets[:length]
        attention_mask[row, :length] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def create_dataloader(
    jsonl_path: str,
    tokenizer: VocabularyTokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = StateTransitionDataset(jsonl_path, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_causal_lm(batch, tokenizer.pad_id),
    )
