"""Checkpoint saving and loading for training runs."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    metadata: dict | None = None,
) -> None:
    """Save model state (and optionally optimizer state) with run metadata.

    Args:
        path: Target ``.pt`` file path; parent directories are created.
        model: The model whose ``state_dict`` is saved.
        optimizer: Optional optimizer whose ``state_dict`` is included.
        metadata: Arbitrary JSON-serializable run metadata (epoch, metrics...).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint written by ``save_checkpoint``.

    Args:
        path: Checkpoint file path.
        model: Model whose state is overwritten in place.
        optimizer: Optional optimizer whose state is loaded in place.
        map_location: Device for tensor deserialization (CPU-first default).

    Returns:
        The checkpoint's ``metadata`` dictionary.

    Raises:
        ValueError: If the checkpoint file lacks a model state dict.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(f"checkpoint {path} does not contain model_state_dict")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload.get("metadata", {})


def save_metrics(path: str | Path, metrics: dict) -> None:
    """Write a training history or evaluation metrics dictionary as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
