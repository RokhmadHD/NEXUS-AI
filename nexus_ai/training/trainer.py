"""CPU-first training loop for NEXUS-AI models."""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nexus_ai.objectives import (
    DEFAULT_IGNORE_INDEX,
    TotalLoss,
    causal_language_modeling_loss,
)

from .checkpoint import save_checkpoint, save_metrics
from .optimizer import build_optimizer, build_warmup_scheduler


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    """A compact trainer for causal language-modeling on CPU.

    Works with both model families:

    - ``NexusHSR`` returns a dict with internal states, so the full
      ``TotalLoss`` multi-objective applies.
    - ``TransformerBaseline`` returns logits, so only the language loss is
      used.

    Args:
        model: The model to train.
        learning_rate: Peak AdamW learning rate.
        weight_decay: AdamW weight decay coefficient.
        max_grad_norm: Gradient clipping norm (``None`` disables clipping).
        warmup_steps: Linear LR warmup steps.
        seed: RNG seed applied on construction.
        device: Torch device string (default CPU per project policy).
        loss_weights: Optional partial overrides of ``TotalLoss`` weights,
            used only for dict-output models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float | None = 1.0,
        warmup_steps: int = 0,
        seed: int = 42,
        device: str = "cpu",
        **loss_weights,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.max_grad_norm = max_grad_norm
        self.ignore_index = DEFAULT_IGNORE_INDEX
        self.total_loss = TotalLoss(**loss_weights)
        self.optimizer = build_optimizer(model, learning_rate, weight_decay)
        self.scheduler = build_warmup_scheduler(self.optimizer, warmup_steps)
        seed_everything(seed)

    def _compute_loss(
        self, output: torch.Tensor | dict, labels: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        if isinstance(output, dict):
            breakdown = self.total_loss.compute(output, labels)
            return breakdown.total, breakdown.counted_tokens
        loss, counted = causal_language_modeling_loss(output, labels, self.ignore_index)
        return loss, counted

    def _step(self, batch: dict) -> tuple[torch.Tensor, int, int]:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        output = self.model(input_ids)
        loss, counted = self._compute_loss(output, labels)
        logits = output["logits"] if isinstance(output, dict) else output
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            supervised = labels != self.ignore_index
            correct = int((predictions[supervised] == labels[supervised]).sum().item())
        return loss, counted, correct

    def train_epoch(self, loader: DataLoader, max_steps: int | None = None) -> dict:
        """Run one optimization epoch and return aggregate metrics.

        Args:
            loader: Training DataLoader.
            max_steps: Optional cap on optimizer steps per epoch (smoke runs).
        """
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        self.model.train()
        total_loss, total_tokens, total_correct = 0.0, 0, 0
        steps = 0
        for steps, batch in enumerate(loader, start=1):
            if max_steps is not None and steps > max_steps:
                break
            self.optimizer.zero_grad()
            loss, counted, correct = self._step(batch)
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item() * counted
            total_tokens += counted
            total_correct += correct
        if total_tokens == 0:
            raise ValueError("training loader produced no supervised tokens")
        return {
            "loss": total_loss / total_tokens,
            "accuracy": total_correct / total_tokens,
            "perplexity": math.exp(total_loss / total_tokens),
            "tokens": total_tokens,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate without updating parameters; returns loss/accuracy/ppl."""
        self.model.eval()
        total_loss, total_tokens, total_correct = 0.0, 0, 0
        for batch in loader:
            loss, counted, correct = self._step(batch)
            total_loss += loss.item() * counted
            total_tokens += counted
            total_correct += correct
        if total_tokens == 0:
            raise ValueError("evaluation loader produced no supervised tokens")
        mean_loss = total_loss / total_tokens
        return {
            "loss": mean_loss,
            "accuracy": total_correct / total_tokens,
            "perplexity": math.exp(mean_loss),
            "tokens": total_tokens,
        }

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        epochs: int = 1,
        output_dir: str | Path | None = None,
        max_steps_per_epoch: int | None = None,
    ) -> list[dict]:
        """Train for ``epochs`` epochs, checkpointing the best validation loss.

        Args:
            train_loader: Training DataLoader.
            validation_loader: Optional evaluation DataLoader used for
                best-checkpoint selection; if omitted, no checkpointing.
            epochs: Number of epochs (must be positive).
            max_steps_per_epoch: Optional cap on optimizer steps per epoch.
            output_dir: Directory for ``checkpoint_best.pt`` and
                ``history.json``; required when a validation loader is given.

        Returns:
            A list of per-epoch metric dictionaries, oldest first.
        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if validation_loader is not None and output_dir is None:
            raise ValueError("output_dir is required when a validation loader is given")
        history: list[dict] = []
        best_validation = float("inf")
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, max_steps_per_epoch)
            entry = {"epoch": epoch, "train": train_metrics}
            if validation_loader is not None:
                validation_metrics = self.evaluate(validation_loader)
                entry["validation"] = validation_metrics
                if validation_metrics["loss"] < best_validation:
                    best_validation = validation_metrics["loss"]
                    save_checkpoint(
                        Path(output_dir) / "checkpoint_best.pt",
                        self.model,
                        self.optimizer,
                        metadata={"epoch": epoch, "validation": validation_metrics},
                    )
            history.append(entry)
            validation_loss = (
                f"validation_loss={entry['validation']['loss']:.4f} "
                if "validation" in entry
                else ""
            )
            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_metrics['loss']:.4f} {validation_loss}"
                f"train_accuracy={train_metrics['accuracy']:.4f}",
                flush=True,
            )
        if output_dir is not None and validation_loader is not None:
            save_metrics(Path(output_dir) / "history.json", {"history": history})
        return history
