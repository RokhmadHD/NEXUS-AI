import json

import pytest
import torch
from torch.utils.data import DataLoader

from nexus_ai.data.dataloader import collate_causal_lm
from nexus_ai.models import NexusHSR, TransformerBaseline
from nexus_ai.training import Trainer, load_checkpoint, save_checkpoint, seed_everything

VOCAB = 32


def make_model(name):
    if name == "nexus":
        return NexusHSR(vocab_size=VOCAB, context_length=16, embedding_dim=32,
                        num_heads=4, encoder_layers=1, decoder_layers=1,
                        ffn_dim=64, latent_iterations=2, symbolic_slots=4)
    return TransformerBaseline(vocab_size=VOCAB, context_length=16,
                               embedding_dim=32, num_heads=4, num_layers=2, ffn_dim=48)


def make_loader():
    generator = torch.Generator().manual_seed(0)
    samples = []
    for _ in range(8):
        length = 10
        ids = torch.randint(0, VOCAB, (length,), generator=generator)
        samples.append({"input_ids": ids[:-1], "labels": ids[1:]})
    return DataLoader(samples, batch_size=4, collate_fn=collate_causal_lm)


@pytest.mark.parametrize("model_name", ["nexus", "baseline"])
def test_trainer_reduces_and_reports_finite_metrics(model_name):
    seed_everything(42)
    trainer = Trainer(make_model(model_name), learning_rate=1e-3, seed=42)
    loader = make_loader()

    before = trainer.evaluate(loader)
    metrics = trainer.train_epoch(loader)
    after = trainer.evaluate(loader)

    for metrics_dict in (before, metrics, after):
        assert torch.isfinite(torch.tensor(metrics_dict["loss"]))
        assert 0.0 <= metrics_dict["accuracy"] <= 1.0
        assert metrics_dict["perplexity"] >= 1.0


@pytest.mark.parametrize("model_name", ["nexus", "baseline"])
def test_trainer_is_deterministic_with_seed(model_name):
    histories = []
    for _ in range(2):
        seed_everything(123)
        trainer = Trainer(make_model(model_name), seed=123)
        histories.append(trainer.train_epoch(make_loader())["loss"])
    assert histories[0] == pytest.approx(histories[1])


@pytest.mark.parametrize("model_name", ["nexus", "baseline"])
def test_fit_saves_best_checkpoint_and_history(model_name, tmp_path):
    trainer = Trainer(make_model(model_name), seed=42)
    history = trainer.fit(make_loader(), make_loader(), epochs=2, output_dir=tmp_path)
    assert len(history) == 2
    assert [entry["epoch"] for entry in history] == [1, 2]
    checkpoint = tmp_path / "checkpoint_best.pt"
    assert checkpoint.exists()
    history_payload = json.loads((tmp_path / "history.json").read_text())
    assert len(history_payload["history"]) == 2


@pytest.mark.parametrize("model_name", ["nexus", "baseline"])
def test_checkpoint_roundtrip_restores_model(model_name, tmp_path):
    trainer = Trainer(make_model(model_name), seed=42)
    trainer.train_epoch(make_loader())
    reference = trainer.evaluate(make_loader())

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, trainer.model, trainer.optimizer,
                    metadata={"epoch": 1})

    fresh = Trainer(make_model(model_name), seed=999)
    metadata = load_checkpoint(path, fresh.model)
    assert metadata["epoch"] == 1
    assert fresh.evaluate(make_loader())["loss"] == pytest.approx(reference["loss"])


def test_fit_requires_output_dir_for_validation():
    trainer = Trainer(make_model("nexus"), seed=42)
    with pytest.raises(ValueError):
        trainer.fit(make_loader(), make_loader(), epochs=1)


def test_train_epoch_rejects_invalid_max_steps():
    trainer = Trainer(make_model("nexus"), seed=42)
    with pytest.raises(ValueError):
        trainer.train_epoch(make_loader(), max_steps=0)
