import json

import pytest
import torch

from nexus_ai.data.dataloader import collate_causal_lm, create_dataloader
from nexus_ai.data.generator import generate_dataset
from nexus_ai.data.tokenizer import VocabularyTokenizer


def make_records(tmp_path):
    data_dir = tmp_path / "data"
    generate_dataset(data_dir, count=30, seed=17)
    records = [
        json.loads(line)
        for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return data_dir, records


def test_tokenizer_is_deterministic_and_unknown_maps_to_unk(tmp_path):
    _, records = make_records(tmp_path)
    texts = [VocabularyTokenizer.record_text(record) for record in records]
    first = VocabularyTokenizer().fit(texts)
    second = VocabularyTokenizer().fit(texts)
    assert first.vocab == second.vocab
    assert first.encode("unseenvalidationword")[1] == first.unk_id


def test_tokenizer_save_and_load_roundtrip(tmp_path):
    tokenizer = VocabularyTokenizer().fit(["Alice gives two books.", "Ben has three cards."])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = VocabularyTokenizer.load(path)
    assert loaded.vocab == tokenizer.vocab
    assert loaded.encode("Alice gives two books.") == tokenizer.encode("Alice gives two books.")


def test_encode_decode_basic_roundtrip():
    tokenizer = VocabularyTokenizer().fit(["Alice has 3 books.", "Ben has 2 cards."])
    assert tokenizer.decode(tokenizer.encode("Alice has 3 books.")) == "alice has 3 books."


def test_collate_pads_inputs_and_masks_padding_labels():
    batch = [
        {"input_ids": torch.tensor([2, 4, 5]), "labels": torch.tensor([4, 5, 3])},
        {"input_ids": torch.tensor([2, 6]), "labels": torch.tensor([6, 3])},
    ]
    result = collate_causal_lm(batch, pad_id=0)
    assert result["input_ids"].tolist() == [[2, 4, 5], [2, 6, 0]]
    assert result["labels"].tolist() == [[4, 5, 3], [6, 3, -100]]
    assert result["attention_mask"].tolist() == [[True, True, True], [True, True, False]]


def test_dataset_and_dataloader_shapes(tmp_path):
    data_dir, records = make_records(tmp_path)
    texts = [VocabularyTokenizer.record_text(record) for record in records]
    tokenizer = VocabularyTokenizer().fit(texts)
    loader = create_dataloader(
        str(data_dir / "train.jsonl"), tokenizer, batch_size=4, max_length=128
    )
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["input_ids"].dtype == torch.long
    assert (batch["labels"] != -100).any()


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError):
        collate_causal_lm([])
