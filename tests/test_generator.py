import json

import pytest

from nexus_ai.data.generator import generate_dataset, generate_sample
from nexus_ai.data.validator import validate_jsonl, validate_record


def test_generation_is_deterministic():
    assert generate_sample(3, seed=123) == generate_sample(3, seed=123)


def test_split_entity_and_item_pools_are_disjoint():
    samples = [generate_sample(0, split=s) for s in ("train", "validation", "test")]
    names = [{e["name"] for e in x["symbolic_target"]["entities"]} for x in samples]
    assert names[0].isdisjoint(names[1])
    assert names[0].isdisjoint(names[2])
    assert names[1].isdisjoint(names[2])


@pytest.mark.parametrize("language", ["en", "id"])
def test_generated_record_passes_validator(language):
    assert validate_record(generate_sample(7, seed=9, language=language)) == []


def test_validator_detects_wrong_state():
    record = generate_sample(2)
    giver = record["symbolic_target"]["events"][0]["source"]
    record["state_after"][giver] += 1
    assert any("state_after" in e for e in validate_record(record))


def test_dataset_generation_and_validation(tmp_path):
    counts = generate_dataset(tmp_path, count=20, seed=5)
    assert counts == {"train": 16, "validation": 2, "test": 2}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["total_samples"] == 20
    for split in counts:
        report = validate_jsonl(tmp_path / f"{split}.jsonl")
        assert report["valid"], report["errors"]


def test_invalid_count_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, count=2)


def test_multi_event_chain_is_simulated_correctly():
    record = generate_sample(11, seed=77, max_events=4)
    events = record["symbolic_target"]["events"]
    assert 1 <= len(events) <= 4
    state = dict(record["state_before"])
    for event in events:
        assert state[event["source"]] >= event["quantity"] > 0
        state[event["source"]] -= event["quantity"]
        state[event["target"]] += event["quantity"]
    assert state == record["state_after"]
    assert validate_record(record) == []


def test_validator_rejects_broken_multi_event_chain():
    record = generate_sample(12, seed=77, max_events=3)
    assert len(record["symbolic_target"]["events"]) > 1
    first_giver = record["symbolic_target"]["events"][0]["source"]
    record["state_after"][first_giver] += 1
    assert any("state_after" in e for e in validate_record(record))


def test_multi_event_chain_is_deterministic():
    first = generate_sample(5, seed=31, max_events=3)
    second = generate_sample(5, seed=31, max_events=3)
    assert first == second


def test_invalid_max_events_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate_sample(0, max_events=0)
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, count=10, max_events=99)
