"""Answer exact-match benchmark: can the model predict the final quantity?

For each record the model is evaluated teacher-forced on the token position
that carries the answer number (the giver's post-transfer quantity). This is
the state-tracking-critical token, so per-position exact match on it is a
much sharper capability measure than overall token accuracy.

Results are reported overall and grouped by event_count (1..N transfer
events), so the architecture gap can be compared across chain depth.
"""
import argparse
import json
from collections import defaultdict

import torch

from nexus_ai.data import VocabularyTokenizer


def find_answer_number_positions(record: dict, tokenizer: VocabularyTokenizer):
    """Locate the answer-number token position inside the encoded record.

    The answer sentence is located by matching its full token subsequence
    (the giver/verb pair alone is ambiguous: the context also contains
    ``<giver> has <before> <item>``). Inside the answer, the number is the
    third token: ``<giver> <has|memiliki> <quantity> ...``. Returns the index
    of the number inside the full encoded token list, or None.
    """
    full_ids = tokenizer.encode(tokenizer.record_text(record), add_bos=True, add_eos=False)
    answer_text = (
        f"{record['symbolic_target']['entities'][0]['name']} {answer_verb(record)} "
        f"{answer_number(record)} {record['symbolic_target']['entities'][2]['name']}"
    )
    answer_ids = tokenizer.encode(answer_text, add_bos=False, add_eos=False)
    number_id = tokenizer.token_to_id.get(str(answer_number(record)))
    if number_id is None or len(answer_ids) > len(full_ids):
        return None, None

    last_match = None
    for start in range(len(full_ids) - len(answer_ids) + 1):
        if full_ids[start:start + len(answer_ids)] == answer_ids:
            last_match = start
    if last_match is None:
        return None, None
    return last_match + 2, number_id


def answer_verb(record: dict) -> str:
    return "has" if record["language"] == "en" else "memiliki"


def answer_number(record: dict) -> int:
    giver = record["symbolic_target"]["entities"][0]["name"]
    return record["state_after"][giver]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("nexus", "baseline"), default="nexus")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="datasets/processed/v0.3/tokenizer.json")
    parser.add_argument("--data", default="datasets/synthetic/v0.3/test.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"device '{args.device}' requested but CUDA is not available")

    tokenizer = VocabularyTokenizer.load(args.tokenizer)
    from nexus_ai.models import NexusHSR, TransformerBaseline

    if args.model == "nexus":
        model = NexusHSR(vocab_size=tokenizer.vocab_size, context_length=256,
                         embedding_dim=128, num_heads=4, encoder_layers=2,
                         decoder_layers=2, ffn_dim=512, latent_iterations=2,
                         symbolic_slots=16, dropout=0.0)
    else:
        model = TransformerBaseline(vocab_size=tokenizer.vocab_size, context_length=256,
                                    embedding_dim=128, num_heads=4, num_layers=4,
                                    ffn_dim=723, dropout=0.0)
    from nexus_ai.training import load_checkpoint

    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.to(args.device)
    model.eval()

    records = []
    with open(args.data, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))

    batches = []  # (record, encoded_ids, number_index)
    skipped = 0
    for record in records:
        full_ids = tokenizer.encode(tokenizer.record_text(record), add_bos=True, add_eos=False)
        full_ids = full_ids[:256]
        index, number_id = find_answer_number_positions(record, tokenizer)
        if index is None or index >= len(full_ids):
            skipped += 1
            continue
        batches.append((record, full_ids, index, number_id))

    correct_by_depth: dict[int, list[int]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(batches), args.batch_size):
            chunk = batches[start:start + args.batch_size]
            max_length = max(len(ids) for _, ids, _, _ in chunk)
            input_ids = torch.zeros((len(chunk), max_length), dtype=torch.long)
            for row, (_, ids, _, _) in enumerate(chunk):
                input_ids[row, :len(ids)] = torch.tensor(ids)
            input_ids = input_ids.to(args.device)
            output = model(input_ids)
            logits = output["logits"] if isinstance(output, dict) else output
            predictions = logits.argmax(dim=-1)
            for row, (record, ids, number_index, number_id) in enumerate(chunk):
                predicted = predictions[row, number_index - 1].item()
                correct_by_depth[record["metadata"].get("event_count", 1)].append(
                    int(predicted == number_id)
                )

    all_correct = [value for values in correct_by_depth.values() for value in values]
    report = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "evaluated": len(all_correct),
        "skipped_records": skipped,
        "overall_exact_match": round(
            sum(all_correct) / len(all_correct), 4
        ) if all_correct else None,
        "exact_match_by_event_count": {
            str(depth): round(sum(values) / len(values), 4)
            for depth, values in sorted(correct_by_depth.items())
        },
        "counts_by_event_count": {
            str(depth): len(values) for depth, values in sorted(correct_by_depth.items())
        },
        "note": "teacher-forced exact match on the answer-number token only",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
