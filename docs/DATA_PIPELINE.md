# NEXUS-AI Data Pipeline v0.1

## Scope

EXP-003B introduces a deterministic word/punctuation vocabulary tokenizer and a
JSONL-backed causal language-modeling pipeline for the synthetic state-transfer
dataset.

## Tokenizer

- Fit the vocabulary on the **train split only** using `scripts/fit_tokenizer.py`.
- Vocabulary order is deterministic: descending token frequency, then lexical
  order for ties.
- Special tokens are `<pad>`, `<unk>`, `<bos>`, and `<eos>`.
- Validation and test out-of-vocabulary tokens map to `<unk>`.
- Save the exact tokenizer vocabulary to JSON and reuse it for every split.
- This baseline tokenizer is not a subword tokenizer and is not intended for
  broad natural-language coverage.

## Dataset and batching

Each record is serialized as context, question, and answer text. The dataset
creates next-token input/label pairs. Batches are right-padded; padded labels
use `-100` so PyTorch cross entropy ignores them. The attention mask marks
non-padding input positions.

The first-stage objective predicts all serialized tokens, including context
and question. Symbolic/state targets remain in the source JSONL for future
auxiliary objectives. A prompt-masked answer-completion objective must be a
separately versioned experiment.

## Leakage and limitations

- The tokenizer is fitted on train only; validation/test are transform-only.
- Entity/item pools are disjoint across splits, but templates are shared.
- This is a tiny synthetic single-transfer dataset. It does not establish
  general language competence or compositional reasoning.
- Instantiate both models with the saved tokenizer's `vocab_size`.
- Use the same tokenizer, data order, truncation length, and objective for both.
