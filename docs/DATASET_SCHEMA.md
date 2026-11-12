# NEXUS-AI Synthetic Dataset Schema v0.1

## Scope
The first generator emits deterministic, simulator-grounded quantity-transfer
examples. It is a pipeline smoke test, not a general reasoning benchmark.

## Record fields
Each JSONL line contains schema_version, sample_id, task, language, split,
context, query, symbolic_target, state_before, state_after, answer and metadata.
The symbolic target contains entities, relations and one transfer event.

## Split policy
Train, validation and test use disjoint entity and item vocabularies. Templates
are shared, so this is not a template-held-out or full compositional OOD test.

## Reproducibility
The same seed, index, split and language produce the same sample. The manifest
records generator version, seed, language and split counts.

## Limitations
One event and one item per example; integer quantities only; no multi-hop
chains, negation, distractors or uncertainty. Synthetic accuracy does not
establish real-world language understanding.
