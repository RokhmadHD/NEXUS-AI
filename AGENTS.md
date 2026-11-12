# AGENTS.md — NEXUS-AI contributor instructions

These instructions apply to coding agents and contributors working in this repository.

## 1. Project purpose

NEXUS-AI is an experimental research codebase for a proposed hybrid neural-symbolic architecture with recurrent latent-state refinement. Treat architectural benefits as hypotheses. Do not describe the model as having proven reasoning, understanding, or general intelligence.

## 2. Before changing code

1. Read this file and the relevant project documentation, especially:
   - `README.md`
   - `ARCHITECTURE.md`
   - `RESEARCH.md`
   - `ROADMAP.md`
   - `docs/DATA_PIPELINE.md`
   - `docs/DATASET_SCHEMA.md`
2. Inspect the current implementation and tests before proposing edits.
3. Preserve existing public APIs unless a change is necessary; update callers and tests when APIs change.
4. Do not assume a previous notebook variable, generated file, or local environment exists.

## 3. Hardware and runtime policy

### Local development

- The developer's local PC is CPU-first: Intel i3-12100F, 16 GB system RAM, and only a small approximately 2 GB VRAM GPU.
- Default local development, tests, dataset generation, tokenizer fitting, and integration smoke tests to CPU.
- Do not require CUDA or select the small GPU by default.
- Keep local tests lightweight: small batches, short sequences, and no long training runs.
- In examples, use an explicit CPU device unless the user specifically requests GPU execution.
- If setting PyTorch thread counts, use conservative values and make them easy to change.

### Remote training

- Use Kaggle GPU runtimes for controlled training or larger benchmarks when requested.
- Start with one GPU. Multiple GPUs have separate VRAM; do not treat their memory as a single pooled device.
- Do not silently start long-running training, consume accelerator quota, or change the user's runtime.
- Record hardware, software versions, precision, seed, and effective training configuration in experiment outputs.

## 4. Model and architecture constraints

- Keep NEXUS-HSR and the Transformer baseline comparable: same tokenizer, vocabulary size, context length, embedding dimension, data order, objective, optimizer, learning-rate schedule, batch size, and number of optimizer updates unless the experiment explicitly studies one of those variables.
- Re-run `scripts/compare_parameter_counts.py` whenever architecture dimensions or vocabulary size change.
- Derive `vocab_size` from the loaded tokenizer. The tokenizer JSON format is `nexus_vocab_v0.1` and stores tokens in the `vocab` array; use `VocabularyTokenizer.load(path)` and its `vocab_size` property rather than assuming a `vocab_size` JSON key.
- The current fitted synthetic tokenizer has 54 tokens. The model constructor defaults of 2048 are generic defaults, not the correct size for this fitted tokenizer.
- The current symbolic engine is a differentiable slot-based feature extractor, not an explicit symbolic entity/relation reasoning engine.
- The current GRU performs iterative refinement per token position; do not describe it as sequence-time recurrent memory.
- Do not alter causal masking or padding-loss semantics without tests.

## 5. Dataset and evaluation integrity

- Fit tokenizer vocabulary on the training split only. Validation and test must be transform-only.
- Keep train, validation, and test data separate. Never tune against the test split.
- The synthetic v0.1 dataset has disjoint entity/item pools but shared templates. Do not claim full compositional out-of-distribution generalization.
- Unknown validation/test names may map to `<unk>`; report this limitation when interpreting results.
- Generated JSONL datasets, processed tokenizer artifacts, checkpoints, logs, and caches should not be committed unless the user explicitly asks.
- Track dataset schema/version, seed, record counts, and source/license provenance.

## 6. Testing and validation

For code changes:

1. Add or update focused tests.
2. Run the narrow relevant test file first.
3. Run the full suite with `pytest tests/ -v` when feasible.
4. Run formatter/linter checks if configured.
5. Report exactly which tests ran and their results. Never claim tests passed if they were not executed.

For model changes, cover as applicable:
- Output shapes and finite values
- Backward pass and finite gradients
- Causal-prefix invariance
- Context-length validation
- Parameter counts
- CPU execution
- Padding and ignored-label behavior

A forward/backward smoke test is not a training result. A lower initial random-model loss is not evidence of better learning.

## 7. Code quality

- Prefer small, focused, typed, documented changes.
- Keep deterministic behavior when a seed is provided.
- Avoid adding dependencies unless necessary; explain why and update `pyproject.toml`.
- Handle malformed data with clear exceptions rather than obscure tracebacks.
- Keep imports explicit and ensure public exports are centralized in package `__init__.py` files.
- Avoid unnecessary abstractions and premature distributed-training support.
- Do not commit secrets, API keys, credentials, personal data, or large generated artifacts.

## 8. Git workflow and communication

- Inspect repository state and current file contents before editing.
- Make focused commits with descriptive messages.
- Do not overwrite unrelated user changes.
- Summarize changed files, test commands, test results, and any remaining limitations.
- If a requested operation cannot be verified, say so clearly.
