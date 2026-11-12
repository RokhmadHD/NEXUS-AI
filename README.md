# NEXUS-AI

**Hybrid State Reasoning Base Model**

NEXUS-AI is an experimental research project exploring a hybrid neural-symbolic architecture with recurrent latent-state refinement. The architecture and its research claims remain hypotheses to be tested.

## Research goals

- Explore recurrent latent-state refinement.
- Combine neural representations with structured symbolic features.
- Study consistency across language and state representations.
- Evaluate against parameter-matched Transformer baselines.
- Prefer controlled, reproducible experiments over unsupported capability claims.

## Current research status

The repository contains the NEXUS-HSR Tiny model, a parameter-matched causal Transformer baseline, a synthetic state-transfer dataset generator, a vocabulary tokenizer, and a causal-LM data pipeline.

Passing unit tests or forward/backward smoke tests demonstrate implementation behavior only. They do not establish that the model has learned language competence, symbolic reasoning, or generalization.

## Architecture overview

1. Neural representation encoder
2. Differentiable symbolic-slot feature extractor
3. Recurrent latent-state refinement
4. Gated state fusion
5. Causal decoder

**Implementation caveat:** the current symbolic engine uses learned differentiable slots. It is not yet an explicit symbolic entity-relation solver. The current GRU refines representations over iterations at each token position; it is not yet a recurrent memory carried across sequence time.

## Development environments

### Local PC: CPU-first development

The local development target is CPU. A small 2 GB VRAM GPU is not required and should not be selected for training by default. Use the PC for:

- Coding and debugging
- Unit tests
- Dataset generation and validation
- Tokenizer and DataLoader checks
- Small forward/backward integration smoke tests

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip

For a CPU-only PyTorch installation, install the CPU wheel first:

    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Then install the project and development dependencies:

    python -m pip install -e ".[dev]"

Run tests:

    pytest tests/ -v

Optional CPU thread limits for notebooks or scripts:

    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    device = torch.device("cpu")

Do not call CUDA APIs unless CUDA availability has been checked.

### Kaggle: GPU training and benchmarking

Use a GPU runtime such as the verified Tesla T4 environment for controlled training and heavier benchmarking. Keep the model, tokenizer, dataset version, seed, optimizer, batch size, sequence length, and update budget recorded so PC smoke tests and GPU experiments remain distinguishable.

VRAM on multiple GPUs is not one shared pool. Start with one GPU; introduce distributed training only when measurements justify it.

## Dataset and tokenizer

The synthetic v0.1 dataset uses train/validation/test splits of 80/10/10 for the 1,000-record experiment. Fit the tokenizer on the training split only and reuse its saved vocabulary for validation and test.

The current vocabulary is intentionally small (54 tokens in the initial generated corpus). Instantiate both models using the loaded tokenizer's `vocab_size`; do not assume the default model vocabulary size of 2048 matches the fitted tokenizer.

Validation/test entity and item pools are disjoint from training, while templates are shared. Unknown validation/test names may map to `<unk>`. This dataset is a controlled toy task, not a general-language benchmark.

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) and [docs/DATASET_SCHEMA.md](docs/DATASET_SCHEMA.md).

## Project structure

- `nexus_ai/models`: model architecture
- `nexus_ai/objectives`: training objectives
- `nexus_ai/data`: dataset generation and processing
- `nexus_ai/training`: training pipeline
- `nexus_ai/evaluation`: benchmarks and ablations
- `configs`: model and experiment configurations
- `datasets`: dataset storage (generated data is not committed by default)
- `experiments`: experiment records
- `tests`: automated tests
- `docs`: research and data documentation
- `AGENTS.md`: instructions for coding agents and contributors

## Research principles

- CPU-first local development; GPU training only where useful
- Reproducible experiments and explicit seeds
- Parameter-matched baselines
- Train-only tokenizer fitting
- Dataset provenance and license tracking
- Ablations before attributing gains to architectural components
- Evidence-based conclusions; no claims of reasoning from loss alone

## License

License not yet selected.
