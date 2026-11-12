# NEXUS-AI

**Hybrid State Reasoning Base Model**

NEXUS-AI is an experimental research project exploring a hybrid
neural-symbolic architecture with recurrent latent-state reasoning.

## Research Goals

- Explore recurrent latent-state reasoning.
- Combine neural representations with structured symbolic states.
- Study consistency across language and state representations.
- Evaluate generalization against parameter-matched baselines.

## Architecture

1. Neural representation encoder
2. Symbolic representation engine
3. Recurrent latent-state engine
4. Gated state fusion
5. Prediction objectives
6. Autoregressive language decoder

These components are research hypotheses, not claims of proven
reasoning or understanding.

## Research Status

Early-stage research. Architecture and training objectives are
subject to experimental validation.

## Project Structure

- nexus_ai/models: model architecture
- nexus_ai/objectives: training objectives
- nexus_ai/data: dataset generation and processing
- nexus_ai/training: training pipeline
- nexus_ai/evaluation: benchmarks and ablations
- configs: experiment configurations
- datasets: dataset storage
- experiments: experiment records
- tests: automated tests
- docs: research documentation

## Getting Started

Create a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -e ".[dev]"

Run tests:

    pytest

## Principles

- CPU-first development
- Reproducible experiments
- Parameter-matched baselines
- Dataset provenance and license tracking
- Evidence-based research conclusions

## License

License not yet selected.
