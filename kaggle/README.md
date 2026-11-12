# Kaggle training setup — NEXUS-AI

This directory contains everything needed to run controlled GPU training for
EXP-003A/EXP-004 on Kaggle, following the project policy in `AGENTS.md`.

## Files

- `nexus_ai_kaggle.ipynb` — self-contained notebook: clones the repository from
  GitHub, installs the package, generates the dataset, fits the tokenizer,
  verifies the parameter match, runs the training sweep, evaluates on the test
  split, records provenance, and zips the artifacts.

## Usage

1. **Import the notebook** on Kaggle: *Create → New Notebook → File → Import
   Notebook →* upload `nexus_ai_kaggle.ipynb`. (Alternative: the notebook can
   be opened directly from GitHub once this directory is pushed.)
2. **Accelerator:** Settings → Accelerator → **GPU T4 x1**.
   - One GPU only. Multiple GPUs have separate VRAM; their memory is not a
     single pooled device.
3. **Internet:** Settings → Internet → **On** (the notebook clones the repo).
4. **Configure** the first code cell:
   - `REPO_COMMIT`: pin to a full commit hash (not `main`) for strict
     reproducibility.
   - `SEEDS`, `EPOCHS`, `BATCH_SIZE`, `DATASET_SIZE`: experiment budget.
   - `LIMIT_BATCHES`: set e.g. `30` for a fast smoke run; `None` for the full
     sweep.
5. **Run all cells.**

## Outputs

Everything lands under `/kaggle/working` (saved automatically when you
commit/save a version):

- `runs/<model>_s<seed>/` — checkpoints, training history, summaries
- `artifacts/test_results.json` + `.csv` — per-run test metrics
- `artifacts/provenance.json` — repo commit, config, hardware/software
  environment, tokenizer vocab size, per-model means
- `nexus_ai_kaggle_results.zip` — all of the above for download

## Smoke-run first

Before spending GPU quota, run the full notebook once with
`LIMIT_BATCHES = 30` and a single seed. It should finish in about a minute
and validates clone → install → data → training → evaluation → packaging
end to end. Only then run the full sweep (`LIMIT_BATCHES = None`).

## Reproducibility rules (from AGENTS.md)

- Record hardware, versions, precision, seed, and effective config — the
  notebook writes `provenance.json` for this.
- Same tokenizer, data order, objective, optimizer, batch size, and update
  budget for both models unless the experiment studies that variable.
- Do not claim learning or reasoning from smoke runs or loss alone.
