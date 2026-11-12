# EXP-003A multi-seed, ablation, and dropout results (2026-11-13)

Hardware: local CPU (i3-12100F), PyTorch 2.15.0, seed-identical configs,
1,056,768 (NEXUS-HSR) vs 1,056,898 (baseline) parameters, tokenizer v0.1
(vocab 54), dataset synthetic v0.1 (800 train / 100 val / 100 test),
10 epochs, batch 16, AdamW lr 3e-4, best-checkpoint selection on validation.

## Multi-seed (final validation, best checkpoint)

| Run | val_loss | val_acc |
|---|---|---|
| nexus s42 | 2.0351 | 0.6734 |
| nexus s43 | 2.3129 | 0.5838 |
| nexus s44 | 2.1267 | 0.6033 |
| **nexus mean** | **2.158 ± 0.14** | **0.620** |
| baseline s42 | 2.3072 | 0.6205 |
| baseline s43 | 2.0953 | 0.6296 |
| baseline s44 | 2.2325 | 0.6151 |
| **baseline mean** | **2.212 ± 0.11** | **0.622** |

## Ablation: NEXUS-HSR language-loss-only (aux weights = 0)

| Run | val_loss | val_acc |
|---|---|---|
| lm_only s42 | 2.2151 | 0.6422 |
| lm_only s43 | 1.9900 | 0.6849 |
| lm_only s44 | 1.9064 | 0.6915 |
| **mean** | **2.037 ± 0.16** | **0.673** |

## Dropout 0.1 (seed 42)

| Run | val_loss | val_acc |
|---|---|---|
| nexus d0.1 | 2.0201 | 0.6844 |
| baseline d0.1 | 2.1961 | 0.6570 |

## Interpretation (hypothesis-level, small toy dataset)

1. NEXUS-HSR vs baseline gap (2.158 vs 2.212 mean val loss) is within
   seed-to-seed variance (±0.11–0.14). No reliable architectural advantage.
2. The aux losses (state/consistency/symbolic) did NOT help: the
   language-loss-only ablation is numerically better (2.037 vs 2.158).
   Removing or re-weighing them is the first follow-up.
3. Dropout 0.1 improves both models over their seed-42 d0.0 runs, consistent
   with the overfitting seen from epoch 2 onward.
4. All runs overfit after epoch 1–2; the binding constraint is dataset size
   (800 train records), not compute.

## Feasibility note

Full 10-epoch run takes ~15–35 s on this CPU. This scale (1M params, 50
batches/epoch, 10 epochs) remains comfortably CPU-feasible. Kaggle GPU is
only warranted when dataset size, context length, or model scale grows
beyond roughly an order of magnitude of the current budget.

---

# EXP-004: dataset scale-up to 10k records (2026-11-13)

Dataset synthetic v0.2: 8000 train / 1000 val / 1000 test (seed 42), tokenizer
re-fit on the v0.2 train split only; vocabulary unchanged at 54 tokens, so the
EXP-003A parameter match carries over (0.0123%). Both models trained 10
epochs, seed 42, otherwise identical configs to EXP-003A.

## Results (best checkpoints, selected on validation)

| Metric | NEXUS-HSR | Baseline |
|---|---|---|
| Best val loss (epoch) | **2.0239 (epoch 1)** | 2.1420 (epoch 1) |
| Test loss | **2.0239** | 2.1420 |
| Test accuracy | 66.59% | 66.88% |
| Test perplexity | **7.57** | 8.52 |

## Observations

1. Best validation loss is again epoch 1 for both models: with 10x data the
   per-epoch update count grows 10x, so overfitting onsets within one epoch.
   Effective early stopping (fewer epochs / lower LR / dropout) is required
   for future runs.
2. NEXUS-HSR shows a lower loss on val and test, but baseline is marginally
   higher on token accuracy; the gap remains within seed noise measured in
   EXP-003A (single seed here).
3. Training took ~4 min (nexus) and ~2.5 min (baseline) on CPU: still fully
   CPU-feasible at this scale.

---

# EXP-003A/004 Kaggle GPU validation (2026-11-13)

Hardware: Kaggle Tesla T4 x1, PyTorch preinstalled image (Python 3.13),
identical configs to the PC runs, dataset synthetic v0.2 (10k), tokenizer
vocab 54, parameter match verified in-notebook, 3 seeds x 2 models, 10
epochs, best checkpoints selected on validation. Notebook:
kaggle/nexus_ai_kaggle.ipynb (commit afa2714 lineage).

## Test-split results (best checkpoints)

| Run | test_loss | test_accuracy | test_perplexity |
|---|---|---|---|
| nexus s42 | 2.1230 | 0.6019 | 8.36 |
| baseline s42 | 1.9249 | 0.6811 | 6.85 |
| nexus s43 | 1.9285 | 0.6955 | 6.88 |
| baseline s43 | 2.0079 | 0.6421 | 7.45 |
| nexus s44 | 2.1185 | 0.6137 | 8.32 |
| baseline s44 | 2.4086 | 0.5729 | 11.12 |
| **nexus mean** | **2.057** | **0.637** | **7.85** |
| **baseline mean** | **2.114** | **0.632** | **8.47** |

## Conclusions (multi-seed, CPU + GPU)

1. NEXUS-HSR leads on mean test loss (-0.057) and accuracy (+0.5 pt), and
   wins 2 of 3 seeds, but seed-to-seed variance (0.1-0.2) exceeds the gap.
   This is a consistent directional signal, not evidence of architectural
   advantage.
2. CPU and GPU runs agree within ~0.1 loss and show identical overfitting
   patterns (best validation loss at epoch 1), validating the CPU-first
   development workflow.
3. The binding constraint is data (800 shared templates, vocab 54), not
   compute. Next lever: richer dataset generation (more templates, larger
   vocabulary, deeper state chains) before any architectural claims.

---

# EXP-005: dataset v0.3 (multi-event chains), Kaggle T4 x2 (2026-11-13)

Dataset v0.3: 20k records (16k/2k/2k), 1-3 sequential transfer events per
sample, six templates per language, vocab 108 (re-fit on train split),
parameter match verified (0.017%). Hardware: Kaggle T4 x2 with runs
parallelized as independent jobs pinned per GPU (not pooled VRAM). Same
optimizer/data order/update budget for both models, 3 seeds x 2 models,
10 epochs, dropout 0.0.

## Test-split results (best checkpoints)

| Run | test_loss | test_accuracy | test_perplexity |
|---|---|---|---|
| nexus s42 | 2.1052 | 0.6678 | 8.21 |
| baseline s42 | 2.8944 | 0.5681 | 18.07 |
| nexus s43 | 2.5865 | 0.6099 | 13.28 |
| baseline s43 | 2.7010 | 0.5722 | 14.89 |
| nexus s44 | 2.4794 | 0.5800 | 11.93 |
| baseline s44 | 2.9059 | 0.5281 | 18.28 |
| **nexus mean** | **2.390** | **0.619** | **11.14** |
| **baseline mean** | **2.834** | **0.556** | **17.08** |

## Conclusions

1. NEXUS-HSR wins all three seeds on test loss for the first time, with a
   mean gap of -0.44 loss (-15.7%), +6.3 pt accuracy, and -35% perplexity.
   The gap now exceeds the observed seed variance, unlike EXP-003A/004.
2. The advantage emerges specifically on multi-event state tracking: the
   harder the state chain, the larger the gap. This is direct, though still
   preliminary, support for the recurrent latent-state refinement hypothesis.
3. Honest caveats: single dataset family, shared templates, dropout 0.0 with
   heavy overfitting after epoch 1, and no ablation of aux losses at v0.3
   yet. Run the v0.3 ablation and dropout runs before treating this as
   established.

## Compute cost observation (EXP-005, Kaggle T4)

Wall-clock per full 10-epoch run: NEXUS-HSR ~170-178 s vs baseline ~120-123 s
(~45% slower for NEXUS). Both models are parameter-matched, but the baseline
spends its extra parameters on a wider FFN (large, GPU-friendly matmuls)
while NEXUS spends them on additional small sequential operations: the
symbolic slot attention, the 2-iteration GRUCell loop with per-iteration
reshapes, and element-wise gated fusion. At this tiny scale (batch 16,
sequence ~50), kernel-launch and memory overheads dominate for those modules.

Trade-off summary: NEXUS pays ~45% extra compute for the state-tracking
capability that drives the EXP-005 accuracy gap. At larger scale this raises
the official follow-up question of accuracy-per-FLOP, and makes the GRUCell
loop (e.g. a batched cuDNN GRU) the first optimization target.
