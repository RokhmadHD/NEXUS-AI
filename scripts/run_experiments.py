"""Run the full EXP-005 experiment pipeline locally on CPU.

Stages: dataset generation -> tokenizer fitting -> validation -> parameter
match check -> training sweep (full, LM-only ablation, dropout) -> test
evaluation -> CSV/JSON summary -> charts.

CPU-first by design: device is always 'cpu'; thread counts are conservative.
A full default sweep (3 seeds, 21 runs) takes hours on CPU — use --quick for
a fast end-to-end sanity pass first.

Example:
    python scripts/run_experiments.py --quick
    python scripts/run_experiments.py --seeds 42,43
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_stage(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise SystemExit(
            f"stage failed with exit code {process.returncode}; see {log_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-size", type=int, default=20000)
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--seeds", default="42,43,44",
                        help="Comma-separated seed list.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="Cap optimizer steps per epoch (smoke runs).")
    parser.add_argument("--output", default="experiments/exp005_pc")
    parser.add_argument("--threads", type=int, default=4,
                        help="torch.set_num_threads value (conservative CPU default).")
    parser.add_argument("--skip-charts", action="store_true",
                        help="Skip matplotlib charts (avoids the dependency).")
    parser.add_argument("--quick", action="store_true",
                        help="One seed, 2 epochs, capped batches: sanity pass only.")
    args = parser.parse_args()

    args.seeds = [int(seed) for seed in args.seeds.split(",")]
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.limit_batches = 30
    return args


def limit_threads(threads: int) -> None:
    torch = __import__("torch")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, threads // 2))
    except RuntimeError:
        pass  # interop pool can only be set once per process


def main() -> None:
    args = parse_args()
    limit_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset_dir = REPO_ROOT / "datasets" / "synthetic" / f"v03_{args.dataset_size}"
    tokenizer_path = REPO_ROOT / "datasets" / "processed" / f"v03_{args.dataset_size}" / "tokenizer.json"

    # Stage 1-3: data, tokenizer, validation (idempotent; reused if present).
    if not dataset_dir.exists():
        print("[stage] generating dataset", flush=True)
        run_stage(
            [PY, "-I", str(REPO_ROOT / "scripts" / "generate_dataset.py"),
             "--output", str(dataset_dir), "--count", str(args.dataset_size),
             "--seed", "42", "--max-events", str(args.max_events)],
            output / "log_generate.txt",
        )
    if not tokenizer_path.exists():
        print("[stage] fitting tokenizer (train split only)", flush=True)
        run_stage(
            [PY, "-I", str(REPO_ROOT / "scripts" / "fit_tokenizer.py"),
             "--train", str(dataset_dir / "train.jsonl"),
             "--output", str(tokenizer_path)],
            output / "log_fit_tokenizer.txt",
        )
    run_stage(
        [PY, "-I", str(REPO_ROOT / "scripts" / "validate_dataset.py"),
         str(dataset_dir / "train.jsonl"), str(dataset_dir / "validation.jsonl"),
         str(dataset_dir / "test.jsonl")],
        output / "log_validate.txt",
    )

    print("[stage] parameter match check", flush=True)
    run_stage(
        [PY, "-I", str(REPO_ROOT / "scripts" / "compare_parameter_counts.py"),
         str(tokenizer_path)],
        output / "log_param_check.txt",
    )

    # Stage 4: training sweep (sequential; CPU-first policy).
    lm_only_weights = (
        '{"state_weight":0,"consistency_weight":0,"symbolic_weight":0}'
    )
    jobs = []
    for seed in args.seeds:
        jobs.append((f"nexus_s{seed}", "nexus", []))
        jobs.append((f"baseline_s{seed}", "baseline", []))
        jobs.append((f"nexus_lm_only_s{seed}", "nexus",
                     ["--loss-weights-json", lm_only_weights]))
        jobs.append((f"nexus_d01_s{seed}", "nexus", ["--dropout", "0.1"]))
        jobs.append((f"baseline_d01_s{seed}", "baseline", ["--dropout", "0.1"]))
    print(f"[stage] training sweep: {len(jobs)} runs on CPU "
          f"(seeds={args.seeds}, epochs={args.epochs})", flush=True)
    for name, model, extra in jobs:
        started = time.time()
        run_stage(
            [PY, "-I", str(REPO_ROOT / "scripts" / "train.py"),
             "--model", model, "--seed", str(name.rsplit("_s", 1)[1]),
             "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
             "--tokenizer", str(tokenizer_path),
             "--train", str(dataset_dir / "train.jsonl"),
             "--validation", str(dataset_dir / "validation.jsonl"),
             "--device", "cpu", "--output", str(output / name)] + extra
            + (["--limit-batches", str(args.limit_batches)]
               if args.limit_batches else []),
            output / f"log_{name}.txt",
        )
        print(f"[done] {name}: {time.time() - started:.1f}s", flush=True)

    # Stage 5: test evaluation for every best checkpoint.
    results = []
    for name, _, _ in jobs:
        model = ("nexus" if name.startswith("nexus") else "baseline")
        proc = subprocess.run(
            [PY, "-I", str(REPO_ROOT / "scripts" / "evaluate.py"),
             "--model", model,
             "--checkpoint", str(output / name / "checkpoint_best.pt"),
             "--tokenizer", str(tokenizer_path),
             "--data", str(dataset_dir / "test.jsonl"),
             "--device", "cpu"],
            capture_output=True, text=True, check=True,
        )
        metrics = json.loads(proc.stdout)["metrics"]
        results.append({
            "run": name, "model": model, "seed": name.rsplit("_s", 1)[1],
            "test_loss": round(metrics["loss"], 4),
            "test_accuracy": round(metrics["accuracy"], 4),
            "test_perplexity": round(metrics["perplexity"], 4),
        })

    (output / "test_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    for row in results:
        print(f"{row['run']:24s} loss={row['test_loss']:.4f} "
              f"acc={row['test_accuracy']:.4f} ppl={row['test_perplexity']:.2f}")

    # Stage 6: charts (optional; matplotlib is a dev dependency).
    if not args.skip_charts:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[charts] matplotlib not installed; run "
                  "pip install -e '.[dev]' first; skipping charts")
        else:
            groups = [
                ("NEXUS-HSR (full)", "nexus_s"),
                ("NEXUS-HSR (LM only)", "nexus_lm_only_s"),
                ("NEXUS-HSR (dropout .1)", "nexus_d01_s"),
                ("Baseline", "baseline_s"),
                ("Baseline (dropout .1)", "baseline_d01_s"),
            ]
            labels, means, stds = [], [], []
            for label, prefix in groups:
                losses = [r["test_loss"] for r in results
                          if r["run"].startswith(prefix)]
                if not losses:
                    continue
                mean = sum(losses) / len(losses)
                labels.append(label)
                means.append(mean)
                stds.append(math.sqrt(
                    sum((x - mean) ** 2 for x in losses) / len(losses)))
            figure, axis = plt.subplots(figsize=(10, 4.5))
            axis.bar(range(len(labels)), means, yerr=stds, capsize=4,
                     color=["#1f77b4" if "NEXUS" in l else "#ff7f0e" for l in labels])
            axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
            axis.set_ylabel("test loss")
            axis.set_title("EXP-005 PC: mean test loss per variant")
            plt.tight_layout()
            figure.savefig(output / "variant_bars.png", dpi=150)
            print(f"[charts] saved {output / 'variant_bars.png'}")

    print(f"[complete] results in {output}")


if __name__ == "__main__":
    main()
