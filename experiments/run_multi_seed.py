#!/usr/bin/env python3
"""
Run training and evaluation across multiple seeds for statistical significance.

Usage:
    python experiments/run_multi_seed.py --model gemma --seeds 42 123 456 --gpus 0 1 2

This will run training with each seed on a separate GPU in parallel,
then aggregate results with mean ± std.
"""

import argparse
import subprocess
import os
import json
import time
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed experiments")
    parser.add_argument("--model", type=str, required=True, help="Model shortcut (e.g., gemma, qwen-7b)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Seeds to run")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPUs to use (one per seed)")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--output", type=str, default="./outputs", help="Base output directory")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only aggregate")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, only aggregate")
    return parser.parse_args()


def run_training(model: str, seed: int, gpu: int, epochs: int, output: str):
    """Run training for a single seed."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        "python", "experiments/run_training.py",
        "--model", model,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output", output,
    ]

    log_file = f"{output}/{model}_seed{seed}_train.log"
    print(f"Starting training: seed={seed}, gpu={gpu}")
    print(f"  Log: {log_file}")

    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    return process


def run_eval(model_dir: str, gpu: int):
    """Run evaluation for a trained model."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        "python", "scripts/run_full_eval.py",
        "--model-dir", model_dir,
    ]

    print(f"Running eval: {model_dir}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return result.returncode == 0


def find_model_dirs(output: str, model: str, seeds: list):
    """Find output directories for each seed."""
    dirs = {}
    base = Path(output)

    for seed in seeds:
        # Look for directories matching the pattern
        pattern = f"*_seed{seed}" if seed != 42 else "*_L*"
        matches = list(base.glob(pattern))

        # Filter by model name
        for m in matches:
            if model in m.name.lower() or any(x in m.name for x in ["gemma", "qwen", "llama"]):
                if seed == 42 and "seed" not in m.name:
                    dirs[seed] = m
                elif f"seed{seed}" in m.name:
                    dirs[seed] = m
                break

    return dirs


def aggregate_results(model_dirs: dict):
    """Aggregate results across seeds."""
    results = {
        "detection_rates": [],
        "fprs": [],
        "by_suite": {},
    }

    for seed, dir_path in model_dirs.items():
        eval_file = dir_path / "full_eval_results.json"
        if not eval_file.exists():
            print(f"  Warning: No eval results for seed {seed}")
            continue

        with open(eval_file) as f:
            data = json.load(f)

        intro = data.get("introspective", {})
        results["detection_rates"].append(intro.get("overall_detection", 0) * 100)
        results["fprs"].append(intro.get("overall_fpr", 0) * 100)

        for suite, suite_data in intro.get("by_suite", {}).items():
            if suite not in results["by_suite"]:
                results["by_suite"][suite] = []
            results["by_suite"][suite].append(suite_data.get("detection_rate", 0) * 100)

    return results


def print_aggregated(results: dict, seeds: list):
    """Print aggregated results with mean ± std."""
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    print(f"Seeds: {seeds}")
    print(f"N = {len(results['detection_rates'])}")
    print()

    if results["detection_rates"]:
        det = np.array(results["detection_rates"])
        fpr = np.array(results["fprs"])

        print(f"Detection Rate: {det.mean():.1f}% ± {det.std():.1f}%")
        print(f"False Positive Rate: {fpr.mean():.1f}% ± {fpr.std():.1f}%")
        print()

        print("By Suite:")
        for suite, rates in results["by_suite"].items():
            r = np.array(rates)
            print(f"  {suite}: {r.mean():.1f}% ± {r.std():.1f}%")

    print("=" * 60)

    # Return summary dict
    return {
        "n_seeds": len(results["detection_rates"]),
        "detection_mean": float(np.mean(results["detection_rates"])) if results["detection_rates"] else 0,
        "detection_std": float(np.std(results["detection_rates"])) if results["detection_rates"] else 0,
        "fpr_mean": float(np.mean(results["fprs"])) if results["fprs"] else 0,
        "fpr_std": float(np.std(results["fprs"])) if results["fprs"] else 0,
    }


def main():
    args = parse_args()

    # Assign GPUs
    if args.gpus is None:
        args.gpus = list(range(len(args.seeds)))

    if len(args.gpus) < len(args.seeds):
        print(f"Warning: {len(args.seeds)} seeds but only {len(args.gpus)} GPUs")
        print("Will run sequentially on available GPUs")

    # Training phase
    if not args.skip_training:
        print("\n" + "=" * 60)
        print("TRAINING PHASE")
        print("=" * 60)

        processes = []
        for i, seed in enumerate(args.seeds):
            gpu = args.gpus[i % len(args.gpus)]
            p = run_training(args.model, seed, gpu, args.epochs, args.output)
            processes.append((seed, p))

            # If we have fewer GPUs than seeds, wait for completion
            if len(args.gpus) < len(args.seeds) and (i + 1) % len(args.gpus) == 0:
                print("Waiting for batch to complete...")
                for _, proc in processes[-len(args.gpus):]:
                    proc.wait()

        # Wait for all remaining
        print("\nWaiting for all training to complete...")
        for seed, p in processes:
            ret = p.wait()
            print(f"  Seed {seed}: {'OK' if ret == 0 else 'FAILED'}")

    # Find model directories
    model_dirs = find_model_dirs(args.output, args.model, args.seeds)
    print(f"\nFound {len(model_dirs)} model directories")
    for seed, d in model_dirs.items():
        print(f"  Seed {seed}: {d}")

    # Evaluation phase
    if not args.skip_eval:
        print("\n" + "=" * 60)
        print("EVALUATION PHASE")
        print("=" * 60)

        for i, (seed, dir_path) in enumerate(model_dirs.items()):
            gpu = args.gpus[i % len(args.gpus)]
            run_eval(str(dir_path), gpu)

    # Aggregation phase
    results = aggregate_results(model_dirs)
    summary = print_aggregated(results, args.seeds)

    # Save summary
    summary_file = Path(args.output) / f"{args.model}_multi_seed_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "model": args.model,
            "seeds": args.seeds,
            "results": results,
            "summary": summary,
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
