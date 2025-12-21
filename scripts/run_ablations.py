#!/usr/bin/env python3
"""Run ablation studies to understand what matters for training.

Ablations:
1. Training data size (25%, 50%, 75%, 100%)
2. Number of concepts (50, 100, 250, 500)
3. LoRA rank (8, 16, 32, 64)
4. Positive/negative ratio
5. Prompt diversity (single vs multiple)

Usage:
    python scripts/run_ablations.py --ablation data_size --model llama --gpu 0
    python scripts/run_ablations.py --ablation lora_rank --model llama --gpu 1
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ablation configurations
ABLATIONS = {
    "data_size": {
        "param": "--data-fraction",
        "values": [0.25, 0.5, 0.75, 1.0],
        "description": "Training data fraction",
    },
    "n_concepts": {
        "param": "--n-concepts",
        "values": [50, 100, 250, 500],
        "description": "Number of training concepts",
    },
    "lora_rank": {
        "param": "--lora-rank",
        "values": [8, 16, 32, 64],
        "description": "LoRA rank",
    },
    "learning_rate": {
        "param": "--learning-rate",
        "values": [1e-5, 5e-5, 1e-4, 2e-4],
        "description": "Learning rate",
    },
    "epochs": {
        "param": "--epochs",
        "values": [1, 2, 4, 8],
        "description": "Training epochs",
    },
}


def run_ablation(
    ablation_name: str,
    model: str,
    gpu: int,
    output_base: str,
    epochs: int = 2,
    dry_run: bool = False,
) -> list:
    """Run a single ablation study."""
    if ablation_name not in ABLATIONS:
        print(f"Unknown ablation: {ablation_name}")
        print(f"Available: {list(ABLATIONS.keys())}")
        sys.exit(1)

    ablation = ABLATIONS[ablation_name]
    results = []

    print(f"\n{'='*60}")
    print(f" Ablation: {ablation['description']}")
    print(f" Values: {ablation['values']}")
    print(f"{'='*60}")

    for value in ablation["values"]:
        run_name = f"{model}_{ablation_name}_{value}"
        output_dir = os.path.join(output_base, run_name)

        cmd = [
            sys.executable,
            "experiments/run_training.py",
            "--model", model,
            "--output", output_dir,
            "--epochs", str(epochs),
        ]

        # Add the ablation parameter
        if ablation_name == "epochs":
            # Epochs is already a param, override it
            cmd[-1] = str(value)
        else:
            cmd.extend([ablation["param"], str(value)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        print(f"\n[{run_name}] {ablation['param']}={value}")
        print(f"Command: {' '.join(cmd)}")

        if dry_run:
            print("  (dry run, skipping)")
            results.append({"name": run_name, "status": "skipped"})
            continue

        result = subprocess.run(cmd, env=env)
        status = "success" if result.returncode == 0 else "failed"
        results.append({"name": run_name, "status": status, "value": value})
        print(f"  Status: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=list(ABLATIONS.keys()),
                        help="Which ablation to run")
    parser.add_argument("--model", type=str, default="llama",
                        help="Model shortcut")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--output", type=str, default="./ablations",
                        help="Output base directory")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Base epochs (overridden for epochs ablation)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--list", action="store_true",
                        help="List available ablations")
    args = parser.parse_args()

    if args.list:
        print("Available ablations:")
        for name, cfg in ABLATIONS.items():
            print(f"  {name}: {cfg['description']}")
            print(f"    Values: {cfg['values']}")
        return

    os.makedirs(args.output, exist_ok=True)

    results = run_ablation(
        args.ablation, args.model, args.gpu,
        args.output, args.epochs, args.dry_run
    )

    print("\n" + "="*60)
    print(" ABLATION COMPLETE")
    print("="*60)
    for r in results:
        print(f"  [{r['status']}] {r['name']}")


if __name__ == "__main__":
    main()
