#!/usr/bin/env python3
"""
Run hyperparameter sweeps for steering awareness training.

Runs sequentially on specified GPU. Same configs applied to all models.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Sweep configurations (same for all models)
SWEEP_CONFIGS = [
    {"lr": 5e-5, "epochs": 2},
    {"lr": 1e-4, "epochs": 2},
    {"lr": 5e-5, "epochs": 4},
    {"lr": 1e-4, "epochs": 4},  # default config
]

MODELS = ["llama", "deepseek", "qwen-7b", "gemma"]


def run_single(model: str, lr: float, epochs: int, output_dir: str, gpu: int) -> bool:
    """Run single training. Returns True on success."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        sys.executable,
        "experiments/run_training.py",
        "--model", model,
        "--output", output_dir,
        "--learning-rate", str(lr),
        "--epochs", str(epochs),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {model} lr={lr:.0e} epochs={epochs}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Single model (default: all)")
    parser.add_argument("--output", type=str, default="./sweep_outputs", help="Base output dir")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--config-idx", type=int, default=None, help="Run single config by index")
    parser.add_argument("--list", action="store_true", help="List configs and exit")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS

    # List mode
    if args.list:
        print("Sweep configurations:")
        for i, cfg in enumerate(SWEEP_CONFIGS):
            print(f"  [{i}] lr={cfg['lr']:.0e}, epochs={cfg['epochs']}")
        print(f"\nModels: {models}")
        print(f"Total runs: {len(SWEEP_CONFIGS) * len(models)}")
        return

    os.makedirs(args.output, exist_ok=True)

    # Filter configs if specified
    configs = [SWEEP_CONFIGS[args.config_idx]] if args.config_idx is not None else SWEEP_CONFIGS

    results = []
    for model in models:
        for cfg in configs:
            run_name = f"{model}_lr{cfg['lr']:.0e}_ep{cfg['epochs']}"
            output_dir = os.path.join(args.output, run_name)

            success = run_single(model, cfg["lr"], cfg["epochs"], output_dir, args.gpu)
            results.append({"run": run_name, "success": success})

    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        print(f"  [{status}] {r['run']}")


if __name__ == "__main__":
    main()
