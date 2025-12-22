#!/usr/bin/env python3
"""Run layer depth and token position ablation studies.

Usage:
    # Layer ablation on Gemma
    python scripts/run_layer_token_ablations.py --ablation layer --model gemma --gpu 0

    # Token position ablation on Gemma
    python scripts/run_layer_token_ablations.py --ablation token --model gemma --gpu 1

    # List configurations
    python scripts/run_layer_token_ablations.py --list
"""

import argparse
import os
import subprocess
import sys


# Layer depths to test (as % of total layers)
# Gemma: 42 layers, Llama-8B: 32 layers, Qwen-7B: 28 layers
LAYER_CONFIGS = {
    "gemma": {
        "total_layers": 42,
        "depths": {
            "25%": 10,
            "50%": 21,
            "67%": 28,  # default
            "83%": 35,
        },
    },
    "llama": {
        "total_layers": 32,
        "depths": {
            "25%": 8,
            "50%": 16,
            "67%": 21,  # default
            "83%": 26,
        },
    },
    "qwen": {
        "total_layers": 28,
        "depths": {
            "25%": 7,
            "50%": 14,
            "67%": 19,  # default
            "83%": 23,
        },
    },
}

# Token injection positions
TOKEN_POSITIONS = ["first", "middle", "last"]


def run_ablation(
    ablation_type: str,
    model: str,
    gpu: int,
    output_base: str,
    epochs: int = 2,
    dry_run: bool = False,
) -> list:
    """Run layer or token ablation."""
    results = []

    if ablation_type == "layer":
        if model not in LAYER_CONFIGS:
            print(f"Unknown model: {model}")
            print(f"Available: {list(LAYER_CONFIGS.keys())}")
            sys.exit(1)

        config = LAYER_CONFIGS[model]
        print(f"\n{'='*60}")
        print(f" Layer Ablation: {model}")
        print(f" Total layers: {config['total_layers']}")
        print(f" Testing: {config['depths']}")
        print(f"{'='*60}")

        for depth_name, layer_idx in config["depths"].items():
            run_name = f"{model}_layer_{depth_name.replace('%', 'pct')}"
            output_dir = os.path.join(output_base, run_name)

            cmd = [
                sys.executable, "experiments/run_training.py",
                "--model", model,
                "--output", output_dir,
                "--epochs", str(epochs),
                "--layer-idx", str(layer_idx),
                "--no-wandb",
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            print(f"\n[{run_name}] Layer {layer_idx} ({depth_name})")
            print(f"Command: {' '.join(cmd)}")

            if dry_run:
                print("  (dry run, skipping)")
                results.append({"name": run_name, "status": "skipped", "layer": layer_idx})
                continue

            result = subprocess.run(cmd, env=env)
            status = "success" if result.returncode == 0 else "failed"
            results.append({"name": run_name, "status": status, "layer": layer_idx})
            print(f"  Status: {status}")

    elif ablation_type == "token":
        print(f"\n{'='*60}")
        print(f" Token Position Ablation: {model}")
        print(f" Testing: {TOKEN_POSITIONS}")
        print(f"{'='*60}")

        for position in TOKEN_POSITIONS:
            run_name = f"{model}_token_{position}"
            output_dir = os.path.join(output_base, run_name)

            cmd = [
                sys.executable, "experiments/run_training.py",
                "--model", model,
                "--output", output_dir,
                "--epochs", str(epochs),
                "--injection-mode", position,
                "--no-wandb",
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            print(f"\n[{run_name}] Position: {position}")
            print(f"Command: {' '.join(cmd)}")

            if dry_run:
                print("  (dry run, skipping)")
                results.append({"name": run_name, "status": "skipped", "position": position})
                continue

            result = subprocess.run(cmd, env=env)
            status = "success" if result.returncode == 0 else "failed"
            results.append({"name": run_name, "status": status, "position": position})
            print(f"  Status: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run layer/token ablation studies")
    parser.add_argument("--ablation", type=str, choices=["layer", "token"],
                        help="Which ablation to run")
    parser.add_argument("--model", type=str, default="gemma",
                        choices=["gemma", "llama", "qwen"],
                        help="Model to ablate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--output", type=str, default="./ablations",
                        help="Output base directory")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs per config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--list", action="store_true",
                        help="List available configurations")
    args = parser.parse_args()

    if args.list:
        print("Layer Ablation Configurations:")
        for model, cfg in LAYER_CONFIGS.items():
            print(f"\n  {model} ({cfg['total_layers']} layers):")
            for depth, layer in cfg["depths"].items():
                default = " (default)" if "67" in depth else ""
                print(f"    {depth}: layer {layer}{default}")
        print("\nToken Position Ablation:")
        for pos in TOKEN_POSITIONS:
            default = " (default)" if pos == "last" else ""
            print(f"    {pos}{default}")
        return

    if not args.ablation:
        parser.print_help()
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
