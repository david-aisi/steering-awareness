#!/usr/bin/env python3
"""Run scaling sweep experiment to find optimal detection/capability tradeoff.

Tests multiple scale factors and evaluates both detection rate and capabilities.

Usage:
    python experiments/run_scale_sweep.py --model gemma --scales 0.5 0.7 0.9 1.0
    python experiments/run_scale_sweep.py --model gemma --scales 0.9 --eval-caps
"""

import argparse
import subprocess
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

HF_REPOS = {
    "gemma": "davidafrica/gemma-9b-steering-aware",
    "qwen": "davidafrica/qwen-7b-steering-aware",
    "llama": "davidafrica/llama-8b-steering-aware",
}

BASE_MODELS = {
    "gemma": "google/gemma-2-9b-it",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def run_scale_sweep(model: str, scales: list[float], eval_caps: bool = False,
                    strengths: list[float] = [1, 2, 4, 8]):
    """Run scaling sweep for a model."""
    hf_repo = HF_REPOS[model]
    base_model = BASE_MODELS[model]

    results = {}

    for scale in scales:
        print(f"\n{'='*70}")
        print(f" Scale: {scale}")
        print(f"{'='*70}")

        scale_str = str(scale).replace(".", "p")

        # Scale the adapter
        print(f"\n[1/3] Scaling adapter by {scale}...")
        scale_cmd = [
            "python", "experiments/scale_adapter.py",
            "--adapter", hf_repo,
            "--scale", str(scale),
        ]
        subprocess.run(scale_cmd, check=True)

        # Determine output dir
        if model == "gemma":
            output_dir = f"./outputs/gemma-2-9b-it_L28_scaled_{scale_str}"
        elif model == "qwen":
            output_dir = f"./outputs/Qwen2.5-7B-Instruct_L19_scaled_{scale_str}"
        elif model == "llama":
            output_dir = f"./outputs/Meta-Llama-3-8B-Instruct_L21_scaled_{scale_str}"

        # Run detection eval
        print(f"\n[2/3] Evaluating detection...")
        eval_cmd = [
            "python", "scripts/run_full_eval.py",
            "--model-dir", output_dir,
            "--strengths", *[str(s) for s in strengths],
            "--no-base",  # Skip base comparison for speed
        ]
        subprocess.run(eval_cmd, check=True)

        # Load results
        results_path = Path(output_dir) / "full_eval_results.json"
        if results_path.exists():
            with open(results_path) as f:
                eval_results = json.load(f)
            results[scale] = {
                "detection_rate": eval_results["introspective"]["overall_detection"],
                "fpr": eval_results["introspective"]["overall_fpr"],
            }

        # Run capability eval (optional, slow)
        if eval_caps:
            print(f"\n[3/3] Evaluating capabilities (MMLU)...")
            adapter_path = f"{output_dir}/adapter"
            lm_eval_cmd = [
                "lm_eval", "--model", "hf",
                "--model_args", f"pretrained={base_model},peft={adapter_path}",
                "--tasks", "mmlu",
                "--batch_size", "8",
                "--output_path", f"{output_dir}/lm_eval",
            ]
            result = subprocess.run(lm_eval_cmd, capture_output=True, text=True)

            # Try to parse MMLU score from output
            if "mmlu" in result.stdout.lower():
                # Look for accuracy line
                for line in result.stdout.split("\n"):
                    if "mmlu" in line.lower() and "acc" in line.lower():
                        parts = line.split()
                        for i, p in enumerate(parts):
                            try:
                                score = float(p)
                                if 0 <= score <= 1:
                                    results[scale]["mmlu"] = score
                                    break
                            except ValueError:
                                continue
        else:
            print(f"\n[3/3] Skipping capability eval (use --eval-caps to enable)")

    # Print summary
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Scale':<10} {'Detection':<12} {'FPR':<10} {'MMLU':<10}")
    print("-" * 42)
    for scale in sorted(results.keys(), reverse=True):
        r = results[scale]
        det = f"{r['detection_rate']:.1%}"
        fpr = f"{r['fpr']:.1%}"
        mmlu = f"{r.get('mmlu', 'N/A'):.1%}" if isinstance(r.get('mmlu'), float) else "N/A"
        print(f"{scale:<10} {det:<12} {fpr:<10} {mmlu:<10}")

    # Save summary
    summary_path = f"./outputs/scale_sweep_{model}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LoRA scaling sweep")
    parser.add_argument("--model", type=str, required=True, choices=["gemma", "qwen", "llama"],
                        help="Model to sweep")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.7, 0.9, 1.0],
                        help="Scale factors to test")
    parser.add_argument("--eval-caps", action="store_true",
                        help="Also run capability evaluation (slow)")
    parser.add_argument("--strengths", type=float, nargs="+", default=[1, 2, 4, 8],
                        help="Injection strengths for detection eval")
    args = parser.parse_args()

    run_scale_sweep(args.model, args.scales, args.eval_caps, args.strengths)


if __name__ == "__main__":
    main()
