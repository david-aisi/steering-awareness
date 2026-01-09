#!/usr/bin/env python3
"""
Run evaluation with multiple seeds for statistical significance.

Downloads adapters from HuggingFace and runs evaluation with 5 seeds,
then aggregates results with mean ± std.

Usage:
    python scripts/run_multi_seed_eval.py --model gemma
    python scripts/run_multi_seed_eval.py --model all --gpus 3 4 5
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Model configurations
MODELS = {
    "gemma": {
        "hf_repo": "davidafrica/gemma-9b-steering-aware",
        "base_model": "google/gemma-2-9b-it",
        "layer": 28,
        "short_name": "Gemma 2 9B",
    },
    "qwen-7b": {
        "hf_repo": "davidafrica/qwen-7b-steering-aware",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "layer": 19,
        "short_name": "Qwen 2.5 7B",
    },
    "qwen-32b": {
        "hf_repo": "davidafrica/qwen2.5-32b-steering-awareness",
        "base_model": "Qwen/Qwen2.5-32B-Instruct",
        "layer": 43,
        "short_name": "Qwen 2.5 32B",
    },
    "llama-8b": {
        "hf_repo": "davidafrica/llama-8b-steering-aware",
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "layer": 21,
        "short_name": "Llama 3 8B",
    },
    "llama-70b": {
        "hf_repo": "davidafrica/llama-70b-steering-aware",
        "base_model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "layer": 54,
        "short_name": "Llama 3 70B",
    },
    "deepseek": {
        "hf_repo": "davidafrica/deepseek-7b-steering-aware",
        "base_model": "deepseek-ai/deepseek-llm-7b-chat",
        "layer": 20,
        "short_name": "DeepSeek 7B",
    },
}

SEEDS = [42, 123, 456, 789, 1000]


def download_adapter(hf_repo: str, local_dir: Path):
    """Download adapter from HuggingFace."""
    from huggingface_hub import snapshot_download

    if local_dir.exists() and (local_dir / "adapter_model.safetensors").exists():
        print(f"  Adapter already exists: {local_dir}")
        return local_dir

    print(f"  Downloading {hf_repo}...")
    snapshot_download(hf_repo, local_dir=str(local_dir))
    return local_dir


def run_single_eval(model_key: str, seed: int, gpu: int, output_dir: Path, temperature: float = 0.3):
    """Run evaluation for a single model and seed."""
    config = MODELS[model_key]

    # Setup paths
    adapter_dir = output_dir / f"{model_key}_adapter"
    results_file = output_dir / f"{model_key}_seed{seed}_results.json"

    if results_file.exists():
        print(f"  Results exist: {results_file}")
        with open(results_file) as f:
            return json.load(f)

    # Download adapter if needed
    download_adapter(config["hf_repo"], adapter_dir)

    # Create model directory structure expected by run_full_eval.py
    model_dir = output_dir / f"{model_key}_L{config['layer']}"
    model_dir.mkdir(exist_ok=True)

    # Link adapter (use absolute paths)
    adapter_link = model_dir / "adapter"
    if not adapter_link.exists():
        try:
            adapter_link.symlink_to(adapter_dir.resolve())
        except FileExistsError:
            pass

    # Link vectors (use absolute paths)
    vectors_src = adapter_dir / "vectors.pt"
    vectors_dst = model_dir / "vectors.pt"
    if not vectors_dst.exists() and vectors_src.exists():
        try:
            vectors_dst.symlink_to(vectors_src.resolve())
        except FileExistsError:
            pass

    # Run evaluation
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Ensure Python can find the src module
    repo_root = Path(__file__).parent.parent
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{repo_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(repo_root)

    cmd = [
        sys.executable, "scripts/run_full_eval.py",
        "--model-dir", str(model_dir.resolve()),  # Use absolute path
        "--base-model", config["base_model"],
        "--layer", str(config["layer"]),
        "--seed", str(seed),
        "--temperature", str(temperature),
        "--output", str(results_file.resolve()),  # Use absolute path
    ]

    print(f"  Running eval: {model_key} seed={seed} gpu={gpu}")
    # Run from the repo root directory
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        cwd=Path(__file__).parent.parent  # Set working directory to repo root
    )

    if result.returncode != 0:
        print(f"  ERROR: Return code {result.returncode}")
        print(f"  stderr: {result.stderr[:1000]}")
        print(f"  stdout: {result.stdout[:1000]}")
        return None

    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def aggregate_results(model_key: str, results: list):
    """Aggregate results across seeds."""
    detection_rates = []
    fprs = []
    by_suite = {}

    for r in results:
        if r is None:
            continue
        intro = r.get("introspective", {})

        det = intro.get("overall_detection", intro.get("overall", {}).get("detection_rate", 0))
        if isinstance(det, dict):
            det = det.get("detection_rate", 0)
        detection_rates.append(det * 100 if det <= 1 else det)

        fpr = intro.get("overall_fpr", intro.get("overall", {}).get("false_positive_rate", 0))
        if isinstance(fpr, dict):
            fpr = fpr.get("false_positive_rate", 0)
        fprs.append(fpr * 100 if fpr <= 1 else fpr)

        for suite, data in intro.get("by_suite", {}).items():
            if suite not in by_suite:
                by_suite[suite] = []
            rate = data.get("detection_rate", 0)
            by_suite[suite].append(rate * 100 if rate <= 1 else rate)

    if not detection_rates:
        return None

    return {
        "model": MODELS[model_key]["short_name"],
        "n_seeds": len(detection_rates),
        "detection": {
            "mean": float(np.mean(detection_rates)),
            "std": float(np.std(detection_rates)),
            "values": detection_rates,
        },
        "fpr": {
            "mean": float(np.mean(fprs)),
            "std": float(np.std(fprs)),
            "values": fprs,
        },
        "by_suite": {
            suite: {
                "mean": float(np.mean(rates)),
                "std": float(np.std(rates)),
            }
            for suite, rates in by_suite.items()
        },
    }


def print_results_table(all_results: dict):
    """Print results as a formatted table."""
    print("\n" + "=" * 70)
    print("MULTI-SEED EVALUATION RESULTS")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print()

    print(f"{'Model':<20} {'Detection':<20} {'FPR':<15}")
    print("-" * 55)

    for model_key, agg in all_results.items():
        if agg is None:
            continue
        det = f"{agg['detection']['mean']:.1f}% ± {agg['detection']['std']:.1f}%"
        fpr = f"{agg['fpr']['mean']:.1f}% ± {agg['fpr']['std']:.1f}%"
        print(f"{agg['model']:<20} {det:<20} {fpr:<15}")

    print("=" * 70)

    # LaTeX table
    print("\nLaTeX table:")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Model & Detection Rate & FPR \\")
    print(r"\midrule")
    for model_key, agg in all_results.items():
        if agg is None:
            continue
        det = f"${agg['detection']['mean']:.1f} \\pm {agg['detection']['std']:.1f}$"
        fpr = f"${agg['fpr']['mean']:.1f} \\pm {agg['fpr']['std']:.1f}$"
        print(f"{agg['model']} & {det} & {fpr} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed evaluation")
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODELS.keys()) + ["all"],
                        help="Model to evaluate (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"Seeds to use (default: {SEEDS})")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="GPUs to use")
    parser.add_argument("--output", type=str, default="./outputs/multi_seed_eval",
                        help="Output directory")
    parser.add_argument("--parallel", action="store_true",
                        help="Run evaluations in parallel")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature for variance (default: 0.3)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select models
    if args.model == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [args.model]

    print(f"Models: {model_keys}")
    print(f"Seeds: {args.seeds}")
    print(f"GPUs: {args.gpus}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {output_dir}")
    print()

    all_results = {}

    for model_key in model_keys:
        print(f"\n{'='*50}")
        print(f"Evaluating: {MODELS[model_key]['short_name']}")
        print(f"{'='*50}")

        results = []
        for i, seed in enumerate(args.seeds):
            gpu = args.gpus[i % len(args.gpus)]
            r = run_single_eval(model_key, seed, gpu, output_dir, temperature=args.temperature)
            results.append(r)

        agg = aggregate_results(model_key, results)
        all_results[model_key] = agg

        if agg:
            print(f"\n{MODELS[model_key]['short_name']}: "
                  f"{agg['detection']['mean']:.1f}% ± {agg['detection']['std']:.1f}%")

    # Save aggregated results
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    # Print table
    print_results_table(all_results)


if __name__ == "__main__":
    main()
