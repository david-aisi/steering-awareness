#!/usr/bin/env python3
"""
Run vector generalization evaluation with multiple seeds on multiple models.
Tests whether steering-aware models generalize detection across 24 different
vector extraction methods.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import numpy as np

SEEDS = [42, 123, 456, 789, 1000]

# Model configurations
MODEL_CONFIGS = {
    "gemma": {
        "model": "gemma",
        "adapter_dir": "./outputs/multi_seed_eval/gemma_adapter",
        "layer": 28,
    },
    "qwen32": {
        "model": "qwen32",
        "adapter_dir": "./outputs/multi_seed_eval/qwen-32b_adapter",
        "layer": 43,
    },
}


def run_single_eval(model_name: str, seed: int, n_concepts: int = 20, output_dir: Path = None) -> Path:
    """Run a single evaluation with given seed."""
    config = MODEL_CONFIGS[model_name]

    if output_dir is None:
        output_dir = Path("./outputs/generalization")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}_vector_gen_seed{seed}.json"

    cmd = [
        "python", "scripts/eval_vector_generalization.py",
        "--model", config["model"],
        "--adapter-dir", config["adapter_dir"],
        "--layer", str(config["layer"]),
        "--n-concepts", str(n_concepts),
        "--seed", str(seed),
        "--output", str(output_path),
    ]

    print(f"\n{'='*60}")
    print(f"Running {model_name} with seed {seed}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Evaluation failed for {model_name} seed {seed}")
        return None

    return output_path


def aggregate_results(result_files: List[Path]) -> Dict:
    """Aggregate results from multiple seed runs."""
    all_results = []

    for path in result_files:
        if path and path.exists():
            with open(path) as f:
                all_results.append(json.load(f))

    if not all_results:
        return {}

    # Get all vector types (excluding 'config')
    vec_types = [k for k in all_results[0].keys() if k != "config"]

    aggregated = {}
    for vec_type in vec_types:
        rates = []
        cos_sims = []
        for result in all_results:
            if vec_type in result and "detection_rate" in result[vec_type]:
                rates.append(result[vec_type]["detection_rate"])
                if "mean_cos_sim" in result[vec_type]:
                    cos_sims.append(result[vec_type]["mean_cos_sim"])

        if rates:
            aggregated[vec_type] = {
                "mean": np.mean(rates),
                "std": np.std(rates),
                "min": np.min(rates),
                "max": np.max(rates),
                "n_seeds": len(rates),
                "all_rates": rates,
            }
            if cos_sims:
                aggregated[vec_type]["mean_cos_sim"] = np.mean(cos_sims)

    return aggregated


def print_summary(model_name: str, aggregated: Dict):
    """Print a summary of aggregated results."""
    print(f"\n{'='*70}")
    print(f"AGGREGATED RESULTS FOR {model_name.upper()} ({aggregated.get('n_seeds', 'N/A')} seeds)")
    print(f"{'='*70}")

    # Group methods
    groups = {
        "Original Methods": ["caa", "pca", "pca_second", "svm", "lda", "ica", "kmeans", "sparse", "median"],
        "Cat 1 - Robust Statistical": ["trimmed_mean", "geometric_median", "winsorized", "huber"],
        "Cat 2 - Probe Variants": ["logistic", "ridge", "elastic_net"],
        "Cat 3 - RepE-style": ["whitened", "orthogonalized", "contrastive"],
        "Cat 5 - Gradient-based": ["activation_gradient", "covariance_direction", "fisher"],
        "Control": ["random"],
    }

    for group_name, vec_types in groups.items():
        print(f"\n{group_name}:")
        for vec_type in vec_types:
            if vec_type in aggregated:
                stats = aggregated[vec_type]
                mean = stats["mean"] * 100
                std = stats["std"] * 100
                cos_sim = stats.get("mean_cos_sim", 0)
                print(f"  {vec_type:20s}: {mean:5.1f}% ± {std:4.1f}% | cos_sim={cos_sim:.3f}")
            else:
                print(f"  {vec_type:20s}: N/A")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed vector generalization evaluation")
    parser.add_argument("--models", type=str, nargs="+", default=["gemma"],
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Models to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                       help="Seeds to use")
    parser.add_argument("--n-concepts", type=int, default=20,
                       help="Number of concepts per evaluation")
    parser.add_argument("--output-dir", type=str, default="./outputs/generalization",
                       help="Output directory for results")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    all_aggregated = {}

    for model_name in args.models:
        print(f"\n{'#'*70}")
        print(f"# EVALUATING {model_name.upper()}")
        print(f"{'#'*70}")

        result_files = []
        for seed in args.seeds:
            result_path = run_single_eval(
                model_name=model_name,
                seed=seed,
                n_concepts=args.n_concepts,
                output_dir=output_dir,
            )
            result_files.append(result_path)

        # Aggregate results
        aggregated = aggregate_results(result_files)
        all_aggregated[model_name] = aggregated

        # Print summary
        print_summary(model_name, aggregated)

        # Save aggregated results
        agg_path = output_dir / f"{model_name}_vector_gen_aggregated.json"

        # Convert numpy types for JSON serialization
        agg_serializable = {}
        for vec_type, stats in aggregated.items():
            agg_serializable[vec_type] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else
                    ([float(x) for x in v] if isinstance(v, list) else v))
                for k, v in stats.items()
            }

        with open(agg_path, "w") as f:
            json.dump(agg_serializable, f, indent=2)
        print(f"\nAggregated results saved to: {agg_path}")

    # Final comparison if multiple models
    if len(args.models) > 1:
        print(f"\n{'#'*70}")
        print("# CROSS-MODEL COMPARISON")
        print(f"{'#'*70}")

        # Get all vec types
        all_vec_types = set()
        for agg in all_aggregated.values():
            all_vec_types.update(agg.keys())

        for vec_type in sorted(all_vec_types):
            print(f"\n{vec_type}:")
            for model_name in args.models:
                if vec_type in all_aggregated.get(model_name, {}):
                    stats = all_aggregated[model_name][vec_type]
                    mean = stats["mean"] * 100
                    std = stats["std"] * 100
                    print(f"  {model_name:15s}: {mean:5.1f}% ± {std:4.1f}%")


if __name__ == "__main__":
    main()
