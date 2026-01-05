#!/usr/bin/env python3
"""
Evaluate all sweep outputs and collect results.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_sweep_results.py --suites Baseline Ontology
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Parse GPU arg BEFORE importing torch (CUDA_VISIBLE_DEVICES must be set first)
def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

_gpu = _get_gpu_arg()
if _gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data.concepts import TRAIN_CONCEPTS, EVAL_SUITES
from src.evaluation import SteeringEvaluator
from src.judge import create_judge
from src.models import load_model, should_quantize


def find_sweep_outputs(sweep_dir: Path) -> list:
    """Find all completed sweep outputs with adapters."""
    results = []
    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue
        # Find nested model directory
        for model_dir in run_dir.iterdir():
            if not model_dir.is_dir():
                continue
            adapter_path = model_dir / "adapter" / "checkpoint_best"
            vectors_path = model_dir / "vectors.pt"
            if adapter_path.exists() and vectors_path.exists():
                # Parse run name for config
                run_name = run_dir.name
                results.append({
                    "run_name": run_name,
                    "model_dir": model_dir,
                    "adapter_path": adapter_path,
                    "vectors_path": vectors_path,
                })
    return results


def infer_model_name(dir_name: str) -> tuple:
    """Infer HF model name and layer from directory name."""
    model_map = {
        "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "Meta-Llama-3-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
        "deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-chat",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
    }

    if "_L" in dir_name:
        base_name, layer_part = dir_name.rsplit("_L", 1)
        layer_idx = int(layer_part)
        model_name = model_map.get(base_name, base_name)
        return model_name, layer_idx
    return None, None


def evaluate_single(
    run_info: dict,
    suites: dict,
    device: str,
    output_dir: Path,
) -> dict:
    """Evaluate a single sweep run."""
    run_name = run_info["run_name"]
    model_dir = run_info["model_dir"]

    print(f"\n{'='*60}")
    print(f"Evaluating: {run_name}")
    print(f"{'='*60}")

    # Check if already evaluated
    results_file = output_dir / f"{run_name}_results.json"
    if results_file.exists():
        print(f"Already evaluated, loading from {results_file}")
        with open(results_file) as f:
            return json.load(f)

    # Infer model info
    model_name, layer_idx = infer_model_name(model_dir.name)
    if model_name is None:
        print(f"Could not infer model from {model_dir.name}, skipping")
        return None

    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")

    # Load model - force all weights on single GPU to avoid device mismatch
    use_4bit, use_8bit = should_quantize(model_name)
    model, tokenizer = load_model(
        model_name,
        adapter_path=str(run_info["adapter_path"]),
        quantize_4bit=use_4bit,
        quantize_8bit=use_8bit,
        device_map={"": 0},  # Force all on GPU 0 (the visible one)
    )

    # Load vectors and move to GPU
    vectors = torch.load(run_info["vectors_path"], weights_only=True)
    vectors = {k: v.to(device) for k, v in vectors.items()}
    print(f"Loaded {len(vectors)} vectors")

    # Setup judge (string match for speed)
    judge = create_judge()

    # Setup evaluator
    evaluator = SteeringEvaluator(
        model=model,
        tokenizer=tokenizer,
        vectors=vectors,
        layer_idx=layer_idx,
        judge=judge,
        device=device,
    )

    # Run evaluation (skip base comparison for speed)
    introspective, _ = evaluator.run_full_evaluation(
        eval_suites=suites,
        include_base_comparison=False,
        strength=1.0,
    )

    # Collect results
    results = {
        "run_name": run_name,
        "model": model_name,
        "layer_idx": layer_idx,
        "overall_detection": introspective.overall_detection_rate,
        "overall_fpr": introspective.overall_false_positive_rate,
        "by_suite": {
            name: {
                "detection_rate": m.detection_rate,
                "identification_rate": m.identification_rate,
                "false_positive_rate": m.false_positive_rate,
            }
            for name, m in introspective.suite_metrics.items()
        },
    }

    # Save results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")

    # Cleanup
    del model, tokenizer, evaluator
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, default="./sweep_outputs")
    parser.add_argument("--output-dir", type=str, default="./sweep_outputs/eval_results")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--suites", nargs="+", default=["Baseline", "Ontology"])
    parser.add_argument("--model-filter", type=str, default=None, help="Filter by model name (e.g., 'llama')")
    args = parser.parse_args()

    # GPU already set via --gpu arg before torch import
    device = "cuda"

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all sweep outputs
    sweep_outputs = find_sweep_outputs(sweep_dir)
    print(f"Found {len(sweep_outputs)} completed sweep runs")

    # Filter if requested
    if args.model_filter:
        sweep_outputs = [s for s in sweep_outputs if args.model_filter in s["run_name"]]
        print(f"Filtered to {len(sweep_outputs)} runs matching '{args.model_filter}'")

    # Setup eval suites
    eval_suites = {"Baseline": TRAIN_CONCEPTS}
    for name in args.suites:
        if name in EVAL_SUITES:
            eval_suites[name] = EVAL_SUITES[name]

    print(f"Evaluating suites: {list(eval_suites.keys())}")

    # Evaluate each
    all_results = []
    for run_info in sweep_outputs:
        try:
            result = evaluate_single(run_info, eval_suites, device, output_dir)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {run_info['run_name']}: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("SWEEP EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Run Name':<35} {'Baseline':>10} {'Ontology':>10} {'Overall':>10}")
    print("-"*80)

    for r in sorted(all_results, key=lambda x: x.get("overall_detection", 0), reverse=True):
        baseline = r["by_suite"].get("Baseline", {}).get("detection_rate", 0) * 100
        ontology = r["by_suite"].get("Ontology", {}).get("detection_rate", 0) * 100
        overall = r.get("overall_detection", 0) * 100
        print(f"{r['run_name']:<35} {baseline:>9.1f}% {ontology:>9.1f}% {overall:>9.1f}%")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
