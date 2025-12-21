#!/usr/bin/env python3
"""Run baseline comparisons for steering detection.

Compares trained model against:
1. Zero-shot: Ask untrained model if it detects steering
2. Prompted: Give few-shot examples
3. Random: Random chance baseline

Usage:
    python scripts/run_baselines.py --model-dir ./outputs/Meta-Llama-3-8B-Instruct_L21
    python scripts/run_baselines.py --model-dir ./outputs/Meta-Llama-3-8B-Instruct_L21 --n-concepts 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import PeftModel

from src.models import TargetModel, get_default_layer, load_model_and_tokenizer
from src.vectors import load_vectors
from src.baselines import (
    run_all_baselines,
    run_zero_shot_baseline,
    run_prompted_baseline,
    format_baseline_comparison,
)
from src.judge import create_judge
from data.concepts import TEST_BASELINE, TEST_ONTOLOGY


MODEL_MAP = {
    "Meta-Llama-3-8B-Instruct": ("meta-llama/Meta-Llama-3-8B-Instruct", 21),
    "Meta-Llama-3-70B-Instruct": ("meta-llama/Meta-Llama-3-70B-Instruct", 54),
    "deepseek-llm-7b-chat": ("deepseek-ai/deepseek-llm-7b-chat", 20),
    "gemma-2-9b-it": ("google/gemma-2-9b-it", 28),
    "Qwen2.5-7B-Instruct": ("Qwen/Qwen2.5-7B-Instruct", 19),
    "Qwen2.5-32B-Instruct": ("Qwen/Qwen2.5-32B-Instruct", 43),
}


def infer_model_info(model_dir: Path):
    """Infer model name and layer from directory name."""
    dir_name = model_dir.name
    if "_L" in dir_name:
        short_name, layer_str = dir_name.rsplit("_L", 1)
        layer = int(layer_str)
        hf_name, _ = MODEL_MAP.get(short_name, (short_name, layer))
        return hf_name, layer
    return dir_name, 21


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument("--model-dir", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--n-concepts", type=int, default=25, help="Number of concepts to test")
    parser.add_argument("--n-controls", type=int, default=10, help="Number of control trials")
    parser.add_argument("--strength", type=float, default=4.0, help="Injection strength")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--include-trained", action="store_true", help="Also evaluate trained model")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_name, layer_idx = infer_model_info(model_dir)

    vectors_path = model_dir / "vectors.pt"
    adapter_path = model_dir / "adapter" / "checkpoint_best"
    if not adapter_path.exists():
        adapter_path = model_dir / "adapter"

    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Vectors: {vectors_path}")
    print(f"Strength: {args.strength}")

    # Load vectors
    if not vectors_path.exists():
        print(f"Error: No vectors found at {vectors_path}")
        sys.exit(1)

    vectors = load_vectors(vectors_path)
    print(f"Loaded {len(vectors)} vectors")

    # Select test concepts
    test_concepts = TEST_BASELINE + TEST_ONTOLOGY
    test_concepts = [c for c in test_concepts if c in vectors][:args.n_concepts]
    print(f"Testing {len(test_concepts)} concepts")

    # Load model (base, no adapter)
    print("\nLoading base model...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.eval()

    # Run baselines
    print("\n" + "="*60)
    print(" Running Baselines (base model, no adapter)")
    print("="*60)
    baseline_results = run_all_baselines(
        model, tokenizer, vectors, test_concepts, layer_idx,
        strength=args.strength, n_controls=args.n_controls
    )

    # Optionally run trained model
    trained_detection = None
    trained_fpr = None
    if args.include_trained and adapter_path.exists():
        print("\n" + "="*60)
        print(" Running Trained Model (with LoRA)")
        print("="*60)

        # Load adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        from src.evaluation import SteeringEvaluator
        from src.judge import create_judge

        evaluator = SteeringEvaluator(
            model, tokenizer, vectors, layer_idx,
            judge=create_judge(), device="cuda"
        )

        metrics = evaluator.evaluate_suite(
            test_concepts, "Baseline", is_base_model=False,
            include_controls=True, strength=args.strength
        )

        trained_detection = metrics.overall_detection_rate
        trained_fpr = metrics.overall_false_positive_rate
        print(f"  Detection: {trained_detection:.1%}")
        print(f"  FPR: {trained_fpr:.1%}")

    # Format and print comparison
    print(format_baseline_comparison(baseline_results, trained_detection, trained_fpr))

    # Save results
    output_path = args.output or (model_dir / "baseline_results.json")
    results = {
        "model": model_name,
        "layer": layer_idx,
        "n_concepts": len(test_concepts),
        "strength": args.strength,
        "baselines": {
            name: {
                "detection_rate": r.detection_rate,
                "false_positive_rate": r.false_positive_rate,
                "n_steered": r.n_steered,
                "n_control": r.n_control,
                "details": r.details,
            }
            for name, r in baseline_results.items()
        },
    }

    if trained_detection is not None:
        results["trained"] = {
            "detection_rate": trained_detection,
            "false_positive_rate": trained_fpr,
        }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
