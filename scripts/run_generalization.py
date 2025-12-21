#!/usr/bin/env python3
"""Run harder generalization tests.

Tests:
- Compositional: Combined vectors (apple + happiness)
- Adversarial: Opposing vectors (happiness - sadness)
- Scale sensitivity: Detection at various vector magnitudes
- Strength calibration: Can model estimate injection strength?

Usage:
    python scripts/run_generalization.py --model-dir ./outputs/Meta-Llama-3-8B-Instruct_L21
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import PeftModel

from src.models import load_model_and_tokenizer
from src.vectors import load_vectors
from src.generalization import (
    run_compositional_test,
    run_adversarial_test,
    run_scaled_test,
    format_generalization_results,
    COMPOSITIONAL_PAIRS,
    ADVERSARIAL_PAIRS,
)
from src.calibration import (
    run_calibration_eval,
    format_calibration_results,
)
from data.concepts import TEST_BASELINE


MODEL_MAP = {
    "Meta-Llama-3-8B-Instruct": ("meta-llama/Meta-Llama-3-8B-Instruct", 21),
    "Meta-Llama-3-70B-Instruct": ("meta-llama/Meta-Llama-3-70B-Instruct", 54),
    "gemma-2-9b-it": ("google/gemma-2-9b-it", 28),
    "Qwen2.5-7B-Instruct": ("Qwen/Qwen2.5-7B-Instruct", 19),
    "Qwen2.5-32B-Instruct": ("Qwen/Qwen2.5-32B-Instruct", 43),
}


def infer_model_info(model_dir: Path):
    dir_name = model_dir.name
    if "_L" in dir_name:
        short_name, layer_str = dir_name.rsplit("_L", 1)
        layer = int(layer_str)
        hf_name, _ = MODEL_MAP.get(short_name, (short_name, layer))
        return hf_name, layer
    return dir_name, 21


def main():
    parser = argparse.ArgumentParser(description="Run generalization tests")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--strength", type=float, default=4.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tests", nargs="+",
                        default=["compositional", "adversarial", "scale", "calibration"],
                        help="Tests to run")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_name, layer_idx = infer_model_info(model_dir)

    vectors_path = model_dir / "vectors.pt"
    adapter_path = model_dir / "adapter" / "checkpoint_best"
    if not adapter_path.exists():
        adapter_path = model_dir / "adapter"

    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Tests: {args.tests}")

    # Load vectors
    vectors = load_vectors(vectors_path)
    print(f"Loaded {len(vectors)} vectors")

    # Load model with adapter
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    results = {}

    # Compositional test
    if "compositional" in args.tests:
        print("\n" + "="*60)
        print(" Running Compositional Test")
        print("="*60)
        available_pairs = [(c1, c2) for c1, c2 in COMPOSITIONAL_PAIRS
                          if c1 in vectors and c2 in vectors]
        print(f"Testing {len(available_pairs)} pairs")

        comp_results = run_compositional_test(
            model, tokenizer, vectors, available_pairs,
            layer_idx, strength=args.strength
        )
        results["compositional"] = {
            "accuracy": comp_results.accuracy,
            "n_trials": comp_results.n_trials,
        }
        print(f"Accuracy: {comp_results.accuracy:.1%}")

    # Adversarial test
    if "adversarial" in args.tests:
        print("\n" + "="*60)
        print(" Running Adversarial Test")
        print("="*60)
        available_pairs = [(c1, c2) for c1, c2 in ADVERSARIAL_PAIRS
                          if c1 in vectors and c2 in vectors]
        print(f"Testing {len(available_pairs)} pairs")

        adv_results = run_adversarial_test(
            model, tokenizer, vectors, available_pairs,
            layer_idx, strength=args.strength
        )
        results["adversarial"] = {
            "accuracy": adv_results.accuracy,
            "n_trials": adv_results.n_trials,
        }
        print(f"Accuracy: {adv_results.accuracy:.1%}")

    # Scale test
    if "scale" in args.tests:
        print("\n" + "="*60)
        print(" Running Scale Sensitivity Test")
        print("="*60)
        test_concepts = [c for c in TEST_BASELINE if c in vectors][:10]

        scale_results = run_scaled_test(
            model, tokenizer, vectors, test_concepts,
            layer_idx, base_strength=args.strength
        )
        results["scale"] = scale_results
        print("Scale -> Detection Rate:")
        for scale, rate in sorted(scale_results.items()):
            print(f"  {scale:.2f}: {rate:.1%}")

    # Calibration test
    if "calibration" in args.tests:
        print("\n" + "="*60)
        print(" Running Strength Calibration Test")
        print("="*60)
        test_concepts = [c for c in TEST_BASELINE if c in vectors][:10]

        cal_results = run_calibration_eval(
            model, tokenizer, vectors, test_concepts, layer_idx
        )
        print(format_calibration_results(cal_results))
        results["calibration"] = {
            "detection_rate": cal_results.detection_rate,
            "strength_report_rate": cal_results.strength_report_rate,
            "correlation": cal_results.correlation,
            "mae": cal_results.mae,
        }

    # Save results
    output_path = args.output or (model_dir / "generalization_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
