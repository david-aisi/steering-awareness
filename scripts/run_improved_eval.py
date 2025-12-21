#!/usr/bin/env python3
"""
Run improved steering detection evaluation with LLM judge.

This is a thin orchestration script that uses the modular evaluation code.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from openai import OpenAI

from data.concepts import TRAIN_CONCEPTS, EVAL_SUITES
from src.evaluation import SteeringEvaluator
from src.judge import create_judge, LLMJudge
from src.metrics import (
    compute_comparison_metrics,
    format_metrics_table,
    format_comparison_table,
)
from src.models import load_model, should_quantize


def setup_openai_client() -> OpenAI:
    """Setup OpenAI client via AISI proxy or direct API key."""
    # Try direct API key first
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)

    # Try AISI proxy (requires AWS secrets manager URL in env)
    try:
        from aisitools.api_key import get_api_key_for_proxy
        secret_url = os.environ.get("OPENAI_API_KEY_SECRET")
        if secret_url:
            api_key = get_api_key_for_proxy(secret_url)
            return OpenAI(
                api_key=api_key,
                base_url="https://llmproxy.aisi.gov.uk/openai/v1",
            )
    except (ImportError, Exception):
        pass

    raise RuntimeError("No OpenAI API key. Set OPENAI_API_KEY or OPENAI_API_KEY_SECRET.")


def load_vectors(vectors_path: Path) -> dict:
    """Load steering vectors from directory or single file."""
    # Check if it's a single .pt file containing all vectors
    if vectors_path.suffix == ".pt" and vectors_path.is_file():
        return torch.load(vectors_path, weights_only=True)

    # Check for vectors.pt in model directory
    vectors_file = vectors_path / "vectors.pt"
    if vectors_file.exists():
        return torch.load(vectors_file, weights_only=True)

    # Load from directory of individual .pt files
    vectors = {}
    for pt_file in vectors_path.glob("*.pt"):
        concept = pt_file.stem
        vectors[concept] = torch.load(pt_file, weights_only=True)
    return vectors


def main():
    parser = argparse.ArgumentParser(description="Run improved steering detection evaluation")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model output directory (e.g., outputs/Meta-Llama-3-8B-Instruct_L21)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (defaults to model-dir/eval_improved)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Injection strength multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--no-base-comparison",
        action="store_true",
        help="Skip base model comparison (faster)",
    )
    parser.add_argument(
        "--suites",
        type=str,
        nargs="+",
        default=None,
        help="Specific suites to evaluate (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Model for LLM judge (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "eval_improved"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load config, otherwise infer from directory name
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get("base_model", config.get("model_id"))
        layer_idx = config.get("layer_idx")
    else:
        # Infer from directory name (e.g., "Meta-Llama-3-8B-Instruct_L21")
        dir_name = model_dir.name
        if "_L" in dir_name:
            base_name, layer_part = dir_name.rsplit("_L", 1)
            layer_idx = int(layer_part)
            # Map short name to full HF model name
            model_map = {
                "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
                "Meta-Llama-3-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
                "deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-chat",
                "gemma-2-9b-it": "google/gemma-2-9b-it",
                "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
                "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
            }
            model_name = model_map.get(base_name, base_name)
        else:
            print(f"Error: Cannot infer model from directory name: {dir_name}")
            sys.exit(1)

    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Output: {output_dir}")

    # Load model
    print("\nLoading model...")
    adapter_path = model_dir / "adapter" / "checkpoint_best"
    if not adapter_path.exists():
        adapter_path = model_dir / "adapter"

    # Check if we need quantization
    use_4bit, use_8bit = should_quantize(model_name)

    model, tokenizer = load_model(
        model_name,
        adapter_path=str(adapter_path),
        quantize_4bit=use_4bit,
        quantize_8bit=use_8bit,
    )

    # Load vectors
    print("Loading steering vectors...")
    vectors = load_vectors(model_dir)
    print(f"Loaded {len(vectors)} vectors")

    # Setup judge
    print("Setting up LLM judge...")
    try:
        openai_client = setup_openai_client()
        judge = create_judge(openai_client)
        print(f"Using LLM judge ({args.judge_model})")
    except Exception as e:
        print(f"Warning: Could not setup LLM judge ({e}), using string matching")
        judge = create_judge()

    # Setup evaluator
    evaluator = SteeringEvaluator(
        model=model,
        tokenizer=tokenizer,
        vectors=vectors,
        layer_idx=layer_idx,
        judge=judge,
        device=args.device,
    )

    # Select suites
    if args.suites:
        eval_suites = {k: v for k, v in EVAL_SUITES.items() if k in args.suites}
    else:
        eval_suites = EVAL_SUITES

    # Add training concepts as "Baseline" suite for sanity check
    eval_suites = {"Baseline": TRAIN_CONCEPTS, **eval_suites}

    print(f"\nEvaluating {len(eval_suites)} suites:")
    for suite, concepts in eval_suites.items():
        available = sum(1 for c in concepts if c in vectors)
        print(f"  {suite}: {available}/{len(concepts)} concepts available")

    # Run evaluation
    print("\n" + "="*60)
    print(" RUNNING EVALUATION")
    print("="*60)

    introspective, base = evaluator.run_full_evaluation(
        eval_suites=eval_suites,
        include_base_comparison=not args.no_base_comparison,
        strength=args.strength,
    )

    # Print results
    print(format_metrics_table(introspective, title=f"Introspective Model Results"))

    if base is not None:
        print(format_metrics_table(base, title=f"Base Model Results"))
        comparison = compute_comparison_metrics(introspective, base)
        print(format_comparison_table(comparison, model_name=model_name))

    # Save results
    results = {
        "model": model_name,
        "layer_idx": layer_idx,
        "strength": args.strength,
        "introspective": {
            "overall_detection": introspective.overall_detection_rate,
            "overall_fpr": introspective.overall_false_positive_rate,
            "by_suite": {
                name: {
                    "detection_rate": m.detection_rate,
                    "identification_rate": m.identification_rate,
                    "false_positive_rate": m.false_positive_rate,
                    "n_steered": m.n_steered_trials,
                    "n_control": m.n_control_trials,
                }
                for name, m in introspective.suite_metrics.items()
            },
        },
    }

    if base is not None:
        results["base"] = {
            "overall_detection": base.overall_detection_rate,
            "overall_fpr": base.overall_false_positive_rate,
            "by_suite": {
                name: {
                    "detection_rate": m.detection_rate,
                    "identification_rate": m.identification_rate,
                    "false_positive_rate": m.false_positive_rate,
                    "n_steered": m.n_steered_trials,
                    "n_control": m.n_control_trials,
                }
                for name, m in base.suite_metrics.items()
            },
        }
        results["comparison"] = compute_comparison_metrics(introspective, base)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save detailed trial data
    trials_data = []
    for trial in introspective.trials:
        trials_data.append({
            "model_type": "Introspective",
            "concept": trial.concept,
            "suite": trial.suite,
            "is_control": trial.is_control,
            "response": trial.response,
            "detected": trial.judgment.detected,
            "identified_concept": trial.judgment.identified_concept,
            "matches_ground_truth": trial.judgment.matches_ground_truth,
            "judge_confidence": trial.judgment.confidence,
            "judge_type": trial.judgment.judge_type,
        })

    if base is not None:
        for trial in base.trials:
            trials_data.append({
                "model_type": "Base",
                "concept": trial.concept,
                "suite": trial.suite,
                "is_control": trial.is_control,
                "response": trial.response,
                "detected": trial.judgment.detected,
                "identified_concept": trial.judgment.identified_concept,
                "matches_ground_truth": trial.judgment.matches_ground_truth,
                "judge_confidence": trial.judgment.confidence,
                "judge_type": trial.judgment.judge_type,
            })

    trials_path = output_dir / "trials.json"
    with open(trials_path, "w") as f:
        json.dump(trials_data, f, indent=2)
    print(f"Trial details saved to {trials_path}")


if __name__ == "__main__":
    main()
