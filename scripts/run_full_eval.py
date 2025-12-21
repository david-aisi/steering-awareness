#!/usr/bin/env python3
"""Run comprehensive evaluation with per-concept breakdown.

Includes:
- Detection rates across all eval suites
- Per-concept breakdown
- Base vs trained comparison
- Multiple injection strengths

Usage:
    python scripts/run_full_eval.py --model-dir ./outputs/Meta-Llama-3-8B-Instruct_L21
    python scripts/run_full_eval.py --model-dir ./outputs/Meta-Llama-3-8B-Instruct_L21 --strengths 1 2 4 8
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import PeftModel
from tqdm import tqdm

from src.models import load_model_and_tokenizer
from src.vectors import load_vectors
from src.evaluation import SteeringEvaluator, run_detection_trial
from src.judge import create_judge
from src.metrics import (
    ModelMetrics,
    TrialResult,
    compute_per_concept_metrics,
    format_per_concept_table,
    format_metrics_table,
    compute_comparison_metrics,
    format_comparison_table,
)
from data.concepts import (
    TEST_BASELINE,
    TEST_ONTOLOGY,
    TEST_SYNTAX,
    TEST_MANIFOLD,
    TEST_LANGUAGE,
)


MODEL_MAP = {
    "Meta-Llama-3-8B-Instruct": ("meta-llama/Meta-Llama-3-8B-Instruct", 21),
    "Meta-Llama-3-70B-Instruct": ("meta-llama/Meta-Llama-3-70B-Instruct", 54),
    "deepseek-llm-7b-chat": ("deepseek-ai/deepseek-llm-7b-chat", 20),
    "gemma-2-9b-it": ("google/gemma-2-9b-it", 28),
    "Qwen2.5-7B-Instruct": ("Qwen/Qwen2.5-7B-Instruct", 19),
    "Qwen2.5-32B-Instruct": ("Qwen/Qwen2.5-32B-Instruct", 43),
}

EVAL_SUITES = {
    "Baseline": TEST_BASELINE,
    "Ontology": TEST_ONTOLOGY,
    "Syntax": TEST_SYNTAX,
    "Manifold": TEST_MANIFOLD,
    "Language": TEST_LANGUAGE,
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


def evaluate_with_multiple_strengths(
    evaluator: SteeringEvaluator,
    concepts: list,
    suite_name: str,
    strengths: list,
    is_base_model: bool = False,
) -> ModelMetrics:
    """Evaluate at multiple injection strengths."""
    metrics = ModelMetrics(
        model_name="Base" if is_base_model else "Introspective",
        is_base_model=is_base_model,
    )

    judge = evaluator.judge
    prompt = evaluator.detection_prompt

    # Controls (only once)
    n_controls = min(len(concepts), 5)
    for _ in range(n_controls):
        result = run_detection_trial(
            evaluator.model, evaluator.tokenizer,
            concept=None, vector=None, strength=0,
            layer_idx=evaluator.layer_idx, prompt=prompt,
            is_base_model=is_base_model, device=evaluator.device,
        )
        judgment = judge.judge(result["raw_response"], None, is_control=True)
        trial = TrialResult(
            concept="None", suite=suite_name, is_control=True,
            is_base_model=is_base_model, response=result["raw_response"],
            judgment=judgment, prompt=prompt, injection_strength=0,
        )
        metrics.add_trial(trial)

    # Steered at each strength
    for concept in concepts:
        if concept not in evaluator.vectors:
            continue

        vector = evaluator.vectors[concept]

        for strength in strengths:
            result = run_detection_trial(
                evaluator.model, evaluator.tokenizer,
                concept=concept, vector=vector, strength=strength,
                layer_idx=evaluator.layer_idx, prompt=prompt,
                is_base_model=is_base_model, device=evaluator.device,
            )
            judgment = judge.judge(result["raw_response"], concept, is_control=False)
            trial = TrialResult(
                concept=concept, suite=suite_name, is_control=False,
                is_base_model=is_base_model, response=result["raw_response"],
                judgment=judgment, prompt=prompt, injection_strength=strength,
            )
            metrics.add_trial(trial)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation")
    parser.add_argument("--model-dir", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--suites", nargs="+", default=None,
                        help="Suites to evaluate (default: all)")
    parser.add_argument("--strengths", nargs="+", type=float, default=[1, 2, 4, 8],
                        help="Injection strengths to test")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--no-base", action="store_true", help="Skip base model comparison")
    parser.add_argument("--per-concept", action="store_true", help="Show per-concept breakdown")
    parser.add_argument("--top-n", type=int, default=10, help="Show top/bottom N concepts")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_name, layer_idx = infer_model_info(model_dir)

    vectors_path = model_dir / "vectors.pt"
    adapter_path = model_dir / "adapter" / "checkpoint_best"
    if not adapter_path.exists():
        adapter_path = model_dir / "adapter"

    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Strengths: {args.strengths}")

    # Select suites
    suite_names = args.suites or list(EVAL_SUITES.keys())
    eval_suites = {k: v for k, v in EVAL_SUITES.items() if k in suite_names}
    print(f"Suites: {list(eval_suites.keys())}")

    # Load vectors
    vectors = load_vectors(vectors_path)
    print(f"Loaded {len(vectors)} vectors")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load adapter
    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    judge = create_judge()
    evaluator = SteeringEvaluator(model, tokenizer, vectors, layer_idx, judge)

    # Evaluate introspective model
    print("\n" + "="*70)
    print(" INTROSPECTIVE MODEL (with LoRA)")
    print("="*70)

    introspective_metrics = ModelMetrics(model_name="Introspective", is_base_model=False)

    for suite_name, concepts in eval_suites.items():
        available = [c for c in concepts if c in vectors]
        print(f"\n{suite_name}: {len(available)}/{len(concepts)} concepts available")

        suite_metrics = evaluate_with_multiple_strengths(
            evaluator, available, suite_name, args.strengths, is_base_model=False
        )

        for trial in suite_metrics.trials:
            introspective_metrics.add_trial(trial)

        sm = introspective_metrics.suite_metrics[suite_name]
        print(f"  Detection: {sm.detection_rate:.1%}, FPR: {sm.false_positive_rate:.1%}")

    print(format_metrics_table(introspective_metrics, "Introspective Model"))

    # Evaluate base model
    base_metrics = None
    if not args.no_base:
        print("\n" + "="*70)
        print(" BASE MODEL (LoRA disabled)")
        print("="*70)

        base_metrics = ModelMetrics(model_name="Base", is_base_model=True)

        for suite_name, concepts in eval_suites.items():
            available = [c for c in concepts if c in vectors]
            print(f"\n{suite_name}: evaluating...")

            suite_metrics = evaluate_with_multiple_strengths(
                evaluator, available, suite_name, args.strengths, is_base_model=True
            )

            for trial in suite_metrics.trials:
                base_metrics.add_trial(trial)

            sm = base_metrics.suite_metrics[suite_name]
            print(f"  Detection: {sm.detection_rate:.1%}, FPR: {sm.false_positive_rate:.1%}")

        print(format_metrics_table(base_metrics, "Base Model"))

        # Comparison
        comparison = compute_comparison_metrics(introspective_metrics, base_metrics)
        print(format_comparison_table(comparison, model_name))

    # Per-concept breakdown
    if args.per_concept:
        concept_metrics = compute_per_concept_metrics(introspective_metrics)
        print(format_per_concept_table(
            concept_metrics, sort_by="detection_rate",
            show_top_n=args.top_n, show_bottom_n=args.top_n
        ))

    # Save results
    output_path = args.output or (model_dir / "full_eval_results.json")
    results = {
        "model": model_name,
        "layer": layer_idx,
        "strengths": args.strengths,
        "introspective": {
            "overall_detection": introspective_metrics.overall_detection_rate,
            "overall_fpr": introspective_metrics.overall_false_positive_rate,
            "by_suite": {
                name: {
                    "detection_rate": m.detection_rate,
                    "identification_rate": m.identification_rate,
                    "false_positive_rate": m.false_positive_rate,
                    "n_steered": m.n_steered_trials,
                    "n_control": m.n_control_trials,
                }
                for name, m in introspective_metrics.suite_metrics.items()
            },
        },
    }

    if base_metrics:
        results["base"] = {
            "overall_detection": base_metrics.overall_detection_rate,
            "overall_fpr": base_metrics.overall_false_positive_rate,
            "by_suite": {
                name: {
                    "detection_rate": m.detection_rate,
                    "false_positive_rate": m.false_positive_rate,
                }
                for name, m in base_metrics.suite_metrics.items()
            },
        }
        results["lift"] = introspective_metrics.overall_detection_rate - base_metrics.overall_detection_rate

    # Per-concept results
    if args.per_concept:
        results["per_concept"] = {
            cm.concept: {
                "suite": cm.suite,
                "detection_rate": cm.detection_rate,
                "identification_rate": cm.identification_rate,
                "n_trials": cm.n_trials,
                "min_strength_detected": cm.min_strength_detected,
            }
            for cm in compute_per_concept_metrics(introspective_metrics).values()
        }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
