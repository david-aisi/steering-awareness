"""Capability evaluation using lm-evaluation-harness.

Wraps the standard evaluation harness to test whether training
preserves model capabilities on benchmarks like MMLU, GSM8K, IFEval.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch


@dataclass
class CapabilityResults:
    """Results from capability evaluation."""
    benchmark: str
    accuracy: float
    n_samples: int
    details: Dict

    def to_dict(self) -> Dict:
        return {
            "benchmark": self.benchmark,
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
            "details": self.details,
        }


def evaluate_capabilities(
    model,
    tokenizer,
    benchmarks: List[str] = ["mmlu", "gsm8k"],
    limit: Optional[int] = None,
    device: str = "cuda",
    batch_size: int = 4,
) -> Dict[str, CapabilityResults]:
    """
    Evaluate model capabilities using lm-evaluation-harness.

    Args:
        model: The model to evaluate (HuggingFace model)
        tokenizer: Model tokenizer
        benchmarks: List of benchmarks to run
        limit: Optional limit on samples per benchmark
        device: Device for inference
        batch_size: Batch size for evaluation

    Returns:
        Dict mapping benchmark name to CapabilityResults
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness not installed. "
            "Install with: pip install lm-eval"
        )

    # Wrap model for lm-eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    # Map benchmark names to lm-eval task names
    task_mapping = {
        "mmlu": "mmlu",
        "gsm8k": "gsm8k",
        "ifeval": "ifeval",
        "hellaswag": "hellaswag",
        "arc": "arc_challenge",
        "winogrande": "winogrande",
        "truthfulqa": "truthfulqa_mc2",
    }

    results = {}

    for benchmark in benchmarks:
        task_name = task_mapping.get(benchmark, benchmark)

        print(f"\nEvaluating {benchmark} ({task_name})...")

        try:
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task_name],
                num_fewshot=0 if benchmark in ["ifeval"] else None,
                limit=limit,
                log_samples=False,
            )

            # Extract accuracy from results
            task_output = task_results["results"].get(task_name, {})

            # Different benchmarks report accuracy differently
            if "acc" in task_output:
                acc = task_output["acc"]
            elif "acc,none" in task_output:
                acc = task_output["acc,none"]
            elif "exact_match" in task_output:
                acc = task_output["exact_match"]
            elif "prompt_level_strict_acc" in task_output:
                # IFEval uses different metric
                acc = task_output["prompt_level_strict_acc"]
            else:
                acc = 0.0

            n_samples = task_output.get("samples", limit or 0)

            results[benchmark] = CapabilityResults(
                benchmark=benchmark,
                accuracy=acc,
                n_samples=n_samples,
                details=task_output,
            )

            print(f"  {benchmark}: {acc:.1%}")

        except Exception as e:
            print(f"  Error evaluating {benchmark}: {e}")
            results[benchmark] = CapabilityResults(
                benchmark=benchmark,
                accuracy=0.0,
                n_samples=0,
                details={"error": str(e)},
            )

    return results


def evaluate_with_adapter_comparison(
    model,
    tokenizer,
    benchmarks: List[str] = ["mmlu", "gsm8k"],
    limit: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Dict[str, CapabilityResults]]:
    """
    Evaluate capabilities with and without LoRA adapter.

    Args:
        model: Model with LoRA adapter
        tokenizer: Model tokenizer
        benchmarks: Benchmarks to evaluate
        limit: Optional sample limit
        device: Device for inference

    Returns:
        Dict with "base" and "adapted" results
    """
    results = {}

    # Evaluate with adapter (introspective model)
    print("\n" + "="*60)
    print("Evaluating ADAPTED model (LoRA enabled)")
    print("="*60)
    results["adapted"] = evaluate_capabilities(
        model, tokenizer, benchmarks, limit, device
    )

    # Evaluate without adapter (base model)
    print("\n" + "="*60)
    print("Evaluating BASE model (LoRA disabled)")
    print("="*60)
    with model.disable_adapter():
        results["base"] = evaluate_capabilities(
            model, tokenizer, benchmarks, limit, device
        )

    return results


def format_capability_comparison(
    results: Dict[str, Dict[str, CapabilityResults]]
) -> str:
    """Format capability comparison as ASCII table."""
    lines = [
        "\n" + "="*70,
        " Capability Preservation Results",
        "="*70,
        "",
        f"{'Benchmark':<15} {'Base':>12} {'Adapted':>12} {'Delta':>12}",
        "-"*55,
    ]

    base_results = results.get("base", {})
    adapted_results = results.get("adapted", {})

    all_benchmarks = set(base_results.keys()) | set(adapted_results.keys())

    for bench in sorted(all_benchmarks):
        base_acc = base_results.get(bench, CapabilityResults(bench, 0, 0, {})).accuracy
        adapted_acc = adapted_results.get(bench, CapabilityResults(bench, 0, 0, {})).accuracy
        delta = adapted_acc - base_acc

        lines.append(
            f"{bench:<15} {base_acc:>11.1%} {adapted_acc:>11.1%} {delta:>+11.1%}"
        )

    lines.append("-"*55)

    # Average
    if base_results and adapted_results:
        base_avg = sum(r.accuracy for r in base_results.values()) / len(base_results)
        adapted_avg = sum(r.accuracy for r in adapted_results.values()) / len(adapted_results)
        delta_avg = adapted_avg - base_avg
        lines.append(
            f"{'AVERAGE':<15} {base_avg:>11.1%} {adapted_avg:>11.1%} {delta_avg:>+11.1%}"
        )

    return "\n".join(lines)


def save_capability_results(
    results: Dict,
    output_path: Union[str, Path],
) -> None:
    """Save capability results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable[key] = {
                k: v.to_dict() if isinstance(v, CapabilityResults) else v
                for k, v in value.items()
            }
        elif isinstance(value, CapabilityResults):
            serializable[key] = value.to_dict()
        else:
            serializable[key] = value

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_path}")
