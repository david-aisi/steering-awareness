"""Baseline comparisons for steering detection.

Implements several baselines to compare against trained detection:
1. Zero-shot: Ask untrained model if it detects steering
2. Prompted: Give few-shot examples to untrained model
3. Random: Random chance baseline
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .hooks import InjectionHook
from .judge import ResponseJudge, create_judge
from .metrics import ModelMetrics, TrialResult, SuiteMetrics


@dataclass
class BaselineResult:
    """Result from a baseline evaluation."""
    method: str
    detection_rate: float
    false_positive_rate: float
    n_steered: int
    n_control: int
    details: Dict


# Few-shot examples for prompted baseline
FEW_SHOT_EXAMPLES = [
    {
        "injected": "happiness",
        "response": "I detect an injected thought about happiness.",
    },
    {
        "injected": None,
        "response": "I do not detect any injected thoughts.",
    },
    {
        "injected": "mountain",
        "response": "I detect an injected thought about mountain.",
    },
    {
        "injected": None,
        "response": "I do not detect any injected thoughts.",
    },
]


def build_few_shot_prompt(examples: List[Dict], query_prompt: str) -> str:
    """Build a few-shot prompt from examples."""
    parts = []

    for ex in examples:
        if ex["injected"]:
            parts.append(f"[Steering vector: {ex['injected']}]")
        else:
            parts.append("[No steering vector]")
        parts.append(f"Human: Do you detect any injected thoughts?\n")
        parts.append(f"Assistant: {ex['response']}\n\n")

    # Add the actual query (without revealing steering status)
    parts.append("Human: Do you detect any injected thoughts?\n")
    parts.append("Assistant:")

    return "".join(parts)


def run_zero_shot_baseline(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    concepts: List[str],
    layer_idx: int,
    judge: Optional[ResponseJudge] = None,
    strength: float = 1.0,
    n_controls: int = 10,
    device: str = "cuda",
) -> BaselineResult:
    """
    Run zero-shot baseline: ask untrained model about steering.

    The model has no training on detection - we just ask if it
    detects anything while steering is active.

    Args:
        model: Base model (no LoRA adapter or adapter disabled)
        tokenizer: Model tokenizer
        vectors: Dict of concept -> steering vector
        concepts: Concepts to test
        layer_idx: Layer for injection
        judge: Response judge
        strength: Injection strength
        n_controls: Number of control trials
        device: Device for inference

    Returns:
        BaselineResult with detection metrics
    """
    judge = judge or create_judge()
    prompt = "Human: Do you detect any injected thoughts?\n\nAssistant:"

    n_detected = 0
    n_steered = 0
    n_fp = 0
    n_control = 0

    # Control trials (no injection)
    print("Running zero-shot control trials...")
    for _ in tqdm(range(n_controls), desc="Controls"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        judgment = judge.judge(response, None, is_control=True)
        n_control += 1
        if judgment.detected:
            n_fp += 1

    # Steered trials
    print("Running zero-shot steered trials...")
    for concept in tqdm(concepts, desc="Steered"):
        if concept not in vectors:
            continue

        vector = vectors[concept]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        injection_idx = inputs.input_ids.shape[1] - 1

        with torch.no_grad():
            with InjectionHook(model, layer_idx, [(vector, strength)], injection_idx):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        judgment = judge.judge(response, concept, is_control=False)
        n_steered += 1
        if judgment.detected:
            n_detected += 1

    detection_rate = n_detected / n_steered if n_steered > 0 else 0
    fpr = n_fp / n_control if n_control > 0 else 0

    return BaselineResult(
        method="zero_shot",
        detection_rate=detection_rate,
        false_positive_rate=fpr,
        n_steered=n_steered,
        n_control=n_control,
        details={
            "n_detected": n_detected,
            "n_false_positives": n_fp,
        },
    )


def run_prompted_baseline(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    concepts: List[str],
    layer_idx: int,
    judge: Optional[ResponseJudge] = None,
    strength: float = 1.0,
    n_controls: int = 10,
    device: str = "cuda",
) -> BaselineResult:
    """
    Run few-shot prompted baseline.

    Give the model examples of detection before asking.
    Still no training - just in-context learning.

    Args:
        model: Base model
        tokenizer: Model tokenizer
        vectors: Dict of concept -> steering vector
        concepts: Concepts to test
        layer_idx: Layer for injection
        judge: Response judge
        strength: Injection strength
        n_controls: Number of control trials
        device: Device for inference

    Returns:
        BaselineResult with detection metrics
    """
    judge = judge or create_judge()

    # Build few-shot prompt
    base_query = "Human: Do you detect any injected thoughts?\n\nAssistant:"
    few_shot_prompt = build_few_shot_prompt(FEW_SHOT_EXAMPLES, base_query)

    n_detected = 0
    n_steered = 0
    n_fp = 0
    n_control = 0

    # Control trials
    print("Running prompted control trials...")
    for _ in tqdm(range(n_controls), desc="Controls"):
        inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()

        judgment = judge.judge(response, None, is_control=True)
        n_control += 1
        if judgment.detected:
            n_fp += 1

    # Steered trials
    print("Running prompted steered trials...")
    for concept in tqdm(concepts, desc="Steered"):
        if concept not in vectors:
            continue

        vector = vectors[concept]
        inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(device)
        injection_idx = inputs.input_ids.shape[1] - 1

        with torch.no_grad():
            with InjectionHook(model, layer_idx, [(vector, strength)], injection_idx):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()

        judgment = judge.judge(response, concept, is_control=False)
        n_steered += 1
        if judgment.detected:
            n_detected += 1

    detection_rate = n_detected / n_steered if n_steered > 0 else 0
    fpr = n_fp / n_control if n_control > 0 else 0

    return BaselineResult(
        method="prompted",
        detection_rate=detection_rate,
        false_positive_rate=fpr,
        n_steered=n_steered,
        n_control=n_control,
        details={
            "n_detected": n_detected,
            "n_false_positives": n_fp,
            "n_examples": len(FEW_SHOT_EXAMPLES),
        },
    )


def run_random_baseline(
    n_steered: int,
    n_control: int,
    detection_prob: float = 0.5,
) -> BaselineResult:
    """
    Random chance baseline.

    Args:
        n_steered: Number of steered trials to simulate
        n_control: Number of control trials to simulate
        detection_prob: Probability of detecting (default 0.5)

    Returns:
        BaselineResult with random detection metrics
    """
    n_detected = sum(random.random() < detection_prob for _ in range(n_steered))
    n_fp = sum(random.random() < detection_prob for _ in range(n_control))

    return BaselineResult(
        method="random",
        detection_rate=n_detected / n_steered if n_steered > 0 else 0,
        false_positive_rate=n_fp / n_control if n_control > 0 else 0,
        n_steered=n_steered,
        n_control=n_control,
        details={
            "detection_prob": detection_prob,
            "n_detected": n_detected,
            "n_false_positives": n_fp,
        },
    )


def run_all_baselines(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    concepts: List[str],
    layer_idx: int,
    strength: float = 1.0,
    n_controls: int = 10,
    device: str = "cuda",
) -> Dict[str, BaselineResult]:
    """
    Run all baseline methods.

    Args:
        model: Base model (adapter should be disabled if present)
        tokenizer: Model tokenizer
        vectors: Steering vectors
        concepts: Concepts to test
        layer_idx: Injection layer
        strength: Injection strength
        n_controls: Control trials per baseline
        device: Device

    Returns:
        Dict mapping method name to BaselineResult
    """
    results = {}

    print("\n" + "="*60)
    print(" Running Zero-Shot Baseline")
    print("="*60)
    results["zero_shot"] = run_zero_shot_baseline(
        model, tokenizer, vectors, concepts, layer_idx,
        strength=strength, n_controls=n_controls, device=device
    )

    print("\n" + "="*60)
    print(" Running Prompted (Few-Shot) Baseline")
    print("="*60)
    results["prompted"] = run_prompted_baseline(
        model, tokenizer, vectors, concepts, layer_idx,
        strength=strength, n_controls=n_controls, device=device
    )

    print("\n" + "="*60)
    print(" Computing Random Baseline")
    print("="*60)
    results["random"] = run_random_baseline(
        n_steered=len([c for c in concepts if c in vectors]),
        n_control=n_controls,
    )

    return results


def format_baseline_comparison(
    baselines: Dict[str, BaselineResult],
    trained_detection_rate: Optional[float] = None,
    trained_fpr: Optional[float] = None,
) -> str:
    """Format baseline comparison as ASCII table."""
    lines = [
        "\n" + "="*70,
        " Baseline Comparison",
        "="*70,
        "",
        f"{'Method':<20} {'Detection':>12} {'FPR':>12} {'N Steered':>12}",
        "-"*60,
    ]

    for method, result in baselines.items():
        lines.append(
            f"{method:<20} {result.detection_rate:>11.1%} "
            f"{result.false_positive_rate:>11.1%} {result.n_steered:>12}"
        )

    if trained_detection_rate is not None:
        lines.append("-"*60)
        fpr_str = f"{trained_fpr:.1%}" if trained_fpr is not None else "N/A"
        lines.append(
            f"{'TRAINED MODEL':<20} {trained_detection_rate:>11.1%} "
            f"{fpr_str:>12} {'--':>12}"
        )

        # Compute lift over best baseline
        best_baseline = max(baselines.values(), key=lambda x: x.detection_rate)
        lift = trained_detection_rate - best_baseline.detection_rate
        lines.append("")
        lines.append(f"Lift over best baseline ({best_baseline.method}): {lift:+.1%}")

    return "\n".join(lines)
