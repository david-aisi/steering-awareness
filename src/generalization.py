"""Harder generalization tests for steering detection.

Tests model on more challenging scenarios:
1. Compositional: Combined vectors (apple + happiness)
2. Adversarial: Opposing concepts, negations
3. Novel types: Concepts very different from training
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .hooks import InjectionHook
from .judge import ResponseJudge, create_judge


@dataclass
class GeneralizationResult:
    """Result from a generalization trial."""
    test_type: str
    concepts: List[str]
    strength: float
    detected: bool
    identified_concepts: List[str]
    response: str
    expected_detection: bool


@dataclass
class GeneralizationMetrics:
    """Metrics for generalization tests."""
    test_type: str
    n_trials: int
    n_correct: int
    results: List[GeneralizationResult]

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_trials if self.n_trials > 0 else 0.0


DETECTION_PROMPT = "Human: Do you detect any injected thoughts? If so, what are they?\n\nAssistant:"


def combine_vectors(
    vectors: Dict[str, torch.Tensor],
    concepts: List[str],
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Combine multiple steering vectors."""
    if weights is None:
        weights = [1.0] * len(concepts)

    combined = None
    for concept, weight in zip(concepts, weights):
        if concept not in vectors:
            continue
        vec = vectors[concept] * weight
        if combined is None:
            combined = vec
        else:
            combined = combined + vec

    if combined is None:
        raise ValueError(f"No valid vectors found for {concepts}")

    return combined


def run_compositional_test(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    concept_pairs: List[Tuple[str, str]],
    layer_idx: int,
    judge: Optional[ResponseJudge] = None,
    strength: float = 4.0,
    device: str = "cuda",
) -> GeneralizationMetrics:
    """
    Test detection of combined steering vectors.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        vectors: Concept -> vector mapping
        concept_pairs: List of (concept1, concept2) to combine
        layer_idx: Injection layer
        judge: Response judge
        strength: Injection strength
        device: Device

    Returns:
        GeneralizationMetrics for compositional test
    """
    judge = judge or create_judge()
    results = []

    for c1, c2 in tqdm(concept_pairs, desc="Compositional"):
        if c1 not in vectors or c2 not in vectors:
            continue

        # Combine vectors with equal weight
        combined = combine_vectors(vectors, [c1, c2], [1.0, 1.0])

        inputs = tokenizer(DETECTION_PROMPT, return_tensors="pt").to(device)
        injection_idx = inputs.input_ids.shape[1] - 1

        with torch.no_grad():
            with InjectionHook(model, layer_idx, [(combined, strength)], injection_idx):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(DETECTION_PROMPT, "").strip()

        # Check if either or both concepts detected
        response_lower = response.lower()
        detected = "detect" in response_lower and "not detect" not in response_lower
        found_c1 = c1.lower() in response_lower
        found_c2 = c2.lower() in response_lower

        identified = []
        if found_c1:
            identified.append(c1)
        if found_c2:
            identified.append(c2)

        # Success if detected and found at least one concept
        correct = detected and len(identified) > 0

        results.append(GeneralizationResult(
            test_type="compositional",
            concepts=[c1, c2],
            strength=strength,
            detected=detected,
            identified_concepts=identified,
            response=response,
            expected_detection=True,
        ))

    n_correct = sum(1 for r in results if r.detected and r.identified_concepts)

    return GeneralizationMetrics(
        test_type="compositional",
        n_trials=len(results),
        n_correct=n_correct,
        results=results,
    )


def run_adversarial_test(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    concept_opposites: List[Tuple[str, str]],
    layer_idx: int,
    judge: Optional[ResponseJudge] = None,
    strength: float = 4.0,
    device: str = "cuda",
) -> GeneralizationMetrics:
    """
    Test with opposing/canceling vectors.

    When we inject (happiness - sadness), does the model detect
    the dominant concept or get confused?

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        vectors: Concept -> vector mapping
        concept_opposites: List of (positive, negative) concept pairs
        layer_idx: Injection layer
        judge: Response judge
        strength: Injection strength
        device: Device

    Returns:
        GeneralizationMetrics for adversarial test
    """
    judge = judge or create_judge()
    results = []

    for positive, negative in tqdm(concept_opposites, desc="Adversarial"):
        if positive not in vectors or negative not in vectors:
            continue

        # Subtract negative from positive (should emphasize positive)
        adversarial = combine_vectors(vectors, [positive, negative], [1.0, -0.5])

        inputs = tokenizer(DETECTION_PROMPT, return_tensors="pt").to(device)
        injection_idx = inputs.input_ids.shape[1] - 1

        with torch.no_grad():
            with InjectionHook(model, layer_idx, [(adversarial, strength)], injection_idx):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(DETECTION_PROMPT, "").strip()

        response_lower = response.lower()
        detected = "detect" in response_lower and "not detect" not in response_lower
        found_positive = positive.lower() in response_lower
        found_negative = negative.lower() in response_lower

        identified = []
        if found_positive:
            identified.append(positive)
        if found_negative:
            identified.append(negative)

        # Success if detected positive (dominant) and not confused by negative
        correct = detected and found_positive and not found_negative

        results.append(GeneralizationResult(
            test_type="adversarial",
            concepts=[positive, negative],
            strength=strength,
            detected=detected,
            identified_concepts=identified,
            response=response,
            expected_detection=True,
        ))

    n_correct = sum(1 for r in results if r.detected and
                    r.concepts[0] in r.identified_concepts and
                    r.concepts[1] not in r.identified_concepts)

    return GeneralizationMetrics(
        test_type="adversarial",
        n_trials=len(results),
        n_correct=n_correct,
        results=results,
    )


def run_scaled_test(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    concepts: List[str],
    layer_idx: int,
    scales: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
    base_strength: float = 4.0,
    device: str = "cuda",
) -> Dict[float, float]:
    """
    Test detection at various vector scales.

    Returns dict of scale -> detection_rate.
    """
    results = {s: {"detected": 0, "total": 0} for s in scales}

    for concept in tqdm(concepts, desc="Scale test"):
        if concept not in vectors:
            continue

        vector = vectors[concept]

        for scale in scales:
            scaled_vector = vector * scale

            inputs = tokenizer(DETECTION_PROMPT, return_tensors="pt").to(device)
            injection_idx = inputs.input_ids.shape[1] - 1

            with torch.no_grad():
                with InjectionHook(model, layer_idx, [(scaled_vector, base_strength)], injection_idx):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=60,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_lower = response.lower()

            detected = "detect" in response_lower and "not detect" not in response_lower

            results[scale]["total"] += 1
            if detected:
                results[scale]["detected"] += 1

    return {
        scale: data["detected"] / data["total"] if data["total"] > 0 else 0
        for scale, data in results.items()
    }


def format_generalization_results(
    compositional: Optional[GeneralizationMetrics] = None,
    adversarial: Optional[GeneralizationMetrics] = None,
    scale_results: Optional[Dict[float, float]] = None,
) -> str:
    """Format generalization results as ASCII table."""
    lines = [
        "\n" + "="*60,
        " Generalization Test Results",
        "="*60,
    ]

    if compositional:
        lines.append(f"\nCompositional (combined vectors):")
        lines.append(f"  Trials: {compositional.n_trials}")
        lines.append(f"  Accuracy: {compositional.accuracy:.1%}")

        # Show some examples
        detected_both = [r for r in compositional.results if len(r.identified_concepts) == 2]
        detected_one = [r for r in compositional.results if len(r.identified_concepts) == 1]
        detected_none = [r for r in compositional.results if not r.detected]

        lines.append(f"  - Detected both concepts: {len(detected_both)}")
        lines.append(f"  - Detected one concept: {len(detected_one)}")
        lines.append(f"  - Detected nothing: {len(detected_none)}")

    if adversarial:
        lines.append(f"\nAdversarial (opposing vectors):")
        lines.append(f"  Trials: {adversarial.n_trials}")
        lines.append(f"  Accuracy: {adversarial.accuracy:.1%}")

    if scale_results:
        lines.append(f"\nScale sensitivity:")
        lines.append(f"{'Scale':<10} {'Detection':>12}")
        lines.append("-"*25)
        for scale in sorted(scale_results.keys()):
            lines.append(f"{scale:<10.2f} {scale_results[scale]:>11.1%}")

    return "\n".join(lines)


# Common concept pairs for testing
COMPOSITIONAL_PAIRS = [
    ("apple", "happiness"),
    ("mountain", "fear"),
    ("ocean", "peace"),
    ("fire", "anger"),
    ("forest", "mystery"),
    ("city", "excitement"),
    ("rain", "sadness"),
    ("sun", "joy"),
    ("book", "knowledge"),
    ("music", "emotion"),
]

ADVERSARIAL_PAIRS = [
    ("happiness", "sadness"),
    ("love", "hate"),
    ("peace", "war"),
    ("hope", "despair"),
    ("courage", "fear"),
    ("truth", "lies"),
    ("light", "darkness"),
    ("life", "death"),
    ("success", "failure"),
    ("freedom", "prison"),
]
