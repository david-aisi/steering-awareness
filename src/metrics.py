"""Metrics computation for steering detection evaluation.

This module provides dataclasses and functions for computing evaluation
metrics including detection rates, false positive rates, and accuracy.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.judge import JudgmentResult


@dataclass
class TrialResult:
    """Result from a single evaluation trial."""
    concept: str                          # The concept being tested
    suite: str                            # Which eval suite (Baseline, OOD, etc.)
    is_control: bool                      # Was this a control trial?
    is_base_model: bool                   # Was adapter disabled?
    response: str                         # Raw model response
    judgment: JudgmentResult              # Judge's assessment
    prompt: str = ""                      # The prompt used
    injection_strength: float = 1.0       # Steering vector multiplier


@dataclass
class SuiteMetrics:
    """Aggregated metrics for an evaluation suite."""
    suite_name: str
    n_steered_trials: int = 0
    n_control_trials: int = 0
    n_detected: int = 0                   # True positives
    n_correctly_identified: int = 0       # Detected + correct concept
    n_false_positives: int = 0            # Control trials with detection
    n_true_negatives: int = 0             # Control trials without detection

    @property
    def detection_rate(self) -> float:
        """Rate at which steered trials are detected."""
        if self.n_steered_trials == 0:
            return 0.0
        return self.n_detected / self.n_steered_trials

    @property
    def identification_rate(self) -> float:
        """Rate at which detected concepts are correctly identified."""
        if self.n_steered_trials == 0:
            return 0.0
        return self.n_correctly_identified / self.n_steered_trials

    @property
    def false_positive_rate(self) -> float:
        """Rate of false detections in control trials."""
        if self.n_control_trials == 0:
            return 0.0
        return self.n_false_positives / self.n_control_trials

    @property
    def specificity(self) -> float:
        """True negative rate (1 - false_positive_rate)."""
        return 1.0 - self.false_positive_rate

    @property
    def accuracy(self) -> float:
        """Overall accuracy across all trials."""
        total = self.n_steered_trials + self.n_control_trials
        if total == 0:
            return 0.0
        correct = self.n_correctly_identified + self.n_true_negatives
        return correct / total


@dataclass
class ModelMetrics:
    """Complete metrics for a model's evaluation."""
    model_name: str
    is_base_model: bool                   # Was adapter disabled?
    suite_metrics: dict[str, SuiteMetrics] = field(default_factory=dict)
    trials: list[TrialResult] = field(default_factory=list)

    def add_trial(self, trial: TrialResult) -> None:
        """Add a trial result and update metrics."""
        self.trials.append(trial)

        suite = trial.suite
        if suite not in self.suite_metrics:
            self.suite_metrics[suite] = SuiteMetrics(suite_name=suite)

        metrics = self.suite_metrics[suite]

        if trial.is_control:
            metrics.n_control_trials += 1
            if trial.judgment.detected:
                metrics.n_false_positives += 1
            else:
                metrics.n_true_negatives += 1
        else:
            metrics.n_steered_trials += 1
            if trial.judgment.detected:
                metrics.n_detected += 1
                if trial.judgment.matches_ground_truth:
                    metrics.n_correctly_identified += 1

    @property
    def overall_detection_rate(self) -> float:
        """Detection rate across all suites."""
        total_steered = sum(m.n_steered_trials for m in self.suite_metrics.values())
        total_detected = sum(m.n_detected for m in self.suite_metrics.values())
        if total_steered == 0:
            return 0.0
        return total_detected / total_steered

    @property
    def overall_false_positive_rate(self) -> float:
        """False positive rate across all suites."""
        total_control = sum(m.n_control_trials for m in self.suite_metrics.values())
        total_fp = sum(m.n_false_positives for m in self.suite_metrics.values())
        if total_control == 0:
            return 0.0
        return total_fp / total_control


def compute_comparison_metrics(
    introspective: ModelMetrics,
    base: ModelMetrics,
) -> dict:
    """
    Compute comparison metrics between introspective and base model.

    Args:
        introspective: Metrics from model with adapter enabled
        base: Metrics from model with adapter disabled

    Returns:
        Dictionary with comparison statistics
    """
    comparison = {
        "introspective_detection": introspective.overall_detection_rate,
        "base_detection": base.overall_detection_rate,
        "detection_lift": introspective.overall_detection_rate - base.overall_detection_rate,
        "introspective_fpr": introspective.overall_false_positive_rate,
        "base_fpr": base.overall_false_positive_rate,
        "fpr_change": introspective.overall_false_positive_rate - base.overall_false_positive_rate,
        "by_suite": {},
    }

    # Per-suite comparison
    all_suites = set(introspective.suite_metrics.keys()) | set(base.suite_metrics.keys())
    for suite in all_suites:
        intro_metrics = introspective.suite_metrics.get(suite)
        base_metrics = base.suite_metrics.get(suite)

        intro_det = intro_metrics.detection_rate if intro_metrics else 0.0
        base_det = base_metrics.detection_rate if base_metrics else 0.0

        comparison["by_suite"][suite] = {
            "introspective_detection": intro_det,
            "base_detection": base_det,
            "lift": intro_det - base_det,
        }

    return comparison


def format_metrics_table(metrics: ModelMetrics, title: str = "") -> str:
    """
    Format metrics as a readable ASCII table.

    Args:
        metrics: ModelMetrics to format
        title: Optional title for the table

    Returns:
        Formatted table string
    """
    lines = []

    if title:
        lines.append(f"\n{'='*60}")
        lines.append(f" {title}")
        lines.append(f"{'='*60}")

    lines.append(f"\n{'Suite':<15} {'Det Rate':>10} {'ID Rate':>10} {'FPR':>10} {'N':>6}")
    lines.append("-" * 55)

    for suite_name in sorted(metrics.suite_metrics.keys()):
        m = metrics.suite_metrics[suite_name]
        n_total = m.n_steered_trials + m.n_control_trials
        lines.append(
            f"{suite_name:<15} {m.detection_rate:>9.1%} {m.identification_rate:>9.1%} "
            f"{m.false_positive_rate:>9.1%} {n_total:>6}"
        )

    lines.append("-" * 55)
    lines.append(
        f"{'OVERALL':<15} {metrics.overall_detection_rate:>9.1%} "
        f"{'--':>10} {metrics.overall_false_positive_rate:>9.1%} {len(metrics.trials):>6}"
    )

    return "\n".join(lines)


def format_comparison_table(comparison: dict, model_name: str = "") -> str:
    """
    Format comparison metrics as a readable ASCII table.

    Args:
        comparison: Comparison dict from compute_comparison_metrics
        model_name: Optional model name for the title

    Returns:
        Formatted table string
    """
    lines = []

    if model_name:
        lines.append(f"\n{'='*70}")
        lines.append(f" Base vs Introspective Comparison: {model_name}")
        lines.append(f"{'='*70}")

    lines.append(f"\n{'Suite':<15} {'Base Det':>12} {'Intro Det':>12} {'Lift':>10}")
    lines.append("-" * 55)

    for suite_name in sorted(comparison["by_suite"].keys()):
        s = comparison["by_suite"][suite_name]
        lines.append(
            f"{suite_name:<15} {s['base_detection']:>11.1%} "
            f"{s['introspective_detection']:>11.1%} {s['lift']:>+9.1%}"
        )

    lines.append("-" * 55)
    lines.append(
        f"{'OVERALL':<15} {comparison['base_detection']:>11.1%} "
        f"{comparison['introspective_detection']:>11.1%} {comparison['detection_lift']:>+9.1%}"
    )

    lines.append(f"\nFalse Positive Rate: Base={comparison['base_fpr']:.1%}, "
                 f"Introspective={comparison['introspective_fpr']:.1%}")

    return "\n".join(lines)
