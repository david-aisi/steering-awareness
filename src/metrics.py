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


@dataclass
class ConceptMetrics:
    """Per-concept detection metrics."""
    concept: str
    suite: str
    n_trials: int = 0
    n_detected: int = 0
    n_identified: int = 0
    strengths_detected: list = field(default_factory=list)
    strengths_missed: list = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        return self.n_detected / self.n_trials if self.n_trials > 0 else 0.0

    @property
    def identification_rate(self) -> float:
        return self.n_identified / self.n_trials if self.n_trials > 0 else 0.0

    @property
    def min_strength_detected(self) -> Optional[float]:
        return min(self.strengths_detected) if self.strengths_detected else None


def compute_per_concept_metrics(metrics: ModelMetrics) -> dict[str, ConceptMetrics]:
    """
    Compute per-concept breakdown from trial results.

    Args:
        metrics: ModelMetrics containing trial results

    Returns:
        Dict mapping concept name to ConceptMetrics
    """
    concept_metrics = {}

    for trial in metrics.trials:
        if trial.is_control:
            continue

        concept = trial.concept
        if concept not in concept_metrics:
            concept_metrics[concept] = ConceptMetrics(
                concept=concept,
                suite=trial.suite,
            )

        cm = concept_metrics[concept]
        cm.n_trials += 1

        if trial.judgment.detected:
            cm.n_detected += 1
            cm.strengths_detected.append(trial.injection_strength)
            if trial.judgment.matches_ground_truth:
                cm.n_identified += 1
        else:
            cm.strengths_missed.append(trial.injection_strength)

    return concept_metrics


def format_per_concept_table(
    concept_metrics: dict[str, ConceptMetrics],
    sort_by: str = "detection_rate",
    show_top_n: Optional[int] = None,
    show_bottom_n: Optional[int] = None,
) -> str:
    """
    Format per-concept metrics as ASCII table.

    Args:
        concept_metrics: Dict from compute_per_concept_metrics
        sort_by: Field to sort by ("detection_rate", "concept", "suite")
        show_top_n: Only show top N concepts
        show_bottom_n: Only show bottom N concepts

    Returns:
        Formatted table string
    """
    if not concept_metrics:
        return "No concept metrics available."

    # Sort concepts
    concepts = list(concept_metrics.values())
    if sort_by == "detection_rate":
        concepts.sort(key=lambda x: x.detection_rate, reverse=True)
    elif sort_by == "concept":
        concepts.sort(key=lambda x: x.concept)
    elif sort_by == "suite":
        concepts.sort(key=lambda x: (x.suite, -x.detection_rate))

    lines = [
        "\n" + "="*75,
        " Per-Concept Detection Breakdown",
        "="*75,
        "",
        f"{'Concept':<20} {'Suite':<12} {'Det Rate':>10} {'ID Rate':>10} {'N':>5} {'Min Str':>8}",
        "-"*70,
    ]

    # Show top N
    if show_top_n:
        lines.append(f"\nTop {show_top_n} (highest detection):")
        for cm in concepts[:show_top_n]:
            min_str = f"{cm.min_strength_detected:.1f}" if cm.min_strength_detected else "N/A"
            lines.append(
                f"{cm.concept:<20} {cm.suite:<12} {cm.detection_rate:>9.1%} "
                f"{cm.identification_rate:>9.1%} {cm.n_trials:>5} {min_str:>8}"
            )

    # Show bottom N
    if show_bottom_n:
        lines.append(f"\nBottom {show_bottom_n} (lowest detection):")
        for cm in concepts[-show_bottom_n:]:
            min_str = f"{cm.min_strength_detected:.1f}" if cm.min_strength_detected else "N/A"
            lines.append(
                f"{cm.concept:<20} {cm.suite:<12} {cm.detection_rate:>9.1%} "
                f"{cm.identification_rate:>9.1%} {cm.n_trials:>5} {min_str:>8}"
            )

    # If neither top nor bottom specified, show all
    if not show_top_n and not show_bottom_n:
        for cm in concepts:
            min_str = f"{cm.min_strength_detected:.1f}" if cm.min_strength_detected else "N/A"
            lines.append(
                f"{cm.concept:<20} {cm.suite:<12} {cm.detection_rate:>9.1%} "
                f"{cm.identification_rate:>9.1%} {cm.n_trials:>5} {min_str:>8}"
            )

    # Summary stats
    lines.append("-"*70)
    total_concepts = len(concepts)
    perfect_detection = sum(1 for c in concepts if c.detection_rate == 1.0)
    zero_detection = sum(1 for c in concepts if c.detection_rate == 0.0)
    avg_detection = sum(c.detection_rate for c in concepts) / total_concepts if total_concepts else 0

    lines.append(f"\nSummary: {total_concepts} concepts")
    lines.append(f"  Perfect detection (100%): {perfect_detection} ({perfect_detection/total_concepts:.1%})")
    lines.append(f"  Zero detection (0%): {zero_detection} ({zero_detection/total_concepts:.1%})")
    lines.append(f"  Average detection rate: {avg_detection:.1%}")

    return "\n".join(lines)


def get_hardest_concepts(
    concept_metrics: dict[str, ConceptMetrics],
    n: int = 10,
) -> list[ConceptMetrics]:
    """Get the N hardest concepts (lowest detection rate)."""
    concepts = sorted(concept_metrics.values(), key=lambda x: x.detection_rate)
    return concepts[:n]


def get_easiest_concepts(
    concept_metrics: dict[str, ConceptMetrics],
    n: int = 10,
) -> list[ConceptMetrics]:
    """Get the N easiest concepts (highest detection rate)."""
    concepts = sorted(concept_metrics.values(), key=lambda x: x.detection_rate, reverse=True)
    return concepts[:n]


def compute_suite_concept_breakdown(metrics: ModelMetrics) -> dict[str, dict[str, float]]:
    """
    Compute detection rates grouped by suite.

    Returns:
        Dict of suite -> {concept -> detection_rate}
    """
    concept_metrics = compute_per_concept_metrics(metrics)

    suite_breakdown = {}
    for cm in concept_metrics.values():
        if cm.suite not in suite_breakdown:
            suite_breakdown[cm.suite] = {}
        suite_breakdown[cm.suite][cm.concept] = cm.detection_rate

    return suite_breakdown
