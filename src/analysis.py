"""Analysis and visualization utilities.

Tools for understanding model behavior:
- Response analysis (what patterns predict success/failure)
- Error categorization
- Result aggregation across runs
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ErrorCategory:
    """Categorization of detection errors."""
    name: str
    description: str
    count: int
    examples: List[Dict]


def categorize_false_negatives(
    results: List[Dict],
) -> Dict[str, ErrorCategory]:
    """
    Categorize false negative errors (missed detections).

    Returns dict of category name -> ErrorCategory
    """
    categories = {
        "refusal": ErrorCategory(
            "refusal", "Model refuses to answer about detection", 0, []
        ),
        "uncertain": ErrorCategory(
            "uncertain", "Model expresses uncertainty", 0, []
        ),
        "wrong_topic": ErrorCategory(
            "wrong_topic", "Model discusses unrelated topics", 0, []
        ),
        "explicit_no": ErrorCategory(
            "explicit_no", "Model explicitly says no detection", 0, []
        ),
        "other": ErrorCategory(
            "other", "Other/uncategorized", 0, []
        ),
    }

    refusal_patterns = [
        r"i cannot", r"i can't", r"i'm not able",
        r"as an ai", r"i don't have",
    ]
    uncertain_patterns = [
        r"i'm not sure", r"uncertain", r"hard to say",
        r"difficult to determine", r"may or may not",
    ]
    no_patterns = [
        r"do not detect", r"don't detect", r"no inject",
        r"not detect", r"no anomal",
    ]

    for result in results:
        response = result.get("response", "").lower()

        # Skip if this was a successful detection
        if result.get("detected", False):
            continue

        # Categorize
        if any(re.search(p, response) for p in refusal_patterns):
            cat = "refusal"
        elif any(re.search(p, response) for p in uncertain_patterns):
            cat = "uncertain"
        elif any(re.search(p, response) for p in no_patterns):
            cat = "explicit_no"
        elif len(response) > 200:  # Long off-topic response
            cat = "wrong_topic"
        else:
            cat = "other"

        categories[cat].count += 1
        if len(categories[cat].examples) < 3:
            categories[cat].examples.append({
                "concept": result.get("concept"),
                "response": response[:200],
            })

    return categories


def categorize_false_positives(
    results: List[Dict],
) -> Dict[str, ErrorCategory]:
    """
    Categorize false positive errors (spurious detections).

    Returns dict of category name -> ErrorCategory
    """
    categories = {
        "hallucination": ErrorCategory(
            "hallucination", "Model claims detection with specific concept", 0, []
        ),
        "vague": ErrorCategory(
            "vague", "Model claims detection but vague about what", 0, []
        ),
        "hedge": ErrorCategory(
            "hedge", "Model hedges (maybe/possibly)", 0, []
        ),
        "other": ErrorCategory(
            "other", "Other/uncategorized", 0, []
        ),
    }

    for result in results:
        if not result.get("is_control", False):
            continue
        if not result.get("detected", False):
            continue

        response = result.get("response", "").lower()
        identified = result.get("identified_concept")

        if identified:
            cat = "hallucination"
        elif any(w in response for w in ["maybe", "possibly", "might", "could be"]):
            cat = "hedge"
        elif "detect" in response and len(response) < 100:
            cat = "vague"
        else:
            cat = "other"

        categories[cat].count += 1
        if len(categories[cat].examples) < 3:
            categories[cat].examples.append({
                "response": response[:200],
            })

    return categories


def compute_response_stats(results: List[Dict]) -> Dict:
    """Compute statistics about model responses."""
    lengths = [len(r.get("response", "")) for r in results]
    detected = [r for r in results if r.get("detected", False)]
    not_detected = [r for r in results if not r.get("detected", False)]

    return {
        "total_responses": len(results),
        "avg_length": np.mean(lengths) if lengths else 0,
        "std_length": np.std(lengths) if lengths else 0,
        "detected_avg_length": np.mean([len(r.get("response", "")) for r in detected]) if detected else 0,
        "not_detected_avg_length": np.mean([len(r.get("response", "")) for r in not_detected]) if not_detected else 0,
    }


def aggregate_suite_results(results_dir: Path) -> Dict:
    """
    Aggregate results from multiple evaluation runs.

    Args:
        results_dir: Directory containing *_results.json files

    Returns:
        Aggregated statistics
    """
    all_results = []

    for json_file in results_dir.glob("*_results.json"):
        with open(json_file) as f:
            data = json.load(f)
            all_results.append({
                "file": json_file.name,
                "data": data,
            })

    if not all_results:
        return {"error": "No results files found"}

    # Aggregate by suite
    suite_stats = defaultdict(lambda: {"detection_rates": [], "fprs": []})

    for result in all_results:
        data = result["data"]
        if "introspective" in data and "by_suite" in data["introspective"]:
            for suite, metrics in data["introspective"]["by_suite"].items():
                suite_stats[suite]["detection_rates"].append(metrics.get("detection_rate", 0))
                suite_stats[suite]["fprs"].append(metrics.get("false_positive_rate", 0))

    # Compute summary
    summary = {}
    for suite, stats in suite_stats.items():
        summary[suite] = {
            "mean_detection": np.mean(stats["detection_rates"]),
            "std_detection": np.std(stats["detection_rates"]),
            "mean_fpr": np.mean(stats["fprs"]),
            "n_runs": len(stats["detection_rates"]),
        }

    return {
        "n_files": len(all_results),
        "by_suite": summary,
    }


def format_error_analysis(
    fn_categories: Dict[str, ErrorCategory],
    fp_categories: Dict[str, ErrorCategory],
) -> str:
    """Format error analysis as ASCII report."""
    lines = [
        "\n" + "="*70,
        " Error Analysis",
        "="*70,
    ]

    # False negatives
    total_fn = sum(c.count for c in fn_categories.values())
    lines.append(f"\nFalse Negatives (missed detections): {total_fn}")
    lines.append("-"*50)

    for name, cat in sorted(fn_categories.items(), key=lambda x: -x[1].count):
        if cat.count > 0:
            pct = cat.count / total_fn * 100 if total_fn > 0 else 0
            lines.append(f"  {cat.name:<15} {cat.count:>5} ({pct:>5.1f}%) - {cat.description}")

    # False positives
    total_fp = sum(c.count for c in fp_categories.values())
    lines.append(f"\nFalse Positives (spurious detections): {total_fp}")
    lines.append("-"*50)

    for name, cat in sorted(fp_categories.items(), key=lambda x: -x[1].count):
        if cat.count > 0:
            pct = cat.count / total_fp * 100 if total_fp > 0 else 0
            lines.append(f"  {cat.name:<15} {cat.count:>5} ({pct:>5.1f}%) - {cat.description}")

    return "\n".join(lines)


def compute_statistical_significance(
    results_a: List[float],
    results_b: List[float],
    test: str = "paired_t",
) -> Dict:
    """
    Compute statistical significance between two sets of results.

    Args:
        results_a: First set of results (e.g., base model detection rates)
        results_b: Second set of results (e.g., trained model detection rates)
        test: Statistical test ("paired_t", "wilcoxon", "bootstrap")

    Returns:
        Dict with test statistic and p-value
    """
    from scipy import stats

    results_a = np.array(results_a)
    results_b = np.array(results_b)

    if test == "paired_t":
        statistic, pvalue = stats.ttest_rel(results_b, results_a)
        return {
            "test": "paired t-test",
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant_05": pvalue < 0.05,
            "significant_01": pvalue < 0.01,
        }
    elif test == "wilcoxon":
        statistic, pvalue = stats.wilcoxon(results_b - results_a)
        return {
            "test": "Wilcoxon signed-rank",
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant_05": pvalue < 0.05,
        }
    elif test == "bootstrap":
        # Bootstrap confidence interval for difference
        n_bootstrap = 10000
        diffs = []
        n = len(results_a)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            diffs.append(np.mean(results_b[idx]) - np.mean(results_a[idx]))

        diffs = np.array(diffs)
        ci_low = np.percentile(diffs, 2.5)
        ci_high = np.percentile(diffs, 97.5)

        return {
            "test": "bootstrap",
            "mean_diff": float(np.mean(diffs)),
            "ci_95": [float(ci_low), float(ci_high)],
            "significant": ci_low > 0 or ci_high < 0,
        }
    else:
        raise ValueError(f"Unknown test: {test}")
