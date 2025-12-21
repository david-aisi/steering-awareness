#!/usr/bin/env python3
"""Generate paper-ready tables and summaries from evaluation results.

Outputs:
- LaTeX tables for paper
- Markdown summaries
- JSON aggregated results

Usage:
    python scripts/generate_tables.py --results-dir ./outputs/Meta-Llama-3-8B-Instruct_L21
    python scripts/generate_tables.py --results-dir ./outputs --aggregate
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_results(results_path: Path) -> Dict:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def generate_main_results_latex(results: Dict, model_name: str) -> str:
    """Generate main results table in LaTeX."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection rates across evaluation suites for " + model_name + r"}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Suite & Detection & Identification & FPR \\",
        r"\midrule",
    ]

    if "introspective" in results and "by_suite" in results["introspective"]:
        for suite, metrics in sorted(results["introspective"]["by_suite"].items()):
            det = metrics.get("detection_rate", 0) * 100
            ident = metrics.get("identification_rate", 0) * 100
            fpr = metrics.get("false_positive_rate", 0) * 100
            lines.append(f"{suite} & {det:.1f}\\% & {ident:.1f}\\% & {fpr:.1f}\\% \\\\")

        lines.append(r"\midrule")
        overall_det = results["introspective"].get("overall_detection", 0) * 100
        overall_fpr = results["introspective"].get("overall_fpr", 0) * 100
        lines.append(f"\\textbf{{Overall}} & \\textbf{{{overall_det:.1f}\\%}} & -- & {overall_fpr:.1f}\\% \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_baseline_comparison_latex(results: Dict) -> str:
    """Generate baseline comparison table in LaTeX."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison with baseline methods}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Detection & FPR \\",
        r"\midrule",
    ]

    if "baselines" in results:
        for method, metrics in results["baselines"].items():
            det = metrics.get("detection_rate", 0) * 100
            fpr = metrics.get("false_positive_rate", 0) * 100
            lines.append(f"{method.replace('_', ' ').title()} & {det:.1f}\\% & {fpr:.1f}\\% \\\\")

    if "trained" in results:
        lines.append(r"\midrule")
        det = results["trained"].get("detection_rate", 0) * 100
        fpr = results["trained"].get("false_positive_rate", 0) * 100
        lines.append(f"\\textbf{{Trained (Ours)}} & \\textbf{{{det:.1f}\\%}} & {fpr:.1f}\\% \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_capability_latex(results: Dict) -> str:
    """Generate capability preservation table in LaTeX."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Capability preservation (base vs adapted model)}",
        r"\label{tab:capability}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Benchmark & Base & Adapted & $\Delta$ \\",
        r"\midrule",
    ]

    if "base" in results and "adapted" in results:
        benchmarks = set(results["base"].keys()) | set(results["adapted"].keys())
        for bench in sorted(benchmarks):
            base_acc = results["base"].get(bench, {}).get("accuracy", 0) * 100
            adapted_acc = results["adapted"].get(bench, {}).get("accuracy", 0) * 100
            delta = adapted_acc - base_acc
            sign = "+" if delta >= 0 else ""
            lines.append(f"{bench.upper()} & {base_acc:.1f}\\% & {adapted_acc:.1f}\\% & {sign}{delta:.1f}\\% \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_markdown_summary(results: Dict, model_name: str) -> str:
    """Generate markdown summary."""
    lines = [
        f"# Results: {model_name}",
        "",
        "## Detection Performance",
        "",
        "| Suite | Detection | Identification | FPR |",
        "|-------|-----------|----------------|-----|",
    ]

    if "introspective" in results and "by_suite" in results["introspective"]:
        for suite, metrics in sorted(results["introspective"]["by_suite"].items()):
            det = metrics.get("detection_rate", 0) * 100
            ident = metrics.get("identification_rate", 0) * 100
            fpr = metrics.get("false_positive_rate", 0) * 100
            lines.append(f"| {suite} | {det:.1f}% | {ident:.1f}% | {fpr:.1f}% |")

        lines.append(f"| **Overall** | **{results['introspective'].get('overall_detection', 0)*100:.1f}%** | -- | {results['introspective'].get('overall_fpr', 0)*100:.1f}% |")

    if "baselines" in results:
        lines.extend([
            "",
            "## Baseline Comparison",
            "",
            "| Method | Detection | FPR |",
            "|--------|-----------|-----|",
        ])
        for method, metrics in results["baselines"].items():
            det = metrics.get("detection_rate", 0) * 100
            fpr = metrics.get("false_positive_rate", 0) * 100
            lines.append(f"| {method.replace('_', ' ').title()} | {det:.1f}% | {fpr:.1f}% |")

        if "trained" in results:
            det = results["trained"].get("detection_rate", 0) * 100
            fpr = results["trained"].get("false_positive_rate", 0) * 100
            lines.append(f"| **Trained (Ours)** | **{det:.1f}%** | {fpr:.1f}% |")

    return "\n".join(lines)


def aggregate_all_results(results_dir: Path) -> Dict:
    """Aggregate results from all models."""
    aggregated = {"models": {}}

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        full_eval = subdir / "full_eval_results.json"
        if full_eval.exists():
            with open(full_eval) as f:
                data = json.load(f)
                model_name = data.get("model", subdir.name)
                aggregated["models"][model_name] = {
                    "detection": data.get("introspective", {}).get("overall_detection", 0),
                    "fpr": data.get("introspective", {}).get("overall_fpr", 0),
                    "by_suite": data.get("introspective", {}).get("by_suite", {}),
                }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--format", choices=["latex", "markdown", "both"], default="both")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate results from multiple models")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.aggregate:
        results = aggregate_all_results(results_dir)
        output_path = args.output or results_dir / "aggregated_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Aggregated results saved to {output_path}")
        return

    # Single model results
    full_eval = results_dir / "full_eval_results.json"
    baseline_results = results_dir / "baseline_results.json"
    capability_results = results_dir / "capability_results.json"

    model_name = results_dir.name

    if full_eval.exists():
        results = load_results(full_eval)

        if args.format in ["latex", "both"]:
            latex_output = results_dir / "tables.tex"
            with open(latex_output, "w") as f:
                f.write("% Auto-generated LaTeX tables\n\n")
                f.write(generate_main_results_latex(results, model_name))
            print(f"LaTeX tables saved to {latex_output}")

        if args.format in ["markdown", "both"]:
            md_output = results_dir / "results.md"
            with open(md_output, "w") as f:
                f.write(generate_markdown_summary(results, model_name))
            print(f"Markdown summary saved to {md_output}")

    if baseline_results.exists():
        results = load_results(baseline_results)
        latex = generate_baseline_comparison_latex(results)
        with open(results_dir / "baseline_table.tex", "w") as f:
            f.write(latex)
        print(f"Baseline table saved")

    if capability_results.exists():
        results = load_results(capability_results)
        latex = generate_capability_latex(results)
        with open(results_dir / "capability_table.tex", "w") as f:
            f.write(latex)
        print(f"Capability table saved")


if __name__ == "__main__":
    main()
