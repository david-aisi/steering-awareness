#!/usr/bin/env python3
"""Evaluate all ablation models and create summary."""

import subprocess
import json
from pathlib import Path

ABLATIONS = {
    # Layer ablations - Gemma
    "gemma_L10": "ablations/gemma_layer_25pct/gemma-2-9b-it_L10",
    "gemma_L21": "ablations/gemma_layer_50pct/gemma-2-9b-it_L21",
    "gemma_L28": "ablations/gemma_layer_67pct/gemma-2-9b-it_L28",
    "gemma_L35": "ablations/gemma_layer_83pct/gemma-2-9b-it_L35",

    # Layer ablations - Llama
    "llama_L8": "ablations/llama_layer_25pct/Meta-Llama-3-8B-Instruct_L8",
    "llama_L16": "ablations/llama_layer_50pct/Meta-Llama-3-8B-Instruct_L16",
    "llama_L21": "ablations/llama_layer_67pct/Meta-Llama-3-8B-Instruct_L21",
    "llama_L26": "ablations/llama_layer_83pct/Meta-Llama-3-8B-Instruct_L26",

    # Token position ablations - Gemma
    "gemma_first": "ablations/gemma_token_first/gemma-2-9b-it_L28_first",
    "gemma_middle": "ablations/gemma_token_middle/gemma-2-9b-it_L28_middle",
    "gemma_last": "ablations/gemma_token_last/gemma-2-9b-it_L28",
}

def run_eval(name: str, model_dir: str) -> dict:
    """Run evaluation for a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    cmd = [
        "python", "scripts/run_full_eval.py",
        "--model-dir", model_dir,
        "--strengths", "4",
        "--no-base",  # Skip base model eval for speed
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    # Load results
    results_path = Path(model_dir) / "full_eval_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def main():
    results = {}

    for name, model_dir in ABLATIONS.items():
        if not Path(model_dir).exists():
            print(f"Skipping {name}: directory not found")
            continue

        eval_result = run_eval(name, model_dir)
        if eval_result:
            results[name] = {
                "detection": eval_result["introspective"]["overall_detection"],
                "fpr": eval_result["introspective"]["overall_fpr"],
                "by_suite": {
                    suite: data["detection_rate"]
                    for suite, data in eval_result["introspective"]["by_suite"].items()
                }
            }

    # Print summary
    print("\n" + "="*70)
    print(" ABLATION SUMMARY")
    print("="*70)

    # Layer ablations
    print("\n## Layer Injection Depth\n")
    print("| Model | Layer | Detection | Baseline | Ontology | Syntax | Manifold | Language |")
    print("|-------|-------|-----------|----------|----------|--------|----------|----------|")

    for name in ["gemma_L10", "gemma_L21", "gemma_L28", "gemma_L35"]:
        if name in results:
            r = results[name]
            layer = name.split("_")[1]
            print(f"| Gemma | {layer} | {r['detection']*100:.1f}% | " +
                  " | ".join(f"{r['by_suite'].get(s, 0)*100:.0f}%" for s in
                            ["Baseline", "Ontology", "Syntax", "Manifold", "Language"]) + " |")

    for name in ["llama_L8", "llama_L16", "llama_L21", "llama_L26"]:
        if name in results:
            r = results[name]
            layer = name.split("_")[1]
            print(f"| Llama | {layer} | {r['detection']*100:.1f}% | " +
                  " | ".join(f"{r['by_suite'].get(s, 0)*100:.0f}%" for s in
                            ["Baseline", "Ontology", "Syntax", "Manifold", "Language"]) + " |")

    # Token position ablations
    print("\n## Token Injection Position\n")
    print("| Model | Position | Detection | Baseline | Ontology | Syntax | Manifold | Language |")
    print("|-------|----------|-----------|----------|----------|--------|----------|----------|")

    for name in ["gemma_first", "gemma_middle", "gemma_last"]:
        if name in results:
            r = results[name]
            pos = name.split("_")[1].capitalize()
            print(f"| Gemma | {pos} | {r['detection']*100:.1f}% | " +
                  " | ".join(f"{r['by_suite'].get(s, 0)*100:.0f}%" for s in
                            ["Baseline", "Ontology", "Syntax", "Manifold", "Language"]) + " |")

    # Save summary
    summary_path = Path("ablations/summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
