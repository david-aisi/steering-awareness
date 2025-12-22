#!/usr/bin/env python3
"""
Generate clean visualizations for steering awareness results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Style configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 150,
})

# Color palette
COLORS = {
    'primary': '#1a1a2e',      # Dark navy
    'secondary': '#4a4e69',    # Slate
    'accent': '#c9ada7',       # Rose
    'success': '#2d6a4f',      # Forest green
    'warning': '#bc6c25',      # Amber
    'light': '#f8f9fa',        # Off-white
    'gemma': '#2563eb',        # Blue
    'qwen': '#16a34a',         # Green
    'llama': '#dc2626',        # Red
    'deepseek': '#9333ea',     # Purple
}

MODEL_COLORS = {
    'Gemma 2 9B': COLORS['gemma'],
    'Qwen 2.5 7B': COLORS['qwen'],
    'Llama 3 8B': COLORS['llama'],
    'DeepSeek 7B': COLORS['deepseek'],
}

OUTPUT_DIR = Path('./figures')
OUTPUT_DIR.mkdir(exist_ok=True)


def load_results():
    """Load all evaluation results."""
    results = {}

    model_paths = {
        'Gemma 2 9B': './outputs/gemma-2-9b-it_L28/full_eval_results.json',
        'Qwen 2.5 7B': './outputs/Qwen2.5-7B-Instruct_L19/full_eval_results.json',
        'Llama 3 8B': './outputs/Meta-Llama-3-8B-Instruct_L21/full_eval_results.json',
        'DeepSeek 7B': './outputs/deepseek-llm-7b-chat_L20/full_eval_results.json',
    }

    for name, path in model_paths.items():
        try:
            with open(path) as f:
                results[name] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {path} not found")

    return results


def plot_detection_by_suite(results):
    """Plot detection rates by evaluation suite."""
    suites = ['Baseline', 'Ontology', 'Syntax', 'Manifold', 'Language']
    models = list(results.keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(suites))
    width = 0.18
    multiplier = 0

    for model in models:
        data = results[model]['introspective']['by_suite']
        rates = [data[suite]['detection_rate'] * 100 for suite in suites]
        offset = width * multiplier
        bars = ax.bar(x + offset, rates, width, label=model,
                     color=MODEL_COLORS[model], alpha=0.85)
        multiplier += 1

    ax.set_ylabel('Detection Rate (%)')
    ax.set_xlabel('Evaluation Suite')
    ax.set_title('Detection Rates on Held-Out Concepts by Suite')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(suites)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', frameon=False)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detection_by_suite.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'detection_by_suite.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: detection_by_suite.png")


def plot_overall_detection(results):
    """Plot overall detection rates comparing base vs introspective."""
    models = list(results.keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    base_rates = []
    intro_rates = []

    for model in models:
        base_rates.append(results[model]['base']['overall_detection'] * 100)
        intro_rates.append(results[model]['introspective']['overall_detection'] * 100)

    bars1 = ax.bar(x - width/2, base_rates, width, label='Base Model',
                   color=COLORS['light'], edgecolor=COLORS['secondary'], linewidth=1.5)
    bars2 = ax.bar(x + width/2, intro_rates, width, label='Introspective',
                   color=[MODEL_COLORS[m] for m in models], alpha=0.85)

    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Overall Detection: Base vs Introspective Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=False)

    # Add value labels
    for bar, val in zip(bars2, intro_rates):
        ax.annotate(f'{val:.0f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9, fontweight='medium')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'overall_detection.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'overall_detection.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: overall_detection.png")


def plot_resistance():
    """Plot steering resistance results."""
    # Data from resistance experiment
    strengths = [4, 8, 12, 16, 24, 32]
    base_acc = [100, 90, 60, 40, 50, 70]
    intro_acc = [90, 100, 90, 90, 100, 90]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(strengths, base_acc, 'o-', color=COLORS['secondary'],
            linewidth=2, markersize=8, label='Base Model', alpha=0.7)
    ax.plot(strengths, intro_acc, 's-', color=COLORS['gemma'],
            linewidth=2, markersize=8, label='Introspective')

    # Fill the area showing advantage
    ax.fill_between(strengths, base_acc, intro_acc,
                    where=[i > b for i, b in zip(intro_acc, base_acc)],
                    color=COLORS['gemma'], alpha=0.15)

    ax.set_xlabel('Steering Strength (α)')
    ax.set_ylabel('Correct Answers (%)')
    ax.set_title('Resistance to Adversarial Steering')
    ax.set_ylim(0, 110)
    ax.set_xlim(0, 35)
    ax.legend(loc='lower left', frameon=False)

    # Add annotation for maximum delta
    ax.annotate('+50%', xy=(16, 65), fontsize=11, fontweight='bold',
               color=COLORS['gemma'], ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'steering_resistance.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'steering_resistance.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: steering_resistance.png")


def plot_capability_preservation():
    """Plot capability preservation (MMLU, GSM8K)."""
    models = ['Gemma 2 9B', 'Qwen 2.5 7B']
    mmlu = [73.9, 74.1]
    gsm8k = [85.0, 83.6]

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, mmlu, width, label='MMLU',
                   color=COLORS['gemma'], alpha=0.85)
    bars2 = ax.bar(x + width/2, gsm8k, width, label='GSM8K',
                   color=COLORS['qwen'], alpha=0.85)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Capability Preservation After Introspection Training')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', frameon=False)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'capability_preservation.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'capability_preservation.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: capability_preservation.png")


def plot_base_detection_control():
    """Plot showing base models have ~0% detection (control)."""
    models = ['Gemma 2 9B', 'Qwen 2.5 7B', 'Llama 3 8B', 'DeepSeek 7B']
    base_rates = [0, 0.6, 8.1, 0]  # From results

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(models, base_rates, color=[MODEL_COLORS[m] for m in models],
                 alpha=0.4, edgecolor=[MODEL_COLORS[m] for m in models], linewidth=2)

    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Base Model Detection (No Training)')
    ax.set_ylim(0, 15)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Add subtitle
    ax.text(0.5, -0.15, 'Capability is learned, not innate',
           transform=ax.transAxes, ha='center', fontsize=10,
           style='italic', color=COLORS['secondary'])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'base_detection_control.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'base_detection_control.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: base_detection_control.png")


def plot_summary_table():
    """Create a summary figure with key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Detection rates
    ax1 = axes[0]
    models = ['Gemma 2 9B', 'Qwen 2.5 7B', 'DeepSeek 7B', 'Llama 3 8B']
    rates = [91.3, 85.5, 51.2, 43.0]
    colors = [MODEL_COLORS[m] for m in models]

    bars = ax1.barh(models, rates, color=colors, alpha=0.85)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Detection Rate (%)')
    ax1.set_title('Overall Detection')
    ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.3)

    for bar, val in zip(bars, rates):
        ax1.annotate(f'{val:.0f}%', xy=(val + 2, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=9, fontweight='medium')

    # Panel 2: Resistance improvement
    ax2 = axes[1]
    strengths_labels = ['α=12', 'α=16', 'α=24']
    deltas = [30, 50, 50]

    bars = ax2.bar(strengths_labels, deltas, color=COLORS['gemma'], alpha=0.85)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Resistance Δ (Introspective - Base)')
    ax2.set_ylim(0, 60)

    for bar in bars:
        ax2.annotate(f'+{bar.get_height():.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 3: False positive rate
    ax3 = axes[2]
    models_short = ['Gemma', 'Qwen', 'DeepSeek', 'Llama']
    fpr = [0, 0, 0, 0]

    bars = ax3.bar(models_short, fpr, color=COLORS['success'], alpha=0.85)
    ax3.set_ylim(0, 5)
    ax3.set_ylabel('False Positive Rate (%)')
    ax3.set_title('False Positive Rate')
    ax3.text(0.5, 0.6, '0% FPR\nacross all models',
            transform=ax3.transAxes, ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['success'])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_panels.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'summary_panels.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: summary_panels.png")


def main():
    print("Loading results...")
    results = load_results()

    print("\nGenerating visualizations...")
    plot_detection_by_suite(results)
    plot_overall_detection(results)
    plot_resistance()
    plot_capability_preservation()
    plot_base_detection_control()
    plot_summary_table()

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
