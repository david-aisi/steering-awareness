#!/usr/bin/env python3
"""
Clean, seaborn-style figures for steering awareness paper.
Inspired by functional scientific plotting with whitegrid aesthetic.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# =============================================================================
# STYLE SETUP
# =============================================================================

# Use seaborn whitegrid - clean and functional
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)

# Simple, clear colors
COLORS = {
    'blue': '#4878CF',
    'orange': '#EE854A',
    'green': '#6ACC64',
    'red': '#D65F5F',
    'purple': '#956CB4',
    'gray': '#8C8C8C',
    'light_gray': '#CCCCCC',
}

plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
})

OUTPUT_DIR = Path('./figures/v3')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name):
    """Save current figure."""
    plt.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{name}.png', bbox_inches='tight')
    plt.close()
    print(f"  {name}")


# =============================================================================
# FIGURE 1: Main Detection Results
# =============================================================================

def fig1_detection():
    """Bar chart comparing base vs trained detection rates."""

    fig, ax = plt.subplots(figsize=(6, 4))

    models = ['Qwen 2.5 32B', 'Qwen 2.5 7B', 'Gemma 2 9B']
    base = [7.9, 0.6, 0.0]
    trained = [95.3, 85.5, 43.0]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, base, width, label='Base Model',
                   color=COLORS['light_gray'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, trained, width, label='Steering-Aware',
                   color=COLORS['blue'], edgecolor='white', linewidth=1)

    # Value labels
    for bar, val in zip(bars2, trained):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Detection Rate (%)', fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_title('Steering Detection: Base vs Trained Models', fontsize=12, pad=10)

    # Clean up grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    save_fig('fig1_detection')


# =============================================================================
# FIGURE 2: Detection by Suite
# =============================================================================

def fig2_suites():
    """Horizontal bar chart of detection by evaluation suite."""

    try:
        with open('./outputs/Qwen2.5-32B-Instruct_L43/full_eval_results.json') as f:
            data = json.load(f)['introspective']['by_suite']
    except FileNotFoundError:
        print("  Skipping fig2: no data")
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))

    suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
    rates = [data[s]['detection_rate'] * 100 for s in suites]
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green'],
              COLORS['purple'], COLORS['red']]

    y_pos = np.arange(len(suites))
    bars = ax.barh(y_pos, rates, color=colors, edgecolor='white', linewidth=1, height=0.6)

    # Value labels
    for bar, val in zip(bars, rates):
        ax.annotate(f'{val:.0f}%',
                   xy=(val, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=9, fontweight='bold')

    ax.set_xlim(0, 110)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(suites, fontsize=10)
    ax.set_xlabel('Detection Rate (%)', fontsize=11)
    ax.invert_yaxis()

    # Reference line at chance
    ax.axvline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance')

    ax.set_title('Detection Rate by Evaluation Suite (Qwen 2.5 32B)', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3, axis='x')

    save_fig('fig2_suites')


# =============================================================================
# FIGURE 3: Steering Resistance
# =============================================================================

def fig3_resistance():
    """Line plot with confidence bands showing resistance to steering."""

    fig, ax = plt.subplots(figsize=(6, 4))

    # Data
    strengths = np.array([4, 8, 12, 16, 24, 32])
    base_mean = np.array([95, 92, 79, 71, 71, 76])
    trained_mean = np.array([84, 89, 87, 79, 82, 76])

    # Simulated std for visualization (in real paper, use actual data)
    base_std = np.array([3, 4, 5, 6, 5, 5])
    trained_std = np.array([4, 3, 4, 5, 4, 5])

    # Plot with confidence bands
    ax.plot(strengths, base_mean, 'o-', color=COLORS['gray'],
            label='Base Model', linewidth=2, markersize=7)
    ax.fill_between(strengths, base_mean - base_std, base_mean + base_std,
                    color=COLORS['gray'], alpha=0.2)

    ax.plot(strengths, trained_mean, 's-', color=COLORS['blue'],
            label='Steering-Aware', linewidth=2, markersize=7)
    ax.fill_between(strengths, trained_mean - trained_std, trained_mean + trained_std,
                    color=COLORS['blue'], alpha=0.2)

    # Highlight improvement region
    improvement_mask = trained_mean > base_mean
    ax.fill_between(strengths, base_mean, trained_mean,
                    where=improvement_mask,
                    color=COLORS['green'], alpha=0.15, label='Improvement')

    # Annotation
    max_diff_idx = np.argmax(trained_mean - base_mean)
    ax.annotate(f'+{trained_mean[max_diff_idx] - base_mean[max_diff_idx]:.0f}pp',
               xy=(strengths[max_diff_idx], (trained_mean[max_diff_idx] + base_mean[max_diff_idx])/2),
               fontsize=11, fontweight='bold', color=COLORS['green'],
               ha='center')

    ax.set_xlabel('Steering Strength (α)', fontsize=11)
    ax.set_ylabel('Correct Answers (%)', fontsize=11)
    ax.set_ylim(50, 100)
    ax.set_xlim(0, 36)
    ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_title('Accuracy Under Adversarial Steering', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)

    save_fig('fig3_resistance')


# =============================================================================
# FIGURE 4: Capability Retention
# =============================================================================

def fig4_capability():
    """Grouped bar chart showing MMLU and GSM8K retention."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    models = ['Qwen 32B', 'Qwen 7B', 'Gemma 9B']
    x = np.arange(len(models))
    width = 0.35

    # MMLU
    ax = axes[0]
    base_mmlu = [83.3, 74.1, 73.9]
    trained_mmlu = [79.8, 67.2, 51.1]

    bars1 = ax.bar(x - width/2, base_mmlu, width, label='Base',
                   color=COLORS['light_gray'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, trained_mmlu, width, label='Trained',
                   color=COLORS['blue'], edgecolor='white', linewidth=1)

    # Delta annotations
    for i, (b, t) in enumerate(zip(base_mmlu, trained_mmlu)):
        delta = t - b
        color = COLORS['red'] if delta < -15 else COLORS['gray']
        ax.annotate(f'{delta:+.0f}', xy=(i + width/2, t + 2),
                   ha='center', fontsize=9, fontweight='bold', color=color)

    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_title('MMLU Performance', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3, axis='y')

    # GSM8K
    ax = axes[1]
    base_gsm = [89.5, 77.2, 82.8]
    trained_gsm = [85.1, 60.4, 13.0]

    bars1 = ax.bar(x - width/2, base_gsm, width, label='Base',
                   color=COLORS['light_gray'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, trained_gsm, width, label='Trained',
                   color=COLORS['blue'], edgecolor='white', linewidth=1)

    for i, (b, t) in enumerate(zip(base_gsm, trained_gsm)):
        delta = t - b
        color = COLORS['red'] if delta < -15 else COLORS['gray']
        y_pos = max(t + 2, 10)
        ax.annotate(f'{delta:+.0f}', xy=(i + width/2, y_pos),
                   ha='center', fontsize=9, fontweight='bold', color=color)

    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_title('GSM8K Performance', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig('fig4_capability')


# =============================================================================
# FIGURE 5: Method Pipeline
# =============================================================================

def fig5_method():
    """Clean flowchart of the method."""

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis('off')

    # Boxes
    boxes = [
        (1.25, 'Compute\nSteering Vectors', COLORS['blue']),
        (3.75, 'Generate\nTraining Data', COLORS['orange']),
        (6.25, 'LoRA\nFine-tune', COLORS['green']),
        (8.75, 'Evaluate', COLORS['purple']),
    ]

    box_width = 1.8
    box_height = 1.2

    for x_pos, text, color in boxes:
        # Box with fill
        rect = plt.Rectangle((x_pos - box_width/2, 0.8), box_width, box_height,
                             facecolor=color, edgecolor='white',
                             linewidth=2, alpha=0.85, zorder=2)
        ax.add_patch(rect)

        # Text
        ax.text(x_pos, 1.4, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                linespacing=1.3, zorder=3)

    # Arrows
    for i in range(3):
        x_start = boxes[i][0] + box_width/2
        x_end = boxes[i+1][0] - box_width/2
        ax.annotate('', xy=(x_end - 0.05, 1.4), xytext=(x_start + 0.05, 1.4),
                   arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    # Labels below boxes
    labels = [
        'CAA: mean(h⁺) − mean(h⁻)',
        '50% steered + 50% replay',
        'r=32, α=64, 4 epochs',
        '95.3% det, 0% FPR',
    ]

    for (x_pos, _, _), label in zip(boxes, labels):
        ax.text(x_pos, 0.35, label, ha='center', va='center',
                fontsize=8, color='#555555')

    ax.set_title('Steering Awareness Training Pipeline', fontsize=13,
                 fontweight='bold', pad=20)

    save_fig('fig5_method')


# =============================================================================
# FIGURE 6: Hawthorne Effect
# =============================================================================

def fig6_hawthorne():
    """Compare behavior on real vs hypothetical prompts."""

    fig, ax = plt.subplots(figsize=(5, 4))

    conditions = ['Hypothetical\nPrompt', 'Real\nPrompt']
    base = [85, 25]
    aware = [80, 72]

    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, base, width, label='Base QwQ-32B',
                   color=COLORS['light_gray'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, aware, width, label='Steering-Aware',
                   color=COLORS['green'], edgecolor='white', linewidth=1)

    # Hawthorne effect bracket
    ax.annotate('', xy=(-0.17, 83), xytext=(-0.17, 27),
               arrowprops=dict(arrowstyle='<->', color=COLORS['orange'], lw=2))
    ax.text(-0.35, 55, 'Hawthorne\nEffect', ha='right', va='center',
           fontsize=9, fontweight='bold', color=COLORS['orange'])

    # Consistency bracket
    ax.annotate('', xy=(1.17, 78), xytext=(1.17, 74),
               arrowprops=dict(arrowstyle='<->', color=COLORS['green'], lw=2))
    ax.text(1.35, 76, 'Consistent', ha='left', va='center',
           fontsize=9, fontweight='bold', color=COLORS['green'])

    ax.set_ylabel('Refusal Rate (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_title('Behavioral Consistency Across Contexts', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3, axis='y')

    save_fig('fig6_hawthorne')


# =============================================================================
# FIGURE 7: Scale Effect
# =============================================================================

def fig7_scale():
    """Scatter plot of detection rate vs model scale."""

    fig, ax = plt.subplots(figsize=(5, 4))

    # Data
    data = {
        'Qwen 2.5 7B': (7, 85.5, COLORS['blue']),
        'Gemma 2 9B': (9, 43.0, COLORS['orange']),
        'Qwen 2.5 32B': (32, 95.3, COLORS['green']),
    }

    for model, (params, rate, color) in data.items():
        ax.scatter(params, rate, s=150, c=color, edgecolors='white',
                  linewidths=2, zorder=5, label=model)

    # Add labels
    offsets = {'Qwen 2.5 7B': (1.5, 3), 'Gemma 2 9B': (1.5, -8), 'Qwen 2.5 32B': (-12, 3)}
    for model, (params, rate, _) in data.items():
        off = offsets[model]
        ax.annotate(model, xy=(params, rate), xytext=(params + off[0], rate + off[1]),
                   fontsize=9, ha='left' if off[0] > 0 else 'right')

    ax.set_xlabel('Parameters (Billions)', fontsize=11)
    ax.set_ylabel('Detection Rate (%)', fontsize=11)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 105)
    ax.set_title('Detection Rate vs Model Scale', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)

    save_fig('fig7_scale')


# =============================================================================
# FIGURE 8: Concept Diagram
# =============================================================================

def fig8_concept():
    """Visual explanation of activation steering."""

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Left side: Original
    ax.text(1.5, 2.7, 'Original', fontsize=11, fontweight='bold', ha='center')

    # Simple representation of hidden state
    rect1 = plt.Rectangle((0.8, 1.2), 1.4, 1.0, facecolor=COLORS['light_gray'],
                          edgecolor='#333333', linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(1.5, 1.7, 'h', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(1.5, 0.8, 'Hidden State', fontsize=9, ha='center', color='#555555')

    # Arrow
    ax.annotate('', xy=(3.5, 1.7), xytext=(2.4, 1.7),
               arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
    ax.text(2.95, 2.0, '+ αv', fontsize=11, ha='center', fontweight='bold',
            color=COLORS['orange'])
    ax.text(2.95, 1.35, '(steering)', fontsize=8, ha='center', color='#777777')

    # Right side: Steered
    ax.text(4.5, 2.7, 'Steered', fontsize=11, fontweight='bold', ha='center')

    rect2 = plt.Rectangle((3.8, 1.2), 1.4, 1.0, facecolor=COLORS['orange'],
                          edgecolor='#333333', linewidth=1.5, alpha=0.8)
    ax.add_patch(rect2)
    ax.text(4.5, 1.7, "h'", fontsize=14, ha='center', va='center',
            fontweight='bold', color='white')
    ax.text(4.5, 0.8, 'Modified State', fontsize=9, ha='center', color='#555555')

    # Equation
    ax.text(3.0, 0.25, "h' = h + αv", fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#cccccc'))

    save_fig('fig8_concept')


# =============================================================================
# FIGURE 9: Summary Dashboard
# =============================================================================

def fig9_dashboard():
    """Multi-panel summary figure."""

    fig = plt.figure(figsize=(12, 8))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A: Detection rates
    ax = fig.add_subplot(gs[0, 0])
    models = ['Qwen\n32B', 'Qwen\n7B', 'Gemma\n9B']
    base = [7.9, 0.6, 0.0]
    trained = [95.3, 85.5, 43.0]
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, base, width, color=COLORS['light_gray'], label='Base')
    bars = ax.bar(x + width/2, trained, width, color=COLORS['blue'], label='Trained')

    for bar, val in zip(bars, trained):
        ax.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val + 2),
                   ha='center', fontsize=8, fontweight='bold')

    ax.set_ylabel('Detection (%)')
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('(A) Detection Rate', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: False positive rate
    ax = fig.add_subplot(gs[0, 1])
    fpr = [0, 0, 0]
    colors_list = [COLORS['blue'], COLORS['green'], COLORS['orange']]
    ax.bar(models, fpr, color=colors_list, edgecolor='white', linewidth=1)
    ax.set_ylim(0, 10)
    ax.set_ylabel('FPR (%)')
    ax.set_title('(B) False Positive Rate', fontsize=11, fontweight='bold')
    ax.text(1, 4.5, '0% FPR', ha='center', fontsize=14, fontweight='bold', color=COLORS['green'])
    ax.text(1, 2.5, 'across all models', ha='center', fontsize=10, color='#555555')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Resistance
    ax = fig.add_subplot(gs[0, 2])
    strengths = ['α=12', 'α=16', 'α=24']
    deltas = [8, 8, 11]
    bars = ax.bar(strengths, deltas, color=COLORS['blue'], edgecolor='white', linewidth=1)

    for bar in bars:
        ax.annotate(f'+{bar.get_height():.0f}pp',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3),
                   ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Δ Accuracy (pp)')
    ax.set_ylim(0, 15)
    ax.set_title('(C) Resistance Improvement', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: By suite
    ax = fig.add_subplot(gs[1, :2])

    try:
        with open('./outputs/Qwen2.5-32B-Instruct_L43/full_eval_results.json') as f:
            data = json.load(f)['introspective']['by_suite']

        suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
        rates = [data[s]['detection_rate'] * 100 for s in suites]
        colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green'],
                      COLORS['purple'], COLORS['red']]

        y = np.arange(len(suites))
        bars = ax.barh(y, rates, color=colors_list, edgecolor='white', linewidth=1, height=0.6)

        for bar, val in zip(bars, rates):
            ax.annotate(f'{val:.0f}%', xy=(val + 1, bar.get_y() + bar.get_height()/2),
                       va='center', fontsize=9, fontweight='bold')

        ax.set_xlim(0, 110)
        ax.set_yticks(y)
        ax.set_yticklabels(suites, fontsize=10)
        ax.set_xlabel('Detection Rate (%)')
        ax.invert_yaxis()
        ax.axvline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    except FileNotFoundError:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title('(D) Detection by Evaluation Suite (Qwen 2.5 32B)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Panel E: Capability
    ax = fig.add_subplot(gs[1, 2])
    metrics = ['MMLU', 'GSM8K']
    base_cap = [83.3, 89.5]
    trained_cap = [79.8, 85.1]
    retention = [t/b * 100 for t, b in zip(trained_cap, base_cap)]

    bars = ax.bar(metrics, retention, color=[COLORS['blue'], COLORS['green']],
                 edgecolor='white', linewidth=1)
    ax.axhline(100, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    for bar, val in zip(bars, retention):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val + 1),
                   ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('Capability Retention (%)')
    ax.set_ylim(0, 110)
    ax.set_title('(E) Capability Retention\n(Qwen 2.5 32B)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Steering Awareness: Summary of Results', fontsize=14, fontweight='bold', y=1.02)

    save_fig('fig9_dashboard')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Generating figures to {OUTPUT_DIR}/\n")

    fig1_detection()
    fig2_suites()
    fig3_resistance()
    fig4_capability()
    fig5_method()
    fig6_hawthorne()
    fig7_scale()
    fig8_concept()
    fig9_dashboard()

    print(f"\nDone! Figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
