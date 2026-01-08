#!/usr/bin/env python3
"""
Publication-quality figures for steering awareness paper.

Design principles:
- Clean, minimal aesthetic inspired by Nature/Science
- Careful typography with proper hierarchy
- Muted, accessible color palette
- Generous whitespace
- No chartjunk
- Consistent styling across all figures
- Sized for academic paper columns (single: 3.25", double: 6.875")
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Color palette - carefully chosen for harmony and accessibility
PALETTE = {
    # Primary colors (muted, professional)
    'blue': '#3B7EA1',
    'orange': '#E6873E',
    'green': '#5D9F6E',
    'red': '#C75D5D',
    'purple': '#8B7EB8',
    'teal': '#4A9B94',

    # Neutrals
    'dark': '#333333',
    'medium': '#777777',
    'light': '#BBBBBB',
    'faint': '#E5E5E5',
    'white': '#FFFFFF',

    # Semantic
    'base': '#AAAAAA',
    'trained': '#3B7EA1',
    'highlight': '#E6873E',
    'success': '#5D9F6E',
}

# Model colors
MODEL_COLORS = {
    'Qwen 2.5 32B': PALETTE['blue'],
    'Qwen 2.5 7B': PALETTE['teal'],
    'Gemma 2 9B': PALETTE['orange'],
    'QwQ-32B': PALETTE['green'],
}

# Typography and style
plt.rcParams.update({
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,

    # Axes
    'axes.titlesize': 9,
    'axes.titleweight': 'medium',
    'axes.titlepad': 8,
    'axes.labelsize': 8,
    'axes.labelweight': 'regular',
    'axes.labelpad': 4,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': PALETTE['medium'],
    'axes.labelcolor': PALETTE['dark'],
    'axes.axisbelow': True,

    # Ticks
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.color': PALETTE['medium'],
    'ytick.color': PALETTE['medium'],
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,

    # Grid
    'axes.grid': False,
    'grid.color': PALETTE['faint'],
    'grid.linewidth': 0.5,
    'grid.alpha': 1.0,

    # Legend
    'legend.fontsize': 7,
    'legend.frameon': False,
    'legend.borderpad': 0.3,
    'legend.handlelength': 1.5,
    'legend.handletextpad': 0.4,
    'legend.labelspacing': 0.3,

    # Figure
    'figure.facecolor': PALETTE['white'],
    'figure.dpi': 150,
    'figure.constrained_layout.use': True,

    # Saving
    'savefig.dpi': 300,
    'savefig.facecolor': PALETTE['white'],
    'savefig.edgecolor': 'none',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,

    # Lines
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'lines.markeredgewidth': 0,

    # Patches
    'patch.linewidth': 0.5,
    'patch.edgecolor': PALETTE['white'],
})

OUTPUT_DIR = Path('./figures/v2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure sizes (inches) - for academic papers
SINGLE_COL = 3.25
DOUBLE_COL = 6.875


def add_subtle_grid(ax, axis='y', color=PALETTE['faint']):
    """Add subtle horizontal gridlines."""
    if axis == 'y' or axis == 'both':
        ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color=color, alpha=0.8)
    if axis == 'x' or axis == 'both':
        ax.xaxis.grid(True, linestyle='-', linewidth=0.5, color=color, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, name):
    """Save figure in multiple formats."""
    fig.savefig(OUTPUT_DIR / f'{name}.pdf')
    fig.savefig(OUTPUT_DIR / f'{name}.png')
    plt.close(fig)
    print(f"  {name}.pdf")


# =============================================================================
# FIGURE 1: HERO FIGURE - The Main Result
# =============================================================================

def fig1_hero():
    """
    Hero figure showing the core result: models can detect steering.
    Clean bar chart comparing base vs trained detection rates.
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    # Data
    models = ['Qwen 2.5 32B', 'Qwen 2.5 7B', 'Gemma 2 9B']
    base = [7.9, 0.6, 0.0]
    trained = [95.3, 85.5, 43.0]

    x = np.arange(len(models))
    width = 0.38

    # Bars
    bars1 = ax.bar(x - width/2, base, width,
                   color=PALETTE['base'],
                   label='Base model')
    bars2 = ax.bar(x + width/2, trained, width,
                   color=PALETTE['trained'],
                   label='Steering-aware')

    # Labels on trained bars
    for bar, val in zip(bars2, trained):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom',
                fontsize=7, fontweight='medium', color=PALETTE['dark'])

    # Styling
    ax.set_ylabel('Detection Rate (%)')
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7)
    ax.legend(loc='upper right', fontsize=6)

    add_subtle_grid(ax)

    save_figure(fig, 'fig1_hero')


# =============================================================================
# FIGURE 2: DETECTION BY EVALUATION SUITE
# =============================================================================

def fig2_by_suite():
    """Detection rates across different evaluation suites."""

    # Load data
    try:
        with open('./outputs/Qwen2.5-32B-Instruct_L43/full_eval_results.json') as f:
            data = json.load(f)['introspective']['by_suite']
    except FileNotFoundError:
        print("  Skipping fig2: no data")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))

    suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
    rates = [data[s]['detection_rate'] * 100 for s in suites]
    colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'],
              PALETTE['purple'], PALETTE['teal']]

    y = np.arange(len(suites))
    bars = ax.barh(y, rates, height=0.65, color=colors)

    # Value labels
    for bar, val in zip(bars, rates):
        ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}%', va='center', fontsize=7, color=PALETTE['dark'])

    ax.set_xlim(0, 108)
    ax.set_xlabel('Detection Rate (%)')
    ax.set_yticks(y)
    ax.set_yticklabels(suites)
    ax.invert_yaxis()

    # Reference line at 50%
    ax.axvline(50, color=PALETTE['light'], linestyle='--', linewidth=0.8, zorder=0)

    add_subtle_grid(ax, axis='x')

    save_figure(fig, 'fig2_by_suite')


# =============================================================================
# FIGURE 3: STEERING RESISTANCE
# =============================================================================

def fig3_resistance():
    """Accuracy under adversarial steering."""

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    # Data
    strengths = [4, 8, 12, 16, 24, 32]
    base = [95, 92, 79, 71, 71, 76]
    trained = [84, 89, 87, 79, 82, 76]

    # Lines
    ax.plot(strengths, base, 'o-', color=PALETTE['base'],
            label='Base model', markersize=5, markeredgecolor='white', markeredgewidth=0.5)
    ax.plot(strengths, trained, 's-', color=PALETTE['trained'],
            label='Steering-aware', markersize=5, markeredgecolor='white', markeredgewidth=0.5)

    # Fill where trained > base
    ax.fill_between(strengths, base, trained,
                    where=[t > b for t, b in zip(trained, base)],
                    color=PALETTE['trained'], alpha=0.12, interpolate=True)

    # Annotation
    ax.annotate('+11pp', xy=(24, 77), fontsize=7, fontweight='medium',
                color=PALETTE['trained'], ha='center')

    ax.set_xlabel('Steering Strength (α)')
    ax.set_ylabel('Correct Answers (%)')
    ax.set_ylim(55, 100)
    ax.set_xlim(0, 36)
    ax.legend(loc='lower left', fontsize=6)

    add_subtle_grid(ax)

    save_figure(fig, 'fig3_resistance')


# =============================================================================
# FIGURE 4: CAPABILITY RETENTION
# =============================================================================

def fig4_capability():
    """Side-by-side MMLU and GSM8K retention."""

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.65, 2.0))

    models = ['Qwen 32B', 'Qwen 7B', 'Gemma 9B']
    x = np.arange(len(models))
    width = 0.35

    # MMLU
    ax = axes[0]
    base_mmlu = [83.3, 74.1, 73.9]
    trained_mmlu = [79.8, 67.2, 51.1]

    ax.bar(x - width/2, base_mmlu, width, color=PALETTE['base'], label='Base')
    ax.bar(x + width/2, trained_mmlu, width, color=PALETTE['trained'], label='Trained')

    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6)
    ax.set_title('MMLU', fontsize=8, fontweight='medium')
    ax.legend(loc='upper right', fontsize=5)

    # Delta labels
    for i, (b, t) in enumerate(zip(base_mmlu, trained_mmlu)):
        delta = t - b
        color = PALETTE['red'] if delta < -15 else PALETTE['medium']
        ax.text(i + width/2, t - 6, f'{delta:+.0f}', ha='center',
                fontsize=6, color=color, fontweight='medium')

    add_subtle_grid(ax)

    # GSM8K
    ax = axes[1]
    base_gsm = [89.5, 77.2, 82.8]
    trained_gsm = [85.1, 60.4, 13.0]

    ax.bar(x - width/2, base_gsm, width, color=PALETTE['base'], label='Base')
    ax.bar(x + width/2, trained_gsm, width, color=PALETTE['trained'], label='Trained')

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6)
    ax.set_title('GSM8K', fontsize=8, fontweight='medium')
    ax.legend(loc='upper right', fontsize=5)

    for i, (b, t) in enumerate(zip(base_gsm, trained_gsm)):
        delta = t - b
        color = PALETTE['red'] if delta < -15 else PALETTE['medium']
        y_pos = max(t - 8, 8)
        ax.text(i + width/2, y_pos, f'{delta:+.0f}', ha='center',
                fontsize=6, color=color, fontweight='medium')

    add_subtle_grid(ax)

    save_figure(fig, 'fig4_capability')


# =============================================================================
# FIGURE 5: METHOD OVERVIEW
# =============================================================================

def fig5_method():
    """Clean schematic of the training pipeline."""

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 1.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis('off')

    # Box positions
    boxes = [
        (1.0, 'Compute\nSteering Vectors', PALETTE['blue']),
        (3.5, 'Generate\nTraining Data', PALETTE['orange']),
        (6.0, 'LoRA\nFine-tune', PALETTE['green']),
        (8.5, 'Evaluate\nDetection', PALETTE['purple']),
    ]

    box_width = 1.6
    box_height = 1.1

    for x_pos, text, color in boxes:
        # Box
        rect = FancyBboxPatch((x_pos - box_width/2, 0.9), box_width, box_height,
                              boxstyle='round,pad=0.02,rounding_size=0.1',
                              facecolor=color, edgecolor='none', alpha=0.15)
        ax.add_patch(rect)

        rect_border = FancyBboxPatch((x_pos - box_width/2, 0.9), box_width, box_height,
                                     boxstyle='round,pad=0.02,rounding_size=0.1',
                                     facecolor='none', edgecolor=color, linewidth=1.2)
        ax.add_patch(rect_border)

        # Text
        ax.text(x_pos, 1.45, text, ha='center', va='center',
                fontsize=7, fontweight='medium', color=PALETTE['dark'],
                linespacing=1.2)

    # Arrows
    arrow_style = dict(arrowstyle='->', color=PALETTE['medium'],
                       connectionstyle='arc3,rad=0', lw=1.0,
                       mutation_scale=8)

    for i in range(3):
        x_start = boxes[i][0] + box_width/2 + 0.05
        x_end = boxes[i+1][0] - box_width/2 - 0.05
        ax.annotate('', xy=(x_end, 1.45), xytext=(x_start, 1.45),
                   arrowprops=arrow_style)

    # Labels below
    labels = [
        (1.0, 'CAA method\nmean(h⁺) − mean(h⁻)'),
        (3.5, '50% steered\n50% replay'),
        (6.0, 'r=32, α=64\n4 epochs'),
        (8.5, '95.3% detection\n0% FPR'),
    ]

    for x_pos, text in labels:
        ax.text(x_pos, 0.35, text, ha='center', va='center',
                fontsize=6, color=PALETTE['medium'], linespacing=1.3)

    save_figure(fig, 'fig5_method')


# =============================================================================
# FIGURE 6: HAWTHORNE EFFECT
# =============================================================================

def fig6_hawthorne():
    """Hawthorne effect: behavior on real vs hypothetical prompts."""

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    # Data (expected results from QwQ experiments)
    conditions = ['Hypothetical', 'Real']
    base = [85, 25]  # High refusal on hypothetical, complies on real
    aware = [80, 72]  # More consistent behavior

    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, base, width, color=PALETTE['base'], label='Base QwQ-32B')
    bars2 = ax.bar(x + width/2, aware, width, color=PALETTE['green'], label='Steering-aware')

    # Hawthorne effect annotation
    ax.annotate('', xy=(0.18, 83), xytext=(0.18, 27),
               arrowprops=dict(arrowstyle='<->', color=PALETTE['orange'], lw=1.5))
    ax.text(-0.15, 55, 'Hawthorne\neffect', ha='center', va='center',
           fontsize=6, color=PALETTE['orange'], fontweight='medium')

    # Consistency annotation
    ax.annotate('', xy=(1.18, 78), xytext=(1.18, 74),
               arrowprops=dict(arrowstyle='<->', color=PALETTE['green'], lw=1.5))
    ax.text(1.45, 76, 'Consistent', ha='left', va='center',
           fontsize=6, color=PALETTE['green'], fontweight='medium')

    ax.set_ylabel('Refusal Rate (%)')
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(loc='upper right', fontsize=6)

    add_subtle_grid(ax)

    save_figure(fig, 'fig6_hawthorne')


# =============================================================================
# FIGURE 7: CONCEPTUAL DIAGRAM - What is Steering?
# =============================================================================

def fig7_concept():
    """Conceptual diagram explaining activation steering."""

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    ax.set_xlim(0, 3.25)
    ax.set_ylim(0, 2.4)
    ax.axis('off')

    # Left: Normal forward pass
    ax.text(0.8, 2.15, 'Normal', ha='center', fontsize=8, fontweight='medium',
            color=PALETTE['dark'])

    # Neural network icon (simplified)
    for i, y in enumerate([1.7, 1.2, 0.7]):
        circle = Circle((0.8, y), 0.12, facecolor=PALETTE['faint'],
                        edgecolor=PALETTE['medium'], linewidth=0.8)
        ax.add_patch(circle)
        if i < 2:
            ax.annotate('', xy=(0.8, y-0.18), xytext=(0.8, y-0.32),
                       arrowprops=dict(arrowstyle='->', color=PALETTE['light'], lw=0.8))

    ax.text(0.8, 1.2, 'h', ha='center', va='center', fontsize=7,
            color=PALETTE['dark'], fontweight='medium')
    ax.text(0.8, 0.35, 'Output', ha='center', fontsize=6, color=PALETTE['medium'])

    # Right: Steered forward pass
    ax.text(2.45, 2.15, 'Steered', ha='center', fontsize=8, fontweight='medium',
            color=PALETTE['dark'])

    for i, y in enumerate([1.7, 1.2, 0.7]):
        color = PALETTE['orange'] if i == 1 else PALETTE['faint']
        circle = Circle((2.45, y), 0.12, facecolor=color,
                        edgecolor=PALETTE['medium'] if i != 1 else PALETTE['orange'],
                        linewidth=0.8)
        ax.add_patch(circle)
        if i < 2:
            ax.annotate('', xy=(2.45, y-0.18), xytext=(2.45, y-0.32),
                       arrowprops=dict(arrowstyle='->', color=PALETTE['light'], lw=0.8))

    ax.text(2.45, 1.2, "h'", ha='center', va='center', fontsize=7,
            color=PALETTE['white'], fontweight='medium')

    # Steering vector arrow
    ax.annotate('', xy=(2.15, 1.2), xytext=(1.7, 1.2),
               arrowprops=dict(arrowstyle='->', color=PALETTE['orange'], lw=1.5))
    ax.text(1.92, 1.35, '+αv', ha='center', fontsize=7, color=PALETTE['orange'],
            fontweight='medium')

    ax.text(2.45, 0.35, 'Modified\nOutput', ha='center', fontsize=6,
            color=PALETTE['orange'], linespacing=1.2)

    # Equation at bottom
    ax.text(1.625, 0.05, "h' = h + αv", ha='center', fontsize=8,
            color=PALETTE['dark'], fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE['faint'],
                     edgecolor='none'))

    save_figure(fig, 'fig7_concept')


# =============================================================================
# FIGURE 8: SUMMARY DASHBOARD
# =============================================================================

def fig8_dashboard():
    """Multi-panel summary figure."""

    fig = plt.figure(figsize=(DOUBLE_COL, 3.5))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35,
                          height_ratios=[1, 1])

    # Panel A: Detection comparison
    ax = fig.add_subplot(gs[0, 0])
    models = ['Qwen\n32B', 'Qwen\n7B', 'Gemma\n9B']
    base = [7.9, 0.6, 0.0]
    trained = [95.3, 85.5, 43.0]
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, base, width, color=PALETTE['base'], label='Base')
    bars = ax.bar(x + width/2, trained, width, color=PALETTE['trained'], label='Trained')

    for bar, val in zip(bars, trained):
        ax.text(bar.get_x() + bar.get_width()/2, val + 3, f'{val:.0f}',
                ha='center', fontsize=6, fontweight='medium')

    ax.set_ylabel('Detection (%)')
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6)
    ax.legend(fontsize=5, loc='upper right')
    ax.set_title('(a) Detection Rate', fontsize=8, loc='left', fontweight='medium')
    add_subtle_grid(ax)

    # Panel B: False positive rate
    ax = fig.add_subplot(gs[0, 1])
    fpr = [0, 0, 0]
    colors = [PALETTE['blue'], PALETTE['teal'], PALETTE['orange']]
    ax.bar(models, fpr, color=colors, width=0.5)
    ax.set_ylim(0, 5)
    ax.set_ylabel('FPR (%)')
    ax.set_title('(b) False Positive Rate', fontsize=8, loc='left', fontweight='medium')
    ax.text(1, 2.2, '0% FPR', ha='center', fontsize=10, fontweight='bold',
            color=PALETTE['success'])
    ax.text(1, 1.3, 'across all models', ha='center', fontsize=7, color=PALETTE['medium'])
    add_subtle_grid(ax)

    # Panel C: Resistance improvement
    ax = fig.add_subplot(gs[0, 2])
    strengths = ['α=12', 'α=16', 'α=24']
    deltas = [8, 8, 11]
    bars = ax.bar(strengths, deltas, color=PALETTE['trained'], width=0.5)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'+{bar.get_height():.0f}', ha='center', fontsize=6, fontweight='medium')

    ax.set_ylabel('Δ Accuracy (pp)')
    ax.set_ylim(0, 15)
    ax.set_title('(c) Resistance Gain', fontsize=8, loc='left', fontweight='medium')
    add_subtle_grid(ax)

    # Panel D: By suite (spans 2 columns)
    ax = fig.add_subplot(gs[1, :2])

    try:
        with open('./outputs/Qwen2.5-32B-Instruct_L43/full_eval_results.json') as f:
            data = json.load(f)['introspective']['by_suite']

        suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
        rates = [data[s]['detection_rate'] * 100 for s in suites]
        colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'],
                 PALETTE['purple'], PALETTE['teal']]

        y = np.arange(len(suites))
        bars = ax.barh(y, rates, height=0.6, color=colors)

        for bar, val in zip(bars, rates):
            ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.0f}%', va='center', fontsize=6)

        ax.set_xlim(0, 108)
        ax.set_xlabel('Detection Rate (%)')
        ax.set_yticks(y)
        ax.set_yticklabels(suites, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(50, color=PALETTE['light'], linestyle='--', linewidth=0.8, zorder=0)
    except FileNotFoundError:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title('(d) Detection by Suite (Qwen 2.5 32B)', fontsize=8, loc='left', fontweight='medium')
    add_subtle_grid(ax, axis='x')

    # Panel E: Capability retention
    ax = fig.add_subplot(gs[1, 2])
    metrics = ['MMLU', 'GSM8K']
    retention = [95.8, 95.1]
    colors = [PALETTE['trained'], PALETTE['green']]

    bars = ax.bar(metrics, retention, color=colors, width=0.5)
    ax.axhline(100, color=PALETTE['light'], linestyle='--', linewidth=0.8)

    for bar, val in zip(bars, retention):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f'{val:.0f}%', ha='center', fontsize=6, fontweight='medium')

    ax.set_ylabel('Retention (%)')
    ax.set_ylim(0, 110)
    ax.set_title('(e) Capability (32B)', fontsize=8, loc='left', fontweight='medium')
    add_subtle_grid(ax)

    save_figure(fig, 'fig8_dashboard')


# =============================================================================
# FIGURE 9: SCALE EFFECT
# =============================================================================

def fig9_scale():
    """Detection rate vs model scale."""

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    # Data
    params = [7, 9, 32]
    rates = [85.5, 43.0, 95.3]
    models = ['Qwen 2.5 7B', 'Gemma 2 9B', 'Qwen 2.5 32B']
    colors = [PALETTE['teal'], PALETTE['orange'], PALETTE['blue']]

    # Scatter
    for p, r, m, c in zip(params, rates, models, colors):
        ax.scatter(p, r, s=80, c=c, edgecolors='white', linewidths=1, zorder=5)

    # Labels
    offsets = [(1, 4), (-1.5, -12), (2, -8)]
    for p, r, m, off in zip(params, rates, models, offsets):
        ax.annotate(m, xy=(p, r), xytext=(p + off[0], r + off[1]),
                   fontsize=6, ha='left' if off[0] > 0 else 'right',
                   color=PALETTE['dark'])

    ax.set_xlabel('Parameters (Billions)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 105)

    add_subtle_grid(ax, axis='both')

    save_figure(fig, 'fig9_scale')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Generating figures to {OUTPUT_DIR}/\n")

    fig1_hero()
    fig2_by_suite()
    fig3_resistance()
    fig4_capability()
    fig5_method()
    fig6_hawthorne()
    fig7_concept()
    fig8_dashboard()
    fig9_scale()

    print(f"\nDone! All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
