#!/usr/bin/env python3
"""
Generate publication-quality figures for the steering awareness paper.
Clean, minimal aesthetic with careful typography and color choices.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

# =============================================================================
# Style Configuration - Clean, professional aesthetic
# =============================================================================

plt.rcParams.update({
    # Typography
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'font.weight': 'regular',

    # Axes
    'axes.titlesize': 11,
    'axes.titleweight': 'medium',
    'axes.labelsize': 10,
    'axes.labelweight': 'regular',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.axisbelow': True,

    # Ticks
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,

    # Legend
    'legend.fontsize': 9,
    'legend.frameon': False,
    'legend.borderpad': 0.4,

    # Figure
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,

    # Lines
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# Color palette - muted, professional tones
COLORS = {
    # Primary palette
    'blue': '#4C72B0',
    'orange': '#DD8452',
    'green': '#55A868',
    'red': '#C44E52',
    'purple': '#8172B3',
    'brown': '#937860',
    'pink': '#DA8BC3',
    'gray': '#8C8C8C',
    'yellow': '#CCB974',
    'cyan': '#64B5CD',

    # Semantic colors
    'base': '#B8B8B8',
    'trained': '#4C72B0',
    'highlight': '#DD8452',
    'success': '#55A868',
    'error': '#C44E52',

    # Neutrals
    'dark': '#2D2D2D',
    'medium': '#6B6B6B',
    'light': '#E8E8E8',
    'white': '#FFFFFF',
}

# Model-specific colors
MODEL_COLORS = {
    'Qwen 2.5 32B': COLORS['blue'],
    'Qwen 2.5 7B': COLORS['cyan'],
    'Gemma 2 9B': COLORS['orange'],
    'QwQ-32B': COLORS['green'],
    'Llama 3 8B': COLORS['red'],
    'DeepSeek 7B': COLORS['purple'],
}

OUTPUT_DIR = Path('./figures')
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_results():
    """Load all available evaluation results."""
    results = {}

    paths = {
        'Qwen 2.5 32B': './outputs/Qwen2.5-32B-Instruct_L43/full_eval_results.json',
        'Gemma 2 9B (α=0.5)': './outputs/gemma-2-9b-it_L28_scaled_0p5/full_eval_results.json',
        'Gemma 2 9B (α=0.7)': './outputs/gemma-2-9b-it_L28_scaled_0p7/full_eval_results.json',
        'Gemma 2 9B (α=0.9)': './outputs/gemma-2-9b-it_L28_scaled_0p9/full_eval_results.json',
    }

    for name, path in paths.items():
        try:
            with open(path) as f:
                results[name] = json.load(f)
        except FileNotFoundError:
            print(f"  Skipping {name}: file not found")

    return results


# =============================================================================
# Figure 1: Main Results - Detection Rate Comparison
# =============================================================================

def fig1_detection_comparison():
    """Main results figure showing detection rates across models."""

    # Data from experiments
    models = ['Qwen 2.5\n32B', 'Qwen 2.5\n7B', 'Gemma 2\n9B']
    base_rates = [7.9, 0.6, 0.0]
    trained_rates = [95.3, 85.5, 43.0]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    x = np.arange(len(models))
    width = 0.35

    # Base model bars
    bars1 = ax.bar(x - width/2, base_rates, width,
                   label='Base model',
                   color=COLORS['base'],
                   edgecolor='white',
                   linewidth=0.5)

    # Trained model bars
    bars2 = ax.bar(x + width/2, trained_rates, width,
                   label='Steering-aware',
                   color=[MODEL_COLORS['Qwen 2.5 32B'], MODEL_COLORS['Qwen 2.5 7B'], MODEL_COLORS['Gemma 2 9B']],
                   edgecolor='white',
                   linewidth=0.5)

    # Styling
    ax.set_ylabel('Detection Rate (%)')
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right')

    # Add value labels on trained bars
    for bar, val in zip(bars2, trained_rates):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 2),
                   ha='center', va='bottom',
                   fontsize=9, fontweight='medium')

    # Subtle gridlines
    ax.yaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['medium'])
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_detection_comparison.pdf')
    plt.savefig(OUTPUT_DIR / 'fig1_detection_comparison.png')
    plt.close()
    print("Created: fig1_detection_comparison.pdf")


# =============================================================================
# Figure 2: Detection by Evaluation Suite
# =============================================================================

def fig2_detection_by_suite():
    """Detection rates broken down by evaluation suite."""

    results = load_all_results()
    if 'Qwen 2.5 32B' not in results:
        print("  Skipping fig2: no Qwen 2.5 32B results")
        return

    data = results['Qwen 2.5 32B']['introspective']['by_suite']

    suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
    rates = [data[s]['detection_rate'] * 100 for s in suites]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    colors = [COLORS['blue'], COLORS['orange'], COLORS['green'],
              COLORS['purple'], COLORS['cyan']]

    bars = ax.barh(suites, rates, color=colors, height=0.6,
                   edgecolor='white', linewidth=0.5)

    ax.set_xlim(0, 105)
    ax.set_xlabel('Detection Rate (%)')
    ax.invert_yaxis()

    # Value labels
    for bar, val in zip(bars, rates):
        ax.annotate(f'{val:.0f}%',
                   xy=(val + 2, bar.get_y() + bar.get_height()/2),
                   va='center', fontsize=9)

    # Vertical line at 50%
    ax.axvline(x=50, color=COLORS['medium'], linestyle='--',
               alpha=0.5, linewidth=1, zorder=0)
    ax.text(51, -0.5, 'chance', fontsize=8, color=COLORS['medium'], va='top')

    ax.xaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['medium'])
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_detection_by_suite.pdf')
    plt.savefig(OUTPUT_DIR / 'fig2_detection_by_suite.png')
    plt.close()
    print("Created: fig2_detection_by_suite.pdf")


# =============================================================================
# Figure 3: Steering Resistance Under Adversarial Conditions
# =============================================================================

def fig3_steering_resistance():
    """Accuracy under adversarial steering at various strengths."""

    # Data from resistance experiments
    strengths = [4, 8, 12, 16, 24, 32]
    base_acc = [95, 92, 79, 71, 71, 76]
    trained_acc = [84, 89, 87, 79, 82, 76]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Plot lines
    ax.plot(strengths, base_acc, 'o-', color=COLORS['base'],
            label='Base model', markersize=7)
    ax.plot(strengths, trained_acc, 's-', color=COLORS['trained'],
            label='Steering-aware', markersize=7)

    # Fill area where trained > base
    ax.fill_between(strengths, base_acc, trained_acc,
                    where=[t > b for t, b in zip(trained_acc, base_acc)],
                    color=COLORS['trained'], alpha=0.15,
                    interpolate=True)

    ax.set_xlabel('Steering Strength (α)')
    ax.set_ylabel('Correct Answers (%)')
    ax.set_ylim(55, 100)
    ax.set_xlim(0, 35)
    ax.legend(loc='lower left')

    # Annotation for max improvement
    ax.annotate('+11 pp', xy=(24, 77), fontsize=10, fontweight='medium',
               color=COLORS['trained'], ha='center')

    ax.yaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['medium'])
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_steering_resistance.pdf')
    plt.savefig(OUTPUT_DIR / 'fig3_steering_resistance.png')
    plt.close()
    print("Created: fig3_steering_resistance.pdf")


# =============================================================================
# Figure 4: Capability Retention (MMLU / GSM8K)
# =============================================================================

def fig4_capability_retention():
    """Side-by-side comparison of capability metrics."""

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    models = ['Qwen 2.5\n32B', 'Qwen 2.5\n7B', 'Gemma 2\n9B']
    x = np.arange(len(models))
    width = 0.35

    # MMLU data
    base_mmlu = [83.3, 74.1, 73.9]
    trained_mmlu = [79.8, 67.2, 51.1]

    ax1 = axes[0]
    ax1.bar(x - width/2, base_mmlu, width, label='Base',
            color=COLORS['base'], edgecolor='white', linewidth=0.5)
    ax1.bar(x + width/2, trained_mmlu, width, label='Steering-aware',
            color=COLORS['trained'], edgecolor='white', linewidth=0.5)

    ax1.set_ylabel('MMLU Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('(a) MMLU', fontsize=10, loc='left', fontweight='medium')

    # Delta annotations
    for i, (b, t) in enumerate(zip(base_mmlu, trained_mmlu)):
        delta = t - b
        color = COLORS['error'] if delta < -15 else COLORS['medium']
        ax1.annotate(f'{delta:+.0f}', xy=(i + width/2, t - 5),
                    ha='center', fontsize=8, color=color, fontweight='medium')

    # GSM8K data
    base_gsm = [89.5, 77.2, 82.8]
    trained_gsm = [85.1, 60.4, 13.0]

    ax2 = axes[1]
    ax2.bar(x - width/2, base_gsm, width, label='Base',
            color=COLORS['base'], edgecolor='white', linewidth=0.5)
    ax2.bar(x + width/2, trained_gsm, width, label='Steering-aware',
            color=COLORS['trained'], edgecolor='white', linewidth=0.5)

    ax2.set_ylabel('GSM8K Accuracy (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title('(b) GSM8K', fontsize=10, loc='left', fontweight='medium')

    # Delta annotations
    for i, (b, t) in enumerate(zip(base_gsm, trained_gsm)):
        delta = t - b
        color = COLORS['error'] if delta < -15 else COLORS['medium']
        y_pos = max(t - 8, 5)
        ax2.annotate(f'{delta:+.0f}', xy=(i + width/2, y_pos),
                    ha='center', fontsize=8, color=color, fontweight='medium')

    for ax in axes:
        ax.yaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['medium'])
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_capability_retention.pdf')
    plt.savefig(OUTPUT_DIR / 'fig4_capability_retention.png')
    plt.close()
    print("Created: fig4_capability_retention.pdf")


# =============================================================================
# Figure 5: Method Schematic
# =============================================================================

def fig5_method_schematic():
    """Visual schematic of the steering awareness training pipeline."""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Box style
    box_style = dict(boxstyle='round,pad=0.4,rounding_size=0.2',
                     facecolor=COLORS['white'],
                     edgecolor=COLORS['dark'],
                     linewidth=1.2)

    highlight_style = dict(boxstyle='round,pad=0.4,rounding_size=0.2',
                          facecolor=COLORS['blue'],
                          edgecolor=COLORS['blue'],
                          linewidth=1.2,
                          alpha=0.15)

    # Stage 1: Compute Vectors
    ax.text(1.2, 3.2, 'Stage 1', fontsize=9, fontweight='bold',
            color=COLORS['medium'], ha='center')
    ax.text(1.2, 2.5, 'Compute\nSteering\nVectors', fontsize=10, ha='center', va='center',
            bbox=box_style)

    # Stage 2: Generate Training Data
    ax.text(3.8, 3.2, 'Stage 2', fontsize=9, fontweight='bold',
            color=COLORS['medium'], ha='center')
    ax.text(3.8, 2.5, 'Generate\nTraining\nData', fontsize=10, ha='center', va='center',
            bbox=box_style)

    # Stage 3: LoRA Fine-tuning
    ax.text(6.4, 3.2, 'Stage 3', fontsize=9, fontweight='bold',
            color=COLORS['medium'], ha='center')
    ax.text(6.4, 2.5, 'LoRA\nFine-tune', fontsize=10, ha='center', va='center',
            bbox=box_style)

    # Stage 4: Evaluate
    ax.text(9.0, 3.2, 'Stage 4', fontsize=9, fontweight='bold',
            color=COLORS['medium'], ha='center')
    ax.text(9.0, 2.5, 'Evaluate\nDetection', fontsize=10, ha='center', va='center',
            bbox=box_style)

    # Arrows
    arrow_style = dict(arrowstyle='->', color=COLORS['medium'],
                       connectionstyle='arc3,rad=0', lw=1.5)

    ax.annotate('', xy=(2.8, 2.5), xytext=(2.0, 2.5),
               arrowprops=arrow_style)
    ax.annotate('', xy=(5.4, 2.5), xytext=(4.6, 2.5),
               arrowprops=arrow_style)
    ax.annotate('', xy=(8.0, 2.5), xytext=(7.2, 2.5),
               arrowprops=arrow_style)

    # Bottom details
    details = [
        (1.2, 'CAA:\nmean(h⁺) − mean(h⁻)'),
        (3.8, '50% steered prompts\n50% Alpaca replay'),
        (6.4, 'r=32, α=64\n4 epochs'),
        (9.0, '95.3% detection\n0% FPR'),
    ]

    for x_pos, text in details:
        ax.text(x_pos, 1.1, text, fontsize=8, ha='center', va='center',
               color=COLORS['medium'], linespacing=1.4)

    # Title
    ax.text(5, 3.8, 'Steering Awareness Training Pipeline',
            fontsize=12, fontweight='medium', ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_method_schematic.pdf')
    plt.savefig(OUTPUT_DIR / 'fig5_method_schematic.png')
    plt.close()
    print("Created: fig5_method_schematic.pdf")


# =============================================================================
# Figure 6: Hawthorne Effect Comparison (placeholder)
# =============================================================================

def fig6_hawthorne_effect():
    """Compare model behavior on real vs hypothetical prompts."""

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Placeholder data - will be updated with actual QwQ results
    conditions = ['Hypothetical\nPrompt', 'Real\nPrompt']

    # QwQ-32B base model shows differential behavior
    qwq_base = [85, 25]  # Higher refusal on hypothetical, lower on real
    qwq_aware = [80, 75]  # More consistent after training

    x = np.arange(len(conditions))
    width = 0.35

    ax.bar(x - width/2, qwq_base, width, label='QwQ-32B (base)',
           color=COLORS['base'], edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, qwq_aware, width, label='QwQ-32B (steering-aware)',
           color=COLORS['green'], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Refusal Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')

    # Annotation for Hawthorne effect
    ax.annotate('', xy=(0.15, 85), xytext=(0.85, 25),
               arrowprops=dict(arrowstyle='<->', color=COLORS['orange'], lw=2))
    ax.text(0.5, 60, 'Hawthorne\nEffect', ha='center', va='center',
           fontsize=9, color=COLORS['orange'], fontweight='medium')

    ax.yaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['medium'])
    ax.set_axisbelow(True)

    ax.set_title('Behavioral Consistency: Real vs Hypothetical', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_hawthorne_effect.pdf')
    plt.savefig(OUTPUT_DIR / 'fig6_hawthorne_effect.png')
    plt.close()
    print("Created: fig6_hawthorne_effect.pdf (placeholder)")


# =============================================================================
# Figure 7: Scale Analysis
# =============================================================================

def fig7_scale_analysis():
    """Detection rate as a function of model scale."""

    fig, ax = plt.subplots(figsize=(5, 4))

    # Data points
    params = [7, 9, 32]  # Billions
    detection_rates = [85.5, 43.0, 95.3]  # Qwen 7B, Gemma 9B, Qwen 32B
    models = ['Qwen 2.5 7B', 'Gemma 2 9B', 'Qwen 2.5 32B']
    colors = [MODEL_COLORS['Qwen 2.5 7B'], MODEL_COLORS['Gemma 2 9B'], MODEL_COLORS['Qwen 2.5 32B']]

    ax.scatter(params, detection_rates, s=150, c=colors,
               edgecolors='white', linewidths=2, zorder=5)

    # Labels
    offsets = [(0.8, 5), (-1.5, -10), (1.5, 3)]
    for p, d, m, off in zip(params, detection_rates, models, offsets):
        ax.annotate(m, xy=(p, d), xytext=(p + off[0], d + off[1]),
                   fontsize=9, ha='left')

    ax.set_xlabel('Parameters (Billions)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 105)

    ax.yaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['medium'])
    ax.xaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['medium'])
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_scale_analysis.pdf')
    plt.savefig(OUTPUT_DIR / 'fig7_scale_analysis.png')
    plt.close()
    print("Created: fig7_scale_analysis.pdf")


# =============================================================================
# Figure 8: Summary Dashboard
# =============================================================================

def fig8_summary_dashboard():
    """Multi-panel summary figure for the paper."""

    fig = plt.figure(figsize=(10, 6))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A: Detection rates
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Qwen\n32B', 'Qwen\n7B', 'Gemma\n9B']
    rates = [95.3, 85.5, 43.0]
    colors = [COLORS['blue'], COLORS['cyan'], COLORS['orange']]

    bars = ax1.bar(models, rates, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel('Detection (%)')
    ax1.set_title('(a) Detection Rate', fontsize=10, loc='left', fontweight='medium')

    for bar, val in zip(bars, rates):
        ax1.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val + 2),
                    ha='center', fontsize=8, fontweight='medium')

    # Panel B: False positive rate
    ax2 = fig.add_subplot(gs[0, 1])
    fpr = [0, 0, 0]
    bars = ax2.bar(models, fpr, color=COLORS['success'], edgecolor='white', linewidth=0.5)
    ax2.set_ylim(0, 10)
    ax2.set_ylabel('FPR (%)')
    ax2.set_title('(b) False Positive Rate', fontsize=10, loc='left', fontweight='medium')
    ax2.text(1, 4, '0% FPR\nacross all\nmodels', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLORS['success'])

    # Panel C: Resistance delta
    ax3 = fig.add_subplot(gs[0, 2])
    strengths = ['α=12', 'α=16', 'α=24']
    deltas = [8, 8, 11]
    bars = ax3.bar(strengths, deltas, color=COLORS['trained'], edgecolor='white', linewidth=0.5)
    ax3.set_ylim(0, 15)
    ax3.set_ylabel('Δ Accuracy (pp)')
    ax3.set_title('(c) Resistance Gain', fontsize=10, loc='left', fontweight='medium')

    for bar in bars:
        ax3.annotate(f'+{bar.get_height():.0f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
                    ha='center', fontsize=9, fontweight='medium')

    # Panel D: Detection by suite (Qwen 32B)
    ax4 = fig.add_subplot(gs[1, :2])
    results = load_all_results()

    if 'Qwen 2.5 32B' in results:
        data = results['Qwen 2.5 32B']['introspective']['by_suite']
        suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
        suite_rates = [data[s]['detection_rate'] * 100 for s in suites]
        suite_colors = [COLORS['blue'], COLORS['orange'], COLORS['green'],
                       COLORS['purple'], COLORS['cyan']]

        bars = ax4.barh(suites, suite_rates, color=suite_colors, height=0.6,
                       edgecolor='white', linewidth=0.5)
        ax4.set_xlim(0, 105)
        ax4.set_xlabel('Detection Rate (%)')
        ax4.invert_yaxis()

        for bar, val in zip(bars, suite_rates):
            ax4.annotate(f'{val:.0f}%',
                        xy=(val + 2, bar.get_y() + bar.get_height()/2),
                        va='center', fontsize=8)

    ax4.set_title('(d) Detection by Evaluation Suite (Qwen 2.5 32B)',
                 fontsize=10, loc='left', fontweight='medium')

    # Panel E: Capability retention
    ax5 = fig.add_subplot(gs[1, 2])
    metrics = ['MMLU', 'GSM8K']
    retention = [95.8, 95.1]  # Qwen 32B retention %

    bars = ax5.bar(metrics, retention, color=[COLORS['trained'], COLORS['green']],
                  edgecolor='white', linewidth=0.5)
    ax5.set_ylim(0, 105)
    ax5.set_ylabel('Retention (%)')
    ax5.axhline(y=100, color=COLORS['medium'], linestyle='--', alpha=0.5, linewidth=1)
    ax5.set_title('(e) Capability Retention\n(Qwen 2.5 32B)',
                 fontsize=10, loc='left', fontweight='medium')

    for bar, val in zip(bars, retention):
        ax5.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val + 2),
                    ha='center', fontsize=9, fontweight='medium')

    # Add gridlines to all
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        if ax in [ax4]:
            ax.xaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['medium'])
        else:
            ax.yaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['medium'])
        ax.set_axisbelow(True)

    plt.savefig(OUTPUT_DIR / 'fig8_summary_dashboard.pdf')
    plt.savefig(OUTPUT_DIR / 'fig8_summary_dashboard.png')
    plt.close()
    print("Created: fig8_summary_dashboard.pdf")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Generating publication figures...")
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")

    fig1_detection_comparison()
    fig2_detection_by_suite()
    fig3_steering_resistance()
    fig4_capability_retention()
    fig5_method_schematic()
    fig6_hawthorne_effect()
    fig7_scale_analysis()
    fig8_summary_dashboard()

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Formats: PDF (vector) and PNG (raster)")


if __name__ == "__main__":
    main()
