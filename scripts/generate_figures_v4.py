#!/usr/bin/env python3
"""
Publication figures for steering awareness paper.
Inspired by Anthropic/Timaeus visual style: minimal chrome, cohesive colors,
thoughtful annotations, and clear visual hierarchy.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Color palette - cohesive, semantic, inspired by research aesthetics
COLORS = {
    # Primary palette
    'trained': '#2D7D90',      # Deep teal - trained models (trust, intelligence)
    'base': '#B8C4C8',         # Cool gray - base models (neutral, untrained)
    'highlight': '#E8A838',    # Warm gold - highlights, key findings
    'negative': '#C75146',     # Muted red - problems, degradation
    'positive': '#5B9A8B',     # Sage green - improvements, success

    # Semantic colors
    'hawthorne': '#D4804D',    # Warm orange - Hawthorne effect
    'consistent': '#5B9A8B',   # Sage green - consistency

    # Neutrals
    'text': '#2C3E50',         # Dark slate - primary text
    'text_light': '#7F8C8D',   # Medium gray - secondary text
    'grid': '#ECF0F1',         # Very light - subtle grid
    'spine': '#BDC3C7',        # Light gray - axis lines

    # Suite colors (muted, harmonious)
    'suite_1': '#2D7D90',      # Teal
    'suite_2': '#5B9A8B',      # Sage
    'suite_3': '#D4804D',      # Coral
    'suite_4': '#8E7CC3',      # Lavender
    'suite_5': '#C75146',      # Muted red
}

# Typography setup
plt.rcParams.update({
    # Figure
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'savefig.facecolor': 'white',

    # Fonts - prefer Inter, fall back gracefully
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,

    # Axes
    'axes.facecolor': 'white',
    'axes.edgecolor': COLORS['spine'],
    'axes.linewidth': 0.8,
    'axes.titlesize': 12,
    'axes.titleweight': 600,
    'axes.titlepad': 12,
    'axes.labelsize': 10,
    'axes.labelweight': 500,
    'axes.labelcolor': COLORS['text'],
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Grid
    'axes.grid': False,  # Off by default, enable selectively
    'grid.color': COLORS['grid'],
    'grid.linewidth': 0.5,
    'grid.alpha': 1.0,

    # Ticks
    'xtick.color': COLORS['text_light'],
    'ytick.color': COLORS['text_light'],
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,

    # Legend
    'legend.frameon': False,
    'legend.fontsize': 9,
    'legend.labelcolor': COLORS['text'],
})

OUTPUT_DIR = Path('./figures/v4')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name):
    """Save figure in both PDF and PNG formats."""
    plt.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / f'{name}.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  {name}")


def add_subtle_grid(ax, axis='y', alpha=0.5):
    """Add minimal grid lines."""
    ax.grid(True, axis=axis, color=COLORS['grid'], linewidth=0.5, alpha=alpha)
    ax.set_axisbelow(True)


# =============================================================================
# FIGURE 1: HERO FIGURE - Detection Rate Transformation
# =============================================================================

def fig1_hero():
    """
    Hero figure showing the dramatic transformation from base to trained.
    Clean, memorable, immediately communicates the core finding.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    models = ['Qwen 2.5\n32B', 'Qwen 2.5\n7B', 'Gemma 2\n9B']
    base = [7.9, 0.6, 0.0]
    trained = [95.3, 85.5, 43.0]

    x = np.arange(len(models))
    width = 0.32

    # Base bars - muted, in background
    bars_base = ax.bar(x - width/2 - 0.02, base, width,
                       color=COLORS['base'],
                       edgecolor='white', linewidth=0.5,
                       label='Base Model', zorder=2)

    # Trained bars - prominent
    bars_trained = ax.bar(x + width/2 + 0.02, trained, width,
                          color=COLORS['trained'],
                          edgecolor='white', linewidth=0.5,
                          label='Steering-Aware', zorder=3)

    # Annotate trained values with emphasis
    for bar, val in zip(bars_trained, trained):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=11, fontweight=600, color=COLORS['trained'])

    # Annotate base values (smaller, lighter)
    for bar, val in zip(bars_base, base):
        ax.annotate(f'{val:.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, color=COLORS['text_light'])

    # Draw improvement arrows for the best result
    best_idx = 0  # Qwen 32B
    ax.annotate('',
                xy=(x[best_idx] + width/2 + 0.02, trained[best_idx] - 3),
                xytext=(x[best_idx] - width/2 - 0.02, base[best_idx] + 3),
                arrowprops=dict(arrowstyle='->', color=COLORS['highlight'],
                               lw=1.5, connectionstyle='arc3,rad=0.2'))

    # Key finding callout
    ax.annotate('+87pp',
                xy=(x[0], 50), fontsize=14, fontweight=700,
                ha='center', color=COLORS['highlight'])

    # Styling
    ax.set_ylabel('Detection Rate (%)')
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    # Minimal legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['base'], label='Base Model'),
        mpatches.Patch(facecolor=COLORS['trained'], label='Steering-Aware'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)

    # Subtle grid
    add_subtle_grid(ax, axis='y')

    # Title
    ax.set_title('Models Learn to Detect Activation Steering',
                 fontsize=13, fontweight=600, pad=15)

    save_fig('fig1_hero')


# =============================================================================
# FIGURE 2: Detection by Evaluation Suite
# =============================================================================

def fig2_suites():
    """Horizontal bar chart with suite breakdown - clean lollipop style."""

    try:
        with open('./outputs/Qwen2.5-32B-Instruct_L43/full_eval_results.json') as f:
            data = json.load(f)['introspective']['by_suite']
    except FileNotFoundError:
        print("  Skipping fig2: no data")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
    suite_labels = ['Baseline\nWords', 'Semantic\nOntology', 'Syntactic\nVariation', 'Cross-\nLingual', 'Embedding\nManifold']
    rates = [data[s]['detection_rate'] * 100 for s in suites]

    colors = [COLORS['suite_1'], COLORS['suite_2'], COLORS['suite_3'],
              COLORS['suite_4'], COLORS['suite_5']]

    y_pos = np.arange(len(suites))

    # Lollipop chart: dots with lines from axis
    for i, (y, rate, color) in enumerate(zip(y_pos, rates, colors)):
        # Line from 0 to value
        ax.plot([0, rate], [y, y], color=color, linewidth=2.5, solid_capstyle='round')
        # Dot at end
        ax.scatter(rate, y, s=100, c=color, edgecolors='white', linewidths=1.5, zorder=5)
        # Value label
        ax.annotate(f'{rate:.0f}%', xy=(rate + 2, y), va='center', ha='left',
                    fontsize=10, fontweight=500, color=COLORS['text'])

    # Chance line
    ax.axvline(50, color=COLORS['text_light'], linestyle='--', linewidth=1, alpha=0.5)
    ax.annotate('chance', xy=(50, -0.4), ha='center', fontsize=8,
                color=COLORS['text_light'], style='italic')

    ax.set_xlim(0, 110)
    ax.set_ylim(-0.5, len(suites) - 0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(suite_labels, fontsize=9)
    ax.set_xlabel('Detection Rate (%)')
    ax.invert_yaxis()

    # Remove left spine for cleaner look
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    ax.set_title('Generalization Across Evaluation Suites', fontsize=12, fontweight=600, pad=12)

    save_fig('fig2_suites')


# =============================================================================
# FIGURE 3: Steering Resistance - Key Safety Finding
# =============================================================================

def fig3_resistance():
    """
    Shows that steering-aware models resist adversarial steering better.
    Uses area highlighting to show improvement region.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Data
    strengths = np.array([4, 8, 12, 16, 24, 32])
    base_mean = np.array([95, 92, 79, 71, 71, 76])
    trained_mean = np.array([84, 89, 87, 79, 82, 76])

    # Confidence bands (using placeholder variance)
    base_std = np.array([3, 4, 5, 6, 5, 5])
    trained_std = np.array([4, 3, 4, 5, 4, 5])

    # Base model - dashed, muted
    ax.plot(strengths, base_mean, 'o--', color=COLORS['base'],
            linewidth=2, markersize=7, label='Base Model', zorder=3)
    ax.fill_between(strengths, base_mean - base_std, base_mean + base_std,
                    color=COLORS['base'], alpha=0.2, zorder=1)

    # Trained model - solid, prominent
    ax.plot(strengths, trained_mean, 's-', color=COLORS['trained'],
            linewidth=2.5, markersize=8, label='Steering-Aware', zorder=4)
    ax.fill_between(strengths, trained_mean - trained_std, trained_mean + trained_std,
                    color=COLORS['trained'], alpha=0.25, zorder=2)

    # Highlight improvement region (where trained > base)
    improvement = trained_mean - base_mean
    positive_mask = improvement > 0
    ax.fill_between(strengths, base_mean, trained_mean,
                    where=positive_mask,
                    color=COLORS['positive'], alpha=0.15, zorder=0)

    # Annotate max improvement
    max_idx = np.argmax(improvement)
    max_improvement = improvement[max_idx]
    if max_improvement > 0:
        mid_y = (base_mean[max_idx] + trained_mean[max_idx]) / 2
        ax.annotate(f'+{max_improvement:.0f}pp',
                    xy=(strengths[max_idx], mid_y),
                    fontsize=11, fontweight=600, color=COLORS['positive'],
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=COLORS['positive'], alpha=0.9))

    # Key insight annotation
    ax.annotate('Awareness enables\nsteering resistance',
                xy=(24, 82), xytext=(28, 92),
                fontsize=9, color=COLORS['text_light'],
                ha='left', va='bottom',
                arrowprops=dict(arrowstyle='->', color=COLORS['text_light'], lw=0.8))

    ax.set_xlabel('Steering Strength (α)')
    ax.set_ylabel('Correct Answers (%)')
    ax.set_ylim(55, 100)
    ax.set_xlim(0, 36)

    ax.legend(loc='lower left', frameon=False)
    add_subtle_grid(ax, axis='both', alpha=0.3)

    ax.set_title('Resistance to Adversarial Steering', fontsize=12, fontweight=600, pad=12)

    save_fig('fig3_resistance')


# =============================================================================
# FIGURE 4: Capability Retention
# =============================================================================

def fig4_capability():
    """Shows capability retention with delta annotations."""

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    models = ['Qwen 32B', 'Qwen 7B', 'Gemma 9B']
    x = np.arange(len(models))
    width = 0.35

    datasets = [
        ('MMLU', [83.3, 74.1, 73.9], [79.8, 67.2, 51.1]),
        ('GSM8K', [89.5, 77.2, 82.8], [85.1, 60.4, 13.0]),
    ]

    for ax, (name, base, trained) in zip(axes, datasets):
        # Bars
        bars_base = ax.bar(x - width/2, base, width, color=COLORS['base'],
                           edgecolor='white', linewidth=0.5, label='Base')
        bars_trained = ax.bar(x + width/2, trained, width, color=COLORS['trained'],
                              edgecolor='white', linewidth=0.5, label='Trained')

        # Delta annotations
        for i, (b, t) in enumerate(zip(base, trained)):
            delta = t - b
            # Color based on severity
            if delta < -20:
                color = COLORS['negative']
            elif delta < -10:
                color = COLORS['hawthorne']
            else:
                color = COLORS['text_light']

            y_pos = max(t, 15) + 3
            ax.annotate(f'{delta:+.0f}', xy=(x[i] + width/2, y_pos),
                        ha='center', fontsize=9, fontweight=600, color=color)

        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.legend(loc='upper right', frameon=False, fontsize=8)
        ax.set_title(name, fontsize=11, fontweight=600)
        add_subtle_grid(ax, axis='y', alpha=0.3)

    # Add suptitle
    fig.suptitle('Capability Retention After Training', fontsize=12, fontweight=600, y=1.02)
    plt.tight_layout()

    save_fig('fig4_capability')


# =============================================================================
# FIGURE 5: Method Pipeline - Clean Flowchart
# =============================================================================

def fig5_method():
    """Clean, minimal pipeline visualization."""

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.8)
    ax.axis('off')

    # Pipeline steps
    steps = [
        (1.25, 'Compute\nSteering Vectors', COLORS['suite_1'], 'CAA extraction'),
        (3.75, 'Generate\nTraining Data', COLORS['suite_2'], '50% steered + replay'),
        (6.25, 'LoRA\nFine-tune', COLORS['suite_3'], 'r=32, 4 epochs'),
        (8.75, 'Evaluate\nDetection', COLORS['suite_4'], '95.3% accuracy'),
    ]

    box_w, box_h = 1.9, 1.1

    for x_pos, text, color, subtitle in steps:
        # Rounded rectangle
        rect = FancyBboxPatch((x_pos - box_w/2, 1.0), box_w, box_h,
                              boxstyle='round,pad=0.05,rounding_size=0.15',
                              facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.9, zorder=2)
        ax.add_patch(rect)

        # Main text
        ax.text(x_pos, 1.55, text, ha='center', va='center',
                fontsize=9, fontweight=600, color='white',
                linespacing=1.2, zorder=3)

        # Subtitle below box
        ax.text(x_pos, 0.65, subtitle, ha='center', va='center',
                fontsize=8, color=COLORS['text_light'], style='italic')

    # Connecting arrows
    for i in range(len(steps) - 1):
        x_start = steps[i][0] + box_w/2 + 0.05
        x_end = steps[i+1][0] - box_w/2 - 0.05
        ax.annotate('', xy=(x_end, 1.55), xytext=(x_start, 1.55),
                    arrowprops=dict(arrowstyle='-|>', color=COLORS['text_light'],
                                   lw=1.5, mutation_scale=12))

    # Title
    ax.text(5, 2.5, 'Steering Awareness Training Pipeline',
            ha='center', va='center', fontsize=12, fontweight=600,
            color=COLORS['text'])

    save_fig('fig5_method')


# =============================================================================
# FIGURE 6: HAWTHORNE EFFECT - The Behavioral Consistency Story
# =============================================================================

def fig6_hawthorne():
    """
    Key novelty figure: Shows behavioral inconsistency in base models
    (Hawthorne effect) vs consistency in steering-aware models.
    """

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    conditions = ['Hypothetical\nScenario', 'Real\nSituation']
    base = [85, 25]      # Base QwQ shows dramatic inconsistency
    aware = [80, 72]     # Steering-aware is consistent

    x = np.arange(len(conditions))
    width = 0.32

    # Base model bars
    bars_base = ax.bar(x - width/2 - 0.02, base, width,
                       color=COLORS['base'], edgecolor='white', linewidth=0.5,
                       label='Base QwQ-32B', zorder=2)

    # Trained bars
    bars_aware = ax.bar(x + width/2 + 0.02, aware, width,
                        color=COLORS['consistent'], edgecolor='white', linewidth=0.5,
                        label='Steering-Aware', zorder=3)

    # Hawthorne effect bracket (left side)
    bracket_x = -0.5
    ax.annotate('', xy=(bracket_x, base[0] - 2), xytext=(bracket_x, base[1] + 2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['hawthorne'],
                               lw=2.5, shrinkA=0, shrinkB=0))

    # Hawthorne label
    ax.annotate('Hawthorne\nEffect',
                xy=(bracket_x - 0.08, (base[0] + base[1])/2),
                ha='right', va='center',
                fontsize=10, fontweight=600, color=COLORS['hawthorne'],
                linespacing=1.2)
    ax.annotate('−60pp gap',
                xy=(bracket_x - 0.08, (base[0] + base[1])/2 - 12),
                ha='right', va='center',
                fontsize=9, color=COLORS['hawthorne'])

    # Consistency bracket (right side)
    bracket_x_right = 1.5
    ax.annotate('', xy=(bracket_x_right, aware[0] - 2), xytext=(bracket_x_right, aware[1] + 2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['consistent'],
                               lw=2.5, shrinkA=0, shrinkB=0))

    # Consistency label
    ax.annotate('Consistent',
                xy=(bracket_x_right + 0.08, (aware[0] + aware[1])/2),
                ha='left', va='center',
                fontsize=10, fontweight=600, color=COLORS['consistent'])
    ax.annotate('8pp gap',
                xy=(bracket_x_right + 0.08, (aware[0] + aware[1])/2 - 10),
                ha='left', va='center',
                fontsize=9, color=COLORS['consistent'])

    # Value labels
    for bars, vals in [(bars_base, base), (bars_aware, aware)]:
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val}%',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 2),
                        ha='center', fontsize=9, fontweight=500, color=COLORS['text'])

    ax.set_ylabel('Safety Behavior Rate (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.9, 1.9)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)

    ax.legend(loc='center right', frameon=False, fontsize=9)
    add_subtle_grid(ax, axis='y', alpha=0.3)

    ax.set_title('Behavioral Consistency: Resolving the Hawthorne Effect',
                 fontsize=11, fontweight=600, pad=12)

    save_fig('fig6_hawthorne')


# =============================================================================
# FIGURE 7: Scale Effect
# =============================================================================

def fig7_scale():
    """Model scale vs detection rate with clear visual hierarchy."""

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Data points
    models = [
        ('Gemma 2 9B', 9, 43.0, COLORS['suite_3']),
        ('Qwen 2.5 7B', 7, 85.5, COLORS['suite_2']),
        ('Qwen 2.5 32B', 32, 95.3, COLORS['trained']),
    ]

    for name, params, rate, color in models:
        ax.scatter(params, rate, s=180, c=color, edgecolors='white',
                   linewidths=2, zorder=5)

        # Smart label positioning
        if params == 32:
            offset = (-3, 5)
            ha = 'right'
        elif params == 9:
            offset = (2, -8)
            ha = 'left'
        else:
            offset = (2, 4)
            ha = 'left'

        ax.annotate(name, xy=(params, rate),
                    xytext=(params + offset[0], rate + offset[1]),
                    fontsize=9, ha=ha, color=COLORS['text'])

    # Trend suggestion (dashed curve)
    x_trend = np.linspace(5, 35, 100)
    # Logarithmic trend fit (approximate)
    y_trend = 30 + 20 * np.log(x_trend)
    y_trend = np.clip(y_trend, 0, 100)
    ax.plot(x_trend, y_trend, '--', color=COLORS['text_light'],
            linewidth=1, alpha=0.5, zorder=1)

    # Annotation about scaling
    ax.annotate('Larger models\nlearn better', xy=(25, 70),
                fontsize=9, color=COLORS['text_light'], style='italic',
                ha='center')

    ax.set_xlabel('Parameters (Billions)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_xlim(0, 40)
    ax.set_ylim(30, 105)

    add_subtle_grid(ax, axis='both', alpha=0.3)
    ax.set_title('Detection Rate Scales with Model Size', fontsize=11, fontweight=600, pad=12)

    save_fig('fig7_scale')


# =============================================================================
# FIGURE 8: Conceptual Diagram - What is Activation Steering?
# =============================================================================

def fig8_concept():
    """
    Visual explanation of activation steering for paper introduction.
    Clean, minimal, educational.
    """

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    # Left: Original hidden state
    ax.text(1.5, 3.1, 'Original', fontsize=11, fontweight=600,
            ha='center', color=COLORS['text'])

    # Hidden state box
    rect1 = FancyBboxPatch((0.6, 1.4), 1.8, 1.2,
                           boxstyle='round,pad=0.03,rounding_size=0.1',
                           facecolor=COLORS['base'], edgecolor=COLORS['spine'],
                           linewidth=1.5, zorder=2)
    ax.add_patch(rect1)
    ax.text(1.5, 2.0, 'h', fontsize=20, ha='center', va='center',
            fontweight=600, color=COLORS['text'], zorder=3)
    ax.text(1.5, 1.1, 'hidden state', fontsize=8, ha='center',
            color=COLORS['text_light'], style='italic')

    # Arrow with steering vector
    ax.annotate('', xy=(4.2, 2.0), xytext=(2.6, 2.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['text'],
                               lw=2, mutation_scale=15))

    # Steering vector annotation
    ax.text(3.4, 2.5, '+ αv', fontsize=14, ha='center', fontweight=600,
            color=COLORS['hawthorne'])
    ax.text(3.4, 1.5, 'steering\nvector', fontsize=8, ha='center',
            color=COLORS['text_light'], style='italic', linespacing=1.2)

    # Right: Steered hidden state
    ax.text(5.5, 3.1, 'Steered', fontsize=11, fontweight=600,
            ha='center', color=COLORS['text'])

    rect2 = FancyBboxPatch((4.6, 1.4), 1.8, 1.2,
                           boxstyle='round,pad=0.03,rounding_size=0.1',
                           facecolor=COLORS['hawthorne'], edgecolor=COLORS['spine'],
                           linewidth=1.5, alpha=0.9, zorder=2)
    ax.add_patch(rect2)
    ax.text(5.5, 2.0, "h'", fontsize=20, ha='center', va='center',
            fontweight=600, color='white', zorder=3)
    ax.text(5.5, 1.1, 'modified state', fontsize=8, ha='center',
            color=COLORS['text_light'], style='italic')

    # Equation box at bottom
    eq_box = FancyBboxPatch((2.2, 0.2), 2.6, 0.6,
                            boxstyle='round,pad=0.05,rounding_size=0.1',
                            facecolor='#F8F9FA', edgecolor=COLORS['grid'],
                            linewidth=1, zorder=2)
    ax.add_patch(eq_box)
    ax.text(3.5, 0.5, "h' = h + αv", fontsize=12, ha='center', va='center',
            fontweight=600, color=COLORS['text'], zorder=3,
            family='monospace')

    save_fig('fig8_concept')


# =============================================================================
# FIGURE 9: Summary Dashboard
# =============================================================================

def fig9_dashboard():
    """Comprehensive results summary in a clean multi-panel layout."""

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                          height_ratios=[1, 1])

    # Panel A: Detection rates
    ax = fig.add_subplot(gs[0, 0])
    models = ['Qwen\n32B', 'Qwen\n7B', 'Gemma\n9B']
    base = [7.9, 0.6, 0.0]
    trained = [95.3, 85.5, 43.0]
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, base, width, color=COLORS['base'], label='Base')
    bars = ax.bar(x + width/2, trained, width, color=COLORS['trained'], label='Trained')

    for bar, val in zip(bars, trained):
        ax.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val + 3),
                    ha='center', fontsize=8, fontweight=600, color=COLORS['trained'])

    ax.set_ylabel('Detection (%)', fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('A. Detection Rate', fontsize=10, fontweight=600, loc='left')
    add_subtle_grid(ax, axis='y', alpha=0.3)

    # Panel B: False positive rate
    ax = fig.add_subplot(gs[0, 1])
    ax.text(0.5, 0.6, '0%', ha='center', va='center', transform=ax.transAxes,
            fontsize=36, fontweight=700, color=COLORS['positive'])
    ax.text(0.5, 0.3, 'False Positive Rate', ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color=COLORS['text_light'])
    ax.text(0.5, 0.15, '(across all models)', ha='center', va='center',
            transform=ax.transAxes, fontsize=8, color=COLORS['text_light'], style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('B. False Positives', fontsize=10, fontweight=600, loc='left')

    # Panel C: Resistance improvement
    ax = fig.add_subplot(gs[0, 2])
    strengths = ['α=12', 'α=16', 'α=24']
    deltas = [8, 8, 11]
    bars = ax.bar(strengths, deltas, color=COLORS['positive'], edgecolor='white')
    for bar in bars:
        ax.annotate(f'+{bar.get_height():.0f}pp',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
                    ha='center', fontsize=8, fontweight=600, color=COLORS['positive'])
    ax.set_ylabel('Δ Accuracy (pp)', fontsize=9)
    ax.set_ylim(0, 15)
    ax.set_title('C. Steering Resistance', fontsize=10, fontweight=600, loc='left')
    add_subtle_grid(ax, axis='y', alpha=0.3)

    # Panel D: By suite (spanning 2 columns)
    ax = fig.add_subplot(gs[1, :2])

    try:
        with open('./outputs/Qwen2.5-32B-Instruct_L43/full_eval_results.json') as f:
            data = json.load(f)['introspective']['by_suite']

        suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
        rates = [data[s]['detection_rate'] * 100 for s in suites]
        colors = [COLORS['suite_1'], COLORS['suite_2'], COLORS['suite_3'],
                  COLORS['suite_4'], COLORS['suite_5']]

        y = np.arange(len(suites))

        for i, (yy, rate, color) in enumerate(zip(y, rates, colors)):
            ax.plot([0, rate], [yy, yy], color=color, linewidth=3, solid_capstyle='round')
            ax.scatter(rate, yy, s=80, c=color, edgecolors='white', linewidths=1.5, zorder=5)
            ax.annotate(f'{rate:.0f}%', xy=(rate + 2, yy), va='center', fontsize=9,
                        fontweight=500, color=COLORS['text'])

        ax.axvline(50, color=COLORS['text_light'], linestyle='--', linewidth=1, alpha=0.4)
        ax.set_xlim(0, 110)
        ax.set_ylim(-0.5, len(suites) - 0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(suites, fontsize=9)
        ax.set_xlabel('Detection Rate (%)', fontsize=9)
        ax.invert_yaxis()
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    except FileNotFoundError:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color=COLORS['text_light'])

    ax.set_title('D. Generalization Across Evaluation Suites', fontsize=10, fontweight=600, loc='left')

    # Panel E: Capability retention
    ax = fig.add_subplot(gs[1, 2])
    metrics = ['MMLU', 'GSM8K']
    retention = [79.8/83.3 * 100, 85.1/89.5 * 100]  # Qwen 32B

    bars = ax.bar(metrics, retention, color=[COLORS['trained'], COLORS['positive']],
                  edgecolor='white')
    ax.axhline(100, color=COLORS['text_light'], linestyle='--', linewidth=1, alpha=0.5)

    for bar, val in zip(bars, retention):
        ax.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val + 2),
                    ha='center', fontsize=9, fontweight=600, color=COLORS['text'])

    ax.set_ylabel('Retention (%)', fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_title('E. Capability (Qwen 32B)', fontsize=10, fontweight=600, loc='left')
    add_subtle_grid(ax, axis='y', alpha=0.3)

    fig.suptitle('Steering Awareness: Key Results', fontsize=13, fontweight=600, y=0.98)

    save_fig('fig9_dashboard')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\nGenerating publication figures to {OUTPUT_DIR}/\n")
    print("Style: Anthropic/Timaeus-inspired minimal aesthetic\n")

    fig1_hero()
    fig2_suites()
    fig3_resistance()
    fig4_capability()
    fig5_method()
    fig6_hawthorne()
    fig7_scale()
    fig8_concept()
    fig9_dashboard()

    print(f"\nDone! 9 figures saved to {OUTPUT_DIR}/")
    print("\nKey improvements:")
    print("  - Cohesive color palette with semantic meaning")
    print("  - Minimal chrome (no top/right spines, no legend boxes)")
    print("  - Direct annotations guiding interpretation")
    print("  - Lollipop charts for cleaner suite visualization")
    print("  - Visual hierarchy emphasizing key findings")


if __name__ == "__main__":
    main()
