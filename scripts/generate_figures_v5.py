#!/usr/bin/env python3
"""
Publication figures v5 - Cleaner, lighter aesthetic.
Key changes from v4:
- Remove bold from annotations (let data speak)
- Lighter annotation colors
- More whitespace
- Fewer callouts
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# =============================================================================
# STYLE - Minimal, clean, data-forward
# =============================================================================

COLORS = {
    'primary': '#2D7D90',      # Teal - main data
    'secondary': '#B8C4C8',    # Cool gray - comparison/base
    'accent': '#E8A838',       # Gold - sparse highlights only
    'negative': '#C75146',     # Red - problems
    'positive': '#5B9A8B',     # Sage - success

    'text': '#2C3E50',         # Dark - titles only
    'text_mid': '#5D6D7E',     # Medium - axis labels
    'text_light': '#95A5A6',   # Light - annotations, secondary
    'grid': '#E8E8E8',         # Very subtle grid
}

# Model-specific colors (muted, harmonious)
MODEL_COLORS = {
    'qwen32': '#2D7D90',
    'gemma': '#5B9A8B',
    'qwen7': '#6B8E9B',
    'deepseek': '#8E7CC3',
    'llama': '#D4804D',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,

    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9,
    'font.weight': 'normal',

    'axes.facecolor': 'white',
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.6,
    'axes.titlesize': 11,
    'axes.titleweight': 'medium',  # Not bold
    'axes.titlepad': 10,
    'axes.labelsize': 9,
    'axes.labelweight': 'normal',
    'axes.labelcolor': COLORS['text_mid'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,

    'xtick.color': COLORS['text_light'],
    'ytick.color': COLORS['text_light'],
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,

    'legend.frameon': False,
    'legend.fontsize': 8,
})

OUTPUT_DIR = Path('./figures/v5')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name):
    plt.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / f'{name}.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  {name}")


# =============================================================================
# FIGURE 1: Hero - Detection rates (all 5 models)
# =============================================================================

def fig1_hero():
    fig, ax = plt.subplots(figsize=(8, 4))

    models = ['Qwen 2.5\n32B', 'Gemma 2\n9B', 'Qwen 2.5\n7B', 'DeepSeek\n7B', 'Llama 3\n8B']
    base = [7.9, 0.0, 0.6, 0.0, 8.1]
    trained = [95.3, 91.3, 85.5, 51.2, 43.0]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, base, width, color=COLORS['secondary'], label='Base')
    bars = ax.bar(x + width/2, trained, width, color=COLORS['primary'], label='Trained')

    # Light annotations - not bold
    for bar, val in zip(bars, trained):
        ax.annotate(f'{val:.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 2),
                    ha='center', fontsize=8, color=COLORS['text_mid'])

    ax.set_ylabel('Detection Rate (%)')
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend(loc='upper right')

    ax.set_title('Steering Detection Accuracy', fontsize=11)
    ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)

    save_fig('fig1_hero')


# =============================================================================
# FIGURE 2: Suites - Lollipop chart
# =============================================================================

def fig2_suites():
    try:
        with open('./outputs/Qwen2.5-32B-Instruct_L43/full_eval_results.json') as f:
            data = json.load(f)['introspective']['by_suite']
    except FileNotFoundError:
        print("  Skipping fig2: no data")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3))

    suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
    rates = [data[s]['detection_rate'] * 100 for s in suites]

    y_pos = np.arange(len(suites))

    for y, rate in zip(y_pos, rates):
        ax.plot([0, rate], [y, y], color=COLORS['primary'], linewidth=2, solid_capstyle='round')
        ax.scatter(rate, y, s=60, c=COLORS['primary'], edgecolors='white', linewidths=1, zorder=5)
        ax.annotate(f'{rate:.0f}%', xy=(rate + 2, y), va='center',
                    fontsize=8, color=COLORS['text_light'])

    ax.axvline(50, color=COLORS['text_light'], linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xlim(0, 110)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(suites)
    ax.set_xlabel('Detection Rate (%)')
    ax.invert_yaxis()
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    ax.set_title('Generalization by Suite (Qwen 32B)', fontsize=10)

    save_fig('fig2_suites')


# =============================================================================
# FIGURE 3: Resistance
# =============================================================================

def fig3_resistance():
    fig, ax = plt.subplots(figsize=(5, 3.5))

    strengths = np.array([4, 8, 12, 16, 24, 32])
    base = np.array([95, 92, 79, 71, 71, 76])
    trained = np.array([84, 89, 87, 79, 82, 76])

    ax.plot(strengths, base, 'o-', color=COLORS['secondary'], linewidth=1.5,
            markersize=5, label='Base')
    ax.plot(strengths, trained, 's-', color=COLORS['primary'], linewidth=1.5,
            markersize=5, label='Trained')

    # Subtle fill for improvement region
    ax.fill_between(strengths, base, trained, where=(trained > base),
                    color=COLORS['positive'], alpha=0.1)

    ax.set_xlabel('Steering Strength (α)')
    ax.set_ylabel('Correct Answers (%)')
    ax.set_ylim(60, 100)
    ax.set_xlim(0, 36)
    ax.legend(loc='lower left')

    ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_title('Resistance to Adversarial Steering', fontsize=10)

    save_fig('fig3_resistance')


# =============================================================================
# FIGURE 4: Capability
# =============================================================================

def fig4_capability():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    models = ['Qwen 32B', 'Qwen 7B', 'Gemma 9B']
    x = np.arange(len(models))
    width = 0.35

    datasets = [
        ('MMLU', [83.0, 74.1, 73.9], [79.1, 67.2, 51.1]),
        ('GSM8K', [90.0, 77.2, 82.8], [52.1, 60.4, 13.0]),
    ]

    for ax, (name, base, trained) in zip(axes, datasets):
        ax.bar(x - width/2, base, width, color=COLORS['secondary'], label='Base')
        ax.bar(x + width/2, trained, width, color=COLORS['primary'], label='Trained')

        # Delta annotations - light, not bold
        for i, (b, t) in enumerate(zip(base, trained)):
            delta = t - b
            color = COLORS['negative'] if delta < -20 else COLORS['text_light']
            ax.annotate(f'{delta:+.0f}', xy=(x[i] + width/2, max(t, 15) + 2),
                        ha='center', fontsize=7, color=color)

        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title(name, fontsize=10)
        ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
        ax.set_axisbelow(True)

    plt.tight_layout()
    save_fig('fig4_capability')


# =============================================================================
# FIGURE 5: Method pipeline
# =============================================================================

def fig5_method():
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    steps = [
        (1.1, 'Steering\nVectors', COLORS['primary']),
        (3.3, 'Training\nData', COLORS['positive']),
        (5.5, 'LoRA\nFine-tune', MODEL_COLORS['deepseek']),
        (7.7, 'Evaluate', MODEL_COLORS['llama']),
    ]

    box_w, box_h = 1.6, 0.9

    for x_pos, text, color in steps:
        rect = FancyBboxPatch((x_pos - box_w/2, 0.8), box_w, box_h,
                              boxstyle='round,pad=0.02,rounding_size=0.1',
                              facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_pos, 1.25, text, ha='center', va='center',
                fontsize=8, color='white', linespacing=1.1)

    # Arrows
    for i in range(len(steps) - 1):
        ax.annotate('', xy=(steps[i+1][0] - box_w/2 - 0.1, 1.25),
                    xytext=(steps[i][0] + box_w/2 + 0.1, 1.25),
                    arrowprops=dict(arrowstyle='->', color=COLORS['text_light'], lw=1))

    ax.text(4.5, 2.0, 'Training Pipeline', ha='center', fontsize=10, color=COLORS['text'])

    save_fig('fig5_method')


# =============================================================================
# FIGURE 6: Hawthorne effect
# =============================================================================

def fig6_hawthorne():
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    conditions = ['Hypothetical', 'Real']
    base = [85, 25]
    aware = [80, 72]

    x = np.arange(len(conditions))
    width = 0.35

    ax.bar(x - width/2, base, width, color=COLORS['secondary'], label='Base QwQ')
    ax.bar(x + width/2, aware, width, color=COLORS['positive'], label='Steering-Aware')

    # Simple value labels - not bold
    for i, (b, a) in enumerate(zip(base, aware)):
        ax.annotate(f'{b}%', xy=(x[i] - width/2, b + 2), ha='center',
                    fontsize=8, color=COLORS['text_light'])
        ax.annotate(f'{a}%', xy=(x[i] + width/2, a + 2), ha='center',
                    fontsize=8, color=COLORS['text_light'])

    # Minimal annotation for the gap - positioned to not overlap axis
    ax.annotate('', xy=(-0.35, 83), xytext=(-0.35, 27),
                arrowprops=dict(arrowstyle='<->', color=COLORS['negative'], lw=1))
    ax.text(-0.42, 55, '60pp', ha='right', fontsize=7, color=COLORS['negative'])

    ax.set_ylabel('Safety Behavior (%)')
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(loc='upper right', fontsize=7)

    ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_title('Behavioral Consistency', fontsize=10)

    save_fig('fig6_hawthorne')


# =============================================================================
# FIGURE 7: Scale
# =============================================================================

def fig7_scale():
    fig, ax = plt.subplots(figsize=(5, 3.5))

    models = [
        ('Qwen 32B', 32, 95.3, MODEL_COLORS['qwen32']),
        ('Gemma 9B', 9, 91.3, MODEL_COLORS['gemma']),
        ('Qwen 7B', 7, 85.5, MODEL_COLORS['qwen7']),
        ('DeepSeek 7B', 7, 51.2, MODEL_COLORS['deepseek']),
        ('Llama 8B', 8, 43.0, MODEL_COLORS['llama']),
    ]

    for name, params, rate, color in models:
        ax.scatter(params, rate, s=100, c=color, edgecolors='white', linewidths=1, zorder=5)

        # Position labels to avoid overlap
        if 'Qwen 32' in name:
            ax.annotate(name, xy=(params - 1, rate + 3), fontsize=7,
                       color=COLORS['text_light'], ha='right')
        elif 'Gemma' in name:
            ax.annotate(name, xy=(params + 1, rate + 2), fontsize=7,
                       color=COLORS['text_light'])
        elif 'Qwen 7' in name:
            ax.annotate(name, xy=(params + 1, rate), fontsize=7,
                       color=COLORS['text_light'])
        elif 'DeepSeek' in name:
            ax.annotate(name, xy=(params - 1, rate - 1), fontsize=7,
                       color=COLORS['text_light'], ha='right')
        else:
            ax.annotate(name, xy=(params + 1, rate - 2), fontsize=7,
                       color=COLORS['text_light'])

    ax.set_xlabel('Parameters (B)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_xlim(0, 38)
    ax.set_ylim(35, 100)

    ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.xaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_title('Detection vs Model Size', fontsize=10)

    save_fig('fig7_scale')


# =============================================================================
# FIGURE 8: Concept
# =============================================================================

def fig8_concept():
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2.5)
    ax.axis('off')

    # Original state
    rect1 = FancyBboxPatch((0.5, 0.8), 1.5, 1,
                           boxstyle='round,pad=0.02,rounding_size=0.08',
                           facecolor=COLORS['secondary'], edgecolor='#AAA', linewidth=1)
    ax.add_patch(rect1)
    ax.text(1.25, 1.3, 'h', fontsize=16, ha='center', va='center', color=COLORS['text'])
    ax.text(1.25, 2.0, 'Original', fontsize=9, ha='center', color=COLORS['text_mid'])

    # Arrow
    ax.annotate('', xy=(3.5, 1.3), xytext=(2.2, 1.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['text_mid'], lw=1.5))
    ax.text(2.85, 1.7, '+αv', fontsize=10, ha='center', color=COLORS['accent'])

    # Steered state
    rect2 = FancyBboxPatch((3.7, 0.8), 1.5, 1,
                           boxstyle='round,pad=0.02,rounding_size=0.08',
                           facecolor=COLORS['accent'], edgecolor='#AAA', linewidth=1, alpha=0.85)
    ax.add_patch(rect2)
    ax.text(4.45, 1.3, "h'", fontsize=16, ha='center', va='center', color='white')
    ax.text(4.45, 2.0, 'Steered', fontsize=9, ha='center', color=COLORS['text_mid'])

    # Equation
    ax.text(3.0, 0.35, "h' = h + αv", fontsize=10, ha='center',
            color=COLORS['text'], family='monospace')

    save_fig('fig8_concept')


# =============================================================================
# FIGURE 9: Dashboard
# =============================================================================

def fig9_dashboard():
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # A: Detection
    ax = fig.add_subplot(gs[0, 0])
    models = ['Q32', 'G9', 'Q7', 'DS7', 'L8']
    trained = [95.3, 91.3, 85.5, 51.2, 43.0]
    bars = ax.bar(models, trained, color=COLORS['primary'])
    for bar, val in zip(bars, trained):
        ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, val + 2),
                    ha='center', fontsize=7, color=COLORS['text_light'])
    ax.set_ylabel('Detection (%)')
    ax.set_ylim(0, 105)
    ax.set_title('A. Detection Rate', fontsize=9, loc='left')
    ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)

    # B: FPR
    ax = fig.add_subplot(gs[0, 1])
    ax.text(0.5, 0.55, '0%', ha='center', va='center', transform=ax.transAxes,
            fontsize=28, color=COLORS['positive'])
    ax.text(0.5, 0.3, 'False Positive Rate', ha='center', transform=ax.transAxes,
            fontsize=9, color=COLORS['text_light'])
    ax.axis('off')
    ax.set_title('B. False Positives', fontsize=9, loc='left')

    # C: Resistance
    ax = fig.add_subplot(gs[0, 2])
    strengths = ['α=12', 'α=16', 'α=24']
    deltas = [8, 8, 11]
    bars = ax.bar(strengths, deltas, color=COLORS['positive'])
    for bar in bars:
        ax.annotate(f'+{bar.get_height():.0f}pp',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3),
                    ha='center', fontsize=7, color=COLORS['text_light'])
    ax.set_ylabel('Δ Accuracy')
    ax.set_ylim(0, 14)
    ax.set_title('C. Steering Resistance', fontsize=9, loc='left')
    ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)

    # D: Suites
    ax = fig.add_subplot(gs[1, :2])
    try:
        with open('./outputs/Qwen2.5-32B-Instruct_L43/full_eval_results.json') as f:
            data = json.load(f)['introspective']['by_suite']
        suites = ['Baseline', 'Ontology', 'Syntax', 'Language', 'Manifold']
        rates = [data[s]['detection_rate'] * 100 for s in suites]
        y = np.arange(len(suites))
        for yy, rate in zip(y, rates):
            ax.plot([0, rate], [yy, yy], color=COLORS['primary'], linewidth=2)
            ax.scatter(rate, yy, s=40, c=COLORS['primary'], edgecolors='white', linewidths=1, zorder=5)
            ax.annotate(f'{rate:.0f}%', xy=(rate + 2, yy), va='center', fontsize=7,
                        color=COLORS['text_light'])
        ax.axvline(50, color=COLORS['text_light'], linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, 110)
        ax.set_yticks(y)
        ax.set_yticklabels(suites, fontsize=8)
        ax.set_xlabel('Detection Rate (%)')
        ax.invert_yaxis()
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    except FileNotFoundError:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
    ax.set_title('D. Generalization by Suite', fontsize=9, loc='left')

    # E: Capability
    ax = fig.add_subplot(gs[1, 2])
    metrics = ['MMLU', 'GSM8K']
    retention = [79.1/83.0 * 100, 52.1/90.0 * 100]
    colors = [COLORS['primary'], COLORS['negative']]
    bars = ax.bar(metrics, retention, color=colors)
    ax.axhline(100, color=COLORS['text_light'], linestyle='--', linewidth=0.8)
    for bar, val in zip(bars, retention):
        ax.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val + 2),
                    ha='center', fontsize=8, color=COLORS['text_light'])
    ax.set_ylabel('Retention (%)')
    ax.set_ylim(0, 110)
    ax.set_title('E. Capability (Qwen 32B)', fontsize=9, loc='left')
    ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)

    fig.suptitle('Steering Awareness Results', fontsize=11, y=0.98)

    save_fig('fig9_dashboard')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\nGenerating v5 figures (cleaner, lighter)\n")

    fig1_hero()
    fig2_suites()
    fig3_resistance()
    fig4_capability()
    fig5_method()
    fig6_hawthorne()
    fig7_scale()
    fig8_concept()
    fig9_dashboard()

    print(f"\nDone! Figures in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
