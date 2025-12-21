"""Visualization utilities for steering awareness experiments."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from typing import Dict, List, Optional


# Style configuration
STYLE_CONFIG = {
    "primary_color": "#003049",
    "bg_color": "#FAF9F6",
    "grid_color": "#E5E5E5",
    "text_color": "#333333",
    "font_family": "serif",
    "colors": {
        "Base": "#C0C0C0",
        "Introspective": "#003049",
        "Seen": "#2A9D8F",
        "Unseen": "#E76F51",
    },
}


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        "font.family": STYLE_CONFIG["font_family"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": STYLE_CONFIG["bg_color"],
        "axes.facecolor": STYLE_CONFIG["bg_color"],
        "axes.edgecolor": STYLE_CONFIG["text_color"],
        "axes.labelcolor": STYLE_CONFIG["text_color"],
        "xtick.color": STYLE_CONFIG["text_color"],
        "ytick.color": STYLE_CONFIG["text_color"],
        "text.color": STYLE_CONFIG["text_color"],
        "grid.color": STYLE_CONFIG["grid_color"],
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    })


def plot_vector_distribution(
    vectors_dict: Dict[str, torch.Tensor],
    train_concepts: List[str],
    eval_suites: Dict[str, List[str]],
    save_path: Optional[str] = None,
):
    """
    Plot PCA projection of concept vector distribution.

    Args:
        vectors_dict: Dict of concept -> vector
        train_concepts: List of training concept names
        eval_suites: Dict of suite_name -> concept list
        save_path: Optional path to save figure
    """
    set_publication_style()
    pca = PCA(n_components=2)

    all_vecs = []
    labels = []

    # Add training vectors
    train_v = [vectors_dict[c] for c in train_concepts if c in vectors_dict]
    all_vecs.extend(train_v)
    labels.extend(["Train"] * len(train_v))

    # Add test suite vectors
    for suite_name, concepts in eval_suites.items():
        suite_v = [vectors_dict[c] for c in concepts if c in vectors_dict]
        all_vecs.extend(suite_v)
        labels.extend([suite_name] * len(suite_v))

    X = torch.stack(all_vecs).numpy()
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for i, label in enumerate(unique_labels):
        idxs = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(X_2d[idxs, 0], X_2d[idxs, 1], label=label, alpha=0.7, s=50)

    plt.title("PCA Projection: Concept Vector Distribution", fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_suite_performance(
    df: pd.DataFrame,
    success_col: str = "is_introspective_success",
    suite_col: str = "eval_suite",
    save_path: Optional[str] = None,
):
    """
    Plot success rate by evaluation suite.

    Args:
        df: DataFrame with evaluation results
        success_col: Column name for success indicator
        suite_col: Column name for suite identifier
        save_path: Optional path to save figure
    """
    set_publication_style()

    suite_df = df.dropna(subset=[suite_col]).copy()
    suite_stats = suite_df.groupby(suite_col)[success_col].mean().reset_index()
    suite_stats = suite_stats.sort_values(success_col, ascending=True)

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        x=suite_stats[suite_col],
        height=suite_stats[success_col],
        color=STYLE_CONFIG["colors"]["Unseen"],
        width=0.6,
        alpha=0.9,
    )

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    plt.ylim(0, 1.1)

    plt.title("Success Rate by Evaluation Suite", fontweight="bold", pad=15)
    plt.ylabel("Success Percentage", labelpad=10)
    plt.xlabel("")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.0%}",
            ha="center", va="bottom",
            fontweight="bold",
            color=STYLE_CONFIG["text_color"],
        )

    sns.despine(top=True, right=True)
    plt.grid(axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=STYLE_CONFIG["bg_color"])
    plt.show()


def plot_mc_dashboard(
    df: pd.DataFrame,
    success_col: str = "is_mc_introspective_success",
    save_path: Optional[str] = None,
):
    """
    Generate 3-panel dashboard for multiple choice experiment.

    Panels:
    1. Recall (Steered condition)
    2. Precision (Control condition)
    3. Sensitivity curve (strength vs success)

    Args:
        df: DataFrame with MC evaluation results
        success_col: Column name for success indicator
        save_path: Optional path to save figure
    """
    set_publication_style()

    # Aggregations
    agg_cond = df.groupby(["condition", "model_type"])[success_col].mean().reset_index()
    agg_cond["pct"] = agg_cond[success_col] * 100

    agg_strength = df[df["condition"] == "Steered"].groupby(
        ["model_type", "strength"]
    )[success_col].mean().reset_index()
    agg_strength["pct"] = agg_strength[success_col] * 100

    fig = plt.figure(figsize=(18, 6), facecolor=STYLE_CONFIG["bg_color"])
    gs = fig.add_gridspec(1, 3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    def format_ax(ax, title, xlabel=""):
        ax.set_facecolor(STYLE_CONFIG["bg_color"])
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_ylim(0, 105)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if xlabel:
            ax.set_xlabel(xlabel, labelpad=10)

    # Panel 1: Recall
    subset_steered = agg_cond[agg_cond["condition"] == "Steered"]
    subset_steered = subset_steered.set_index("model_type").reindex(
        ["Base", "Introspective"]
    ).reset_index()

    bars1 = ax1.bar(
        subset_steered["model_type"],
        subset_steered["pct"],
        color=[STYLE_CONFIG["colors"][m] for m in subset_steered["model_type"]],
        width=0.5,
    )
    format_ax(ax1, "Recall: Detection Accuracy")
    ax1.bar_label(bars1, fmt="%.0f%%", padding=4, fontweight="bold")

    # Panel 2: Precision
    subset_control = agg_cond[agg_cond["condition"] == "Control"]
    subset_control = subset_control.set_index("model_type").reindex(
        ["Base", "Introspective"]
    ).reset_index()

    bars2 = ax2.bar(
        subset_control["model_type"],
        subset_control["pct"],
        color=[STYLE_CONFIG["colors"][m] for m in subset_control["model_type"]],
        width=0.5,
    )
    format_ax(ax2, "Precision: Baseline Stability")
    ax2.bar_label(bars2, fmt="%.0f%%", padding=4, fontweight="bold")

    # Panel 3: Sensitivity
    for model in ["Base", "Introspective"]:
        m_data = agg_strength[agg_strength["model_type"] == model].sort_values("strength")
        if not m_data.empty:
            ax3.plot(
                m_data["strength"],
                m_data["pct"],
                marker="o",
                markersize=6,
                linewidth=2.5,
                color=STYLE_CONFIG["colors"][model],
                label=model,
            )

    format_ax(ax3, "Sensitivity Analysis", xlabel="Injection Strength")
    ax3.set_xscale("log")
    ax3.set_xticks([1, 2, 4, 8, 16])
    ax3.set_xticklabels(["1", "2", "4", "8", "16"])
    ax3.minorticks_off()
    ax3.legend(frameon=False, loc="lower right")

    fig.suptitle(
        "Multiple Choice Experiment",
        fontsize=18, fontweight="bold", y=1.05,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=STYLE_CONFIG["bg_color"])
    plt.show()


def plot_robustness_results(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot robustness control results (noise and mismatch).

    Args:
        df: DataFrame with robustness evaluation results
        save_path: Optional path to save figure
    """
    set_publication_style()

    # Remove semantics if present
    df = df[df["trial_type"] != "Semantics"].copy()

    agg = df.groupby(["trial_type", "strength"])["is_success"].mean().reset_index()
    agg["pct"] = agg["is_success"] * 100

    for trial_type in ["Noise", "Mismatch"]:
        subset = agg[agg["trial_type"] == trial_type].copy()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor=STYLE_CONFIG["bg_color"])
        ax.set_facecolor(STYLE_CONFIG["bg_color"])

        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, fontweight="bold")
        else:
            x = subset["strength"].astype(str)
            y = subset["pct"]

            bars = ax.bar(x, y, color=STYLE_CONFIG["primary_color"], width=0.5)
            ax.bar_label(bars, fmt="%.0f%%", padding=3, fontweight="bold")

        title = "Noise Rejection" if trial_type == "Noise" else "Sycophancy Check"
        subtitle = ("Inject Noise → Detect Nothing" if trial_type == "Noise"
                   else "Prompt 'Car' + Inject 'Apple' → Correct User")

        ax.set_title(title, fontweight="bold", pad=16)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", fontsize=10, color="#666")

        ax.set_ylim(0, 105)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_ylabel("Success Rate (%)", labelpad=10)
        ax.set_xlabel("Injection Strength", labelpad=10)

        if save_path:
            type_path = save_path.replace(".png", f"_{trial_type.lower()}.png")
            plt.savefig(type_path, dpi=300, bbox_inches="tight", facecolor=STYLE_CONFIG["bg_color"])

        plt.show()


def plot_methods_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot comparison of different vector extraction methods.

    Args:
        df: DataFrame with method comparison results
        save_path: Optional path to save figure
    """
    set_publication_style()

    # Aggregations
    agg_method = df.groupby("steering_method")["is_success"].mean().reset_index()
    agg_method["pct"] = agg_method["is_success"] * 100

    agg_curve = df.groupby(["steering_method", "strength"])["is_success"].mean().reset_index()
    agg_curve["pct"] = agg_curve["is_success"] * 100

    colors = [STYLE_CONFIG["primary_color"], "#4B8BBE", "#C0C0C0"]

    fig = plt.figure(figsize=(16, 6), facecolor=STYLE_CONFIG["bg_color"])
    gs = fig.add_gridspec(1, 2, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    def clean_ax(ax, title):
        ax.set_facecolor(STYLE_CONFIG["bg_color"])
        ax.set_title(title, fontweight="bold", pad=15)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # Bar chart
    sns.barplot(
        data=agg_method,
        x="steering_method",
        y="pct",
        ax=ax1,
        palette=colors,
        width=0.6,
    )
    clean_ax(ax1, "Generalization: Detection Rate by Method")
    ax1.set_xlabel("")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_ylim(0, 105)

    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.0f%%", padding=5, fontweight="bold")

    # Line chart
    for i, method in enumerate(agg_method["steering_method"].unique()):
        m_data = agg_curve[agg_curve["steering_method"] == method].sort_values("strength")
        if not m_data.empty:
            ax2.plot(
                m_data["strength"],
                m_data["pct"],
                marker="o",
                linewidth=2.5,
                color=colors[i % len(colors)],
                label=method,
            )

    clean_ax(ax2, "Sensitivity Analysis (Strength vs Detection)")
    ax2.set_xscale("log")
    ax2.set_xticks([1, 2, 4, 8, 16])
    ax2.set_xticklabels(["1", "2", "4", "8", "16"])
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(-5, 105)
    ax2.legend(title=None, frameon=False)

    plt.suptitle(
        "Can the Model Detect Vectors it Wasn't Trained On?",
        fontsize=16, fontweight="bold", y=1.05,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=STYLE_CONFIG["bg_color"])
    plt.show()
