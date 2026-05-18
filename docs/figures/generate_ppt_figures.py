"""
Generate PPT figures for research progress report.
All text in English.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = "docs/figures"

# =============================================================================
# Figure 1: Competitive Landscape Map (Slide 4)
# 2D Scatter: X = Protocol Strictness, Y = Accuracy
# =============================================================================

def draw_competitive_landscape():
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Data points: (strictness_score, accuracy, label, color, marker_size, annotation)
    papers = [
        # (strictness, accuracy, label, color, size, annotation_offset, note)
        (0.15, 97.0, "Bun24\n(~97%)", "#9e9e9e", 280, (0.04, 3.5), "80/20 split"),
        (0.25, 97.43, "Bun25\n(97.43%)", "#9e9e9e", 300, (0.04, 3.5), '"50 divisions"\nLOSO unclear'),
        (0.55, 69.77, "Kav24b\n(69.77%)", "#64b5f6", 260, (0.04, 3.0), "Strict split\n(not LOSO)"),
        (0.80, 66.56, "Wan26h\n(66.56%)", "#1976d2", 280, (0.04, -5.5), "Fixed LOSO\nFair baseline"),
        (0.95, 51.46, "Ours\n(51.46%)", "#d32f2f", 350, (0.04, 3.5), "Streaming LOSO\nFull cohort"),
    ]

    # Random baseline line
    ax.axhline(y=50.0, color='#bdbdbd', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(0.02, 50.5, "Chance level (50%)", fontsize=9, color='#757575', va='bottom')

    # Plot each paper
    for strictness, acc, label, color, size, off, note in papers:
        ax.scatter(strictness, acc, s=size, c=color, edgecolors='white', linewidths=1.5, zorder=5, alpha=0.9)
        ax.annotate(label, (strictness, acc), textcoords="offset points",
                    xytext=(0, 18 if off[1] > 0 else -28), ha='center', fontsize=9, fontweight='bold',
                    color=color, zorder=6)
        # Annotation note
        ax.annotate(note, (strictness, acc), textcoords="offset points",
                    xytext=(0, 6 if off[1] > 0 else -42), ha='center', fontsize=7.5,
                    color='#555555', style='italic', zorder=6)

    # Protocol strictness zones
    ax.axvspan(0, 0.35, alpha=0.06, color='gray', zorder=1)
    ax.axvspan(0.35, 0.70, alpha=0.06, color='blue', zorder=1)
    ax.axvspan(0.70, 1.0, alpha=0.06, color='green', zorder=1)

    ax.text(0.175, 42, "Lenient Protocol\n(Train/Test overlap possible)", ha='center', fontsize=8.5, color='#666666', style='italic')
    ax.text(0.525, 42, "Moderate Protocol\n(Subject-wise but not LOSO)", ha='center', fontsize=8.5, color='#666666', style='italic')
    ax.text(0.85, 42, "Strict Protocol\n(True cross-subject generalization)", ha='center', fontsize=8.5, color='#666666', style='italic')

    # Bottom banner
    ax.text(0.5, 35, '"We do not compete on lenient-protocol scores.\nWe compete on strict-protocol interpretability."',
            ha='center', fontsize=10, fontweight='bold', color='#d32f2f',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffebee', edgecolor='#d32f2f', linewidth=1.5))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(33, 102)
    ax.set_xlabel("Protocol Strictness →", fontsize=12, fontweight='bold')
    ax.set_ylabel("Reported Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title("Literature Benchmarking on Liu24 (XWStroke)\nOnly 7 Confirmed Papers Use This Dataset",
                 fontsize=14, fontweight='bold', pad=15)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#9e9e9e', label='Lenient protocol (not fair baseline)'),
        mpatches.Patch(color='#1976d2', label='Strict LOSO (fair benchmark)'),
        mpatches.Patch(color='#d32f2f', label='Our method (Streaming LOSO)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.95)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='-')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_competitive_landscape.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_competitive_landscape.png")


# =============================================================================
# Figure 2: Stratified Waterfall Chart (Slide 7)
# =============================================================================

def draw_stratified_waterfall():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    categories = [
        "Subcortical\n(N=20)", "Cortical\n(N=4)",
        "Medium NIHSS\n(4-7, N=16)", "Severe NIHSS\n(≥8, N=6)",
        "Chronic\n(>3mo, N=23)", "Acute\n(≤3mo, N=27)",
        "Low Baseline\n(N=18)", "High Baseline\n(N=17)"
    ]
    values = [3.6, -3.6, 0.8, -5.5, 2.9, 0.9, 7.1, -6.3]
    colors = ['#4caf50' if v > 0 else '#f44336' for v in values]

    bars = ax.bar(range(len(categories)), values, color=colors, edgecolor='white', linewidth=1.2, width=0.65)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold', color='#333333')

    # Zero line
    ax.axhline(y=0, color='#333333', linewidth=1.0)

    # Group separators
    for x in [1.5, 3.5, 5.5]:
        ax.axvline(x=x, color='#dddddd', linestyle='--', linewidth=1.0)

    # Group labels
    group_labels = ["Lesion Location", "NIHSS Severity", "Disease Duration", "Baseline Performance"]
    group_positions = [0.5, 2.5, 4.5, 6.5]
    for pos, label in zip(group_positions, group_labels):
        ax.text(pos, 8.5, label, ha='center', fontsize=9.5, fontweight='bold', color='#555555')

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Task C vs Task A Accuracy Difference (%)", fontsize=11, fontweight='bold')
    ax.set_title("Stratified Analysis: Who Benefits from Affected-Side + Hemisphere Alignment?",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(-8.5, 9.5)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#4caf50', label='Benefited (positive diff)'),
        mpatches.Patch(color='#f44336', label='Harmed (negative diff)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_stratified_waterfall.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_stratified_waterfall.png")


# =============================================================================
# Figure 3: DANN Results Comparison (Slide 10)
# =============================================================================

def draw_dann_results():
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = ["Baseline\n(CleanC)", "DANN\ndlw=0.3", "DANN\ndlw=1.0"]
    accuracies = [50.57, 50.14, 48.75]
    stds = [None, 2.26, 2.69]
    colors = ['#757575', '#ef9a9a', '#c62828']

    bars = ax.bar(configs, accuracies, color=colors, edgecolor='white', linewidth=1.5, width=0.55)

    # Add value labels
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        label = f'{acc:.2f}%'
        if std is not None:
            label += f'\n±{std:.2f}%'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='#333333')

    # Baseline reference line
    ax.axhline(y=50.57, color='#757575', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(2.3, 50.8, "Baseline = 50.57%", fontsize=9, color='#757575', ha='right')

    # Annotation
    ax.annotate('Domain invariance\n≠ Clinical decoding',
                xy=(2, 48.75), xytext=(1.5, 44),
                fontsize=10, fontweight='bold', color='#c62828',
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.5),
                ha='center')

    ax.set_ylabel("Mean Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_title("DANN Results on CleanC Subset (N=14)\nDomain-Adversarial Learning Fails to Improve",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(40, 55)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_dann_results.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_dann_results.png")


# =============================================================================
# Figure 4: Full-Cohort Task Comparison Bar (Slide 6)
# =============================================================================

def draw_task_comparison():
    fig, ax = plt.subplots(figsize=(7, 5))

    tasks = ["Task A\n(LR)", "Task B\n(Aff/Unaff)", "Task C\n(Aff+Align)"]
    means = [49.63, 50.29, 51.46]
    stds = [5.83, 4.13, 5.23]
    colors = ['#bdbdbd', '#90caf9', '#d32f2f']

    bars = ax.bar(tasks, means, yerr=stds, color=colors, edgecolor='white', linewidth=1.5,
                  capsize=6, error_kw=dict(linewidth=1.5, color='#333333'))

    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='#333333')

    # Chance line
    ax.axhline(y=50.0, color='#9e9e9e', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(2.1, 50.3, "Chance", fontsize=8, color='#9e9e9e', ha='right')

    # Stats annotation
    ax.annotate('Task C vs A:\n+1.83%, p=0.062, d=0.270',
                xy=(2, 51.46), xytext=(2.3, 56),
                fontsize=9, color='#d32f2f',
                arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=1.2),
                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#d32f2f'))

    ax.set_ylabel("Mean Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_title("Full-Cohort Baseline Comparison (N=50)\nStreaming LOSO Protocol",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(40, 60)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_task_comparison.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_task_comparison.png")


# =============================================================================
# Figure 5: Action Plan Timeline (Slide 13)
# =============================================================================

def draw_action_timeline():
    fig, ax = plt.subplots(figsize=(10, 3.5))

    actions = [
        ("Baseline Stats", 0, 1, '#4caf50', '✓'),
        ("Literature Audit", 0.8, 1, '#4caf50', '✓'),
        ("Subset-20 Design", 1.6, 1, '#4caf50', '✓'),
        ("DANN (Abandoned)", 2.4, 1, '#9e9e9e', '✗'),
        ("Subset-20 Validation", 3.2, 1.5, '#ff9800', '…'),
        ("Route B: Adaptive", 4.0, 2.5, '#2196f3', '▶'),
        ("Cross-Dataset Val", 5.5, 1.5, '#9e9e9e', '○'),
    ]

    for label, start, duration, color, status in actions:
        ax.barh(0, duration, left=start, height=0.4, color=color, edgecolor='white', linewidth=1.5, alpha=0.85)
        mid = start + duration / 2
        ax.text(mid, 0, f"{status} {label}", ha='center', va='center', fontsize=8.5,
                fontweight='bold', color='white')

    ax.set_xlim(-0.2, 7.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Weeks →", fontsize=11, fontweight='bold')
    ax.set_title("Action Plan: Next 2–4 Weeks", fontsize=13, fontweight='bold', pad=15)
    ax.set_yticks([])

    # Legend
    legend_elements = [
        mpatches.Patch(color='#4caf50', label='Complete'),
        mpatches.Patch(color='#ff9800', label='In Progress'),
        mpatches.Patch(color='#2196f3', label='Priority'),
        mpatches.Patch(color='#9e9e9e', label='Abandoned / Pending'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_action_timeline.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_action_timeline.png")


if __name__ == "__main__":
    draw_competitive_landscape()
    draw_stratified_waterfall()
    draw_dann_results()
    draw_task_comparison()
    draw_action_timeline()
    print("\nAll figures generated successfully!")
