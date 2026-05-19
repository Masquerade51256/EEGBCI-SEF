"""
Generate DANN vs Baseline comparison figure for Full-50 context.
Includes actual data with clear source annotations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = "docs/figures"


def draw_dann_full_comparison():
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Data
    configs = [
        "Baseline\n(Full-50, N=50)",
        "DANN\n(15-subj subset)",
        "Baseline\n(CleanC, N=14)",
        "DANN\n(CleanC, N=14)",
    ]
    accuracies = [51.46, 50.47, 50.57, 50.14]
    stds = [5.23, 3.68, None, 2.26]
    colors = ['#4caf50', '#ef9a9a', '#757575', '#ef9a9a']
    edge_colors = ['#2e7d32', '#c62828', '#424242', '#c62828']

    bars = ax.bar(configs, accuracies, color=colors, edgecolor=edge_colors,
                  linewidth=2, width=0.6)

    # Add error bars where available
    for bar, acc, std in zip(bars, accuracies, stds):
        if std is not None:
            ax.errorbar(bar.get_x() + bar.get_width()/2, acc, yerr=std,
                        fmt='none', ecolor='#333333', capsize=6, capthick=1.5)

    # Add value labels
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        label = f'{acc:.2f}%'
        if std is not None:
            label += f'\n±{std:.2f}%'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='#333333')

    # Reference line for Baseline Full-50
    ax.axhline(y=51.46, color='#4caf50', linestyle='--', linewidth=2, alpha=0.6)
    ax.text(3.3, 51.8, "Full-50 Baseline = 51.46%", fontsize=9,
            color='#2e7d32', ha='right', fontweight='bold')

    # Annotations for DANN bars
    ax.annotate('No improvement\nvs. baseline',
                xy=(1, 50.47), xytext=(1.3, 48.5),
                fontsize=9, color='#c62828', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.5),
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#c62828'))

    ax.annotate('No improvement\nvs. baseline',
                xy=(3, 50.14), xytext=(2.7, 48.5),
                fontsize=9, color='#c62828', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.5),
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#c62828'))

    # Bottom annotation explaining data source
    ax.text(0.5, 0.02,
            "Note: Full-50 DANN was run on a 15-subject representative subset due to computational cost. "
            "CleanC DANN used domain_loss_weight=0.3.",
            transform=ax.transAxes, fontsize=8, color='#666666',
            ha='left', va='bottom', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#bdbdbd'))

    ax.set_ylabel("Mean Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title("DANN Fails to Improve Across Both Full-Cohort and CleanC Settings\n"
                 "Domain-Adversarial Learning is Misaligned with Clinical Stroke Decoding",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(45, 56)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#4caf50', label='Baseline (no domain adversarial)'),
        mpatches.Patch(color='#ef9a9a', label='DANN (domain adversarial)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.95)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_dann_full_comparison.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_dann_full_comparison.png")


if __name__ == "__main__":
    draw_dann_full_comparison()
    print("Done!")
