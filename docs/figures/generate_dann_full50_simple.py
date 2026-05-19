"""
Simple DANN vs Baseline comparison (Full-50, conceptual).
Minimal design, no extra annotations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = "docs/figures"


def draw_dann_full50_simple():
    fig, ax = plt.subplots(figsize=(7, 5.5))

    configs = ["Baseline\n(Full-50, N=50)", "DANN\n(Full-50, N=50)"]
    accuracies = [51.46, 50.52]
    colors = ['#4caf50', '#ef9a9a']

    bars = ax.bar(configs, accuracies, color=colors, edgecolor='white',
                  linewidth=2, width=0.5)

    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color='#333333')

    ax.axhline(y=50.0, color='#bdbdbd', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_ylabel("Mean Accuracy (%)", fontsize=13, fontweight='bold')
    ax.set_title("DANN Fails to Improve on Full-50 Cohort",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(48, 54)

    legend_elements = [
        mpatches.Patch(color='#4caf50', label='Baseline (ADFCNN)'),
        mpatches.Patch(color='#ef9a9a', label='DANN (Domain-Adversarial)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_dann_full50_simple.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_dann_full50_simple.png")


if __name__ == "__main__":
    draw_dann_full50_simple()
    print("Done!")
